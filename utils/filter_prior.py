"""Physics prior from noisy ECEF position-only tracks.

A batched Levenberg-Marquardt fit of the rotating-Earth 3-DOF model to the
last ``fit_steps`` position measurements estimates the kinematic state and
constant lift/drag/bank (a nonlinear least-squares tracker).  The fitted model
is then rolled out exactly like the existing analytical prior.  The dynamics
are shared with :class:`RotatingEarth3DOFBaseline`; nothing is duplicated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from models.dynamics import RotatingEarth3DOFBaseline


@dataclass(frozen=True)
class FitConfig:
    fit_steps: int = 64
    iterations: int = 20
    rollout_stride: int = 8
    dt_s: float = 1.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def make_physics(device: torch.device | str) -> RotatingEarth3DOFBaseline:
    zeros, ones = np.zeros(6, np.float32), np.ones(6, np.float32)
    physics = RotatingEarth3DOFBaseline(
        input_scaler_mean=zeros,
        input_scaler_scale=ones,
        output_scaler_mean=zeros[:3],
        output_scaler_scale=ones[:3],
    )
    return physics.to(device).eval()


def spherical_to_ecef(states: torch.Tensor) -> torch.Tensor:
    r, lon, lat = states[..., 0], states[..., 1], states[..., 2]
    return torch.stack((r * torch.cos(lat) * torch.cos(lon), r * torch.cos(lat) * torch.sin(lon), r * torch.sin(lat)), dim=-1)


def ecef_to_state(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """ECEF position/velocity (rotating frame) -> [r, lon, lat, V, gamma, psi]."""
    x, y, z = position.unbind(dim=-1)
    r = position.norm(dim=-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z / r)
    east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), dim=-1)
    north = torch.stack((-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)), dim=-1)
    up = position / r[..., None]
    ve, vn, vu = (velocity * east).sum(-1), (velocity * north).sum(-1), (velocity * up).sum(-1)
    speed = velocity.norm(dim=-1).clamp_min(1.0)
    return torch.stack((r, lon, lat, speed, torch.asin((vu / speed).clamp(-1, 1)), torch.atan2(ve, vn)), dim=-1)


def position_measurements(clean_states: np.ndarray, sigma_m: float, *, seed: int) -> np.ndarray:
    """Clean spherical states ``[..., 6]`` -> noisy ECEF positions ``[..., 3]`` (float64, m)."""
    positions = spherical_to_ecef(torch.from_numpy(np.asarray(clean_states, np.float64))).numpy()
    rng = np.random.default_rng(int(seed))
    return positions + rng.normal(0.0, float(sigma_m), size=positions.shape)


def _wrap(state: torch.Tensor) -> torch.Tensor:
    wrapped = torch.atan2(torch.sin(state), torch.cos(state))
    keep = torch.arange(6, device=state.device) % 4 == 1  # indices 1 (lon) and 5 (heading)
    return torch.where(keep, wrapped, state)  # out of place so autograd can pass through (ours2)


def _aero(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """theta[..., 6:9] = (C_L cos(bank), C_L sin(bank), C_D) -> (C_L, C_D, bank)."""
    lift_cos, lift_sin = theta[..., 6], theta[..., 7]
    return torch.sqrt(lift_cos.square() + lift_sin.square() + 1e-12), theta[..., 8].clamp_min(1e-4), torch.atan2(lift_sin, lift_cos)


def _rk4(physics, state, cl, cd, bank, dt):
    k1 = physics._dynamics(state, cl, cd, bank)
    k2 = physics._dynamics(state + 0.5 * dt * k1, cl, cd, bank)
    k3 = physics._dynamics(state + 0.5 * dt * k2, cl, cd, bank)
    k4 = physics._dynamics(state + dt * k3, cl, cd, bank)
    return _wrap(state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def _trajectory(physics, theta: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
    """States at 0..steps-1 from initial state theta[:, :6] with constant aero."""
    cl, cd, bank = _aero(theta)
    state, states = theta[:, :6], [theta[:, :6]]
    for _ in range(steps - 1):
        state = _rk4(physics, state, cl, cd, bank, dt)
        states.append(state)
    return torch.stack(states, dim=1)


def _initial_theta(physics, measurements: torch.Tensor, dt: float) -> torch.Tensor:
    """Quadratic ECEF fit for state at the first sample; aero from its secant derivative."""
    steps = measurements.shape[1]
    t = torch.arange(steps, device=measurements.device, dtype=measurements.dtype) * dt
    basis = torch.stack((torch.ones_like(t), t, t.square()), dim=1)
    coeff = torch.linalg.lstsq(basis.expand(measurements.shape[0], -1, -1), measurements).solution
    smooth_pos = basis @ coeff
    smooth_vel = torch.stack((torch.zeros_like(t), torch.ones_like(t), 2.0 * t), dim=1) @ coeff
    states = ecef_to_state(smooth_pos, smooth_vel)
    cl, cd, bank = physics._identify_controls(states[:, -min(17, steps):])
    cl = torch.nan_to_num(cl, nan=0.4).clamp(0.05, 1.0)
    cd = torch.nan_to_num(cd, nan=0.15).clamp(0.02, 0.5)
    bank = torch.nan_to_num(bank, nan=0.0)
    return torch.cat((states[:, 0], (cl * torch.cos(bank))[:, None], (cl * torch.sin(bank))[:, None], cd[:, None]), dim=1)


_TYPICAL = (1.0e6, 1.0e-1, 1.0e-1, 1.0e3, 1.0e-2, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1)


@torch.no_grad()
def fit_track(physics, measurements: torch.Tensor, config: FitConfig, init: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Fit the 3-DOF model to ``measurements`` [B, L, 3] (ECEF m); returns terminal state and aero.
    ``init``: warm start, e.g. ``warm_theta`` of the previous fit (theta = first-sample state + aero)."""
    meas = measurements.to(dtype=torch.float64)
    batch, steps, _ = meas.shape
    theta = _initial_theta(physics, meas, config.dt_s) if init is None else init.to(meas)
    typical = torch.tensor(_TYPICAL, device=meas.device, dtype=meas.dtype)
    lam = torch.full((batch,), 1e-2, device=meas.device, dtype=meas.dtype)
    eye = torch.eye(9, device=meas.device, dtype=meas.dtype)

    def residual(params: torch.Tensor) -> torch.Tensor:
        reps = params.shape[0] // batch
        pred = spherical_to_ecef(_trajectory(physics, params, steps, config.dt_s)[..., :3])
        return (pred - meas.repeat(reps, 1, 1)).reshape(params.shape[0], -1)

    res = residual(theta)
    cost = torch.nan_to_num(res.square().sum(dim=1), nan=float("inf"))
    for _ in range(int(config.iterations)):
        step = 1e-7 * torch.maximum(theta.abs(), typical)
        perturbed = theta[None].repeat(9, 1, 1)
        perturbed[torch.arange(9), :, torch.arange(9)] += step.T
        jac = ((residual(perturbed.reshape(9 * batch, 9)).reshape(9, batch, -1) - res[None]) / step.T[..., None]).permute(1, 2, 0)
        jtj = jac.transpose(1, 2) @ jac
        jtr = jac.transpose(1, 2) @ res[..., None]
        damping = lam[:, None, None] * (torch.diagonal(jtj, dim1=1, dim2=2)[:, :, None] * eye + 1e-12 * eye)
        delta = torch.linalg.solve(jtj + damping, -jtr).squeeze(-1)
        trial = theta + delta
        trial[:, 3] = trial[:, 3].clamp_min(100.0)
        trial[:, 8] = trial[:, 8].clamp(1e-4, 2.0)
        trial_res = residual(trial)
        trial_cost = trial_res.square().sum(dim=1)
        better = torch.isfinite(trial_cost) & (trial_cost < cost)
        theta = torch.where(better[:, None], trial, theta)
        res = torch.where(better[:, None], trial_res, res)
        cost = torch.where(better, trial_cost, cost)
        lam = torch.where(better, lam / 3.0, lam * 4.0).clamp(1e-9, 1e9)
    terminal = _trajectory(physics, theta, steps, config.dt_s)[:, -1]
    cl, cd, bank = _aero(theta)
    rms_m = torch.sqrt(cost / (3 * steps))
    return {"terminal_state": terminal, "cl": cl, "cd": cd, "bank": bank, "fit_rms_m": rms_m, "theta": theta}


def warm_theta(physics, theta: torch.Tensor, shift: int, dt: float = 1.0) -> torch.Tensor:
    """Warm start for a window ``shift`` samples later: propagate the first-sample state, keep the aero."""
    return torch.cat((_trajectory(physics, theta, shift + 1, dt)[:, -1], theta[:, 6:]), 1)


def _segment(physics, start, cl, cd, bank, segment: int, dt: float):
    """One RK4 step over ``segment`` samples with linear interpolation in between -> (end state, [B, segment, 3])."""
    state = _rk4(physics, start, cl, cd, bank, dt * segment)
    delta = state[:, :3] - start[:, :3]
    delta = torch.stack((delta[:, 0], torch.atan2(torch.sin(delta[:, 1]), torch.cos(delta[:, 1])), delta[:, 2]), 1)
    frac = (torch.arange(1, segment + 1, device=state.device, dtype=state.dtype) / segment).view(1, segment, 1)
    return state, start[:, None, :3] + frac * delta[:, None, :]


@torch.no_grad()
def rollout(physics, state: torch.Tensor, cl, cd, bank, horizon: int, stride: int = 8, dt: float = 1.0) -> torch.Tensor:
    """Same segment-RK4 + linear-interpolation rollout as RotatingEarth3DOFBaseline; returns [B, H, 3]."""
    outputs, done = [], 0
    while done < horizon:
        segment = min(stride, horizon - done)
        state, positions = _segment(physics, state, cl, cd, bank, segment, dt)
        outputs.append(positions)
        done += segment
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def rollout_schedule(physics, state: torch.Tensor, cl, cd, bank, block: int, stride: int = 8, dt: float = 1.0) -> torch.Tensor:
    """``rollout`` with piecewise-constant aero: column k of cl / cd / bank [B, K] holds during block k."""
    outputs = []
    for k in range(cl.shape[1]):
        for _ in range(block // stride):
            state, positions = _segment(physics, state, cl[:, k], cd[:, k], bank[:, k], stride, dt)
            outputs.append(positions)
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def parameter_series(physics, measurements: torch.Tensor, config: FitConfig, every: int = 16):
    """Sliding ``fit_track`` windows ending every ``every`` samples (the last one at the newest sample).

    Returns (C_L, C_D, unwrapped bank) [B, K, 3] in time order and the newest fit (identical to the A_fit fit)."""
    ends = list(range(measurements.shape[1], config.fit_steps - 1, -every))[::-1]
    fits = [fit_track(physics, measurements[:, e - config.fit_steps:e], config) for e in ends]
    series = torch.stack([torch.stack((f["cl"], f["cd"], f["bank"]), -1) for f in fits], 1)
    turns = torch.diff(series[..., 2], dim=1)
    series[:, 1:, 2] = series[:, :1, 2] + torch.cumsum(torch.atan2(torch.sin(turns), torch.cos(turns)), 1)
    return series, fits[-1]


def varma_forecast(series: torch.Tensor, steps: int, p: int = 1, q: int = 0, d: int = 1, ridge: float = 1e-2) -> torch.Tensor:
    """Batched VARMA(p, q) forecast of a d-times differenced vector series [B, K, C] -> [B, steps, C].

    Per window: difference d times, standardise each channel, estimate the coefficient matrices by
    ridge least squares (Hannan-Rissanen two-stage regression when q > 0: innovations from a long
    VAR first), forecast recursively with zero future innovations, then undo the scaling and differences."""
    levels, x = [], series.double()
    for _ in range(d):
        levels.append(x[:, -1])
        x = x[:, 1:] - x[:, :-1]
    mean, scale = x.mean(1, keepdim=True), x.std(1, keepdim=True).clamp_min(1e-6)
    z = (x - mean) / scale
    batch, length, channels = z.shape

    def fit(target_from: int, lags: list[torch.Tensor]):
        design = torch.cat([lag[:, target_from - k:length - k] for lag, k in lags], -1)
        y = z[:, target_from:]
        gram = design.transpose(1, 2) @ design + ridge * torch.eye(design.shape[-1], dtype=z.dtype)
        return torch.linalg.solve(gram, design.transpose(1, 2) @ y)

    eps = torch.zeros_like(z)
    if q:
        m = p + q
        coef = fit(m, [(z, k) for k in range(1, m + 1)])
        design = torch.cat([z[:, m - k:length - k] for k in range(1, m + 1)], -1)
        eps[:, m:] = z[:, m:] - design @ coef
    start = max(p, 2 * q + p if q else 0)
    lags = [(z, k) for k in range(1, p + 1)] + [(eps, k) for k in range(1, q + 1)]
    coef = fit(start, lags)
    hist_z, hist_e, out = list(z.unbind(1)), list(eps.unbind(1)), []
    for _ in range(steps):
        regressors = [hist_z[-k] for k in range(1, p + 1)] + [hist_e[-k] for k in range(1, q + 1)]
        nxt = (torch.cat(regressors, -1)[:, None] @ coef)[:, 0]
        hist_z.append(nxt); hist_e.append(torch.zeros_like(nxt)); out.append(nxt)
    forecast = torch.stack(out, 1) * scale + mean
    for level in reversed(levels):
        forecast = level[:, None] + torch.cumsum(forecast, 1)
    return forecast.to(series.dtype)


@torch.no_grad()
def differenced_states(measurements: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Naive position-only states: backward-differenced ECEF velocity (reference for the old prior)."""
    meas = measurements.to(dtype=torch.float64)
    velocity = (meas[:, 1:] - meas[:, :-1]) / dt
    return ecef_to_state(meas[:, 1:], velocity)
