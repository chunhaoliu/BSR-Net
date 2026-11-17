import numpy as np
import torch


def test_ecef_state_round_trip():
    from utils.filter_prior import ecef_to_state, spherical_to_ecef

    state = torch.tensor([[6_438_000.0, 0.3, 0.4, 6_000.0, -0.02, 1.2]], dtype=torch.float64)
    r, lon, lat, v, gamma, psi = state.unbind(-1)
    east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), -1)
    north = torch.stack((-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)), -1)
    pos = spherical_to_ecef(state)
    up = pos / r[:, None]
    vel = v[:, None] * (torch.cos(gamma)[:, None] * (torch.sin(psi)[:, None] * east + torch.cos(psi)[:, None] * north) + torch.sin(gamma)[:, None] * up)
    torch.testing.assert_close(ecef_to_state(pos, vel), state)


def test_position_measurements_are_deterministic():
    from utils.filter_prior import position_measurements

    clean = np.zeros((2, 8, 6), dtype=np.float32)
    clean[..., 0], clean[..., 1], clean[..., 2] = 6_438_000.0, 0.2, 0.3
    first = position_measurements(clean, 50.0, seed=123)
    np.testing.assert_array_equal(first, position_measurements(clean, 50.0, seed=123))
    assert first.shape == (2, 8, 3) and 20.0 < (first - position_measurements(clean, 0.0, seed=0)).std() < 80.0


def test_fit_recovers_noise_free_model_trajectory():
    from utils.filter_prior import FitConfig, _trajectory, fit_track, make_physics, rollout, spherical_to_ecef

    physics = make_physics("cpu")
    cl, cd, bank = 0.4, 0.15, 0.3
    theta = torch.tensor([[6_418_000.0, 0.1, 0.2, 5_000.0, -0.002, 1.0, cl * np.cos(bank), cl * np.sin(bank), cd]], dtype=torch.float64)
    states = _trajectory(physics, theta, 40, 1.0)
    fitted = fit_track(physics, spherical_to_ecef(states[..., :3]), FitConfig(fit_steps=40))
    assert fitted["fit_rms_m"].item() < 5.0
    assert abs(fitted["cl"].item() - cl) < 0.02 and abs(fitted["bank"].item() - bank) < 0.05
    pred = rollout(physics, fitted["terminal_state"], fitted["cl"], fitted["cd"], fitted["bank"], 16)
    assert pred.shape == (1, 16, 3) and torch.isfinite(pred).all()


def test_varma_forecast_continues_linear_trend_and_schedule_matches_rollout():
    from utils.filter_prior import make_physics, rollout, rollout_schedule, varma_forecast

    t = torch.arange(12.0, dtype=torch.float64)
    series = torch.stack((0.3 + 0.01 * t, 0.1 + 0.0 * t, -0.5 + 0.02 * t), -1)[None]
    for p, q in ((1, 0), (2, 0), (1, 1)):
        future = varma_forecast(series, 4, p=p, q=q, d=1)
        torch.testing.assert_close(future, series[:, -1:] + torch.arange(1.0, 5.0, dtype=torch.float64)[None, :, None] * (series[:, 1] - series[:, 0])[:, None])
    physics = make_physics("cpu").double()
    state = torch.tensor([[6_438_000.0, 0.3, 0.4, 6_000.0, -0.002, 1.2]], dtype=torch.float64)
    cl, cd, bank = torch.tensor([0.4], dtype=torch.float64), torch.tensor([0.15], dtype=torch.float64), torch.tensor([0.3], dtype=torch.float64)
    torch.testing.assert_close(rollout_schedule(physics, state, cl[:, None].repeat(1, 4), cd[:, None].repeat(1, 4), bank[:, None].repeat(1, 4), 16),
                               rollout(physics, state, cl, cd, bank, 64))
