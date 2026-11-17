"""Rotating-Earth 3-DOF point-mass propagation used by the identification and the bank-schedule rollout."""

import math

import numpy as np
import torch
import torch.nn as nn


class RotatingEarth3DOFBaseline(nn.Module):
    """Parameter-free rotating-Earth 3-DOF propagation baseline.

    Equivalent lift/drag coefficients and bank angle are identified from the
    final observed state differences. They are then held fixed during an RK4
    rollout of the same rotating spherical-Earth point-mass equations used by
    the simulator. This is a transparent numerical physics baseline, not a
    claim to reproduce a published analytical predictor.
    """

    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        input_scaler_mean=None,
        input_scaler_scale=None,
        output_scaler_mean=None,
        output_scaler_scale=None,
        sampling_interval_s=1.0,
        identification_steps=16,
        integration_stride=8,
        **kwargs,
    ):
        super().__init__()
        if input_dim < 6 or output_dim != 3:
            raise ValueError("RotatingEarth3DOFBaseline requires 6-D state input.")
        arrays = (input_scaler_mean, input_scaler_scale, output_scaler_mean, output_scaler_scale)
        if any(value is None for value in arrays):
            raise ValueError("Physical scaler metadata is required for the 3-DOF baseline.")
        self.register_buffer("input_scaler_mean", torch.as_tensor(input_scaler_mean, dtype=torch.float32))
        self.register_buffer("input_scaler_scale", torch.as_tensor(input_scaler_scale, dtype=torch.float32))
        self.register_buffer("output_scaler_mean", torch.as_tensor(output_scaler_mean, dtype=torch.float32))
        self.register_buffer("output_scaler_scale", torch.as_tensor(output_scaler_scale, dtype=torch.float32))
        self.sampling_interval_s = float(sampling_interval_s)
        self.identification_steps = max(2, int(identification_steps))
        self.integration_stride = max(1, int(integration_stride))
        self.earth_radius = 6_378_000.0
        self.mu = 3.986004418e14
        self.omega = 7.292115e-5
        self.mass = 907.2
        self.reference_area = 0.4839
        # These atmosphere tables are immutable model constants. Registering
        # them once avoids rebuilding and transferring four tensors for every
        # RK4 dynamics evaluation (128 evaluations for a 256-step rollout at
        # the default integration stride).
        self.register_buffer(
            "_atmosphere_base_h",
            torch.tensor(
                [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_base_t",
            torch.tensor(
                [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_base_p",
            torch.tensor(
                [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3734],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_lapse",
            torch.tensor(
                [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _safe_signed(value, floor=1e-8):
        sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
        return sign * value.abs().clamp_min(float(floor))

    @staticmethod
    def _wrapped_difference(value):
        delta = value[:, 1:] - value[:, :-1]
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _atmosphere(self, altitude):
        dtype, device = altitude.dtype, altitude.device
        base_h = self._atmosphere_base_h.to(device=device, dtype=dtype)
        base_t = self._atmosphere_base_t.to(device=device, dtype=dtype)
        base_p = self._atmosphere_base_p.to(device=device, dtype=dtype)
        lapse = self._atmosphere_lapse.to(device=device, dtype=dtype)
        h = altitude.clamp_min(0.0)
        layer = torch.bucketize(h, base_h[1:-1]).clamp(max=len(lapse) - 1)
        h0 = base_h[layer]
        t0 = base_t[layer]
        p0 = base_p[layer]
        rate = lapse[layer]
        gas_constant = 287.05287
        gravity = 9.80665
        temperature = t0 + rate * (h - h0)
        pressure_gradient = p0 * torch.pow(
            t0 / temperature.clamp_min(1.0), gravity / (gas_constant * self._safe_signed(rate, 1e-12))
        )
        pressure_isothermal = p0 * torch.exp(-gravity * (h - h0) / (gas_constant * t0))
        pressure = torch.where(rate.abs() < 1e-12, pressure_isothermal, pressure_gradient)
        above = h >= base_h[-1]
        pressure_above = base_p[-1] * torch.exp(
            -gravity * (h - base_h[-1]) / (gas_constant * base_t[-1])
        )
        pressure = torch.where(above, pressure_above, pressure)
        temperature = torch.where(above, base_t[-1], temperature)
        return pressure / (gas_constant * temperature)

    def _rotation_terms(self, state):
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        sin_gamma, cos_gamma = torch.sin(gamma), torch.cos(gamma)
        sin_heading, cos_heading = torch.sin(heading), torch.cos(heading)
        sin_lat, cos_lat = torch.sin(latitude), torch.cos(latitude)
        safe_speed = speed.clamp_min(1.0)
        safe_cos_gamma = self._safe_signed(cos_gamma)
        velocity_term = self.omega**2 * radius * cos_lat * (
            sin_gamma * cos_lat - cos_gamma * sin_lat * cos_heading
        )
        gamma_term = (
            2.0 * self.omega * cos_lat * sin_heading
            + self.omega**2 * radius * cos_lat / safe_speed
            * (cos_gamma * cos_lat + sin_gamma * sin_lat * cos_heading)
        )
        heading_term = (
            2.0 * self.omega * sin_lat
            + self.omega**2 * radius * sin_heading * sin_lat * cos_lat
            / (safe_speed * safe_cos_gamma)
            - 2.0 * self.omega * cos_lat * cos_heading * torch.tan(gamma)
        )
        return velocity_term, gamma_term, heading_term

    def _identify_controls(self, history):
        dt = self.sampling_interval_s
        diffs = history[:, 1:] - history[:, :-1]
        diffs[..., 1] = self._wrapped_difference(history[..., 1])
        diffs[..., 5] = self._wrapped_difference(history[..., 5])
        # Averaging wrapped finite differences is the deterministic secant
        # slope over the identification window and avoids CUDA median kernels.
        derivative = torch.mean(diffs / dt, dim=1)
        state = history[:, -1]
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        gravity = self.mu / radius.square()
        velocity_rotation, gamma_rotation, heading_rotation = self._rotation_terms(state)
        drag = -self.mass * (derivative[:, 3] + gravity * torch.sin(gamma) - velocity_rotation)
        gamma_base = -(gravity / speed - speed / radius) * torch.cos(gamma) + gamma_rotation
        heading_base = (
            speed * torch.cos(gamma) * torch.sin(heading) * torch.tan(latitude) / radius
            + heading_rotation
        )
        lift_cos_bank = self.mass * speed * (derivative[:, 4] - gamma_base)
        lift_sin_bank = (
            self.mass * speed * self._safe_signed(torch.cos(gamma))
            * (derivative[:, 5] - heading_base)
        )
        lift = torch.sqrt(lift_cos_bank.square() + lift_sin_bank.square())
        bank = torch.atan2(lift_sin_bank, lift_cos_bank)
        density = self._atmosphere(radius - self.earth_radius)
        dynamic_area = 0.5 * density * speed.square() * self.reference_area
        # Explicit admissibility bounds protect inverse identification from
        # sensor-difference noise; these are reported as part of the baseline.
        cl = (lift / dynamic_area.clamp_min(1e-6)).clamp(0.0, 2.0)
        cd = (drag / dynamic_area.clamp_min(1e-6)).clamp(1e-4, 2.0)
        return cl, cd, bank

    def _dynamics(self, state, cl, cd, bank):
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        safe_speed = speed.clamp_min(1.0)
        safe_cos_lat = self._safe_signed(torch.cos(latitude))
        safe_cos_gamma = self._safe_signed(torch.cos(gamma))
        density = self._atmosphere(radius - self.earth_radius)
        dynamic_pressure = 0.5 * density * speed.square()
        lift = cl * dynamic_pressure * self.reference_area
        drag = cd * dynamic_pressure * self.reference_area
        gravity = self.mu / radius.square()
        velocity_rotation, gamma_rotation, heading_rotation = self._rotation_terms(state)
        return torch.stack(
            [
                speed * torch.sin(gamma),
                speed * torch.cos(gamma) * torch.sin(heading) / (radius * safe_cos_lat),
                speed * torch.cos(gamma) * torch.cos(heading) / radius,
                -drag / self.mass - gravity * torch.sin(gamma) + velocity_rotation,
                lift * torch.cos(bank) / (self.mass * safe_speed)
                - (gravity / safe_speed - safe_speed / radius) * torch.cos(gamma)
                + gamma_rotation,
                lift * torch.sin(bank) / (self.mass * safe_speed * safe_cos_gamma)
                + safe_speed * torch.cos(gamma) * torch.sin(heading) * torch.tan(latitude) / radius
                + heading_rotation,
            ],
            dim=-1,
        )

    def forward(self, src, tgt=None, tgt_mask=None, target_length=None, decoder_context=None):
        del tgt_mask, decoder_context
        if target_length is None:
            if tgt is None:
                raise ValueError("target_length or tgt is required.")
            target_length = tgt.size(1)
        target_length = int(target_length)
        mean = self.input_scaler_mean.to(device=src.device, dtype=src.dtype)
        scale = self.input_scaler_scale.to(device=src.device, dtype=src.dtype)
        history = src[:, -min(self.identification_steps + 1, src.size(1)):, :6] * scale + mean
        cl, cd, bank = self._identify_controls(history)
        state = history[:, -1]
        output_segments = []
        base_dt = self.sampling_interval_s
        completed_steps = 0
        while completed_steps < target_length:
            segment_steps = min(self.integration_stride, target_length - completed_steps)
            dt = base_dt * float(segment_steps)
            start_state = state
            k1 = self._dynamics(start_state, cl, cd, bank)
            k2 = self._dynamics(start_state + 0.5 * dt * k1, cl, cd, bank)
            k3 = self._dynamics(start_state + 0.5 * dt * k2, cl, cd, bank)
            k4 = self._dynamics(start_state + dt * k3, cl, cd, bank)
            state = start_state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            state = torch.cat(
                [
                    state[:, :1],
                    torch.atan2(torch.sin(state[:, 1:2]), torch.cos(state[:, 1:2])),
                    state[:, 2:5],
                    torch.atan2(torch.sin(state[:, 5:6]), torch.cos(state[:, 5:6])),
                ],
                dim=-1,
            )
            delta = state - start_state
            delta = torch.cat(
                [
                    delta[:, :1],
                    torch.atan2(torch.sin(delta[:, 1:2]), torch.cos(delta[:, 1:2])),
                    delta[:, 2:5],
                    torch.atan2(torch.sin(delta[:, 5:6]), torch.cos(delta[:, 5:6])),
                ],
                dim=-1,
            )
            fractions = (
                torch.arange(1, segment_steps + 1, device=state.device, dtype=state.dtype)
                / float(segment_steps)
            ).view(1, segment_steps, 1)
            interpolated = start_state.unsqueeze(1) + fractions * delta.unsqueeze(1)
            output_segments.append(interpolated[..., :3])
            completed_steps += segment_steps
        physical = torch.cat(output_segments, dim=1)
        # A single synchronized check preserves the numerical guard without
        # forcing a CPU/GPU barrier after every integration segment.
        if not torch.isfinite(physical).all():
            raise FloatingPointError("Non-finite rotating-Earth 3-DOF rollout.")
        output_mean = self.output_scaler_mean.to(device=src.device, dtype=src.dtype)
        output_scale = self.output_scaler_scale.to(device=src.device, dtype=src.dtype)
        return (physical - output_mean) / output_scale


