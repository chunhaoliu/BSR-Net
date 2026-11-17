import math
import os

import torch
import pytest


def test_zero_initialized_residual_reproduces_prior():
    from models.bsrnet import ResidualAST
    from scripts.pipeline import to_prediction

    model = ResidualAST().eval()
    history, prior_n, features = torch.randn(2, 256, 6), torch.randn(2, 256, 3), torch.randn(2, 10)
    prior_km, scale = 100 * torch.randn(2, 256, 3), torch.rand(256, 3) + 0.5
    with torch.no_grad():
        out, _ = model(history, prior_n, features)
    torch.testing.assert_close(to_prediction(out, prior_km, {"res_std": scale}, direct=False), prior_km, rtol=0.0, atol=0.0)


def test_untrained_bsrnet_returns_zero_bank_offset_and_zero_correction():
    from models.bsrnet import ManeuverAST, count_parameters

    model = ManeuverAST(series_dim=6).eval()
    history, features, series = torch.randn(2, 256, 6), torch.randn(2, 10), torch.randn(2, 15, 6)
    with torch.no_grad():
        memory, offset = model.encode(history, features, series)
        out = model.correct(memory, torch.randn(2, 256, 3), torch.zeros(2, 256, 3))
    assert offset.shape == (2, 32) and not offset.any() and not out.any()
    assert abs(count_parameters(model) - 1.32e6) < 0.01e6  # paper: 1.32 M parameters


def test_residual_model_stays_under_parameter_budget():
    from models.bsrnet import ResidualAST, count_parameters

    assert count_parameters(ResidualAST()) <= 2_000_000


@pytest.mark.parametrize("cls", [
    pytest.param("af_ciln", id="af-ciln"),
    pytest.param("pit", id="pit"),
])
def test_hgv_baseline_forward_shape_and_parameter_budget(cls):
    from models.hgv_baselines import AFCILN, PIT, count_parameters
    model = {"af_ciln": AFCILN, "pit": PIT}[cls]().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 256, 6), None, None)
    assert out.shape == (2, 256, 3 if cls == "af_ciln" else 5)  # PIT adds (V, gamma) for its physics loss
    assert count_parameters(model) <= 6_000_000


def test_pit_filter_recovers_speed_and_climb_angle():
    from models.hgv_baselines import PIT
    model = PIT().eval()
    model.radius_km = 1e12  # flat Earth for this check
    t = torch.arange(-255.0, 1.0)
    v, g = 2.0, 0.01  # km/s, rad
    pos = torch.stack((v * math.cos(g) * t, torch.zeros_like(t), v * math.sin(g) * t), -1)
    history = torch.cat((pos, torch.zeros_like(pos)), -1)[None]  # identity normalisation (mean 0, std 1)
    smooth, speed, gamma = model.filtered_track(history)
    torch.testing.assert_close(smooth, pos[None], atol=1e-3, rtol=0)
    torch.testing.assert_close(speed, torch.full_like(speed, v), atol=1e-4, rtol=0)
    torch.testing.assert_close(gamma, torch.full_like(gamma, g), atol=1e-4, rtol=0)


def test_pit_physics_loss_vanishes_on_consistent_track():
    from models.hgv_baselines import pit_physics_loss
    t = torch.arange(256.0)
    v, g = torch.full((1, 256), 5.0), torch.full((1, 256), -0.002)
    pred = torch.stack((v * torch.cos(g) * t, torch.zeros_like(t)[None], v * torch.sin(g) * t), -1)  # straight, R -> inf
    vg = torch.stack((v, g), -1)
    mean, std = torch.tensor([5.0, 0.0]), torch.tensor([1.0, 0.01])
    out = torch.cat((pred, (vg - mean) / std), -1)
    assert pit_physics_loss(out, pred, vg, mean, std, vz_std=0.01, radius_km=1e12).item() < 1e-8


@pytest.mark.parametrize("variant", ["transformer", "dlinear", "patchtst", "itransformer"])
def test_tslib_baseline_forward_shape(variant):
    from models.public_baselines import TSLibForecastAdapter, TSLIB_ADAPTER_DEFAULTS
    try:
        model = TSLibForecastAdapter(model_type=variant, source_root=os.getenv("HGV_TSLIB_ROOT", "third_party/Time-Series-Library"),
                                     **TSLIB_ADAPTER_DEFAULTS[variant]).eval()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("official TSLib checkout unavailable")
    with torch.no_grad():
        out = model(torch.randn(2, 256, 6))
    assert out.shape == (2, 256, 3)
