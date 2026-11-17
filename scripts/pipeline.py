#!/usr/bin/env python3
"""BSR-Net pipeline: identification cache, training, evaluation, and the physics baselines.

Subcommands
  cache     batch LM identification (A_fit, utils/filter_prior.py) for one split and some sigmas -> OUT/cache
  analytic  sliding-fit parameter series (needed by BSR-Net, CPU) and the VARMA-PM baseline
            (--select orders on val, --evaluate on the test set)
  train     train one model and keep the best-on-validation checkpoint (BSR-Net: --variant ours2s --aux 1)
  evaluate  evaluate the baseline variants on the test set
  stress    evaluate A_fit and every trained method on the test set or a density-mismatch set -> <split>_summary.csv
  errors    per-window test errors for significance tests and error-vs-time curves -> holdout_errors.npz
  latency   network and identification timing -> latency.csv

Variant names (paper name): ours2s (BSR-Net), ours2s_nb (w/o bank-schedule rollout), ours2 (w/o identified-parameter
sequence), ours (w/o both, residual on prior), direct (w/o physics prior); --aux is the bank-term weight lambda.
Train windows: every train trajectory, origins 0..488 with stride 5 (val/holdout keep their stored windows).
Each epoch draws a noise level per train window from SIGMAS; noise realisations use fixed per-sigma seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.bsrnet import ManeuverAST, ResidualAST, count_parameters
from models.hgv_baselines import AFCILN, PIT, PriorInput, pit_physics_loss
from models.public_baselines import TSLibForecastAdapter, TSLIB_ADAPTER_DEFAULTS, TSLIB_OFFICIAL_COMMIT
from utils.filter_prior import (_segment, FitConfig, fit_track, make_physics, parameter_series, position_measurements, rollout, rollout_schedule,
                                spherical_to_ecef, varma_forecast)

DATA = Path(os.getenv("HGV_PROCESSED_DIR", PROJECT_ROOT / "data_generation" / "data" / "processed"))
OUT = Path(os.getenv("HGV_OUT", PROJECT_ROOT / "outputs"))
SIGMAS = (0, 50, 100, 200)
HORIZONS = (32, 64, 128, 256)
FIT_STEPS = {0: 24, 50: 48, 100: 64, 200: 96}
NOISE_SEED = {"train": 20000, "val": 120000, "holdout": 220000, "mismatch15": 220000, "mismatch30": 220000}  # stress sets: holdout noise
TRAIN_STRIDE = 5
SEEDS = (42, 123, 456)
AUX_WEIGHTS = (0.1, 0.3)  # weight 0 is the noaux variant
BASELINE_VARIANTS = ("transformer", "dlinear", "patchtst", "itransformer", "af_ciln", "pit", "itransformer_prior", "itransformer_priorf", "itransformer_priors", "pit_prior")
BASELINE_LABELS = {"transformer": "Transformer", "dlinear": "DLinear", "patchtst": "PatchTST",
                   "itransformer": "iTransformer", "af_ciln": "AF-CILN", "pit": "PIT",
                   "itransformer_prior": "iTransformer+prior", "itransformer_priorf": "iTransformer+prior+feats",
                   "itransformer_priors": "iTransformer+prior+series", "pit_prior": "PIT+prior"}  # +prior: A_fit rollout in, residual out


def is_baseline(variant: str) -> bool:
    return variant in BASELINE_VARIANTS


def build_model(variant: str, device: torch.device, input_dim: int = 6, dropout: float = 0.1):
    if variant == "af_ciln":
        return AFCILN().to(device)
    if variant in ("ours2", "ours2s", "ours2s_nb"):  # ours2s also reads the identified parameter series;
        # ours2s_nb = ablation without the bank rollout (offset fixed at 0, so the shift is 0)
        return ManeuverAST(series_dim=0 if variant == "ours2" else 6, bank_scale=0.0 if variant == "ours2s_nb" else 0.6,
                           dropout=dropout).to(device)
    if variant in ("pit", "pit_prior"):
        return PIT(prior_input=variant == "pit_prior").to(device)
    if variant in ("itransformer_prior", "itransformer_priorf", "itransformer_priors"):  # 3 rollout + 6 history channels (+ feats, + series)
        return PriorInput(build_model("itransformer", device, input_dim=9), use_features=variant != "itransformer_prior",
                          use_series=variant == "itransformer_priors").to(device)
    if variant in {"transformer", "dlinear", "patchtst", "itransformer"}:
        kwargs = dict(TSLIB_ADAPTER_DEFAULTS[variant], input_dim=input_dim)
        if variant == "patchtst":
            kwargs.update(patch_len=16, stride=8)
        model = TSLibForecastAdapter(model_type=variant, source_root=os.getenv("HGV_TSLIB_ROOT", str(PROJECT_ROOT / "third_party" / "Time-Series-Library")), **kwargs)
        if variant == "patchtst":  # official PatchTST head_dropout default is 0; TSLib reuses dropout there, which
            model.external_model.head.dropout.p = 0.0  # zeroes whole forecast steps of our level-valued outputs in training
        return model.to(device)
    return ResidualAST(direct=variant == "direct", use_aux=variant == "ours").to(device)


def model_output(model, hist, prior_n, feats, series=None):
    if isinstance(model, TSLibForecastAdapter):
        return model(hist)
    return model(hist, prior_n, feats, series) if series is not None else model(hist, prior_n, feats)


def norm_series(model, e, stats):
    return (e["series"] - stats["series_mean"]) / stats["series_std"] if getattr(model, "use_series", False) else None


def load_split(split: str) -> dict[str, np.ndarray]:
    if split == "holdout" or split.startswith("mismatch"):  # mismatch<100A>: density-bias stress holdout
        with np.load(DATA / ("hgv_confirmatory_holdout" + ("" if split == "holdout" else "_" + split) + ".npz")) as d:
            ids = d["trajectory_ids_confirmatory"]
            return {"clean": d["clean_trajectories"], "rows": ids - ids.min(), "starts": d["window_starts_confirmatory"],
                    "strata": d["joint_strata_confirmatory"], "bank": d["bank_commands"], "bank_achieved": d["bank_achieved"]}
    with np.load(DATA / "hgv_multiregime_dataset_v2_1.npz") as d:
        rows, starts = d[f"trajectory_ids_{split}"], d[f"window_starts_{split}"]
        if split == "train":
            traj, grid = np.unique(rows), np.arange(0, 1000 - 512 + 1, TRAIN_STRIDE)
            rows, starts = np.repeat(traj, len(grid)), np.tile(grid, len(traj))
        return {"clean": d["clean_trajectories"], "rows": rows, "starts": starts, "strata": d["joint_strata"][rows],
                "bank": d["bank_commands"], "bank_achieved": d["bank_achieved"]}


def measurements(data, split, sigma):
    return position_measurements(data["clean"], sigma, seed=NOISE_SEED[split] + sigma)


def cache_path(split, sigma):
    return OUT / "cache" / f"{split}_sigma{sigma}.npz"


def build_cache(split, sigma, device, batch_size=4096):
    data, path = load_split(split), cache_path(split, sigma)
    if path.exists():
        with np.load(path) as c:
            if "starts" in c and np.array_equal(c["starts"], data["starts"]):
                return
    meas, physics, cfg = measurements(data, split, sigma), make_physics(device), FitConfig(fit_steps=FIT_STEPS[sigma])
    keys = ("terminal_state", "cl", "cd", "bank", "fit_rms_km", "rollout_ecef_m")
    parts = {k: [] for k in keys}
    for begin in range(0, len(data["starts"]), batch_size):
        r, s = data["rows"][begin:begin + batch_size, None], data["starts"][begin:begin + batch_size, None]
        fit = fit_track(physics, torch.from_numpy(meas[r, s + np.arange(256 - cfg.fit_steps, 256)]).to(device), cfg)
        pred = rollout(physics, fit["terminal_state"], fit["cl"], fit["cd"], fit["bank"], 256)
        for k, v in zip(keys, (fit["terminal_state"], fit["cl"], fit["cd"], fit["bank"], fit["fit_rms_m"] / 1000, spherical_to_ecef(pred))):
            parts[k].append(v.cpu().numpy().astype(np.float32))
        print(split, sigma, begin, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, starts=data["starts"], **{k: np.concatenate(v) for k, v in parts.items()})


def localize(points, origin):
    """ECEF m -> local ENU km at ``origin`` [N, 3]."""
    lon, lat = np.arctan2(origin[:, 1], origin[:, 0]), np.arcsin(origin[:, 2] / np.linalg.norm(origin, axis=-1))
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), -1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), -1)
    basis = np.stack((east, north, origin / np.linalg.norm(origin, axis=-1, keepdims=True)), 1)
    return np.einsum("nhj,nij->nhi", points - origin[:, None], basis) / 1000.0


def examples(data, split, sigma, meas=None):
    """Raw (unnormalised) arrays in local ENU km for every window of ``split`` at ``sigma``."""
    meas = measurements(data, split, sigma) if meas is None else meas
    with np.load(cache_path(split, sigma)) as c:
        cache = {k: c[k] for k in c.files}
    assert np.array_equal(cache["starts"], data["starts"]), f"stale cache {split} sigma {sigma}"
    r, s = data["rows"][:, None], data["starts"][:, None]
    origin = meas[data["rows"], data["starts"] + 255]
    hist = localize(meas[r, s + np.arange(256)], origin)
    hist = np.concatenate((hist, np.diff(hist, axis=1, prepend=hist[:, :1])), -1)
    target = localize(spherical_to_ecef(torch.from_numpy(data["clean"][r, s + 256 + np.arange(256), :3].astype(np.float64))).numpy(), origin)
    st = cache["terminal_state"]
    feats = np.stack((cache["cl"], cache["cd"], np.sin(cache["bank"]), np.cos(cache["bank"]), st[:, 3] / 1000, np.sin(st[:, 4]),
                      np.sin(st[:, 5]), np.cos(st[:, 5]), cache["fit_rms_km"], np.full(len(st), sigma / 1000)), 1)
    label = np.any(np.abs(np.diff(data["bank"][r, s + 256 + np.arange(256)], axis=1)) > 1e-5, axis=1)
    f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    bank_future = data["bank_achieved"][r, s + 256 + np.arange(256)].reshape(len(st), BANK_BLOCKS, -1).mean(-1)  # ours2 target
    ex = {"hist": f32(hist), "prior": f32(localize(cache["rollout_ecef_m"].astype(np.float64), origin)), "target": f32(target),
          "feats": f32(feats), "label": f32(label), "term": f32(st), "aero": f32(np.stack((cache["cl"], cache["cd"], cache["bank"]), 1)),
          "origin": f32(origin), "bank_future": f32(bank_future)}
    series = series_tokens(split, sigma, len(st))
    if series is not None:
        ex["series"] = f32(series)
    return ex


SERIES_LEN = 15  # ours2s: sliding fits every 16 s over the history (11-15 by sigma), left edge-padded


def series_tokens(split, sigma, n):
    """ours2s input: identified (C_L, C_D, sin/cos bank, bank change vs newest fit, valid flag) per 16 s fit, or None."""
    path = series_path(split, sigma, 5)
    if not path.exists():
        return None
    with np.load(path) as c:
        series = c["series"]
    assert len(series) == n, f"stale series {split} sigma {sigma}"
    pad = SERIES_LEN - series.shape[1]
    series = np.concatenate((np.repeat(series[:, :1], pad, 1), series), 1)
    bank, valid = series[..., 2], np.broadcast_to(np.arange(SERIES_LEN) >= pad, series.shape[:2])
    return np.stack((series[..., 0], series[..., 1], np.sin(bank), np.cos(bank), bank - bank[:, -1:], valid), -1)


BANK_BLOCKS = 32  # ours2: one bank value per 8 s of the 256 s horizon
SHIFT_BATCH = 512  # ours2: fixed batch of the CUDA-graphed rollout
_SHIFT = {}
_INFER = {}


def localize_t(points, origin):
    """Torch ``localize``: ECEF m [B, H, 3] -> local ENU km at ``origin`` [B, 3]."""
    lon, lat = torch.atan2(origin[:, 1], origin[:, 0]), torch.asin(origin[:, 2] / origin.norm(dim=-1))
    east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), -1)
    north = torch.stack((-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)), -1)
    basis = torch.stack((east, north, origin / origin.norm(dim=-1, keepdim=True)), 1)
    return torch.einsum("bhj,bij->bhi", points - origin[:, None], basis) / 1000.0


def bank_shift_fn(device):
    """CUDA-graphed fp64 rollout difference roll(held + offset) - roll(held) [SHIFT_BATCH, 256, 3] km.
    The ~10^5 small kernels of the 3-DOF RK4 make eager autograd launch-bound; one graph replay removes that."""
    if device not in _SHIFT:
        physics = make_physics(device).double()

        def shift(term, cl, cd, held, origin, offset):
            sched = torch.cat((held[:, None].expand_as(offset), held[:, None] + offset))
            state, cl2, cd2, out = torch.cat((term, term)), torch.cat((cl, cl)), torch.cat((cd, cd)), []
            for k in range(offset.shape[1]):
                state, pos = _segment(physics, state, cl2, cd2, sched[:, k], 256 // offset.shape[1], 1.0)
                out.append(pos)
            base, rolled = localize_t(spherical_to_ecef(torch.cat(out, 1)), torch.cat((origin, origin))).chunk(2)
            return rolled - base.detach()
        f64 = lambda *v: torch.tensor(v, device=device, dtype=torch.float64).expand(SHIFT_BATCH, len(v)).clone()
        sample = (f64(6.43e6, 0.1, 0.1, 5e3, 0.0, 0.5), f64(0.4)[:, 0], f64(0.15)[:, 0], f64(0.0)[:, 0], f64(5e6, 1e6, 1e6),
                  torch.zeros(SHIFT_BATCH, BANK_BLOCKS, device=device, dtype=torch.float64, requires_grad=True))
        _SHIFT[device] = shift, torch.cuda.make_graphed_callables(shift, sample)
    return _SHIFT[device]


def shift_infer(device, args):
    """Forward-only CUDA graph of the bank shift (no autograd), captured once per (device, batch size)."""
    key = (device, len(args[0]))
    if key not in _INFER:
        eager, static = bank_shift_fn(device)[0], [a.clone() for a in args]
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                eager(*static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = eager(*static)
        _INFER[key] = graph, static, out
    graph, static, out = _INFER[key]
    for buf, a in zip(static, args):
        buf.copy_(a)
    graph.replay()
    return out.clone()


def maneuver_forward(model, e, stats, hist, prior_n, feats):
    """ours2: predicted bank schedule -> differentiable 3-DOF rollout (identified C_L, C_D) -> residual decoder.
    The rollout enters as its difference to the held-bank rollout from the same state, so a zero bank offset
    reproduces the cached A_fit prior exactly.  Returns (prediction km, change score, bank blocks rad)."""
    series = norm_series(model, e, stats)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        memory, offset = model.encode(hist, feats, series)
    with torch.autocast("cuda", enabled=False):
        n, pad = len(offset), -len(offset) % SHIFT_BATCH
        args = [e["term"], e["aero"][:, 0], e["aero"][:, 1], e["aero"][:, 2], e["origin"], offset.float()]
        args = [torch.cat((a, a[:1].expand(pad, *a.shape[1:]))).double() for a in args]  # pad to the graph batch
        eager, graph = bank_shift_fn(hist.device)
        if torch.is_grad_enabled():  # graph outputs are static buffers: clone before the next replay
            shift = torch.cat([graph(*(a[b:b + SHIFT_BATCH] for a in args)).clone() for b in range(0, n + pad, SHIFT_BATCH)])
        else:  # the training graph needs its backward; inference replays a forward-only graph per chunk size
            shift = torch.cat([shift_infer(hist.device, [a[b:b + SHIFT_BATCH] for a in (args if n >= SHIFT_BATCH else [a[:n] for a in args])])
                               for b in range(0, n + pad if n >= SHIFT_BATCH else n, SHIFT_BATCH)])
        shift = shift[:n].float()
        bank = e["aero"][:, 2:3] + offset.float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.correct(memory, prior_n, shift / stats["res_std"])
    return e["prior"] + shift + out.float() * stats["res_std"], offset.float().abs().amax(1), bank.float()


def load_examples(split):
    data = load_split(split)
    return data, {sigma: examples(data, split, sigma) for sigma in SIGMAS}


def fit_stats(train):
    cat = lambda k: torch.cat([train[s][k] for s in SIGMAS]).double()
    hist, prior, target, feats = cat("hist"), cat("prior"), cat("target"), cat("feats")
    std = lambda x, dims: x.std(dim=dims).clamp_min(1e-6)
    return {k: v.float() for k, v in {
        "hist_mean": hist.mean((0, 1)), "hist_std": std(hist, (0, 1)),
        "prior_mean": prior.mean(0), "prior_std": std(prior, 0),
        "feats_mean": feats.mean(0), "feats_std": std(feats, 0),
        "res_std": (target - prior).square().mean(0).sqrt().clamp_min(1e-6),
        "dir_mean": target.mean(0), "dir_std": std(target, 0),
        **({"series_mean": cat("series").mean((0, 1)), "series_std": std(cat("series"), (0, 1))} if "series" in train[SIGMAS[0]] else {})}.items()}


def normalize(ex, stats):
    return ((ex["hist"] - stats["hist_mean"]) / stats["hist_std"], (ex["prior"] - stats["prior_mean"]) / stats["prior_std"],
            (ex["feats"] - stats["feats_mean"]) / stats["feats_std"])


def output_mode(model):
    """TSLib forecasters continue the history position channels (their RevIN de-normalises in that space)."""
    return "hist" if isinstance(model, TSLibForecastAdapter) else getattr(model, "direct", False)


def to_prediction(out, prior_km, stats, direct):
    """Model output -> local ENU km.  Residual: scale only, so a zero output is exactly the prior."""
    out = out[..., :3]  # PIT also predicts (V, gamma)
    if direct == "prior":  # "+prior" baselines: correction in prior-normalised units
        return prior_km + out * stats["prior_std"]
    if direct == "hist":
        return out * stats["hist_std"][:3] + stats["hist_mean"][:3]
    if direct:
        return out * stats["dir_std"] + stats["dir_mean"]
    return prior_km + out * stats["res_std"]


def predict(model, ex, stats, idx, device, batch=2048):
    """Errors [N, 256] (km) and logits [N] for windows ``idx``."""
    model.eval(); errs, logits = [], []
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            e = {k: v[j].to(device) for k, v in ex.items()}
            hist, prior_n, feats = normalize(e, stats)
            if getattr(model, "maneuver", False):
                pred, logit, _ = maneuver_forward(model, e, stats, hist, prior_n, feats)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model_output(model, hist, prior_n, feats, norm_series(model, e, stats))
                if isinstance(raw, (tuple, list)):
                    out, logit = raw
                else:
                    out, logit = raw, torch.zeros(hist.size(0), device=device)
                pred = to_prediction(out.float(), e["prior"], stats, output_mode(model))
            errs.append((pred - e["target"]).norm(dim=-1).cpu()); logits.append(logit.float().cpu())
    return torch.cat(errs).numpy(), torch.cat(logits).numpy()


def auroc(labels, scores):
    from scipy.stats import rankdata
    labels = labels.astype(bool); pos, neg = labels.sum(), (~labels).sum()
    return float((rankdata(scores)[labels].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))


def run_name(variant, aux, seed):
    return f"{variant}_aux{aux:g}_seed{seed}"


def train(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    _, tr = load_examples("train"); _, va = load_examples("val")
    stats = {k: v.to(device) for k, v in fit_stats(tr).items()}
    tr = {k: torch.stack([tr[s][k] for s in SIGMAS]).to(device) for k in tr[0]}  # [4, N, ...] on GPU
    n = tr["label"].shape[1] if args.limit is None else args.limit
    aux = args.aux if args.variant in ("ours", "ours2", "ours2s") else 0.0
    model = build_model(args.variant, device, dropout=args.dropout)
    direct = output_mode(model)
    if args.variant.startswith("pit"):  # PIT supervises (V km/s, gamma) and adds its physics residual
        data = load_split("train")
        t = data["starts"][:, None] + 256 + np.arange(256)
        vg = torch.from_numpy(np.stack((data["clean"][data["rows"][:, None], t, 3] / 1000, data["clean"][data["rows"][:, None], t, 4]),
                                       -1).astype(np.float32)).to(device)
        vg_mean, vg_std = vg.mean((0, 1)), vg.std((0, 1))
        vz_std = float((vg[..., 0] * torch.sin(vg[..., 1])).std())
        model.set_stats(stats["hist_mean"], stats["hist_std"], vg_mean, vg_std, stats["prior_mean"], stats["prior_std"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * -(-n // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    name = run_name(args.variant, aux, args.seed) + (f"_limit{args.limit}" if args.limit else "") + args.tag
    log = OUT / "logs" / f"{name}.jsonl"; log.parent.mkdir(parents=True, exist_ok=True); log.write_text("")
    val_idx = torch.arange(len(va[0]["label"]))
    best, stale, started = np.inf, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        sig = torch.from_numpy(rng.integers(0, len(SIGMAS), n)).to(device)
        perm = torch.from_numpy(rng.permutation(n)).to(device)
        for b in range(0, n, args.batch_size):
            j = perm[b:b + args.batch_size]; s = sig[j]
            e = {k: v[s, j] for k, v in tr.items()}
            hist, prior_n, feats = normalize(e, stats)
            if getattr(model, "maneuver", False):  # ours2: aux weight scales the bank-schedule loss (rad / 0.3)
                pred, logit, bank = maneuver_forward(model, e, stats, hist, prior_n, feats)
                loss = (pred - e["target"]).norm(dim=-1).mean() + aux * ((bank - e["bank_future"]) / 0.3).square().mean()
                aux_bce = False
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model_output(model, hist, prior_n, feats, norm_series(model, e, stats))
                if isinstance(raw, (tuple, list)):
                    out, logit = raw
                else:
                    out, logit = raw, torch.zeros(hist.size(0), device=device)
                pred = to_prediction(out.float(), e["prior"], stats, direct)
                loss = (pred - e["target"]).norm(dim=-1).mean()
                aux_bce = bool(aux)
            if args.variant.startswith("pit"):
                loss = loss + pit_physics_loss(out, pred, vg[j], vg_mean, vg_std, vz_std)
            if aux_bce:
                loss = loss + aux * torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), e["label"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "elapsed_s": time.time() - started}
        if args.limit:  # overfit check: error on the same training windows (sigma 0 slice)
            err, _ = predict(model, {k: v[0, :n] for k, v in tr.items()}, stats, torch.arange(n), device)
            rec["train_ade256_sigma0_km"] = float(err.mean())
        for sigma in SIGMAS:
            err, logit = predict(model, va[sigma], stats, val_idx, device)
            rec[f"val_ade256_sigma{sigma}_km"] = float(err.mean())
            rec[f"val_auroc_sigma{sigma}"] = auroc(va[sigma]["label"].numpy(), logit)
        rec["val_ade256_km"] = float(np.mean([rec[f"val_ade256_sigma{s}_km"] for s in SIGMAS]))
        print(json.dumps(rec), flush=True)
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["val_ade256_km"] < best:
            best, stale = rec["val_ade256_km"], 0
            if not args.limit:
                (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "stats": {k: v.cpu() for k, v in stats.items()}, "variant": args.variant,
                            "aux": aux, "seed": args.seed, "lr": args.lr, "epoch": epoch, "val_ade256_km": best}, OUT / "checkpoints" / f"{name}.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break


def best_val(name):
    recs = [json.loads(line) for line in (OUT / "logs" / f"{name}.jsonl").read_text().splitlines()]
    return min(r["val_ade256_km"] for r in recs), len(recs)


def summarize(err, logit, data, ex, **keys):
    label = ex["label"].numpy().astype(bool)
    groups = [("overall", np.ones(len(label), bool)), ("bank_change", label), ("no_bank_change", ~label)]
    groups += [(str(s), data["strata"] == s) for s in np.unique(data["strata"])]
    finite, rows = np.isfinite(err).all(1), []
    for name, mask in groups:
        ok = mask & finite
        for h in HORIZONS:
            rows.append({**keys, "group": name, "horizon_s": h, "ade_km": float(err[ok, :h].mean()), "fde_km": float(err[ok, h - 1].mean()),
                         "windows": int(mask.sum()), "diverged": int((mask & ~finite).sum())})
    return rows, groups


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def evaluate(args):
    device = torch.device("cuda")
    result_dir = OUT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_path, summary_path, tuning_path = result_dir / "metrics.csv", result_dir / "summary.csv", result_dir / "tuning.csv"
    existing = _read_csv(metric_path)
    present = {r["method"] for r in existing}
    requested = [v for v in BASELINE_VARIANTS if args.methods is None or v in args.methods]
    if args.only_missing:
        requested = [v for v in requested if BASELINE_LABELS[v] not in present]
    data, ho = load_examples("holdout")
    idx = torch.arange(len(data["starts"]))
    new_rows, tuning = [], _read_csv(tuning_path)
    afit = {s: (ho[s]["prior"] - ho[s]["target"]).norm(dim=-1).numpy() for s in SIGMAS}
    for variant in requested:
        label = BASELINE_LABELS[variant]
        seed42_val = None
        for seed in SEEDS:
            name = run_name(variant, 0, seed)
            ckpt_path = OUT / "checkpoints" / f"{name}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"missing baseline checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model = build_model(variant, device); model.load_state_dict(ckpt["model"]); model.eval()
            stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
            if seed == 42:
                seed42_val = ckpt.get("val_ade256_km", "")
            for sigma in SIGMAS:
                err, logit = predict(model, ho[sigma], stats, idx, device)
                rows, groups = summarize(err, logit, data, ho[sigma], method=label, seed=seed, sigma_m=sigma)
                new_rows += rows
                new_rows.append({"method": label, "seed": seed, "sigma_m": sigma, "group": "aux_auroc", "horizon_s": 256,
                                 "ade_km": auroc(ho[sigma]["label"].numpy(), logit), "fde_km": "", "windows": len(logit), "diverged": ""})
                win, base = err.mean(1), afit[sigma].mean(1)
                new_rows += [{"method": label, "seed": seed, "sigma_m": sigma, "group": f"beats_afit:{g}", "horizon_s": 256,
                              "ade_km": float((win[m] < base[m]).mean()), "fde_km": "", "windows": int(m.sum()), "diverged": ""} for g, m in groups]
        tuning.append({"variant": variant, "aux": 0.0, "lr": ckpt.get("lr", ""), "seed42_val_ade256_km": seed42_val})
    all_rows = existing + new_rows
    if all_rows:
        write_csv(metric_path, all_rows)
    # Existing summary rows are preserved byte-for-byte in values; only new methods are added.
    old_summary = _read_csv(summary_path)
    old_methods = {r["method"] for r in old_summary}
    summary_source = all_rows
    missing_methods = {r["method"] for r in summary_source} - old_methods
    added_summary = [r for r in seed_summary([r for r in summary_source if r["method"] in missing_methods])
                     if r["method"] not in old_methods]
    if old_summary or added_summary:
        write_csv(summary_path, old_summary + added_summary)
    if tuning:
        write_csv(tuning_path, tuning)
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update({"baseline_settings": {v: {"label": BASELINE_LABELS[v], "lr_candidates": [1e-3, 3e-4]} for v in BASELINE_VARIANTS},
                   "tslib_commit": TSLIB_OFFICIAL_COMMIT,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()})
    config_path.write_text(json.dumps(config, indent=2))


def seed_summary(rows):
    keyed = {}
    for r in rows:
        keyed.setdefault((r["method"], r["sigma_m"], r["group"], r["horizon_s"]), []).append(r)
    out = []
    for (method, sigma, group, h), rs in keyed.items():
        ade = np.array([float(r["ade_km"]) for r in rs])
        fde = np.array([float(r["fde_km"]) for r in rs if r["fde_km"] != ""])
        out.append({"method": method, "sigma_m": sigma, "group": group, "horizon_s": h, "ade_mean": ade.mean(), "ade_std": ade.std(ddof=1) if len(ade) > 1 else 0.0,
                    "fde_mean": fde.mean() if len(fde) else "", "fde_std": fde.std(ddof=1) if len(fde) > 1 else ""})
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _timed(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        start = time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(values))


def latency(args):
    device = torch.device("cuda")
    data, ex = load_examples("holdout")
    result = []
    model_specs = [("Ours-MS", "ours2s", "ours2s_aux1_seed42"), ("Ours-MS w/o bank rollout", "ours2s_nb", "ours2s_nb_aux0_seed42"),
                   ("Ours", "ours", "ours_aux0.1_seed42"), ("Direct", "direct", "direct_aux0_seed42")]
    model_specs += [(BASELINE_LABELS[v], v, run_name(v, 0, 42)) for v in BASELINE_VARIANTS]
    for label, variant, name in model_specs:
        ckpt_path = OUT / "checkpoints" / f"{name}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(variant, device); model.load_state_dict(ckpt["model"]); model.eval()
        stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
        params = count_parameters(model)
        for batch in (1, 1024):
            idx = torch.arange(batch) % len(ex[0]["hist"])
            e = {k: v[idx].to(device) for k, v in ex[0].items()}
            hist, prior_n, feats = normalize(e, stats)
            maneuver = getattr(model, "maneuver", False)  # bf16 network + fp64 bank rollout (eager), as in predict()
            for dtype_name, autocast_dtype in ((("bf16", torch.bfloat16),) if maneuver else (("fp32", None), ("bf16", torch.bfloat16))):
                def run_network():
                    with torch.no_grad():
                        if maneuver:
                            maneuver_forward(model, e, stats, hist, prior_n, feats)
                            return
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                            model_output(model, hist, prior_n, feats, norm_series(model, e, stats))
                median = _timed(run_network, runs=max(50, args.runs))
                result.append({"component": label, "mode": "network", "dtype": dtype_name, "batch": batch,
                               "fit_sigma_m": "", "fit_steps": "", "median_ms_per_window": median / batch, "parameters": params})
    # A_fit (LM fit, always float64) on the GPU and on the CPU; VARMA-PM (sliding fits + VARMA + rollout) on the CPU.
    torch.set_num_threads(8)
    for sigma in (50, 200):
        raw, cfg = measurements(data, "holdout", sigma), FitConfig(fit_steps=FIT_STEPS[sigma], iterations=args.fit_iterations)
        for dev_name, batch in (("cuda", 1), ("cuda", 1024), ("cpu", 1), ("cpu", 256)):
            dev = torch.device(dev_name)
            physics = make_physics(dev).double()
            rr, ss = np.resize(data["rows"], batch)[:, None], np.resize(data["starts"], batch)[:, None]
            track = torch.from_numpy(raw[rr, ss + np.arange(256)]).to(dev)

            def run_fit():
                fit = fit_track(physics, track[:, -cfg.fit_steps:], cfg)
                rollout(physics, fit["terminal_state"], fit["cl"], fit["cd"], fit["bank"], 256)
            # calls take seconds (LM iterations), so a few repeats already give a stable median
            result.append({"component": "A_fit", "mode": f"fit_track+rollout ({dev_name}, {cfg.iterations} LM it)", "dtype": "fp64", "batch": batch,
                           "fit_sigma_m": sigma, "fit_steps": cfg.fit_steps,
                           "median_ms_per_window": _timed(run_fit, warmup=1, runs=10 if batch == 1 else 5) / batch, "parameters": 0})
            print("A_fit", sigma, dev_name, batch, f"{result[-1]['median_ms_per_window']:.2f} ms/window", time.strftime("%H:%M"), flush=True)
            if dev_name == "cpu":
                rule = tuple(json.loads((OUT / "config.json").read_text()).get("varma", {}).get("chosen", ("varma", 1, 1, 0)))

                def run_varma():
                    series, fit = parameter_series(physics, track, cfg, VARMA_EVERY)
                    analytic_prediction(series, fit["terminal_state"], rule)
                result.append({"component": "VARMA-PM", "mode": f"sliding fits+VARMA+rollout (cpu, {cfg.iterations} LM it)", "dtype": "fp64",
                               "batch": batch, "fit_sigma_m": sigma, "fit_steps": cfg.fit_steps,
                               "median_ms_per_window": _timed(run_varma, warmup=0, runs=3 if batch == 1 else 1) / batch,
                               "parameters": 0})
                print("VARMA-PM", sigma, batch, f"{result[-1]['median_ms_per_window']:.2f} ms/window", time.strftime("%H:%M"), flush=True)
    # End-to-end = network + the faster measured A_fit device for that regime (batch 1: cuda or cpu; batched: cuda 1024 or
    # cpu 256 per window). Ours-MS also reads the parameter series: in tracking, its earlier sliding fits were computed
    # when those measurements arrived, so each new window adds one A_fit fit (the one timed here).
    for label, sigma in ((label, sigma) for label in ("Ours-MS", "Ours") for sigma in (50, 200)):
        for batch in (1, 1024):
            for dtype_name in ("fp32", "bf16"):
                fit_rows = sorted((r for r in result if r["component"] == "A_fit" and r["fit_sigma_m"] == sigma
                                   and (r["batch"] == batch or (batch > 1 and r["batch"] > 1))),
                                  key=lambda r: r["median_ms_per_window"])
                net_rows = [r for r in result if r["component"] == label and r["mode"] == "network"
                            and r["batch"] == batch and r["dtype"] == dtype_name]
                if fit_rows and net_rows:
                    result.append({"component": label, "mode": "A_fit+network: " + fit_rows[0]["mode"], "dtype": dtype_name, "batch": batch,
                                   "fit_sigma_m": sigma, "fit_steps": FIT_STEPS[sigma],
                                   "median_ms_per_window": fit_rows[0]["median_ms_per_window"] + net_rows[0]["median_ms_per_window"],
                                   "parameters": net_rows[0]["parameters"]})
    OUT.joinpath("results").mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "results" / "latency.csv", result)


VARMA_GRID = [("hold", 0, 0, 0)] + [("varma", d, p, q) for d in (0, 1) for p, q in ((1, 0), (2, 0), (1, 1))]
VARMA_EVERY = 16  # s between sliding fits = length of each forecast parameter block
BANK_MAX = np.deg2rad(35.0)  # bank clip of data_generation/data_generator.py


def series_path(split, sigma, iterations=FitConfig.iterations):
    return OUT / "cache" / f"{split}_sigma{sigma}_series{'' if iterations == FitConfig.iterations else f'_it{iterations}'}.npz"


def build_series(split, sigma, limit=None, batch_size=256, iterations=FitConfig.iterations):
    """Sliding A_fit parameter series for every window (CPU, resumable)."""
    data, path = load_split(split), series_path(split, sigma, iterations)
    n = len(data["starts"]) if limit is None else limit
    done = {"series": [], "terminal_state": []}
    if path.exists():
        with np.load(path) as c:
            done = {k: [c[k]] for k in done}
    start = sum(len(v) for v in done["series"])
    meas, physics, cfg = measurements(data, split, sigma), make_physics("cpu").double(), FitConfig(fit_steps=FIT_STEPS[sigma], iterations=iterations)
    for begin in range(start, n, batch_size):
        r, s = data["rows"][begin:min(begin + batch_size, n), None], data["starts"][begin:min(begin + batch_size, n), None]
        series, fit = parameter_series(physics, torch.from_numpy(meas[r, s + np.arange(256)]), cfg, VARMA_EVERY)
        done["series"].append(series.numpy()); done["terminal_state"].append(fit["terminal_state"].numpy())
        if (begin // batch_size) % 8 == 7 or begin + batch_size >= n:
            np.savez(path, **{k: np.concatenate(v) for k, v in done.items()})
            print(split, sigma, min(begin + batch_size, n), "/", n, time.strftime("%H:%M"), flush=True)


def analytic_prediction(series, terminal, rule):
    """Analytical predictor: parameters held (A_fit) or VARMA-forecast per 16 s block."""
    kind, d, p, q = rule
    blocks = 256 // VARMA_EVERY
    if kind == "hold":
        future = series[:, -1:].repeat(1, blocks, 1)
    else:
        future = varma_forecast(series, blocks, p=p, q=q, d=d)  # 11-15 fits per window: AR extrapolation can run away,
        future[..., 2] = future[..., 2].clamp(-BANK_MAX, BANK_MAX)  # so keep the bank in the simulator's envelope
    cl, cd, bank = future[..., 0].clamp(1e-4, 2.0), future[..., 1].clamp(1e-4, 2.0), future[..., 2]  # fit_track's C_D bound
    return spherical_to_ecef(rollout_schedule(make_physics("cpu").double(), terminal, cl, cd, bank, VARMA_EVERY))


def analytic_errors(split, sigma, rule, limit=None):
    data = load_split(split)
    with np.load(series_path(split, sigma)) as c:
        series, terminal = torch.from_numpy(c["series"]).double(), torch.from_numpy(c["terminal_state"]).double()
    n = len(series) if limit is None else min(limit, len(series))
    meas = measurements(data, split, sigma)
    ex = examples(data, split, sigma, meas)
    origin = meas[data["rows"][:n], data["starts"][:n] + 255]
    pred = localize(analytic_prediction(series[:n], terminal[:n], rule).numpy(), origin)
    return np.linalg.norm(pred - ex["target"][:n].numpy(), axis=-1), data, ex


def analytic(args):
    """Sliding-fit parameter series (BSR-Net input) and the VARMA-PM baseline: VARMA forecast + 3-DOF rollout."""
    torch.set_num_threads(args.threads)
    if not args.select and not args.evaluate:
        for sigma in args.sigmas:
            build_series(args.split, sigma, args.limit, iterations=args.iterations)
        return
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if args.select:  # validation: VARMA orders / differencing by mean ADE256 over sigma (paper: min trajectory RMSE)
        scores = {str(rule): float(np.mean([analytic_errors("val", s, rule, args.limit)[0].mean() for s in SIGMAS])) for rule in VARMA_GRID}
        best = min((r for r in VARMA_GRID if r[0] == "varma"), key=lambda r: np.nan_to_num(scores[str(r)], nan=np.inf))  # NaN = rollout failed
        print(json.dumps(scores, indent=1), "\nbest", best)
        config["varma"] = {"every_s": VARMA_EVERY, "fit_steps": FIT_STEPS, "val_ade256_km": scores, "chosen": list(best),
                           "val_windows": args.limit}
        config_path.write_text(json.dumps(config, indent=2))
        return
    rule, rows = tuple(config["varma"]["chosen"]), []
    for sigma in SIGMAS:
        err, data, ex = analytic_errors("holdout", sigma, rule)
        if sigma == 0:  # the held-parameter rule must reproduce the cached A_fit prior
            hold, _, _ = analytic_errors("holdout", sigma, ("hold", 0, 0, 0), 2048)
            gap = float(np.abs(hold - (ex["prior"][:2048] - ex["target"][:2048]).norm(dim=-1).numpy()).max())
            print("hold vs A_fit max gap km", gap)
        rows += summarize(err, np.zeros(len(err)), data, ex, method="VARMA-PM", seed=0, sigma_m=sigma)[0]
    result_dir = OUT / "results"
    metrics = [r for r in _read_csv(result_dir / "metrics.csv") if r["method"] != "VARMA-PM"] + rows
    write_csv(result_dir / "metrics.csv", metrics)
    summary = [r for r in _read_csv(result_dir / "summary.csv") if r["method"] != "VARMA-PM"]
    write_csv(result_dir / "summary.csv", summary + seed_summary(rows))


METHOD_SPECS = [("Ours-MS", "ours2s", 1), ("Ours-M", "ours2", 1), ("Ours-MS w/o bank rollout", "ours2s_nb", 0), ("Ours", "ours", 0.1),
                ("NoAux", "noaux", 0), ("Direct", "direct", 0)] + [(BASELINE_LABELS[v], v, 0) for v in BASELINE_VARIANTS]
EPOCH100 = {"dlinear": "_ep100p20", "af_ciln": "_ep100p20"}  # still improving at epoch 50 -> 100-epoch runs reported


def errors(args):
    """Per-window ADE256 [seeds, N] and mean per-step error [seeds, 256] (km) on the holdout, for paired significance tests
    and error-vs-time curves -> results/holdout_errors.npz (keys "<variant>/sigma<s>/ade|step"; "afit" has one row)."""
    device = torch.device("cuda")
    data, ex = load_examples("holdout")
    idx, out = torch.arange(len(data["starts"])), {"windows": np.asarray(len(data["starts"]))}
    for sigma in SIGMAS:
        err = (ex[sigma]["prior"] - ex[sigma]["target"]).norm(dim=-1).numpy()
        out[f"afit/sigma{sigma}/ade"], out[f"afit/sigma{sigma}/step"] = err.mean(1)[None], err.mean(0)[None]
    for label, variant, aux in METHOD_SPECS:
        if variant not in args.methods:
            continue
        for sigma in SIGMAS:
            ade, step = [], []
            for seed in SEEDS:
                ckpt = torch.load(OUT / "checkpoints" / f"{run_name(variant, aux, seed)}{EPOCH100.get(variant, '')}.pt", map_location=device)
                model = build_model(variant, device); model.load_state_dict(ckpt["model"])
                err, _ = predict(model, ex[sigma], {k: v.to(device) for k, v in ckpt["stats"].items()}, idx, device)
                ade.append(err.mean(1)); step.append(err.mean(0))
            out[f"{variant}/sigma{sigma}/ade"], out[f"{variant}/sigma{sigma}/step"] = np.stack(ade), np.stack(step)
            print(label, sigma, float(np.mean(ade)), flush=True)
    np.savez_compressed(OUT / "results" / "holdout_errors.npz", **{k: np.asarray(v, np.float32) for k, v in out.items()})


def stress(args):
    """Evaluate A_fit and every trained method (METHOD_SPECS) on a holdout split, no retraining or re-tuning: holdout (final
    table) or a model-mismatch stress set (truth density biased, see generate_confirmatory_holdout.py --density-mismatch)."""
    device = torch.device("cuda")
    data, ex = load_examples(args.split)
    idx, rows = torch.arange(len(data["starts"])), []
    for sigma in SIGMAS:
        err = (ex[sigma]["prior"] - ex[sigma]["target"]).norm(dim=-1).numpy()
        rows += summarize(err, np.zeros(len(err)), data, ex[sigma], method="A_fit", seed=0, sigma_m=sigma)[0]
    for label, variant, aux in METHOD_SPECS:
        for seed in SEEDS:
            path = OUT / "checkpoints" / f"{run_name(variant, aux, seed)}{EPOCH100.get(variant, '')}.pt"
            if not path.exists():
                print("missing", path.name); continue
            ckpt = torch.load(path, map_location=device)
            model = build_model(variant, device); model.load_state_dict(ckpt["model"])
            stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
            for sigma in SIGMAS:
                err, logit = predict(model, ex[sigma], stats, idx, device)
                rows += summarize(err, logit, data, ex[sigma], method=label, seed=seed, sigma_m=sigma)[0]
    write_csv(OUT / "results" / f"{args.split}_metrics.csv", rows)
    write_csv(OUT / "results" / f"{args.split}_summary.csv", seed_summary(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--split", required=True); c.add_argument("--sigmas", type=int, nargs="+", default=list(SIGMAS))
    t = sub.add_parser("train"); t.add_argument("--variant", choices=("ours", "ours2", "ours2s", "ours2s_nb", "noaux", "direct", *BASELINE_VARIANTS), required=True)
    t.add_argument("--aux", type=float, default=0.0); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=50); t.add_argument("--patience", type=int, default=7)
    t.add_argument("--tag", default="", help="name suffix, e.g. _lr0.001 for lr tuning runs"); t.add_argument("--batch-size", type=int, default=512); t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--limit", type=int, default=None, help="train on the first N windows only (overfit check; no checkpoint)")
    t.add_argument("--dropout", type=float, default=0.1, help="ours2/ours2s only"); t.add_argument("--weight-decay", type=float, default=1e-4)
    e = sub.add_parser("evaluate"); e.add_argument("--only-missing", action="store_true"); e.add_argument("--methods", nargs="*", choices=BASELINE_VARIANTS)
    l = sub.add_parser("latency"); l.add_argument("--runs", type=int, default=50)
    l.add_argument("--fit-iterations", type=int, default=FitConfig.iterations, help="LM iterations for A_fit timing (converged by 5)")
    a = sub.add_parser("analytic"); a.add_argument("--split", default="val"); a.add_argument("--sigmas", type=int, nargs="+", default=list(SIGMAS))
    a.add_argument("--limit", type=int, default=None); a.add_argument("--threads", type=int, default=6)
    a.add_argument("--select", action="store_true"); a.add_argument("--evaluate", action="store_true")
    a.add_argument("--iterations", type=int, default=FitConfig.iterations, help="LM iterations of the sliding fits (series cache)")
    x = sub.add_parser("errors"); x.add_argument("--methods", nargs="+", default=["ours2s", "ours2s_nb", "ours2", "ours", "noaux",
                                                                               "itransformer_priors", "itransformer_priorf", "pit_prior"])
    r = sub.add_parser("stress"); r.add_argument("--split", required=True, help="holdout (final table), mismatch15 or mismatch30")
    args = ap.parse_args()
    if args.cmd == "cache":
        for sigma in args.sigmas:
            build_cache(args.split, sigma, torch.device("cuda"))
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    elif args.cmd == "analytic":
        analytic(args)
    elif args.cmd == "errors":
        errors(args)
    elif args.cmd == "stress":
        stress(args)
    else:
        latency(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
