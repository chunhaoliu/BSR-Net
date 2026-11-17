#!/usr/bin/env python3
"""Paper analyses from trained checkpoints and outputs/results/holdout_errors.npz (no retraining).

  python scripts/analysis.py wilcoxon   # trajectory-level paired tests of BSR-Net vs every method in holdout_errors.npz
  python scripts/analysis.py lambda0    # lambda = 0 vs lambda = 1: accuracy, Wilcoxon, bank-schedule RMS error vs held bank
  python scripts/analysis.py extras     # per-trajectory ADE percentiles, along/cross/vertical split, bank-change AUROC
  python scripts/analysis.py flops      # GFLOPs per window (matrix and convolution operations)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import pipeline as T  # noqa: E402

DEV = torch.device("cuda")
AUX = {"ours2s": 1, "itransformer_priors": 0, "pit": 0, "af_ciln": 0}


def load(variant, seed, aux=None):
    name = T.run_name(variant, AUX.get(variant, 0) if aux is None else aux, seed) + T.EPOCH100.get(variant, "")
    ck = torch.load(T.OUT / "checkpoints" / f"{name}.pt", map_location=DEV)
    model = T.build_model(variant, DEV); model.load_state_dict(ck["model"]); model.eval()
    return model, {k: v.to(DEV) for k, v in ck["stats"].items()}


def forward(model, stats, ex, batch=2048):
    """Predictions [N, 256, 3] km, bank-change score [N], bank schedule [N, 32] rad (zeros for non-BSR-Net models)."""
    preds, scores, banks = [], [], []
    with torch.no_grad():
        for b in range(0, len(ex["hist"]), batch):
            e = {k: v[b:b + batch].to(DEV) for k, v in ex.items()}
            hist, prior_n, feats = T.normalize(e, stats)
            if getattr(model, "maneuver", False):
                y, s, bank = T.maneuver_forward(model, e, stats, hist, prior_n, feats)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = T.model_output(model, hist, prior_n, feats, T.norm_series(model, e, stats))
                raw = raw[0] if isinstance(raw, (tuple, list)) else raw
                y = T.to_prediction(raw.float(), e["prior"], stats, T.output_mode(model))
                s, bank = torch.zeros(len(y), device=DEV), torch.zeros(len(y), T.BANK_BLOCKS, device=DEV)
            preds.append(y.cpu()); scores.append(s.float().cpu()); banks.append(bank.float().cpu())
    return torch.cat(preds).numpy(), torch.cat(scores).numpy(), torch.cat(banks)


def per_trajectory(rows, ade):
    return np.bincount(rows, ade) / np.bincount(rows)


def wilcoxon_table():
    z, rows = np.load(T.OUT / "results" / "holdout_errors.npz"), np.asarray(T.load_split("holdout")["rows"])
    others = sorted({k.split("/")[0] for k in z.files if "/" in k} - {"ours2s"})
    for sigma in T.SIGMAS:
        ours = per_trajectory(rows, z[f"ours2s/sigma{sigma}/ade"].mean(0))
        for name in others:
            key = f"{name}/sigma{sigma}/ade"
            if key in z.files:
                other = per_trajectory(rows, z[key].mean(0))
                print(f"sigma {sigma:3d} vs {name:22s} win {np.mean(ours < other):.2f}  p {wilcoxon(ours, other).pvalue:.1e}")


def lambda0():
    data, ex = T.load_examples("holdout")
    rows, idx = np.asarray(data["rows"]), torch.arange(len(data["starts"]))
    for sigma in T.SIGMAS:
        ade = {aux: np.stack([_ade(load("ours2s", s, aux), ex[sigma], idx) for s in T.SEEDS]) for aux in (0, 1)}
        t0, t1 = per_trajectory(rows, ade[0].mean(0)), per_trajectory(rows, ade[1].mean(0))
        print(f"sigma {sigma:3d}: lambda0 {ade[0].mean():.3f} | lambda1 {ade[1].mean():.3f} | "
              f"lambda1 wins {np.mean(t1 < t0):.2f} p {wilcoxon(t1, t0).pvalue:.2f}")
    for sigma in (0, 50, 200):
        target = ex[sigma]["bank_future"]
        held = np.degrees((ex[sigma]["aero"][:, 2:3] - target).square().mean().sqrt().item())
        rms = {aux: np.mean([np.degrees((forward(*load("ours2s", s, aux), ex[sigma])[2] - target).square().mean().sqrt().item())
                             for s in T.SEEDS]) for aux in (0, 1)}
        print(f"sigma {sigma:3d}: bank RMS error held {held:.1f} | lambda0 {rms[0]:.1f} | lambda1 {rms[1]:.1f} deg")


def _ade(loaded, ex, idx):
    model, stats = loaded
    return T.predict(model, ex, stats, idx, DEV)[0].mean(1)


def directions(pred, target, origin):
    """Mean |error| km along track, cross track, vertical (axes from the true trajectory; vertical = local radial)."""
    up = target + np.array([0, 0, 1]) * (np.linalg.norm(origin, axis=-1)[:, None, None] / 1000)
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    along = np.gradient(target, axis=1); along -= (along * up).sum(-1, keepdims=True) * up
    along /= np.linalg.norm(along, axis=-1, keepdims=True)
    d = pred - target
    return [float(np.abs((d * a).sum(-1)).mean()) for a in (along, np.cross(up, along), up)]


def extras():
    z, rows = np.load(T.OUT / "results" / "holdout_errors.npz"), np.asarray(T.load_split("holdout")["rows"])
    for sigma in (0, 50, 200):
        line = [f"{n} " + "/".join(f"{np.percentile(per_trajectory(rows, z[f'{n}/sigma{sigma}/ade'].mean(0)), q):.2f}" for q in (50, 90, 95))
                for n in ("afit", "pit_prior", "itransformer_priors", "ours2s") if f"{n}/sigma{sigma}/ade" in z.files]
        print(f"sigma {sigma:3d} per-trajectory ADE median/P90/P95: " + "  ".join(line))
    _, ex = T.load_examples("holdout")
    for sigma in (0, 50, 200):
        target, origin = ex[sigma]["target"].numpy().astype(np.float64), ex[sigma]["origin"].numpy().astype(np.float64)
        print(f"sigma {sigma:3d} A_fit along/cross/vertical {np.round(directions(ex[sigma]['prior'].numpy(), target, origin), 3)}")
        for variant in ("itransformer_priors", "ours2s"):
            split, auc = [], []
            for seed in T.SEEDS:
                pred, score, _ = forward(*load(variant, seed), ex[sigma])
                split.append(directions(pred, target, origin))
                auc.append(T.auroc(ex[sigma]["label"].numpy(), score))
            print(f"sigma {sigma:3d} {variant} along/cross/vertical {np.round(np.mean(split, 0), 3)}"
                  + (f"  bank-change AUROC {np.mean(auc):.3f}" if variant == "ours2s" else ""))


def flops():
    from torch.utils.flop_counter import FlopCounterMode
    _, ex = T.load_examples("holdout")
    for variant in ("ours2s", "itransformer_priors", "pit", "af_ciln"):
        model, stats = load(variant, 42)
        e = {k: v[:1].to(DEV) for k, v in ex[50].items()}
        hist, prior_n, feats = T.normalize(e, stats)
        with FlopCounterMode(display=False) as counter:  # grad mode on: FlopCounterMode needs it for some kernels
            if variant == "ours2s":  # encoder and both decoders; the elementwise fp64 rollout is not counted
                memory, _ = model.encode(hist, feats, T.norm_series(model, e, stats))
                model.correct(memory, prior_n, torch.zeros_like(e["prior"]))
            else:
                T.model_output(model, hist, prior_n, feats, T.norm_series(model, e, stats))
        print(f"{variant}: {counter.get_total_flops() / 1e9:.3f} GFLOPs per window")


if __name__ == "__main__":
    {"wilcoxon": wilcoxon_table, "lambda0": lambda0, "extras": extras, "flops": flops}[sys.argv[1]]()
