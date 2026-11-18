"""Result figures of the BSR-Net paper (vector PDF).

    python scripts/plot_figures.py --out figures
Reads outputs/results/{holdout_errors.npz, <split>_summary.csv} and the BSR-Net checkpoints (ours2s_aux1_seed*.pt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import pipeline as T  # noqa: E402

RES = T.OUT / "results"
# methods differ by color only (solid lines); colors match Fig. 1 of the paper (orange BSR-Net, blue physics prior, black truth)
STYLE = {"A_fit": ("#0070C0", "-"), "iTransformer+prior+series": ("#00B050", "-"),
         "Ours-MS w/o bank rollout": ("#7030A0", "-"), "Ours-MS": ("#ED7D31", "-")}  # BSR-Net drawn last (on top)
KEYS = {"Ours-MS": "ours2s", "iTransformer+prior+series": "itransformer_priors", "Ours-MS w/o bank rollout": "ours2s_nb", "A_fit": "afit"}
LABEL = {"A_fit": r"$A_{\mathrm{fit}}$ (physics prior)", "Ours-MS w/o bank rollout": "BSR-Net w/o bank-schedule rollout",
         "iTransformer+prior+series": "iTransformer + prior + series", "Ours-MS": "BSR-Net"}  # paper name of ours2s
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def fig_error_time(out):
    z = np.load(RES / "holdout_errors.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    t = np.arange(1, 257)
    for ax, sigma in zip(axes, (50, 200)):
        for name, (color, ls) in STYLE.items():
            ax.plot(t, z[f"{KEYS[name]}/sigma{sigma}/step"].mean(0), color=color, ls=ls, label=LABEL[name])
        ax.set_title(rf"$\sigma = {sigma}$ m", fontsize=9)
        ax.set_xlabel("Prediction time (s)"); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    axes[0].set_ylabel("Position error (km)"); axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.savefig(out / "fig_error_time.pdf"); plt.close(fig)


def fig_noise(out):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey=True)
    panels = (("holdout", "Nominal dynamics"), ("mismatch15", "Density mismatch 15%"), ("mismatch30", "Density mismatch 30%"),
              ("mismatchaero110_90", r"$C_L\times1.1$, $C_D\times0.9$"), ("mismatchaero90_110", r"$C_L\times0.9$, $C_D\times1.1$"))
    for ax, (split, title) in zip(axes.flat, panels):
        d = pd.read_csv(RES / f"{split}_summary.csv")
        d = d[(d.group == "overall") & (d.horizon_s == 256)]
        for name, (color, ls) in STYLE.items():
            r = d[d.method == name].sort_values("sigma_m")
            ax.plot(r.sigma_m, r.ade_mean, color=color, ls=ls, marker="o", ms=3.5, mec="white", mew=0.6, label=LABEL[name])
        ax.set_title(title, fontsize=9); ax.set_xlabel(r"Measurement noise $\sigma$ (m)"); ax.set_xticks([0, 50, 100, 200])
    for ax in axes[:, 0]:
        ax.set_ylabel("ADE over 256 s (km)")
    axes[0, 0].set_ylim(bottom=0); axes[1, 2].axis("off")
    axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="center", frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_noise_mismatch.pdf"); plt.close(fig)


def fig_case(out, sigma=50):
    """Typical bank-change window: Ours-MS gain closest to the median gain over bank-change windows."""
    device = torch.device("cuda")
    data = T.load_split("holdout"); ex = T.examples(data, "holdout", sigma)
    ck = torch.load(T.OUT / "checkpoints" / "ours2s_aux1_seed42.pt", map_location=device)
    model = T.build_model("ours2s", device); model.load_state_dict(ck["model"]); model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    idx = np.flatnonzero(ex["label"].numpy() > 0)
    with torch.no_grad():
        e = {k: v[idx].to(device) for k, v in ex.items()}
        e = T.rel_aero_view(e) if ck.get("rel_aero", False) else e
        pred, _, bank = T.maneuver_forward(model, e, stats, *T.normalize(e, stats))
    err_o = (pred - e["target"]).norm(dim=-1).mean(1).cpu().numpy(); err_a = (e["prior"] - e["target"]).norm(dim=-1).mean(1).cpu().numpy()
    mid = np.flatnonzero((err_a > np.percentile(err_a, 40)) & (err_a < np.percentile(err_a, 60)))  # typical prior error
    ratio = err_o[mid] / err_a[mid]
    w = mid[np.argmin(np.abs(ratio - np.median(ratio)))]  # typical relative gain
    i = idx[w]; p, b = pred[w].cpu().numpy(), bank[w].cpu().numpy()
    tgt, pri = ex["target"][i].numpy(), ex["prior"][i].numpy()
    along = np.gradient(pri[:, :2], axis=0); along /= np.linalg.norm(along, axis=1, keepdims=True)
    normal = np.stack((-along[:, 1], along[:, 0]), 1)
    cross = lambda x: ((x[:, :2] - pri[:, :2]) * normal).sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    tf = np.arange(1, 257)
    for ax, f, ylabel in ((axes[0], cross, "Cross-track offset (km)"), (axes[1], lambda x: x[:, 2] - pri[:, 2], "Vertical offset (km)")):
        ax.axhline(0, color=STYLE["A_fit"][0], lw=1.4, label=LABEL["A_fit"])
        ax.plot(tf, f(tgt), color="black", lw=1.2, label="Truth"); ax.plot(tf, f(p), color=STYLE["Ours-MS"][0], label=LABEL["Ours-MS"])
        ax.set_xlabel("Prediction time (s)"); ax.set_ylabel(ylabel); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    ax = axes[2]
    blocks = np.arange(T.BANK_BLOCKS + 1) * (256 // T.BANK_BLOCKS)
    deg = lambda x: np.rad2deg(np.append(x, x[-1]))
    ax.step(blocks, deg(ex["bank_future"][i].numpy()), where="post", color="black", lw=1.2, label="Truth")
    ax.step(blocks, deg(np.full(T.BANK_BLOCKS, ex["aero"][i, 2].item())), where="post", color=STYLE["A_fit"][0], label="Held (prior)")
    ax.step(blocks, deg(b), where="post", color=STYLE["Ours-MS"][0], label="Predicted plan")
    ax.set_xlabel("Prediction time (s)"); ax.set_ylabel("Bank angle (deg)"); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    handles = axes[2].get_legend_handles_labels()[0]
    fig.legend(handles, ["Truth", LABEL["A_fit"] + ", held bank", LABEL["Ours-MS"] + ", predicted bank plan"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "fig_case_bank.pdf"); plt.close(fig)
    print(f"case window {i}: ADE A_fit {err_a[w]:.2f} km, Ours-MS {err_o[w]:.2f} km")


STRATA = [f"{v}__{l}" for v in ("quasi_equilibrium", "skip_glide") for l in ("longitudinal", "turning", "weaving")]
# color only: hue = lateral maneuver, dark/light shade = vertical profile (quasi-equilibrium / skip glide)
LATERAL = {"longitudinal": ("#0070C0", "#8FC1EA"), "turning": ("#ED7D31", "#F6BE93"), "weaving": ("#00B050", "#8FDCAE")}
VERTICAL = {"quasi_equilibrium": (0, "quasi-equilibrium glide"), "skip_glide": (1, "skip glide")}
R_EARTH = 6.378e6


def fig_dataset(out):
    """One test trajectory per joint stratum (median total heading change within the stratum): ground track, altitude, bank."""
    with np.load(T.DATA / "hgv_confirmatory_holdout.npz") as d:
        clean, strata, bank = d["clean_trajectories"], d["joint_strata"], d["bank_achieved"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    t = np.arange(clean.shape[1])
    for s in STRATA:
        idx = np.flatnonzero(strata == s)
        turn = np.abs(np.unwrap(clean[idx, :, 5], axis=1)[:, -1] - clean[idx, 0, 5])
        i = idx[np.argsort(turn)[len(idx) // 2]]
        v, l = s.split("__"); ls, c = "-", LATERAL[l][VERTICAL[v][0]]
        lon, lat, psi0 = clean[i, :, 1], clean[i, :, 2], clean[i, 0, 5]
        dist = np.arccos(np.clip(np.sin(lat[0]) * np.sin(lat) + np.cos(lat[0]) * np.cos(lat) * np.cos(lon - lon[0]), -1, 1))
        bearing = np.arctan2(np.sin(lon - lon[0]) * np.cos(lat), np.cos(lat[0]) * np.sin(lat) - np.sin(lat[0]) * np.cos(lat) * np.cos(lon - lon[0]))
        xt = np.arcsin(np.sin(dist) * np.sin(bearing - psi0))  # cross-range from the initial great circle (rad)
        at = np.arccos(np.clip(np.cos(dist) / np.cos(xt), -1, 1))
        axes[0].plot(at * R_EARTH / 1000, xt * R_EARTH / 1000, color=c, ls=ls, lw=1.1)
        axes[1].plot(t, (clean[i, :, 0] - R_EARTH) / 1000, color=c, ls=ls, lw=1.1)
        axes[2].plot(t, np.rad2deg(bank[i]), color=c, ls=ls, lw=1.1)
    axes[0].set_xlabel("Downrange (km)"); axes[0].set_ylabel("Cross-range (km)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Altitude (km)")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Bank angle (deg)")
    for ax in axes[1:]:
        ax.set_xlim(0, clean.shape[1] - 1)
    handles = [plt.Line2D([], [], color=c[k], lw=1.4, label=f"{l}, {n}") for k, n in VERTICAL.values() for l, c in LATERAL.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_dataset.pdf"); plt.close(fig)


def _load(variant, aux, seed, device):
    ck = torch.load(T.OUT / "checkpoints" / f"{T.run_name(variant, aux, seed)}.pt", map_location=device)
    model = T.build_model(variant, device); model.load_state_dict(ck["model"]); model.eval(); model.rel_aero = ck.get("rel_aero", False)
    return model, {k: v.to(device) for k, v in ck["stats"].items()}


def _predict(model, stats, ex, idx, device):
    """Positions [n, 256, 3] km for windows ``idx``."""
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), 2048):
            e = {k: v[idx[b:b + 2048]].to(device) for k, v in ex.items()}
            e = T.rel_aero_view(e) if model.rel_aero else e
            h, p, f = T.normalize(e, stats)
            if getattr(model, "maneuver", False):
                y = T.maneuver_forward(model, e, stats, h, p, f)[0]
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = T.model_output(model, h, p, f, T.norm_series(model, e, stats))
                y = T.to_prediction((raw[0] if isinstance(raw, (tuple, list)) else raw).float(), e["prior"], stats, T.output_mode(model))
            out.append(y.cpu())
    return torch.cat(out).numpy()


def _directions(pred, tgt, origin):
    """Mean |error| along track, cross track, vertical (km); axes from the true trajectory, vertical = local radial."""
    up = tgt + np.array([0.0, 0.0, 1.0]) * (np.linalg.norm(origin, axis=-1)[:, None, None] / 1000)
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    al = np.gradient(tgt, axis=1); al -= (al * up).sum(-1, keepdims=True) * up; al /= np.linalg.norm(al, axis=-1, keepdims=True)
    d = pred - tgt
    return [np.abs((d * a).sum(-1)).mean() for a in (al, np.cross(up, al), up)]


def fig_tail_direction(out):
    """Per-trajectory ADE distribution (sigma 50, 200) and the along/cross/vertical split at sigma 200 (three seeds)."""
    device = torch.device("cuda")
    z = np.load(RES / "holdout_errors.npz"); data = T.load_split("holdout"); rows = np.asarray(data["rows"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    names = ["A_fit", "iTransformer+prior+series", "Ours-MS"]
    for ax, sigma in zip(axes[:2], (50, 200)):
        for name in names:
            a = z[f"{KEYS[name]}/sigma{sigma}/ade"].mean(0)
            traj = np.sort(np.bincount(rows, a) / np.bincount(rows))
            color, ls = STYLE[name]
            ax.plot(traj, np.arange(1, len(traj) + 1) / len(traj), color=color, ls=ls, label=LABEL[name])
        ax.set_title(rf"$\sigma = {sigma}$ m", fontsize=9); ax.set_xlabel("Per-trajectory ADE over 256 s (km)")
        ax.set_xlim(0, 20); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of test trajectories")
    ex = T.examples(data, "holdout", 200); idx = torch.arange(len(data["starts"]))
    tgt, org = ex["target"].numpy().astype(np.float64), ex["origin"].numpy().astype(np.float64)
    split = {"A_fit": _directions(ex["prior"].numpy(), tgt, org)}
    for name, (variant, aux) in (("iTransformer+prior+series", ("itransformer_priors", 0)), ("Ours-MS", ("ours2s", 1))):
        split[name] = np.mean([_directions(_predict(*_load(variant, aux, s, device), ex, idx, device), tgt, org) for s in T.SEEDS], 0)
    x, w = np.arange(3), 0.26
    for k, name in enumerate(names):
        axes[2].bar(x + (k - 1) * w, split[name], w - 0.02, color=STYLE[name][0], label=LABEL[name])
    axes[2].set_xticks(x, ["Along", "Cross", "Vertical"]); axes[2].set_ylabel("Mean |error| (km)")
    axes[2].set_title(r"$\sigma = 200$ m", fontsize=9); axes[2].grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_tail_direction.pdf"); plt.close(fig)
    print("direction split sigma 200:", {k: np.round(v, 3) for k, v in split.items()})


def fig_cases(out, sigma=50):
    """Three bank-change windows (QEG turning, QEG weaving, skip-glide weaving), seed 42. In each stratum, among bank-change
    windows whose prior error lies between the 40th and 60th percentiles, the window with the median paired error difference
    between BSR-Net and iTransformer+prior+series, i.e. a typical comparison of the two."""
    device = torch.device("cuda")
    data = T.load_split("holdout"); ex = T.examples(data, "holdout", sigma)
    strata = np.asarray(data["strata"]); label = ex["label"].numpy() > 0
    ours, base = _load("ours2s", 1, 42, device), _load("itransformer_priors", 0, 42, device)
    cases = ["quasi_equilibrium__turning", "quasi_equilibrium__weaving", "skip_glide__weaving"]
    titles = ["Quasi-equilibrium glide, turning", "Quasi-equilibrium glide, weaving", "Skip glide, weaving"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), gridspec_kw={"height_ratios": [1.25, 1]})
    for col, (s, title) in enumerate(zip(cases, titles)):
        idx = np.flatnonzero(label & (strata == s))
        po, pball = _predict(*ours, ex, idx, device), _predict(*base, ex, idx, device)
        tgt, pri = ex["target"][idx].numpy(), ex["prior"][idx].numpy()
        err = lambda y: np.linalg.norm(y - tgt, axis=-1).mean(1)
        ea, diff = err(pri), err(po) - err(pball)
        mid = np.flatnonzero((ea > np.percentile(ea, 40)) & (ea < np.percentile(ea, 60)))
        w = mid[np.argmin(np.abs(diff[mid] - np.median(diff[mid])))]
        i = idx[w]; pb = pball[w]
        hist = ex["hist"][i, :, :3].numpy()
        center = np.array([0.0, 0.0, np.linalg.norm(ex["origin"][i].numpy().astype(np.float64)) / 1000])
        alt = lambda y: np.linalg.norm(y + center, axis=-1) - R_EARTH / 1000  # ENU km -> altitude km
        tracks = {"Truth": tgt[w], "A_fit": pri[w], "iTransformer+prior+series": pb, "Ours-MS": po[w]}
        along = np.gradient(pri[w][:, :2], axis=0); along /= np.linalg.norm(along, axis=1, keepdims=True)
        normal = np.stack((-along[:, 1], along[:, 0]), 1)
        cross = lambda y: ((y[:, :2] - pri[w][:, :2]) * normal).sum(1)  # horizontal offset from the prior, left positive
        ax, axh = axes[0, col], axes[1, col]
        axh.plot(np.arange(-255, 1), alt(hist), color="#00B0F0", lw=1.0, label="Observed track")
        tf = np.arange(1, 257)
        for name, y in tracks.items():
            color, ls = ("black", "-") if name == "Truth" else STYLE[name]
            ax.plot(tf, cross(y), color=color, ls=ls, lw=1.2)
            axh.plot(tf, alt(y), color=color, ls=ls, lw=1.2, label="Truth" if name == "Truth" else LABEL[name])
        ax.set_title(title, fontsize=9); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xlabel("Prediction time (s)")
        axh.set_xlabel("Time from newest measurement (s)"); axh.set_xlim(-255, 256)
        ade = {k: np.linalg.norm(v - tgt[w], axis=-1).mean() for k, v in tracks.items() if k != "Truth"}
        print(f"{s}: window {i}, ADE " + ", ".join(f"{k} {v:.2f}" for k, v in ade.items()))
    axes[0, 0].set_ylabel("Cross-track offset from prior (km)"); axes[1, 0].set_ylabel("Altitude (km)")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.065))
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    fig.savefig(out / "fig_cases.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="subset of figure functions, e.g. fig_cases")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_error_time, fig_noise, fig_case, fig_dataset, fig_tail_direction, fig_cases):
        if not args.only or f.__name__ in args.only:
            f(args.out)
