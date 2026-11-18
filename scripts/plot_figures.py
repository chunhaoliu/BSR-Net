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
# methods differ by color only (solid lines); colors match Fig. 1 (orange BSR-Net, blue physics prior, black truth)
# three method colors with one meaning across the paper (Fig. 1): orange BSR-Net, blue physics prior, green strongest
# baseline; truth black, observed track gray. Solid lines only; BSR-Net drawn last, on top, and heavier.
STYLE = {"A_fit": ("#0070C0", "-"), "iTransformer+prior+series": ("#00B050", "-"), "Ours-MS": ("#ED7D31", "-")}
KEYS = {"Ours-MS": "ours2s", "iTransformer+prior+series": "itransformer_priors", "A_fit": "afit"}
LABEL = {"A_fit": r"$A_{\mathrm{fit}}$ (physics prior)", "iTransformer+prior+series": "iTransformer+prior+series",
         "Ours-MS": "BSR-Net"}  # paper name of ours2s
LW = {"Ours-MS": 1.6, "Truth": 1.6}  # every other line 1.3
OBSERVED = "#A6A6A6"


def lw(name):
    return LW.get(name, 1.3)


def legend(ax, *args, **kw):
    """Legend in a thin black frame."""
    leg = ax.legend(*args, **kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


WIDTH = 6.45  # drawn at the final text width (468 pt) and included at natural size, one PDF per panel (IEEE subfigures)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.linewidth": 0.7, "axes.grid": False,  # journal style: white, boxed axes, no grid
                     "axes.spines.top": True, "axes.spines.right": True, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True,
                     "lines.linewidth": 1.8, "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "black", "legend.fancybox": False,
                     "legend.framealpha": 1.0, "legend.borderpad": 0.3, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
                     "legend.handlelength": 1.6, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "figure.constrained_layout.use": True})


def save_panels(fig, axes, out, name):
    """One PDF per panel, name_a.pdf, name_b.pdf, ...; LaTeX sets the (a), (b) labels below each subfloat."""
    axes = list(np.ravel(axes))
    fig.canvas.draw(); fig.set_layout_engine("none")  # freeze the joint layout so every panel keeps its size and scale
    box = [ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) for ax in axes]
    row = [ax.get_subplotspec().rowspan.start for ax in axes]
    for k, ax in enumerate(axes):
        for a in axes:
            a.set_visible(a is ax)
        same = [b for b, r in zip(box, row) if r == row[k]]  # one height per row, so the subfloats align
        crop = matplotlib.transforms.Bbox([[box[k].x0, min(b.y0 for b in same)], [box[k].x1, max(b.y1 for b in same)]])
        fig.savefig(out / f"{name}_{chr(97 + k)}.pdf", bbox_inches=crop.padded(0.02))
    plt.close(fig)


def fig_error_time(out):
    z = np.load(RES / "holdout_errors.npz")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    t = np.arange(1, 257)
    for ax, sigma in zip(axes, (50, 200)):
        ins = ax.inset_axes([0.14, 0.26, 0.24, 0.26] if sigma == 50 else [0.14, 0.52, 0.24, 0.26])  # last 32 s
        lo, hi = np.inf, -np.inf
        for name, (color, ls) in STYLE.items():
            e = z[f"{KEYS[name]}/sigma{sigma}/step"].mean(0)
            ax.plot(t, e, color=color, ls=ls, lw=lw(name), label=LABEL[name])
            if name != "A_fit":
                ins.plot(t[223:], e[223:], color=color, ls=ls, lw=lw(name))
                lo, hi = min(lo, e[223:].min()), max(hi, e[223:].max())
        ins.set_xlim(224, 256); ins.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)); ins.set_xticks([224, 256])
        for c in ax.indicate_inset_zoom(ins, edgecolor="black", alpha=1.0, lw=0.6).connectors:
            c.set_visible(False)
        ax.set_xlabel("Prediction time (s)"); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    axes[0].set_ylabel("Position error (km)"); axes[0].set_ylim(bottom=0)
    legend(axes[0], loc="upper left")
    save_panels(fig, axes, out, "fig_error_time")


def fig_noise(out):
    """Five conditions (ADE vs noise) and, in the sixth panel, the gain of BSR-Net over iTransformer+prior+series at 200 m."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.6))
    panels = (("holdout", "Nominal"), ("mismatch15", "Density 15%"), ("mismatch30", "Density 30%"),
              ("mismatchaero110_90", "$L/D$ +22%"),
              ("mismatchaero90_110", "$L/D$ −18%"))
    gain = []
    for ax, (split, _) in zip(axes.flat, panels):
        d = pd.read_csv(RES / f"{split}_summary.csv")
        d = d[(d.group == "overall") & (d.horizon_s == 256)]
        for name, (color, ls) in STYLE.items():
            r = d[d.method == name].sort_values("sigma_m")
            ax.plot(r.sigma_m, r.ade_mean, color=color, ls=ls, lw=lw(name), marker="o", ms=4, mec="white", mew=0.6, label=LABEL[name])
        at = lambda m: d[(d.method == m) & (d.sigma_m == 200)].ade_mean.item()
        gain.append(100 * (1 - at("Ours-MS") / at("iTransformer+prior+series")))
        ax.set_xlabel(r"Measurement noise $\sigma$ (m)"); ax.set_xticks([0, 50, 100, 200]); ax.set_ylim(0, 14)
    for ax in axes[:, 0]:
        ax.set_ylabel("ADE over 256 s (km)")
    ax = axes[1, 2]
    x = np.arange(len(gain))
    ax.bar(x, gain, 0.6, color=STYLE["Ours-MS"][0])
    for k, g in enumerate(gain):
        ax.text(k, g + 0.4, f"{g:.1f}", ha="center", va="bottom")
    ax.set_xticks(x, [p[1] for p in panels], rotation=35, ha="right", rotation_mode="anchor"); ax.set_ylim(0, 21); ax.tick_params(axis="x", bottom=False, top=False)
    ax.set_ylabel("ADE reduction vs.\niTransformer+prior+series (%)")
    legend(axes[0, 0], loc="upper left")
    save_panels(fig, axes, out, "fig_noise_mismatch")


def _typical_case(sigma=50):
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
    return ex, idx[w], pred[w].cpu().numpy(), bank[w].cpu().numpy(), err_a[w], err_o[w]


def fig_case(out, sigma=50):
    ex, i, p, b, ea, eo = _typical_case(sigma)
    tgt, pri = ex["target"][i].numpy(), ex["prior"][i].numpy()
    along = np.gradient(pri[:, :2], axis=0); along /= np.linalg.norm(along, axis=1, keepdims=True)
    normal = np.stack((-along[:, 1], along[:, 0]), 1)
    cross = lambda x: ((x[:, :2] - pri[:, :2]) * normal).sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    tf = np.arange(1, 257)
    for ax, f, ylabel in ((axes[0], cross, "Cross-track offset (km)"), (axes[1], lambda x: x[:, 2] - pri[:, 2], "Vertical offset (km)")):
        ax.axhline(0, color=STYLE["A_fit"][0], lw=lw("A_fit"), label=LABEL["A_fit"])
        ax.plot(tf, f(tgt), color="black", lw=lw("Truth"), label="Truth")
        ax.plot(tf, f(p), color=STYLE["Ours-MS"][0], lw=lw("Ours-MS"), label=LABEL["Ours-MS"])
        ax.set_xlabel("Prediction time (s)"); ax.set_ylabel(ylabel); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    ax = axes[2]
    blocks = np.arange(T.BANK_BLOCKS + 1) * (256 // T.BANK_BLOCKS)
    deg = lambda x: np.rad2deg(np.append(x, x[-1]))
    ax.step(blocks, deg(ex["bank_future"][i].numpy()), where="post", color="black", lw=lw("Truth"), label="Truth")
    ax.step(blocks, deg(np.full(T.BANK_BLOCKS, ex["aero"][i, 2].item())), where="post", color=STYLE["A_fit"][0],
            lw=lw("A_fit"), label=LABEL["A_fit"])
    ax.step(blocks, deg(b), where="post", color=STYLE["Ours-MS"][0], lw=lw("Ours-MS"), label=LABEL["Ours-MS"])
    ax.set_xlabel("Prediction time (s)"); ax.set_ylabel("Bank angle (deg)"); ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
    axes[1].set_ylim(-9.5, 0.5)
    legend(axes[1], *ax.get_legend_handles_labels(), loc="lower right")
    save_panels(fig, axes, out, "fig_case_bank")
    print(f"case window {i}: ADE A_fit {ea:.2f} km, Ours-MS {eo:.2f} km")


STRATA = [f"{v}__{l}" for v in ("quasi_equilibrium", "skip_glide") for l in ("longitudinal", "turning", "weaving")]
LATERAL = {"longitudinal": "#7030A0", "turning": "#C00000", "weaving": "#00827F"}  # data strata: colors no method uses
VERTICAL = {"quasi_equilibrium": 0, "skip_glide": 1}  # row
R_EARTH = 6.378e6


def fig_dataset(out):
    """One test trajectory per joint stratum (median total heading change within the stratum): ground track, altitude, bank."""
    with np.load(T.DATA / "hgv_confirmatory_holdout.npz") as d:
        clean, strata, bank = d["clean_trajectories"], d["joint_strata"], d["bank_achieved"]
    fig, grid = plt.subplots(2, 3, figsize=(WIDTH, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    t = np.arange(clean.shape[1])
    for s in STRATA:
        idx = np.flatnonzero(strata == s)
        turn = np.abs(np.unwrap(clean[idx, :, 5], axis=1)[:, -1] - clean[idx, 0, 5])
        i = idx[np.argsort(turn)[len(idx) // 2]]
        v, l = s.split("__"); ls, c, axes = "-", LATERAL[l], grid[VERTICAL[v]]
        lon, lat, psi0 = clean[i, :, 1], clean[i, :, 2], clean[i, 0, 5]
        dist = np.arccos(np.clip(np.sin(lat[0]) * np.sin(lat) + np.cos(lat[0]) * np.cos(lat) * np.cos(lon - lon[0]), -1, 1))
        bearing = np.arctan2(np.sin(lon - lon[0]) * np.cos(lat), np.cos(lat[0]) * np.sin(lat) - np.sin(lat[0]) * np.cos(lat) * np.cos(lon - lon[0]))
        xt = np.arcsin(np.sin(dist) * np.sin(bearing - psi0))  # cross-range from the initial great circle (rad)
        at = np.arccos(np.clip(np.cos(dist) / np.cos(xt), -1, 1))
        axes[0].plot(at * R_EARTH / 1000, xt * R_EARTH / 1000, color=c, ls=ls, lw=1.5, label=l.capitalize())
        axes[1].plot(t, (clean[i, :, 0] - R_EARTH) / 1000, color=c, ls=ls, lw=1.5)
        axes[2].plot(t, np.rad2deg(bank[i]), color=c, ls=ls, lw=1.5)
    for axes in grid:
        axes[0].set_xlabel("Downrange (km)"); axes[0].set_ylabel("Cross-range (km)")
        axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Altitude (km)")
        axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Bank angle (deg)")
        for ax in axes[1:]:
            ax.set_xlim(0, clean.shape[1] - 1)
    legend(grid[0][0], loc="upper left")
    save_panels(fig, grid, out, "fig_dataset")


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
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5))
    names = ["A_fit", "iTransformer+prior+series", "Ours-MS"]
    for ax, sigma in zip(axes[:2], (50, 200)):
        for name in names:
            a = z[f"{KEYS[name]}/sigma{sigma}/ade"].mean(0)
            traj = np.sort(np.bincount(rows, a) / np.bincount(rows))
            color, ls = STYLE[name]
            ax.plot(traj, np.arange(1, len(traj) + 1) / len(traj), color=color, ls=ls, lw=lw(name), label=LABEL[name])
        ax.set_xlabel("Per-trajectory ADE over 256 s (km)")
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
    axes[2].tick_params(axis="x", bottom=False, top=False)
    axes[2].set_ylim(0, 10); legend(axes[2], *axes[0].get_legend_handles_labels(), loc="upper right")
    save_panels(fig, axes, out, "fig_tail_direction")
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
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.6), gridspec_kw={"height_ratios": [1.25, 1]})
    for col, s in enumerate(cases):
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
        axh.plot(np.arange(-255, 1), alt(hist), color=OBSERVED, lw=lw("Observed"), label="Observed track")
        tf = np.arange(1, 257)
        for name, y in tracks.items():
            color, ls = ("black", "-") if name == "Truth" else STYLE[name]
            ax.plot(tf, cross(y), color=color, ls=ls, lw=lw(name))
            axh.plot(tf, alt(y), color=color, ls=ls, lw=lw(name), label="Truth" if name == "Truth" else LABEL[name])
        ax.set_xlim(0, 256); ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xlabel("Prediction time (s)")
        axh.set_xlabel("Time from newest measurement (s)"); axh.set_xlim(-255, 256)
        ade = {k: np.linalg.norm(v - tgt[w], axis=-1).mean() for k, v in tracks.items() if k != "Truth"}
        print(f"{s}: window {i}, ADE " + ", ".join(f"{k} {v:.2f}" for k, v in ade.items()))
    axes[0, 0].set_ylabel("Cross-track offset from prior (km)"); axes[1, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_ylim(top=27); legend(axes[0, 0], *axes[1, 0].get_legend_handles_labels(), loc="upper left")
    save_panels(fig, axes, out, "fig_cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="subset of figure functions, e.g. fig_cases")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_error_time, fig_noise, fig_case, fig_dataset, fig_tail_direction, fig_cases):
        if not args.only or f.__name__ in args.only:
            f(args.out)
