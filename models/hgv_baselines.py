"""HGV-specific published baselines reimplemented for the noisy position-only protocol.

Neither paper has public code.  Both are built on the official TSLib layers /
algorithms they extend (AF-CILN cites SCINet, PIT extends Informer),
with only the changes our protocol forces: 256 s in / 256 s out at 1 Hz, and
the shared normalised 6-channel noisy history (local ENU km + first
differences, i.e. position plus a velocity auxiliary feature) as input.
Training uses the common protocol of ``scripts/pipeline.py``
(AdamW + OneCycle, lr from {1e-3, 3e-4} on validation, early stopping,
L2 displacement loss) instead of each paper's schedule.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path

import torch
from torch import nn

TSLIB_ROOT = os.getenv("HGV_TSLIB_ROOT", str(Path(__file__).resolve().parents[1] / "third_party" / "Time-Series-Library"))


def _haar(x: torch.Tensor) -> torch.Tensor:
    """Recursive 1-D Haar transform along the last axis (AF-CILN Eq. 12)."""
    if x.size(-1) == 1:
        return x
    if x.size(-1) % 2:
        return torch.cat((_haar(x[..., :-1]), x[..., -1:]), dim=-1)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.cat((_haar((even + odd) / math.sqrt(2)), (even - odd) / math.sqrt(2)), dim=-1)


class _Conv(nn.Sequential):
    """Conv1d-LeakyReLU-Dropout-Conv1d-Tanh, length preserving (AF-CILN Fig. 7 / SCINet)."""

    def __init__(self, channels: int, kernel: int, dropout: float):
        pad = kernel - 1
        super().__init__(nn.ReplicationPad1d((pad, pad)), nn.Conv1d(channels, channels, kernel), nn.LeakyReLU(0.01),
                         nn.Dropout(dropout), nn.Conv1d(channels, channels, kernel), nn.Tanh())


class _CILTree(nn.Module):
    """Binary tree of CIL-Blocks (Eqs. 20-22); returns the re-interleaved sequence."""

    def __init__(self, channels: int, level: int, kernel: int, dropout: float):
        super().__init__()
        self.psi, self.phi, self.rho, self.eta = (_Conv(channels, kernel, dropout) for _ in range(4))
        self.subtrees = nn.ModuleList([_CILTree(channels, level - 1, kernel, dropout) for _ in range(2)]) if level > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, L], L divisible by 2**level
        even, odd = x[..., ::2], x[..., 1::2]
        s_even, s_odd = even * torch.exp(self.psi(odd)), odd * torch.exp(self.phi(even))
        even, odd = s_even + self.rho(s_odd), s_odd - self.eta(s_even)
        if self.subtrees is not None:
            even, odd = self.subtrees[0](even), self.subtrees[1](odd)
        return torch.stack((even, odd), dim=-1).flatten(-2)


class AFCILN(nn.Module):
    """AF-CILN (Cai et al., AST 177 (2026) 112222).

    As in the paper: per-sample normalisation of the input and
    de-normalisation of the output (Eqs. 8-11); AHWT channel attention
    (recursive Haar transform of every channel, FC block with ReLU, dropout
    and sigmoid, element-wise scaling of the input, Eqs. 12-15); sinusoidal
    positional encoding with log-spaced time scales (Eqs. 16-19); a K = 4
    level binary tree of CIL-Blocks (kernel 3, dropout 0.2); residual
    connection and linear decoder over time (Eq. 23); C_in = 6, C_out = 3.
    Deviations: there is no missing data here, so the mask is all ones and
    mask normalisation reduces to per-sample normalisation; the auxiliary
    channels are first differences of the noisy positions (the only velocity
    information a position-only radar track gives) instead of simulator
    velocity / attitude; the FC block acts along time on the C x L
    coefficient matrix with hidden size L/4 (not given in the paper).
    Outputs continue the history position channels, so the training script
    maps them with the history statistics (``direct == "hist"``).
    """

    direct = "hist"

    def __init__(self, input_dim: int = 6, output_dim: int = 3, seq_len: int = 256, horizon: int = 256,
                 levels: int = 4, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(seq_len, seq_len // 4), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(seq_len // 4, seq_len), nn.Sigmoid())
        n = (input_dim + 1) // 2
        inv = torch.exp(torch.arange(n, dtype=torch.float32) * -(math.log(1e4) / max(n - 1, 1)))
        t = torch.arange(seq_len, dtype=torch.float32)[:, None] * inv[None]
        self.register_buffer("pe", torch.cat((t.sin(), t.cos()), dim=1)[:, :input_dim].T[None], persistent=False)
        self.tree = _CILTree(input_dim, levels, kernel, dropout)
        self.decoder = nn.Linear(seq_len, horizon)

    def forward(self, history: torch.Tensor, prior=None, features=None) -> torch.Tensor:
        x = history.transpose(1, 2)  # [B, C, L]
        mean = x.mean(-1, keepdim=True).detach()
        std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - mean) / std
        x = x * self.fc(_haar(x)) + self.pe
        y = self.decoder(x + self.tree(x)) * std + mean
        return y[:, :self.output_dim].transpose(1, 2)


def _tslib_layers():
    """Official TSLib Informer layers (files hash-checked by the public adapter)."""
    from models.public_baselines import _tslib_import_context, resolve_tslib_source

    source = resolve_tslib_source("transformer", root=TSLIB_ROOT)
    with _tslib_import_context(Path(str(source["root"]))):
        return tuple(importlib.import_module(m) for m in ("layers.Transformer_EncDec", "layers.SelfAttention_Family",
                                                          "layers.Embed"))


def _top_mean_attention(base):
    class TopMeanAttention(base):
        """TSLib ProbSparse attention with PIT's top-delta-mean sparsity measure (Eq. 10)."""

        def __init__(self, *args, delta: int = 2, **kwargs):
            super().__init__(*args, **kwargs)
            self.delta = delta

        def _prob_QK(self, Q, K, sample_k, n_top):
            B, H, L_K, _ = K.shape
            idx = torch.randint(L_K, (Q.size(2), sample_k), device=Q.device)
            qk = torch.einsum("bhqd,bhqkd->bhqk", Q, K[:, :, idx])
            M = qk.topk(min(self.delta, sample_k), dim=-1)[0].mean(-1) - qk.sum(-1) / L_K
            top = M.topk(n_top, sorted=False)[1]
            q = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], top]
            return q @ K.transpose(-2, -1), top

    return TopMeanAttention


def _savgol_operators(length: int, window: int = 21, order: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Savitzky-Golay smoothing and first derivative (dt = 1 s) as [L, L] matrices (scipy 'interp' edges)."""
    from scipy.signal import savgol_filter

    eye = torch.eye(length, dtype=torch.float64).numpy()
    return tuple(torch.from_numpy(savgol_filter(eye, window, order, deriv=d, axis=0).copy()).float() for d in (0, 1))


class PIT(nn.Module):
    """Physics-informed Transformer (Ren et al., IEEE TAES 59(6), 2023; Ren Jihuan, PhD thesis, NJUST, ch. 4).

    As in the paper / thesis: the radar track is first denoised with a
    Savitzky-Golay filter (window 21, order 3) and fed as five channels
    (three positions, speed V, flight-path angle gamma); Informer-style
    encoder (four layers of multi-head top-delta-mean self-attention,
    delta = 2, with Conv-pooling distillation) and generative decoder (two
    layers; masked top-delta-mean self-attention and full cross-attention as
    in Informer; start token = second half of the encoder input followed by
    zero placeholders; one forward pass); GELU; stage encoding PE +/- eps for
    rising / falling samples (Eq. 14, eps "a very small constant");
    physics-informed loss of Eq. 13 on the last m predicted steps
    (``pit_physics_loss``).  Five channels are predicted per step.
    Deviations: positions are local ENU km of the noisy position-only track
    (the paper uses lon / lat / h) and V, gamma come from the filtered ENU
    velocity (gamma w.r.t. the local vertical, h = U + (E^2 + N^2) / 2R);
    rising / falling = sign of the filtered gamma; d_model 256 / 8 heads /
    d_ff 1024 as for the TSLib baselines, eps = 0.01 and m = 32 (not given);
    256 s history at 1 Hz like every method here (the thesis uses 64 samples
    at 2 s for L = 256); training uses the common protocol plus the PIT terms
    instead of Adam with 0.8 per-epoch decay, batch 32 and patience 15.
    ``set_stats`` stores the normalisation constants (saved with the checkpoint).
    """

    direct = True
    stage_offset = 0.01
    physics_steps = 32
    radius_km = 6411.0

    def __init__(self, input_dim: int = 5, output_dim: int = 5, seq_len: int = 256, horizon: int = 256,
                 d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, e_layers: int = 4, d_layers: int = 2,
                 factor: int = 5, dropout: float = 0.05, prior_input: bool = False):
        super().__init__()
        self.prior_input = prior_input
        if prior_input:  # "+prior" ablation: A_fit rollout fills the decoder placeholders, output = prior + gated correction
            self.direct = "prior"
            self.gate = nn.Parameter(torch.zeros(1))
            self.register_buffer("prior_mean", torch.zeros(horizon, 3))
            self.register_buffer("prior_std", torch.ones(horizon, 3))
        enc_dec, attn, embed = _tslib_layers()
        top_mean = _top_mean_attention(attn.ProbAttention)
        self.label_len, self.horizon = seq_len // 2, horizon
        smooth, deriv = _savgol_operators(seq_len)
        self.register_buffer("smooth", smooth, persistent=False)
        self.register_buffer("deriv", deriv, persistent=False)
        for name, size in (("hist_mean", 6), ("hist_std", 6), ("vg_mean", 2), ("vg_std", 2)):
            self.register_buffer(name, torch.zeros(size) if name.endswith("mean") else torch.ones(size))
        self.enc_value = embed.TokenEmbedding(input_dim, d_model)
        self.dec_value = embed.TokenEmbedding(input_dim, d_model)
        self.position = embed.PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        sparse = lambda mask: attn.AttentionLayer(top_mean(mask, factor, attention_dropout=dropout), d_model, n_heads)
        full = lambda: attn.AttentionLayer(attn.FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads)
        self.encoder = enc_dec.Encoder([enc_dec.EncoderLayer(sparse(False), d_model, d_ff, dropout=dropout, activation="gelu")
                                        for _ in range(e_layers)],
                                       [enc_dec.ConvLayer(d_model) for _ in range(e_layers - 1)],
                                       norm_layer=nn.LayerNorm(d_model))
        self.decoder = enc_dec.Decoder([enc_dec.DecoderLayer(sparse(True), full(), d_model, d_ff, dropout=dropout,
                                                             activation="gelu") for _ in range(d_layers)],
                                       norm_layer=nn.LayerNorm(d_model), projection=nn.Linear(d_model, output_dim))

    def set_stats(self, hist_mean, hist_std, vg_mean, vg_std, prior_mean=None, prior_std=None):
        values = (("hist_mean", hist_mean), ("hist_std", hist_std), ("vg_mean", vg_mean), ("vg_std", vg_std))
        if self.prior_input:
            values += (("prior_mean", prior_mean), ("prior_std", prior_std))
        for name, value in values:
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def filtered_track(self, history: torch.Tensor, km=None):
        """Normalised history [B, L, 6] (or ENU km ``km`` [B, L, 3]) -> SG-filtered ENU km [B, L, 3], speed V (km/s)
        and gamma (rad) [B, L]."""
        with torch.autocast(history.device.type, enabled=False):
            pos = history[..., :3].float() * self.hist_std[:3] + self.hist_mean[:3] if km is None else km.float()
            smooth, vel = self.smooth @ pos, self.deriv @ pos
            speed = vel.norm(dim=-1).clamp_min(1e-6)
            climb = vel[..., 2] + (smooth[..., 0] * vel[..., 0] + smooth[..., 1] * vel[..., 1]) / self.radius_km
            return smooth, speed, torch.asin((climb / speed).clamp(-1.0, 1.0))

    def channels(self, pos, speed, gamma):
        return torch.cat(((pos - self.hist_mean[:3]) / self.hist_std[:3],
                          (torch.stack((speed, gamma), -1) - self.vg_mean) / self.vg_std), -1)

    def forward(self, history: torch.Tensor, prior=None, features=None) -> torch.Tensor:
        pos, speed, gamma = self.filtered_track(history)
        x = self.channels(pos, speed, gamma)
        stage = self.stage_offset * torch.sign(gamma)[..., None]
        enc = self.drop(self.enc_value(x) + self.position(x) + stage)
        future = x.new_zeros(x.size(0), self.horizon, x.size(2))
        if self.prior_input:
            future = self.channels(*self.filtered_track(history, prior.float() * self.prior_std + self.prior_mean)).to(x.dtype)
        x_dec = torch.cat((x[:, -self.label_len:], future), 1)
        dec_stage = torch.cat((stage[:, -self.label_len:], torch.zeros_like(future[..., :1])), 1)
        dec = self.drop(self.dec_value(x_dec) + self.position(x_dec) + dec_stage)
        memory, _ = self.encoder(enc, attn_mask=None)
        out = self.decoder(dec, memory, x_mask=None, cross_mask=None)[:, -self.horizon:]
        if self.prior_input:  # positions: correction in prior-normalised units on top of the rollout; (V, gamma) as before
            out = torch.cat((self.gate * out[..., :3], out[..., 3:]), -1)
        return out


class PriorInput(nn.Module):
    """"+prior" ablation for a TSLib forecaster: it sees the normalised A_fit rollout as three extra input channels
    (placed first, so its RevIN restores them) and its output becomes prior + gate * (output - prior); gate starts
    at 0, so the untrained model returns the rollout exactly, like Ours."""

    direct = "prior"

    def __init__(self, base: nn.Module, use_features: bool = False, use_series: bool = False):
        super().__init__()
        self.base, self.gate, self.use_features, self.use_series = base, nn.Parameter(torch.zeros(1)), use_features, use_series

    def forward(self, history, prior, features=None, series=None):
        # use_features: Ours' scalar features enter as iTransformer covariate tokens (x_mark, not RevIN-normalised);
        # use_series: the identified parameter series (B, 15, 6), held over the history steps, adds 6 more tokens
        mark = features[:, None].expand(-1, history.size(1), -1) if self.use_features else None
        if series is not None:
            series = nn.functional.interpolate(series.transpose(1, 2), size=history.size(1), mode="nearest").transpose(1, 2)
            mark = series if mark is None else torch.cat((mark, series.to(mark.dtype)), -1)
        return self.gate * (self.base(torch.cat((prior, history), -1), x_mark=mark) - prior)


def pit_physics_loss(out: torch.Tensor, pred_km: torch.Tensor, vg: torch.Tensor, vg_mean: torch.Tensor,
                     vg_std: torch.Tensor, vz_std: float, m: int = PIT.physics_steps, radius_km: float = PIT.radius_km):
    """PIT terms of Eq. 13 beyond the position error: MSE on (V, gamma) and the
    physics residual dh/dt - V sin(gamma) on the last ``m`` steps (1 Hz).
    ``pred_km`` is local ENU km at the last measurement; h adds the spherical
    drop (E^2 + N^2) / 2R.  Every term is normalised to unit variance."""
    vg_n = out[..., 3:5].float()
    mse_vg = (vg_n - (vg - vg_mean) / vg_std).square().mean()
    v, g = (vg_n * vg_std + vg_mean).unbind(-1)
    vz = v * torch.sin(g)
    h = pred_km[..., 2] + (pred_km[..., 0].square() + pred_km[..., 1].square()) / (2 * radius_km)
    dh = h[:, -m:] - h[:, -m - 1:-1]
    phy = ((dh - 0.5 * (vz[:, -m:] + vz[:, -m - 1:-1])) / vz_std).square().mean()
    return mse_vg + phy


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["AFCILN", "PIT", "PriorInput", "pit_physics_loss", "count_parameters"]
