"""BSR-Net models: ManeuverAST (BSR-Net and its rollout/series ablations) and ResidualAST (residual-on-prior and direct ablations)."""

from __future__ import annotations

import torch
from torch import nn


class ResidualAST(nn.Module):
    """Transformer encoder over the measured history, horizon-query decoder.

    All inputs are normalised by the training script.  The model returns a
    normalised per-step output ``[B, H, 3]`` and a bank-change logit ``[B]``.
    For the residual variant the script maps the output to kilometres by a
    per-step scale only (no offset), and the output layer is zero initialised,
    so an untrained model reproduces the physics prior exactly.  ``direct``
    drops the prior input (matched pure-learning baseline).  With ``use_aux``
    the decoder queries are conditioned on the predicted change probability.
    """

    def __init__(self, *, d_model: int = 128, layers: int = 3, nhead: int = 4, history_dim: int = 6,
                 feature_dim: int = 10, horizon: int = 256, direct: bool = False, use_aux: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.direct, self.use_aux = bool(direct), bool(use_aux)
        self.dropout = dropout
        self.history = nn.Linear(history_dim, d_model)
        self.positional = nn.Parameter(0.02 * torch.randn(1, 256, d_model))
        self.meta = nn.Sequential(nn.Linear(feature_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = dict(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=self.dropout, batch_first=True,
                     activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), num_layers=layers)
        self.pool = nn.Linear(d_model, 1)
        self.aux_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.queries = nn.Parameter(0.02 * torch.randn(1, horizon, d_model))
        self.prior_in = None if direct else nn.Linear(3, d_model)
        self.p_in = nn.Linear(1, d_model)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), num_layers=2)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, history: torch.Tensor, prior: torch.Tensor | None, features: torch.Tensor):
        tokens = self.encoder(self.history(history) + self.positional[:, :history.size(1)])
        weights = torch.softmax(self.pool(tokens).squeeze(-1), dim=1)
        context = (tokens * weights[..., None]).sum(dim=1)
        meta = self.meta(features)
        logit = self.aux_head(torch.cat((context, meta), dim=-1)).squeeze(-1)
        query = self.queries.expand(history.size(0), -1, -1)
        if self.prior_in is not None:
            query = query + self.prior_in(prior)
        if self.use_aux:
            query = query + self.p_in(torch.sigmoid(logit).detach()[:, None, None])
        memory = torch.cat((meta[:, None], tokens), dim=1)
        return self.out(self.decoder(query, memory)), logit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ManeuverAST(ResidualAST):
    """Maneuver-aware variant: predict the future bank schedule, integrate the 3-DOF model, learn the remainder.

    A block decoder reads the encoded history and predicts a bank offset for each ``block``-second slot of the
    horizon (zero initialised).  The training script adds it to the identified bank, rolls the shared rotating-
    Earth 3-DOF model out with the identified C_L, C_D (differentiable), and the residual decoder corrects that
    maneuver-aware rollout.  Untrained, both offsets are zero, so the output is exactly the A_fit prior.
    """

    maneuver = True

    def __init__(self, *, blocks: int = 32, bank_scale: float = 0.6, series_dim: int = 0, series_len: int = 15, **kwargs):
        super().__init__(use_aux=False, **kwargs)
        d_model = self.queries.size(-1)
        layer = dict(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, dropout=self.dropout, batch_first=True,
                     activation="gelu", norm_first=True)
        self.bank_scale = bank_scale
        self.block_queries = nn.Parameter(0.02 * torch.randn(1, blocks, d_model))
        self.block_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), num_layers=2)
        self.bank_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.shift_in = nn.Linear(3, d_model)
        self.use_series = series_dim > 0  # identified parameter series as extra encoder tokens
        if self.use_series:
            self.series_in = nn.Linear(series_dim, d_model)
            self.series_pos = nn.Parameter(0.02 * torch.randn(1, series_len, d_model))
        nn.init.zeros_(self.bank_out[-1].weight)
        nn.init.zeros_(self.bank_out[-1].bias)

    def encode(self, history: torch.Tensor, features: torch.Tensor, series: torch.Tensor | None = None):
        tokens = self.history(history) + self.positional[:, :history.size(1)]
        if self.use_series:
            tokens = torch.cat((tokens, self.series_in(series) + self.series_pos), dim=1)
        tokens = self.encoder(tokens)
        memory = torch.cat((self.meta(features)[:, None], tokens), dim=1)
        query = self.block_queries.expand(history.size(0), -1, -1)
        return memory, self.bank_scale * self.bank_out(self.block_decoder(query, memory)).squeeze(-1)  # rad offsets [B, K]

    def correct(self, memory: torch.Tensor, prior: torch.Tensor, shift: torch.Tensor):
        """prior: normalised A_fit rollout; shift: maneuver rollout minus A_fit rollout, in residual units."""
        query = self.queries.expand(memory.size(0), -1, -1) + self.prior_in(prior) + self.shift_in(shift)
        return self.out(self.decoder(query, memory))
