"""Uncertainty-aware Transformer architecture for weighted LS experiments."""

from __future__ import annotations

import torch
from torch import nn

from thesis_transformer_v1.models.transformer import TokenEncoder, TransformerConfig


class UncertaintyHybridTransformer(nn.Module):
    """Predict nonlinear parameters plus diagonal uncertainty estimates.

    This keeps the original mean-pooling encoder style while adding uncertainty
    heads for the weighted LS plugin.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TokenEncoder(config)
        out_dim = 2 + 2 * config.l_eff + config.n_rx + 2 * config.l_eff
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.head(self.encoder(tokens))
        cursor = 0
        tau0_raw = raw[:, cursor]
        cursor += 1
        cfo_raw = raw[:, cursor]
        cursor += 1
        path_raw = raw[:, cursor : cursor + 2 * self.config.l_eff].reshape(
            -1, self.config.l_eff, 2
        )
        cursor += 2 * self.config.l_eff
        rx_raw = raw[:, cursor : cursor + self.config.n_rx]
        cursor += self.config.n_rx
        uncertainty_raw = raw[:, cursor : cursor + 2 * self.config.l_eff].reshape(
            -1, self.config.l_eff, 2
        )

        rel_delay = torch.sigmoid(path_raw[..., 0]) * self.config.max_rel_delay_s
        doppler = torch.tanh(path_raw[..., 1]) * self.config.max_doppler_hz
        rx_offsets = torch.tanh(rx_raw) * self.config.max_rx_time_offset_s
        rx_offsets = rx_offsets.clone()
        rx_offsets[:, 0] = 0.0

        rel_delay_var = (
            torch.nn.functional.softplus(uncertainty_raw[..., 0])
            * max(self.config.max_rel_delay_s, 1e-12) ** 2
        )
        doppler_var = (
            torch.nn.functional.softplus(uncertainty_raw[..., 1])
            * max(self.config.max_doppler_hz, 1e-6) ** 2
        )
        return {
            "total_delay_s": torch.sigmoid(tau0_raw) * self.config.max_total_delay_s,
            "cfo_hz": torch.tanh(cfo_raw) * self.config.max_cfo_hz,
            "rel_delay_s": rel_delay,
            "doppler_hz": doppler,
            "rx_time_offsets_s": rx_offsets,
            "rel_delay_log_var": torch.log(rel_delay_var.clamp_min(1e-30)),
            "doppler_log_var": torch.log(doppler_var.clamp_min(1e-30)),
        }

