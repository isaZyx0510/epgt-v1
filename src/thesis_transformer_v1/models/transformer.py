"""Transformer models for hybrid and direct-H baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    input_dim: int
    l_eff: int = 12
    n_rx: int = 4
    n_tx: int = 4
    n_sym: int = 8
    n_sc: int = 48
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_rel_delay_s: float = 3.0e-6
    max_doppler_hz: float = 200.0
    max_total_delay_s: float = 1.0e-6
    max_cfo_hz: float = 500.0
    max_rx_time_offset_s: float = 200.0e-9
    predict_path_uncertainty: bool = False


class TokenEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_proj(tokens))
        return encoded.mean(dim=1)


class HybridTransformer(nn.Module):
    """Predict nonlinear parameters only; path gains are recovered by LS."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TokenEncoder(config)
        out_dim = 2 + 2 * config.l_eff + config.n_rx
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

        rel_delay = torch.sigmoid(path_raw[..., 0]) * self.config.max_rel_delay_s
        doppler = torch.tanh(path_raw[..., 1]) * self.config.max_doppler_hz
        rx_offsets = torch.tanh(rx_raw) * self.config.max_rx_time_offset_s
        rx_offsets = rx_offsets.clone()
        rx_offsets[:, 0] = 0.0
        return {
            "total_delay_s": torch.sigmoid(tau0_raw) * self.config.max_total_delay_s,
            "cfo_hz": torch.tanh(cfo_raw) * self.config.max_cfo_hz,
            "rel_delay_s": rel_delay,
            "doppler_hz": doppler,
            "rx_time_offsets_s": rx_offsets,
        }


class DirectHTransformer(nn.Module):
    """Directly regress full-grid complex H as a black-box baseline."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TokenEncoder(config)
        out_dim = 2 * config.n_rx * config.n_tx * config.n_sym * config.n_sc
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(tokens))
        shape = (
            tokens.shape[0],
            self.config.n_rx,
            self.config.n_tx,
            self.config.n_sym,
            self.config.n_sc,
            2,
        )
        return raw.reshape(shape)
