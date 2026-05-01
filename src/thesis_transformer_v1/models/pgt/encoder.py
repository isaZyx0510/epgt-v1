"""Observation and query encoders for EPGT-v1."""

from __future__ import annotations

import torch
from torch import nn

from thesis_transformer_v1.models.transformer import TransformerConfig
from thesis_transformer_v1.physics.priors import normalized_ofdm_grid


class ObservationEncoder(nn.Module):
    """Encode sparse observation tokens with a learnable global token."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.global_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
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

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.input_proj(tokens)
        global_token = self.global_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([global_token, context], dim=1))
        return encoded[:, 0], encoded[:, 1:]


class FullGridQueryEncoder(nn.Module):
    """Create query embeddings for every OFDM grid point."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        grid = normalized_ofdm_grid(config.n_sym, config.n_sc)
        self.register_buffer("grid_coords", grid, persistent=False)
        self.query_proj = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

    def forward(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self.grid_coords.to(device)
        query = self.query_proj(coords).unsqueeze(0).expand(batch_size, -1, -1)
        return query, coords

