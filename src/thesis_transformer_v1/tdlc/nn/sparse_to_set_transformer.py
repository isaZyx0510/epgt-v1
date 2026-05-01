"""Sparse-to-set Transformer for nonlinear channel parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class SparseToSetTransformerConfig:
    input_dim: int
    max_paths: int = 8
    n_rx: int = 4
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    path_output_dim: int = 2


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseToSetTransformer(nn.Module):
    """Encode sparse pilot tokens and decode a fixed-size set of paths.

    Inputs:
        tokens: [B, N_obs, input_dim]
        token_padding_mask: optional bool [B, N_obs], True means padding.

    Outputs:
        path_params: [B, L, 2], default columns are normalized [mu, rho].
        path_logits: [B, L], useful for variable effective-path masks.
        delta_r: [B, N_r].
    """

    def __init__(self, config: SparseToSetTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.path_queries = nn.Parameter(torch.randn(config.max_paths, config.d_model) * 0.02)
        self.path_head = MLP(
            config.d_model,
            config.dim_feedforward,
            config.path_output_dim,
            dropout=config.dropout,
        )
        self.path_logit_head = nn.Linear(config.d_model, 1)
        self.delta_head = MLP(config.d_model, config.dim_feedforward, config.n_rx, dropout=config.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        memory = self.input_proj(tokens)
        memory = self.encoder(memory, src_key_padding_mask=token_padding_mask)

        bsz = tokens.shape[0]
        queries = self.path_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=token_padding_mask,
        )
        pooled = masked_mean(memory, token_padding_mask)
        return {
            "path_params": self.path_head(decoded),
            "path_logits": self.path_logit_head(decoded).squeeze(-1),
            "delta_r": self.delta_head(pooled),
        }


def masked_mean(tokens: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None:
        return tokens.mean(dim=1)
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / denom
