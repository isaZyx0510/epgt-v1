"""Guided cross-attention blocks for EPGT-v1."""

from __future__ import annotations

import torch
from torch import nn


class GuidedCrossAttention(nn.Module):
    """Multi-head cross-attention with additive physics bias."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = None
        if attn_bias is not None:
            attn_mask = attn_bias.repeat_interleave(self.nhead, dim=0)
        output, weights = self.attention(
            query,
            context,
            context,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return output, weights

