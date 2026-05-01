"""Attention mask helpers for physics-guided models."""

from __future__ import annotations

import torch


def reliability_attention_mask(
    reliability: torch.Tensor,
    *,
    min_reliability: float = 0.0,
    mask_value: float = -1.0e4,
) -> torch.Tensor:
    """Return an additive context mask from token reliability values.

    The output shape is `[B, 1, C]` so it can broadcast over query positions.
    """
    invalid = reliability < float(min_reliability)
    return invalid[:, None, :].to(dtype=reliability.dtype) * float(mask_value)


def ensure_at_least_one_valid(mask: torch.Tensor) -> torch.Tensor:
    """Avoid fully masked rows by clearing the first context position if needed."""
    fully_masked = torch.all(mask < 0.0, dim=-1, keepdim=True)
    if not torch.any(fully_masked):
        return mask
    first_clear = torch.zeros_like(mask[..., :1])
    return torch.cat([torch.where(fully_masked, first_clear, mask[..., :1]), mask[..., 1:]], dim=-1)

