"""Diagnostics for physics-guided attention."""

from __future__ import annotations

import torch


def attention_entropy(attention_weights: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Compute entropy over the context dimension of attention weights."""
    weights = attention_weights.clamp_min(float(eps))
    return -torch.sum(weights * torch.log(weights), dim=-1)


def prior_alignment(
    attention_weights: torch.Tensor,
    kernel: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Measure normalized overlap between attention and a physical prior kernel."""
    prior = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return torch.sum(attention_weights * prior, dim=-1)
