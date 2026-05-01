"""Physics regularizers for EPGT-style models."""

from __future__ import annotations

import torch


def center_doppler(
    doppler_hz: torch.Tensor,
    path_gates: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Center residual Dopplers using path gates as weights."""
    denom = torch.sum(path_gates, dim=-1, keepdim=True).clamp_min(float(eps))
    mean = torch.sum(path_gates * doppler_hz, dim=-1, keepdim=True) / denom
    return doppler_hz - mean


def path_gate_sparsity_loss(path_gates: torch.Tensor) -> torch.Tensor:
    """L1 sparsity on effective-path gates."""
    return path_gates.mean()


def group_gain_sparsity_loss(path_gains_ri: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Group sparsity for path gains represented as real/imag tensors."""
    power = path_gains_ri.square().sum(dim=(-3, -2, -1))
    return torch.sqrt(power + float(eps)).mean()
