"""Losses for sparse-to-set nonlinear parameter learning."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def sparse_to_set_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    path_weight: float = 1.0,
    delta_weight: float = 0.1,
    existence_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    path_mask = batch["path_mask"].to(outputs["path_params"].dtype)
    path_error = (outputs["path_params"] - batch["path_params"]) ** 2
    path_loss = (path_error.sum(dim=-1) * path_mask).sum() / path_mask.sum().clamp_min(1.0)

    delta_loss = F.mse_loss(outputs["delta_r"], batch["delta_r"])
    existence_loss = F.binary_cross_entropy_with_logits(outputs["path_logits"], path_mask)
    total = path_weight * path_loss + delta_weight * delta_loss + existence_weight * existence_loss
    return {
        "loss": total,
        "path_loss": path_loss,
        "delta_loss": delta_loss,
        "existence_loss": existence_loss,
    }
