"""NMSE metrics."""

from __future__ import annotations

import numpy as np


def nmse(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.sum(np.abs(prediction - target) ** 2)
    denominator = np.sum(np.abs(target) ** 2) + eps
    return float(numerator / denominator)


def nmse_db(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return float(10.0 * np.log10(nmse(prediction, target, eps=eps) + eps))

