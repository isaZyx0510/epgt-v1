"""Sparse pilot observations as point-cloud tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class SparsePilotTokenConfig:
    include_rx: bool = True
    include_pilot: bool = True
    include_coords: bool = True
    coord_range: str = "minus_one_to_one"


def complex_to_channels(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array.real, array.imag], axis=-1).astype(np.float32)


def normalized_coords(symbol_idx: np.ndarray, sc_idx: np.ndarray, n_sym: int, n_sc: int) -> np.ndarray:
    if n_sym <= 1:
        n_coord = np.zeros_like(symbol_idx, dtype=np.float32)
    else:
        n_coord = 2.0 * symbol_idx.astype(np.float32) / float(n_sym - 1) - 1.0
    if n_sc <= 1:
        k_coord = np.zeros_like(sc_idx, dtype=np.float32)
    else:
        k_coord = 2.0 * sc_idx.astype(np.float32) / float(n_sc - 1) - 1.0
    return np.stack([k_coord, n_coord], axis=-1).astype(np.float32)


def sparse_pilot_mask(pilot_mask: np.ndarray) -> np.ndarray:
    """Collapse per-Tx pilot mask [N_t, N_sym, N_sc] into observed RE mask."""
    if pilot_mask.ndim != 3:
        raise ValueError(f"pilot_mask must be [N_t, N_sym, N_sc], got {pilot_mask.shape}")
    return np.any(pilot_mask, axis=0)


def build_sparse_pilot_tokens(
    rx_grid: np.ndarray,
    pilot_symbols: np.ndarray,
    pilot_mask: np.ndarray,
    config: SparsePilotTokenConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Build token tensor from one sample.

    Args:
        rx_grid: complex array [N_r, N_sym, N_sc].
        pilot_symbols: complex array [N_t, N_sym, N_sc].
        pilot_mask: bool array [N_t, N_sym, N_sc].

    Returns:
        dict with:
            tokens: float tensor [N_obs, feature_dim]
            coords: float tensor [N_obs, 2], columns are normalized [k, n]
            indices: long tensor [N_obs, 2], columns are [n, k]
    """
    config = config or SparsePilotTokenConfig()
    if rx_grid.ndim != 3:
        raise ValueError(f"rx_grid must be [N_r, N_sym, N_sc], got {rx_grid.shape}")
    if pilot_symbols.ndim != 3:
        raise ValueError(f"pilot_symbols must be [N_t, N_sym, N_sc], got {pilot_symbols.shape}")

    _, n_sym, n_sc = rx_grid.shape
    observed = sparse_pilot_mask(pilot_mask)
    symbol_idx, sc_idx = np.nonzero(observed)
    if len(symbol_idx) == 0:
        raise ValueError("No observed pilot REs found")

    features = []
    if config.include_rx:
        y = rx_grid[:, symbol_idx, sc_idx].T
        features.append(complex_to_channels(y))
    if config.include_pilot:
        x = pilot_symbols[:, symbol_idx, sc_idx].T
        features.append(complex_to_channels(x))
    coords = normalized_coords(symbol_idx, sc_idx, n_sym, n_sc)
    if config.include_coords:
        features.append(coords)

    tokens = np.concatenate(features, axis=-1).astype(np.float32)
    indices = np.stack([symbol_idx, sc_idx], axis=-1).astype(np.int64)
    return {
        "tokens": torch.from_numpy(tokens),
        "coords": torch.from_numpy(coords),
        "indices": torch.from_numpy(indices),
    }


def token_feature_dim(n_rx: int, n_tx: int, config: SparsePilotTokenConfig | None = None) -> int:
    config = config or SparsePilotTokenConfig()
    dim = 0
    if config.include_rx:
        dim += 2 * n_rx
    if config.include_pilot:
        dim += 2 * n_tx
    if config.include_coords:
        dim += 2
    return dim
