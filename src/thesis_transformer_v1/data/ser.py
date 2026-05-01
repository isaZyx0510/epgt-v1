"""Symbol-error corruption for data-aided observations."""

from __future__ import annotations

import numpy as np


def corrupt_data_symbols(
    rng: np.random.Generator,
    tx_grid: np.ndarray,
    data_mask: np.ndarray,
    symbol_indices: tuple[int, ...],
    symbol_error_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt data REs in selected OFDM symbols while leaving pilots untouched.

    The replacement `-x` stays on the square-QAM constellation used by the
    generator and is guaranteed to differ from `x`.
    """
    if not 0.0 <= symbol_error_rate <= 1.0:
        raise ValueError("symbol_error_rate must be in [0, 1]")
    corrupted = np.asarray(tx_grid).copy()
    changed = np.zeros(tx_grid.shape, dtype=bool)
    if symbol_error_rate == 0.0:
        return corrupted, changed

    selected = np.zeros(tx_grid.shape[2:], dtype=bool)
    selected[np.asarray(symbol_indices, dtype=int), :] = True
    candidate = data_mask[None, :, :, :] & selected[None, None, :, :]
    draw = rng.random(tx_grid.shape) < symbol_error_rate
    changed = candidate & draw
    corrupted[changed] = -corrupted[changed]
    return corrupted.astype(np.complex64), changed

