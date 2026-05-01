"""Digital modulation helpers."""

from __future__ import annotations

import numpy as np


def modulation_order(name: str) -> int:
    table = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    try:
        return table[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported modulation: {name}") from exc


def bits_per_symbol(name: str) -> int:
    return int(np.log2(modulation_order(name)))


def _gray_to_binary(gray: np.ndarray) -> np.ndarray:
    binary = gray.copy()
    shift = 1
    while (1 << shift) <= int(np.max(gray, initial=0)):
        binary ^= gray >> shift
        shift += 1
    return binary


def qam_modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map bits to unit-average-power QPSK/16QAM/64QAM symbols."""
    order = modulation_order(modulation)
    m = int(np.log2(order))
    if bits.shape[-1] != m:
        raise ValueError(f"Expected last bit dimension {m}, got {bits.shape[-1]}")

    bits = bits.astype(np.int64, copy=False)
    if order == 4:
        i = 1.0 - 2.0 * bits[..., 0]
        q = 1.0 - 2.0 * bits[..., 1]
        return (i + 1j * q) / np.sqrt(2.0)

    axis_bits = m // 2
    weights = 1 << np.arange(axis_bits - 1, -1, -1)
    gray_i = np.sum(bits[..., :axis_bits] * weights, axis=-1)
    gray_q = np.sum(bits[..., axis_bits:] * weights, axis=-1)
    idx_i = _gray_to_binary(gray_i)
    idx_q = _gray_to_binary(gray_q)

    sqrt_order = int(np.sqrt(order))
    levels_i = 2.0 * idx_i - (sqrt_order - 1)
    levels_q = 2.0 * idx_q - (sqrt_order - 1)
    norm = np.sqrt((2.0 / 3.0) * (order - 1))
    return (levels_i + 1j * levels_q) / norm
