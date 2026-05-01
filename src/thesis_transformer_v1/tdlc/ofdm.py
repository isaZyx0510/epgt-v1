"""OFDM modulation and demodulation."""

from __future__ import annotations

import numpy as np


def ofdm_modulate(tx_grid: np.ndarray, cp_len: int) -> np.ndarray:
    """IFFT and CP insertion. Input [B, N_ant, N_sym, N_sc]."""
    time_no_cp = np.fft.ifft(tx_grid, axis=-1) * np.sqrt(tx_grid.shape[-1])
    cp = time_no_cp[..., -cp_len:]
    with_cp = np.concatenate([cp, time_no_cp], axis=-1)
    bsz, n_ant, n_sym, n_total = with_cp.shape
    return with_cp.reshape(bsz, n_ant, n_sym * n_total).astype(np.complex64)


def ofdm_demodulate(rx_time: np.ndarray, n_sym: int, n_sc: int, cp_len: int) -> np.ndarray:
    """Remove CP and FFT. Output [B, N_ant, N_sym, N_sc]."""
    n_total = n_sc + cp_len
    rx = rx_time.reshape(rx_time.shape[0], rx_time.shape[1], n_sym, n_total)
    rx = rx[..., cp_len:]
    return (np.fft.fft(rx, axis=-1) / np.sqrt(n_sc)).astype(np.complex64)
