"""Receiver impairments."""

from __future__ import annotations

import numpy as np


def add_awgn(
    rng: np.random.Generator, signal: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray]:
    signal_power = np.mean(np.abs(signal) ** 2, axis=(1, 2, 3), keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return (signal + noise).astype(np.complex64), noise_power.squeeze().astype(np.float32)


def apply_cfo_time(
    waveform: np.ndarray,
    cfo_hz: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    sample_index = np.arange(waveform.shape[-1], dtype=np.float64)
    phase = np.exp(1j * 2.0 * np.pi * cfo_hz[:, None, None] * sample_index / sample_rate_hz)
    return (waveform * phase).astype(np.complex64)
