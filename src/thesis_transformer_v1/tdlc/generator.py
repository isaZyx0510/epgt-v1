"""End-to-end MIMO-OFDM data generation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from .channel import apply_channel_frequency, generate_channel
from .config import DatasetConfig
from .impairments import add_awgn, apply_cfo_time
from .modulation import bits_per_symbol, qam_modulate
from .ofdm import ofdm_demodulate, ofdm_modulate
from .pilots import generate_pilot_pattern


def fill_resource_grid(
    rng: np.random.Generator,
    cfg: DatasetConfig,
    pilot_mask: np.ndarray,
    pilot_symbols: np.ndarray,
    data_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz = cfg.batch_size
    ofdm = cfg.ofdm
    k = bits_per_symbol(ofdm.modulation)

    n_data = int(np.sum(data_mask[0]))
    if n_data == 0:
        raise ValueError(
            "Pilot pattern leaves no data REs. Increase pilot spacing or reduce pilot density."
        )

    bits = rng.integers(0, 2, size=(bsz, ofdm.n_tx, n_data, k), dtype=np.int8)
    data_symbols = qam_modulate(bits, ofdm.modulation).astype(np.complex64)
    tx_grid = np.broadcast_to(
        pilot_symbols[None, :, :, :],
        (bsz, ofdm.n_tx, ofdm.n_sym, ofdm.n_sc),
    ).copy()

    for b in range(bsz):
        for tx in range(ofdm.n_tx):
            tx_grid[b, tx, data_mask[tx]] = data_symbols[b, tx]
            tx_grid[b, tx, pilot_mask[tx]] = pilot_symbols[tx, pilot_mask[tx]]

    return bits, data_symbols, tx_grid


def generate_cfo(
    rng: np.random.Generator, cfg: DatasetConfig
) -> np.ndarray:
    imp = cfg.impairment
    if imp.random_cfo_hz > 0.0:
        return rng.uniform(-imp.random_cfo_hz, imp.random_cfo_hz, size=(cfg.batch_size,))
    return np.full((cfg.batch_size,), imp.cfo_hz, dtype=np.float64)


def generate_dataset(cfg: DatasetConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    pilot_mask, pilot_symbols, data_mask = generate_pilot_pattern(cfg.ofdm, cfg.pilot)
    bits, data_symbols, tx_grid = fill_resource_grid(
        rng, cfg, pilot_mask, pilot_symbols, data_mask
    )
    tx_time = ofdm_modulate(tx_grid, cfg.ofdm.cp_len)

    h_freq, channel_labels = generate_channel(rng, cfg)
    rx_grid_clean = apply_channel_frequency(tx_grid, h_freq)
    rx_grid_noisy, noise_power = add_awgn(rng, rx_grid_clean, cfg.impairment.snr_db)

    rx_time = ofdm_modulate(rx_grid_noisy, cfg.ofdm.cp_len)
    cfo_hz = generate_cfo(rng, cfg)
    sample_rate_hz = cfg.ofdm.n_sc * cfg.ofdm.subcarrier_spacing_hz
    if np.any(cfo_hz != 0.0):
        rx_time = apply_cfo_time(rx_time, cfo_hz, sample_rate_hz)
        rx_grid = ofdm_demodulate(rx_time, cfg.ofdm.n_sym, cfg.ofdm.n_sc, cfg.ofdm.cp_len)
    else:
        rx_grid = rx_grid_noisy

    return {
        "bits": bits,
        "data_symbols": data_symbols,
        "tx_grid": tx_grid,
        "tx_time": tx_time,
        "pilot_mask": pilot_mask,
        "pilot_symbols": pilot_symbols,
        "data_mask": data_mask,
        "rx_time": rx_time,
        "rx_grid": rx_grid.astype(np.complex64),
        "rx_grid_clean": rx_grid_clean,
        "h_freq": h_freq,
        "noise_power": noise_power,
        "cfo_hz": cfo_hz.astype(np.float32),
        "channel_labels": channel_labels,
        "config": asdict(cfg),
    }


def print_summary(data: Dict[str, Any]) -> None:
    keys = [
        "bits",
        "data_symbols",
        "tx_grid",
        "tx_time",
        "pilot_mask",
        "pilot_symbols",
        "data_mask",
        "rx_time",
        "rx_grid",
        "rx_grid_clean",
        "h_freq",
    ]
    for key in keys:
        value = data[key]
        print(f"{key:18s} shape={value.shape} dtype={value.dtype}")
    print("channel label shapes:")
    for key, value in data["channel_labels"].items():
        print(f"  {key:20s} shape={value.shape} dtype={value.dtype}")
