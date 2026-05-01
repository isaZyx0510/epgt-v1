"""Thesis-specific dataset generation on top of copied TDLC modules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from thesis_transformer_v1.data.config import ExperimentConfig, ThesisImpairmentConfig
from thesis_transformer_v1.tdlc.channel import apply_channel_frequency, channel_axes, generate_channel
from thesis_transformer_v1.tdlc.generator import fill_resource_grid
from thesis_transformer_v1.tdlc.impairments import add_awgn
from thesis_transformer_v1.tdlc.ofdm import ofdm_modulate
from thesis_transformer_v1.tdlc.pilots import generate_pilot_pattern


def generate_shared_impairments(
    rng: np.random.Generator,
    batch_size: int,
    n_rx: int,
    cfg: ThesisImpairmentConfig,
) -> dict[str, np.ndarray]:
    tau_low, tau_high = cfg.tau0_range_s
    cfo_low, cfo_high = cfg.cfo_range_hz
    rx_low, rx_high = cfg.rx_time_offset_range_s
    total_delay_s = rng.uniform(tau_low, tau_high, size=(batch_size,)).astype(np.float32)
    cfo_hz = rng.uniform(cfo_low, cfo_high, size=(batch_size,)).astype(np.float32)
    rx_time_offsets_s = np.zeros((batch_size, n_rx), dtype=np.float32)
    if n_rx > 1 and (rx_low != 0.0 or rx_high != 0.0):
        rx_time_offsets_s[:, 1:] = rng.uniform(
            rx_low,
            rx_high,
            size=(batch_size, n_rx - 1),
        ).astype(np.float32)
    return {
        "total_delay_s": total_delay_s,
        "cfo_hz": cfo_hz,
        "rx_time_offsets_s": rx_time_offsets_s,
    }


def timing_cfo_phase(
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    total_delay_s: np.ndarray,
    cfo_hz: np.ndarray,
    rx_time_offsets_s: np.ndarray,
) -> np.ndarray:
    delay = total_delay_s[:, None] + rx_time_offsets_s
    freq_phase = np.exp(-1j * 2.0 * np.pi * delay[:, :, None] * freq_hz[None, None, :])
    time_phase = np.exp(1j * 2.0 * np.pi * cfo_hz[:, None] * time_s[None, :])
    return (freq_phase[:, :, None, :] * time_phase[:, None, :, None]).astype(np.complex64)


def apply_timing_cfo(
    h_freq: np.ndarray,
    path_contrib_h_freq: np.ndarray | None,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    impairments: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    phase = timing_cfo_phase(
        freq_hz,
        time_s,
        impairments["total_delay_s"],
        impairments["cfo_hz"],
        impairments["rx_time_offsets_s"],
    )
    h_out = h_freq * phase[:, :, None, :, :]
    contrib_out = None
    if path_contrib_h_freq is not None:
        contrib_out = path_contrib_h_freq * phase[:, None, :, None, :, :]
    return h_out.astype(np.complex64), None if contrib_out is None else contrib_out.astype(np.complex64)


def generate_thesis_dataset(config: ExperimentConfig) -> dict[str, Any]:
    """Generate one in-memory thesis dataset batch.

    The copied TDLC generator creates the base common-delay channel. This wrapper
    adds the thesis impairments directly to `H_freq`, then derives the received
    grid from the impaired channel.
    """
    cfg = config.dataset
    rng = np.random.default_rng(cfg.seed)
    pilot_mask, pilot_symbols, data_mask = generate_pilot_pattern(cfg.ofdm, cfg.pilot)
    bits, data_symbols, tx_grid = fill_resource_grid(rng, cfg, pilot_mask, pilot_symbols, data_mask)
    tx_time = ofdm_modulate(tx_grid, cfg.ofdm.cp_len)

    h_freq, channel_labels = generate_channel(rng, cfg)
    freq_hz, time_s = channel_axes(cfg)
    shared_impairments = generate_shared_impairments(
        rng,
        cfg.batch_size,
        cfg.ofdm.n_rx,
        config.thesis_impairment,
    )
    h_freq, path_contrib = apply_timing_cfo(
        h_freq,
        channel_labels.get("path_contrib_h_freq"),
        freq_hz,
        time_s,
        shared_impairments,
    )
    channel_labels["path_contrib_h_freq"] = path_contrib
    channel_labels.update(shared_impairments)

    rx_grid_clean = apply_channel_frequency(tx_grid, h_freq)
    if config.thesis_impairment.awgn_snr_db is None:
        rx_grid = rx_grid_clean.astype(np.complex64)
        noise_power = np.zeros((cfg.batch_size,), dtype=np.float32)
    else:
        rx_grid, noise_power = add_awgn(rng, rx_grid_clean, config.thesis_impairment.awgn_snr_db)
    rx_time = ofdm_modulate(rx_grid, cfg.ofdm.cp_len)

    return {
        "bits": bits,
        "data_symbols": data_symbols,
        "tx_grid": tx_grid.astype(np.complex64),
        "tx_time": tx_time,
        "pilot_mask": pilot_mask,
        "pilot_symbols": pilot_symbols,
        "data_mask": data_mask,
        "rx_time": rx_time,
        "rx_grid": rx_grid.astype(np.complex64),
        "rx_grid_clean": rx_grid_clean.astype(np.complex64),
        "h_freq": h_freq.astype(np.complex64),
        "noise_power": noise_power,
        "channel_labels": channel_labels,
        "freq_hz": freq_hz.astype(np.float32),
        "time_s": time_s.astype(np.float32),
        "config": {
            "dataset": asdict(cfg),
            "thesis_impairment": asdict(config.thesis_impairment),
            "observation": asdict(config.observation),
            "l_eff": config.l_eff,
            "ridge": config.ridge,
        },
    }

