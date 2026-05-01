"""Pilot pattern generation."""

from __future__ import annotations

import numpy as np

from .config import OFDMConfig, PilotConfig


def generate_pilot_pattern(
    ofdm: OFDMConfig, pilot: PilotConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pilot_mask, pilot_symbols, data_mask with shape [N_t, N_sym, N_sc]."""
    n_tx, n_sym, n_sc = ofdm.n_tx, ofdm.n_sym, ofdm.n_sc
    pilot_mask = np.zeros((n_tx, n_sym, n_sc), dtype=bool)
    pilot_symbols = np.zeros((n_tx, n_sym, n_sc), dtype=np.complex64)

    symbol_spacing = max(1, int(pilot.symbol_spacing))
    subcarrier_spacing = max(1, int(pilot.subcarrier_spacing))
    time_period = symbol_spacing
    freq_period = subcarrier_spacing
    if pilot.mode == "block" or pilot.orthogonal_axis == "time":
        time_period = max(symbol_spacing, n_tx)
    if pilot.mode == "comb" or pilot.orthogonal_axis == "frequency":
        freq_period = max(subcarrier_spacing, n_tx)

    for tx in range(n_tx):
        time_shift = tx if pilot.orthogonal_axis == "time" else 0
        freq_shift = tx if pilot.orthogonal_axis == "frequency" else 0

        if pilot.mode == "block":
            sym0 = (pilot.symbol_offset + tx) % time_period
            sym_idx = np.arange(sym0, n_sym, time_period)
            pilot_mask[tx, sym_idx, :] = True
        elif pilot.mode == "comb":
            sc0 = (pilot.subcarrier_offset + tx) % freq_period
            sc_idx = np.arange(sc0, n_sc, freq_period)
            pilot_mask[tx, :, sc_idx] = True
        elif pilot.mode == "sparse2d":
            sym0 = (pilot.symbol_offset + time_shift) % time_period
            sc0 = (pilot.subcarrier_offset + freq_shift) % freq_period
            sym_idx = np.arange(sym0, n_sym, time_period)
            sc_idx = np.arange(sc0, n_sc, freq_period)
            pilot_mask[tx, sym_idx[:, None], sc_idx[None, :]] = True
        else:
            raise ValueError(f"Unsupported pilot mode: {pilot.mode}")

        tx_phase = np.exp(1j * 2.0 * np.pi * tx / max(1, n_tx))
        pilot_symbols[tx, pilot_mask[tx]] = np.sqrt(pilot.pilot_power) * tx_phase

    occupied_by_pilot = np.any(pilot_mask, axis=0, keepdims=True)
    data_mask = np.broadcast_to(~occupied_by_pilot, (n_tx, n_sym, n_sc)).copy()
    return pilot_mask, pilot_symbols, data_mask
