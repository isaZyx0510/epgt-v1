"""Nonlinear labels for the hybrid Transformer."""

from __future__ import annotations

import numpy as np


def effective_path_indices(channel_labels: dict, l_eff: int, rule: str = "strongest") -> np.ndarray:
    powers_db = np.asarray(channel_labels["path_powers_db"])
    delays_s = np.asarray(channel_labels["path_delays_s"])
    if powers_db.ndim == 2:
        powers_ref = powers_db[0]
        delays_ref = delays_s[0]
    else:
        powers_ref = powers_db
        delays_ref = delays_s
    count = min(int(l_eff), powers_ref.shape[0])
    if rule == "strongest":
        chosen = np.argsort(powers_ref)[::-1][:count]
        return chosen[np.argsort(delays_ref[chosen])]
    if rule == "lowest_delay":
        return np.argsort(delays_ref)[:count]
    raise ValueError(f"Unsupported effective path rule: {rule}")


def nonlinear_oracle_params(
    channel_labels: dict,
    l_eff: int,
    rule: str = "strongest",
) -> dict[str, np.ndarray]:
    idx = effective_path_indices(channel_labels, l_eff, rule=rule)
    delays = np.asarray(channel_labels["path_delays_s"])[:, idx]
    doppler = np.asarray(channel_labels["path_doppler_hz"])[:, idx]
    total_delay = np.asarray(channel_labels["total_delay_s"])
    cfo = np.asarray(channel_labels["cfo_hz"])
    rx_time_offsets = np.asarray(channel_labels["rx_time_offsets_s"])
    return {
        "path_indices": idx.astype(np.int64),
        "rel_delay_s": delays.astype(np.float32),
        "doppler_hz": doppler.astype(np.float32),
        "total_delay_s": total_delay.astype(np.float32),
        "cfo_hz": cfo.astype(np.float32),
        "rx_time_offsets_s": rx_time_offsets.astype(np.float32),
    }

