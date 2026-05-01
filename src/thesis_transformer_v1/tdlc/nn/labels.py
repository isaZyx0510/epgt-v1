"""Build nonlinear physical-parameter labels for the hybrid architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class PhysicalLabelConfig:
    max_paths: int = 8
    max_delay_s: float = 3.0e-6
    max_doppler_hz: float = 200.0
    n_rx: int = 4
    include_delta_r: bool = True


def normalize_delay(delays_s: np.ndarray, max_delay_s: float) -> np.ndarray:
    return np.clip(delays_s / max(max_delay_s, 1e-12), 0.0, 1.0).astype(np.float32)


def normalize_doppler(doppler_hz: np.ndarray, max_doppler_hz: float) -> np.ndarray:
    denom = max(max_doppler_hz, 1e-12)
    return np.clip(doppler_hz / denom, -1.0, 1.0).astype(np.float32)


def pad_1d(values: np.ndarray, length: int, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((length,), fill, dtype=np.float32)
    mask = np.zeros((length,), dtype=bool)
    count = min(length, values.shape[0])
    out[:count] = values[:count]
    mask[:count] = True
    return out, mask


def build_physical_labels(
    label_dict: Dict[str, np.ndarray],
    config: PhysicalLabelConfig,
) -> Dict[str, torch.Tensor]:
    """Build labels for theta = {mu_l, rho_l}_{l=1..L}, {delta_r}_{r=1..N_r}.

    Current mapping:
        mu_l   = normalized path delay.
        rho_l  = normalized path Doppler.
        delta_r = receive-side offset placeholder. If the generator later outputs
                  a real calibration/position offset, add it to label_dict as
                  "channel_labels.rx_offsets" and this function will use it.
    """
    delays = np.asarray(label_dict["channel_labels.path_delays_s"])
    doppler = np.asarray(label_dict.get("channel_labels.path_doppler_hz", np.zeros_like(delays)))

    mu = normalize_delay(delays, config.max_delay_s)
    rho = normalize_doppler(doppler, config.max_doppler_hz)
    mu, path_mask = pad_1d(mu, config.max_paths)
    rho, _ = pad_1d(rho, config.max_paths)
    path_params = np.stack([mu, rho], axis=-1).astype(np.float32)

    if config.include_delta_r and "channel_labels.rx_offsets" in label_dict:
        delta_r = np.asarray(label_dict["channel_labels.rx_offsets"], dtype=np.float32)
    else:
        delta_r = np.zeros((config.n_rx,), dtype=np.float32)
    delta_r, _ = pad_1d(delta_r, config.n_rx)

    return {
        "path_params": torch.from_numpy(path_params),
        "path_mask": torch.from_numpy(path_mask),
        "delta_r": torch.from_numpy(delta_r),
    }
