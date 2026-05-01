"""Pluggable LS recovery backends."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from thesis_transformer_v1.estimation.common_delay import (
    recover_path_gains_ls,
    reconstruct_channel,
)


LS_MODES = ("traditional_ls", "learnable_weighted_ls")


def _to_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype)
    return torch.as_tensor(value, dtype=dtype)


def _phase_terms_torch(
    params: dict[str, Any],
    freq_hz: np.ndarray,
    time_s: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_delay = _to_tensor(params["rel_delay_s"], torch.float32)
    doppler = _to_tensor(params["doppler_hz"], torch.float32)
    total_delay = _to_tensor(params["total_delay_s"], torch.float32)
    cfo = _to_tensor(params["cfo_hz"], torch.float32)
    rx_offsets = _to_tensor(params["rx_time_offsets_s"], torch.float32)
    freq = _to_tensor(freq_hz, torch.float32)
    time = _to_tensor(time_s, torch.float32)

    delay_phase = torch.exp(
        -1j * 2.0 * torch.pi * rel_delay[:, :, None] * freq[None, None, :]
    )
    doppler_phase = torch.exp(
        1j * 2.0 * torch.pi * doppler[:, :, None] * time[None, None, :]
    )
    path_phase = doppler_phase[:, :, :, None] * delay_phase[:, :, None, :]

    delay = total_delay[:, None] + rx_offsets
    freq_phase = torch.exp(-1j * 2.0 * torch.pi * delay[:, :, None] * freq[None, None, :])
    cfo_phase = torch.exp(1j * 2.0 * torch.pi * cfo[:, None] * time[None, :])
    shared_phase = freq_phase[:, :, None, :] * cfo_phase[:, None, :, None]
    return path_phase.to(torch.complex64), shared_phase.to(torch.complex64)


def uncertainty_observation_weights(
    params: dict[str, Any],
    observation_indices: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    eps: float = 1e-8,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build observation weights from diagonal phase uncertainty.

    Returns [B, N_obs] weights normalized to mean 1 for each sample.
    """
    rel_delay = _to_tensor(params["rel_delay_s"], torch.float32)
    bsz = rel_delay.shape[0]
    n_obs = observation_indices.shape[0]
    if "rel_delay_log_var" not in params or "doppler_log_var" not in params:
        return torch.ones((bsz, n_obs), dtype=torch.float32), {"weights_source": "uniform"}

    delay_var = torch.exp(_to_tensor(params["rel_delay_log_var"], torch.float32))
    doppler_var = torch.exp(_to_tensor(params["doppler_log_var"], torch.float32))
    freq = _to_tensor(freq_hz, torch.float32)
    time = _to_tensor(time_s, torch.float32)
    n_idx = torch.as_tensor(observation_indices[:, 0], dtype=torch.long)
    k_idx = torch.as_tensor(observation_indices[:, 1], dtype=torch.long)
    freq_obs = freq[k_idx][None, None, :]
    time_obs = time[n_idx][None, None, :]
    phase_var = (
        (2.0 * torch.pi * freq_obs) ** 2 * delay_var[:, :, None]
        + (2.0 * torch.pi * time_obs) ** 2 * doppler_var[:, :, None]
    )
    obs_var = phase_var.mean(dim=1)
    obs_scale = obs_var.mean(dim=1, keepdim=True).clamp_min(eps)
    weights = 1.0 / (1.0 + obs_var / obs_scale)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(eps)
    weights = weights.clamp(min_weight, max_weight)
    return weights.to(torch.float32), {
        "weights_source": "uncertainty",
        "weights_mean": float(weights.mean().detach()),
        "weights_min": float(weights.min().detach()),
        "weights_max": float(weights.max().detach()),
    }


def recover_path_gains_weighted_torch(
    rx_grid: np.ndarray,
    tx_grid_observed: np.ndarray,
    observation_indices: np.ndarray,
    params: dict[str, Any],
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover path gains with weighted complex ridge LS using PyTorch ops."""
    rx = _to_tensor(rx_grid, torch.complex64)
    tx = _to_tensor(tx_grid_observed, torch.complex64)
    bsz, n_rx, _, _ = rx.shape
    _, n_tx, _, _ = tx.shape
    l_eff = _to_tensor(params["rel_delay_s"]).shape[1]
    n_idx = torch.as_tensor(observation_indices[:, 0], dtype=torch.long)
    k_idx = torch.as_tensor(observation_indices[:, 1], dtype=torch.long)
    path_phase, shared_phase = _phase_terms_torch(params, freq_hz, time_s)
    weights, extra = uncertainty_observation_weights(params, observation_indices, freq_hz, time_s)

    gains = torch.zeros((bsz, l_eff, n_rx, n_tx), dtype=torch.complex64)
    ridge_value = float(max(ridge, 1e-10))

    for b in range(bsz):
        x_obs = tx[b][:, n_idx, k_idx].T
        p_obs = path_phase[b][:, n_idx, k_idx].T
        w = weights[b].to(torch.complex64)
        for r in range(n_rx):
            shared_obs = shared_phase[b, r, n_idx, k_idx]
            columns = []
            for l in range(l_eff):
                for t in range(n_tx):
                    columns.append(shared_obs * p_obs[:, l] * x_obs[:, t])
            a = torch.stack(columns, dim=1)
            y = rx[b, r, n_idx, k_idx]
            sqrt_w = torch.sqrt(w.real.clamp_min(1e-8)).to(torch.complex64)
            aw = a * sqrt_w[:, None]
            yw = y * sqrt_w
            if ridge_value > 0.0:
                ridge_rows = torch.sqrt(torch.tensor(ridge_value, dtype=torch.float32)).to(
                    torch.complex64
                ) * torch.eye(l_eff * n_tx, dtype=torch.complex64)
                aw = torch.cat([aw, ridge_rows], dim=0)
                yw = torch.cat([yw, torch.zeros(l_eff * n_tx, dtype=torch.complex64)], dim=0)
            solution = torch.linalg.lstsq(aw, yw).solution
            gains[b, :, r, :] = solution.reshape(l_eff, n_tx)
    return gains.detach().cpu().numpy().astype(np.complex64), extra


def recover_with_ls_plugin(
    mode: str,
    rx_grid: np.ndarray,
    tx_grid_observed: np.ndarray,
    observation_indices: np.ndarray,
    params: dict[str, Any],
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run selected LS backend and reconstruct full-grid H."""
    if mode == "traditional_ls":
        gains = recover_path_gains_ls(
            rx_grid,
            tx_grid_observed,
            observation_indices,
            params,
            freq_hz,
            time_s,
            ridge=ridge,
        )
        h_hat = reconstruct_channel(gains, params, freq_hz, time_s)
        return gains, h_hat, {"ls_mode": mode}
    if mode == "learnable_weighted_ls":
        gains, extra = recover_path_gains_weighted_torch(
            rx_grid,
            tx_grid_observed,
            observation_indices,
            params,
            freq_hz,
            time_s,
            ridge=max(ridge, 1e-8),
        )
        h_hat = reconstruct_channel(gains, params, freq_hz, time_s)
        return gains, h_hat, {"ls_mode": mode, **extra}
    raise ValueError(f"Unsupported LS mode {mode!r}; choose {LS_MODES}")
