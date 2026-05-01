"""Differentiable LS layer for reconstruction-driven hybrid training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _complex_tensor(value: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex64)
    return torch.as_tensor(value, device=device, dtype=torch.complex64)


def _float_tensor(value: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def phase_terms_torch(
    params: dict[str, torch.Tensor],
    freq_hz: np.ndarray | torch.Tensor,
    time_s: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return path and shared phase terms in torch complex tensors."""
    device = params["rel_delay_s"].device
    freq = _float_tensor(freq_hz, device)
    time = _float_tensor(time_s, device)
    rel_delay = params["rel_delay_s"].to(torch.float32)
    doppler = params["doppler_hz"].to(torch.float32)
    total_delay = params["total_delay_s"].to(torch.float32)
    cfo = params["cfo_hz"].to(torch.float32)
    rx_offsets = params["rx_time_offsets_s"].to(torch.float32)

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


def reconstruct_channel_torch(
    path_gains: torch.Tensor,
    params: dict[str, torch.Tensor],
    freq_hz: np.ndarray | torch.Tensor,
    time_s: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Reconstruct H with shape [B, N_r, N_t, N_sym, N_sc]."""
    path_phase, shared = phase_terms_torch(params, freq_hz, time_s)
    h = torch.einsum("blrt,blnk->brtnk", path_gains, path_phase)
    return (h * shared[:, :, None, :, :]).to(torch.complex64)


def uncertainty_observation_weights_torch(
    params: dict[str, torch.Tensor],
    observation_indices: np.ndarray,
    freq_hz: np.ndarray | torch.Tensor,
    time_s: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> torch.Tensor:
    """Return [B, N_obs] weights from predicted delay/Doppler uncertainty."""
    device = params["rel_delay_s"].device
    bsz = params["rel_delay_s"].shape[0]
    n_obs = observation_indices.shape[0]
    if "rel_delay_log_var" not in params or "doppler_log_var" not in params:
        return torch.ones((bsz, n_obs), dtype=torch.float32, device=device)

    delay_var = torch.exp(params["rel_delay_log_var"].to(torch.float32))
    doppler_var = torch.exp(params["doppler_log_var"].to(torch.float32))
    freq = _float_tensor(freq_hz, device)
    time = _float_tensor(time_s, device)
    n_idx = torch.as_tensor(observation_indices[:, 0], dtype=torch.long, device=device)
    k_idx = torch.as_tensor(observation_indices[:, 1], dtype=torch.long, device=device)
    freq_obs = freq[k_idx][None, None, :]
    time_obs = time[n_idx][None, None, :]
    phase_var = (
        (2.0 * torch.pi * freq_obs).square() * delay_var[:, :, None]
        + (2.0 * torch.pi * time_obs).square() * doppler_var[:, :, None]
    )
    obs_var = phase_var.mean(dim=1)
    obs_scale = obs_var.mean(dim=1, keepdim=True).clamp_min(eps)
    weights = 1.0 / (1.0 + obs_var / obs_scale)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(eps)
    return weights.clamp(min_weight, max_weight).to(torch.float32)


def recover_path_gains_lstsq_torch(
    rx_grid: np.ndarray | torch.Tensor,
    tx_grid_observed: np.ndarray | torch.Tensor,
    observation_indices: np.ndarray,
    params: dict[str, torch.Tensor],
    freq_hz: np.ndarray | torch.Tensor,
    time_s: np.ndarray | torch.Tensor,
    ridge: float = 1e-8,
    use_uncertainty_weights: bool = False,
) -> torch.Tensor:
    """Recover path gains with differentiable complex least squares."""
    device = params["rel_delay_s"].device
    rx = _complex_tensor(rx_grid, device)
    tx = _complex_tensor(tx_grid_observed, device)
    bsz, n_rx, _, _ = rx.shape
    _, n_tx, _, _ = tx.shape
    l_eff = params["rel_delay_s"].shape[1]
    n_idx = torch.as_tensor(observation_indices[:, 0], dtype=torch.long, device=device)
    k_idx = torch.as_tensor(observation_indices[:, 1], dtype=torch.long, device=device)
    path_phase, shared_phase = phase_terms_torch(params, freq_hz, time_s)
    if use_uncertainty_weights:
        weights = uncertainty_observation_weights_torch(params, observation_indices, freq_hz, time_s)
    else:
        weights = torch.ones((bsz, observation_indices.shape[0]), dtype=torch.float32, device=device)

    gains: list[torch.Tensor] = []
    ridge_value = float(max(ridge, 0.0))
    for b in range(bsz):
        x_obs = tx[b][:, n_idx, k_idx].T
        p_obs = path_phase[b][:, n_idx, k_idx].T
        rx_gains: list[torch.Tensor] = []
        for r in range(n_rx):
            shared_obs = shared_phase[b, r, n_idx, k_idx]
            columns = [
                shared_obs * p_obs[:, path_idx] * x_obs[:, tx_idx]
                for path_idx in range(l_eff)
                for tx_idx in range(n_tx)
            ]
            a = torch.stack(columns, dim=1)
            y = rx[b, r, n_idx, k_idx]
            sqrt_w = torch.sqrt(weights[b].clamp_min(1e-8)).to(torch.complex64)
            a = a * sqrt_w[:, None]
            y = y * sqrt_w
            if ridge_value > 0.0:
                eye = torch.eye(l_eff * n_tx, dtype=torch.complex64, device=device)
                ridge_rows = torch.sqrt(
                    torch.tensor(ridge_value, dtype=torch.float32, device=device)
                ).to(torch.complex64) * eye
                a = torch.cat([a, ridge_rows], dim=0)
                y = torch.cat(
                    [y, torch.zeros(l_eff * n_tx, dtype=torch.complex64, device=device)],
                    dim=0,
                )
            solution = torch.linalg.lstsq(a, y).solution
            rx_gains.append(solution.reshape(l_eff, n_tx))
        gains.append(torch.stack(rx_gains, dim=1))
    return torch.stack(gains, dim=0).to(torch.complex64)


def reconstruct_via_differentiable_ls(
    rx_grid: np.ndarray | torch.Tensor,
    tx_grid_observed: np.ndarray | torch.Tensor,
    observation_indices: np.ndarray,
    params: dict[str, torch.Tensor],
    freq_hz: np.ndarray | torch.Tensor,
    time_s: np.ndarray | torch.Tensor,
    ridge: float = 1e-8,
    use_uncertainty_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run differentiable LS and return `(path_gains, h_hat)`."""
    gains = recover_path_gains_lstsq_torch(
        rx_grid,
        tx_grid_observed,
        observation_indices,
        params,
        freq_hz,
        time_s,
        ridge=ridge,
        use_uncertainty_weights=use_uncertainty_weights,
    )
    h_hat = reconstruct_channel_torch(gains, params, freq_hz, time_s)
    return gains, h_hat


def complex_nmse_loss(prediction: torch.Tensor, target: Any, eps: float = 1e-12) -> torch.Tensor:
    """Linear NMSE loss for complex tensors."""
    target_tensor = _complex_tensor(target, prediction.device)
    numerator = torch.sum(torch.abs(prediction - target_tensor).square())
    denominator = torch.sum(torch.abs(target_tensor).square()).clamp_min(eps)
    return numerator / denominator
