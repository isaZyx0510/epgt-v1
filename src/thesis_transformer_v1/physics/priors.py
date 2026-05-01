"""Physical coordinate helpers for EPGT-v1."""

from __future__ import annotations

import torch


def normalized_ofdm_grid(n_sym: int, n_sc: int, device: torch.device | None = None) -> torch.Tensor:
    """Return full-grid coordinates `[N_sym * N_sc, 2]` in `[k, n]` order."""
    n = torch.arange(n_sym, dtype=torch.float32, device=device)
    k = torch.arange(n_sc, dtype=torch.float32, device=device)
    nn, kk = torch.meshgrid(n, k, indexing="ij")
    k_coord = 2.0 * kk / float(max(n_sc - 1, 1)) - 1.0
    n_coord = 2.0 * nn / float(max(n_sym - 1, 1)) - 1.0
    return torch.stack([k_coord, n_coord], dim=-1).reshape(-1, 2)


def normalized_to_physical_grid(
    coords: torch.Tensor,
    n_sym: int,
    n_sc: int,
    symbol_period_s: float,
    subcarrier_spacing_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized `[k, n]` coordinates to `(time_s, freq_hz)` values."""
    k_index = 0.5 * (coords[..., 0] + 1.0) * float(max(n_sc - 1, 1))
    n_index = 0.5 * (coords[..., 1] + 1.0) * float(max(n_sym - 1, 1))
    centered_k = k_index - 0.5 * float(max(n_sc - 1, 1))
    return n_index * symbol_period_s, centered_k * subcarrier_spacing_hz


def token_coordinate_slice(n_rx: int, n_tx: int) -> slice:
    """Return the token feature slice containing normalized `[k, n]` coordinates."""
    start = 2 * n_rx + 2 * n_tx
    return slice(start, start + 2)

