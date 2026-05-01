"""Effective-path attention bias for EPGT-v1."""

from __future__ import annotations

import math

import torch

from thesis_transformer_v1.physics.priors import normalized_to_physical_grid


def delay_doppler_correlation_kernel(
    query_coords: torch.Tensor,
    context_coords: torch.Tensor,
    rel_delay_s: torch.Tensor,
    doppler_hz: torch.Tensor,
    path_gates: torch.Tensor,
    *,
    n_sym: int,
    n_sc: int,
    symbol_period_s: float,
    subcarrier_spacing_hz: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Build `K(q,i)` from gated delay-Doppler effective paths.

    Args:
        query_coords: `[Q, 2]` or `[B, Q, 2]` normalized `[k, n]` coordinates.
        context_coords: `[B, C, 2]` or `[C, 2]` normalized `[k, n]` coordinates.
        rel_delay_s: `[B, L]` relative delays.
        doppler_hz: `[B, L]` residual Dopplers.
        path_gates: `[B, L]` path gates in `[0, 1]`.

    Returns:
        Kernel magnitude with shape `[B, Q, C]`.
    """
    if query_coords.dim() == 2:
        query_coords = query_coords.unsqueeze(0).expand(rel_delay_s.shape[0], -1, -1)
    if context_coords.dim() == 2:
        context_coords = context_coords.unsqueeze(0).expand(rel_delay_s.shape[0], -1, -1)

    q_time, q_freq = normalized_to_physical_grid(
        query_coords, n_sym, n_sc, symbol_period_s, subcarrier_spacing_hz
    )
    c_time, c_freq = normalized_to_physical_grid(
        context_coords, n_sym, n_sc, symbol_period_s, subcarrier_spacing_hz
    )
    delta_t = q_time[:, :, None] - c_time[:, None, :]
    delta_f = q_freq[:, :, None] - c_freq[:, None, :]

    phase = (
        2.0 * math.pi * doppler_hz[:, :, None, None] * delta_t[:, None, :, :]
        - 2.0 * math.pi * rel_delay_s[:, :, None, None] * delta_f[:, None, :, :]
    )
    weights = path_gates[:, :, None, None]
    real = torch.sum(weights * torch.cos(phase), dim=1)
    imag = torch.sum(weights * torch.sin(phase), dim=1)
    return torch.sqrt(real.square() + imag.square() + eps)


def delay_doppler_attention_bias(
    query_coords: torch.Tensor,
    context_coords: torch.Tensor,
    rel_delay_s: torch.Tensor,
    doppler_hz: torch.Tensor,
    path_gates: torch.Tensor,
    *,
    n_sym: int,
    n_sc: int,
    symbol_period_s: float,
    subcarrier_spacing_hz: float,
    bias_scale: float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return `Gamma_qi = lambda * log(eps + K(q,i))` with shape `[B, Q, C]`."""
    kernel = delay_doppler_correlation_kernel(
        query_coords,
        context_coords,
        rel_delay_s,
        doppler_hz,
        path_gates,
        n_sym=n_sym,
        n_sc=n_sc,
        symbol_period_s=symbol_period_s,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        eps=eps,
    )
    return float(bias_scale) * torch.log(kernel + float(eps))

