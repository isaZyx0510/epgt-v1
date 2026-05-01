"""Parameterized channel decoder helpers for EPGT-v1."""

from __future__ import annotations

import torch


def path_phase(
    rel_delay_s: torch.Tensor,
    doppler_hz: torch.Tensor,
    freq_hz: torch.Tensor,
    time_s: torch.Tensor,
) -> torch.Tensor:
    """Return complex path phase `[B, L, N_sym, N_sc]`."""
    delay_phase = torch.exp(
        -1j * 2.0 * torch.pi * rel_delay_s[:, :, None] * freq_hz[None, None, :]
    )
    doppler_phase = torch.exp(
        1j * 2.0 * torch.pi * doppler_hz[:, :, None] * time_s[None, None, :]
    )
    return doppler_phase[:, :, :, None] * delay_phase[:, :, None, :]


def global_phase(
    total_delay_s: torch.Tensor,
    cfo_hz: torch.Tensor,
    rx_time_offsets_s: torch.Tensor,
    freq_hz: torch.Tensor,
    time_s: torch.Tensor,
) -> torch.Tensor:
    """Return complex global phase `[B, N_rx, N_sym, N_sc]`."""
    delay = total_delay_s[:, None] + rx_time_offsets_s
    freq_phase = torch.exp(-1j * 2.0 * torch.pi * delay[:, :, None] * freq_hz[None, None, :])
    cfo_phase = torch.exp(1j * 2.0 * torch.pi * cfo_hz[:, None] * time_s[None, :])
    return freq_phase[:, :, None, :] * cfo_phase[:, None, :, None]

