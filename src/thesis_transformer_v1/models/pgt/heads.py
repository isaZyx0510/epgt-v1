"""Prediction heads for EPGT-v1."""

from __future__ import annotations

import torch
from torch import nn

from thesis_transformer_v1.models.transformer import TransformerConfig
from thesis_transformer_v1.physics.losses import center_doppler


class GlobalParameterHead(nn.Module):
    """Predict global delay/CFO and relative RX timing offsets."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 2 + config.n_rx),
        )

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.head(state)
        tau0_raw = raw[:, 0]
        cfo_raw = raw[:, 1]
        rx_raw = raw[:, 2:]
        rx_offsets = torch.tanh(rx_raw) * self.config.max_rx_time_offset_s
        rx_offsets = rx_offsets.clone()
        rx_offsets[:, 0] = 0.0
        return {
            "total_delay_s": torch.sigmoid(tau0_raw) * self.config.max_total_delay_s,
            "cfo_hz": torch.tanh(cfo_raw) * self.config.max_cfo_hz,
            "rx_time_offsets_s": rx_offsets,
        }


class EffectivePathHead(nn.Module):
    """Predict effective-path gates, delays, residual Dopplers, and optional uncertainty."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        center_doppler_values: bool = True,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.center_doppler_values = center_doppler_values
        self.predict_uncertainty = predict_uncertainty
        values_per_path = 5 if predict_uncertainty else 3
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, values_per_path * config.l_eff),
        )

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        values_per_path = 5 if self.predict_uncertainty else 3
        raw = self.head(state).reshape(-1, self.config.l_eff, values_per_path)
        path_gates = torch.sigmoid(raw[..., 0])
        rel_delay_s = torch.sigmoid(raw[..., 1]) * self.config.max_rel_delay_s
        doppler_hz = torch.tanh(raw[..., 2]) * self.config.max_doppler_hz
        rel_delay_log_var = None
        doppler_log_var = None
        if self.predict_uncertainty:
            rel_delay_var = (
                torch.nn.functional.softplus(raw[..., 3])
                * max(self.config.max_rel_delay_s, 1e-12) ** 2
            )
            doppler_var = (
                torch.nn.functional.softplus(raw[..., 4])
                * max(self.config.max_doppler_hz, 1e-6) ** 2
            )
            rel_delay_log_var = torch.log(rel_delay_var.clamp_min(1e-30))
            doppler_log_var = torch.log(doppler_var.clamp_min(1e-30))
        if self.center_doppler_values:
            doppler_hz = center_doppler(doppler_hz, path_gates)
        rel_delay_s, order = torch.sort(rel_delay_s, dim=1)
        doppler_hz = torch.gather(doppler_hz, 1, order)
        path_gates = torch.gather(path_gates, 1, order)
        outputs = {
            "path_gates": path_gates,
            "rel_delay_s": rel_delay_s,
            "doppler_hz": doppler_hz,
        }
        if rel_delay_log_var is not None and doppler_log_var is not None:
            outputs["rel_delay_log_var"] = torch.gather(rel_delay_log_var, 1, order)
            outputs["doppler_log_var"] = torch.gather(doppler_log_var, 1, order)
        return outputs
