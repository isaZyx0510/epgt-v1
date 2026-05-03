"""Query-based hybrid Transformer for reconstruction-driven training."""
#不再对所有 token 做 mean pooling，而是保留 token 序列，再用 learned queries 去 cross-attend。
from __future__ import annotations

import torch
from torch import nn

from thesis_transformer_v1.models.transformer import TransformerConfig


class SequenceTokenEncoder(nn.Module):
    """Encode observation tokens while preserving token-level structure."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_proj(tokens))


class QueryHybridTransformer(nn.Module):
    """Predict nonlinear parameters from learned global/path/RX queries.

    Unlike the original v1 model, this model does not collapse all encoded
    observation tokens with mean pooling. Learned queries cross-attend to the
    encoded token sequence, giving path parameters and RX offsets their own
    decoder states.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceTokenEncoder(config)
        self.global_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.path_queries = nn.Parameter(torch.randn(1, config.l_eff, config.d_model) * 0.02)
        self.rx_queries = nn.Parameter(torch.randn(1, config.n_rx, config.d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, config.num_layers))
        self.global_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 2),
        )
        self.path_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 4),
        )
        self.rx_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 1),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encoder(tokens)
        bsz = tokens.shape[0]
        queries = torch.cat(
            [
                self.global_query.expand(bsz, -1, -1),
                self.path_queries.expand(bsz, -1, -1),
                self.rx_queries.expand(bsz, -1, -1),
            ],
            dim=1,
        )
        decoded = self.decoder(queries, memory)
        global_state = decoded[:, 0]
        path_state = decoded[:, 1 : 1 + self.config.l_eff]
        rx_state = decoded[:, 1 + self.config.l_eff :]

        global_raw = self.global_head(global_state)
        path_raw = self.path_head(path_state)
        rx_raw = self.rx_head(rx_state).squeeze(-1)

        rel_delay = torch.sigmoid(path_raw[..., 0]) * self.config.max_rel_delay_s
        doppler = torch.tanh(path_raw[..., 1]) * self.config.max_doppler_hz
        rx_offsets = torch.tanh(rx_raw) * self.config.max_rx_time_offset_s
        rx_offsets = rx_offsets.clone()
        rx_offsets[:, 0] = 0.0

        rel_delay_var = (
            torch.nn.functional.softplus(path_raw[..., 2])
            * max(self.config.max_rel_delay_s, 1e-12) ** 2
        )
        doppler_var = (
            torch.nn.functional.softplus(path_raw[..., 3])
            * max(self.config.max_doppler_hz, 1e-6) ** 2
        )
        return {
            "total_delay_s": torch.sigmoid(global_raw[:, 0]) * self.config.max_total_delay_s,
            "cfo_hz": torch.tanh(global_raw[:, 1]) * self.config.max_cfo_hz,
            "rel_delay_s": rel_delay,
            "doppler_hz": doppler,
            "rx_time_offsets_s": rx_offsets,
            "rel_delay_log_var": torch.log(rel_delay_var.clamp_min(1e-30)),
            "doppler_log_var": torch.log(doppler_var.clamp_min(1e-30)),
        }
