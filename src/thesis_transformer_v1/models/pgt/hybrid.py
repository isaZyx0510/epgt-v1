"""EPGT-v1 hybrid model with effective-path guided cross-attention."""
#主模型
from __future__ import annotations

import torch
from torch import nn

from thesis_transformer_v1.models.pgt.attention import GuidedCrossAttention
from thesis_transformer_v1.models.pgt.config import EPGTGuidanceConfig
from thesis_transformer_v1.models.pgt.encoder import FullGridQueryEncoder, ObservationEncoder
from thesis_transformer_v1.models.pgt.heads import EffectivePathHead, GlobalParameterHead
from thesis_transformer_v1.models.transformer import TransformerConfig
from thesis_transformer_v1.physics.attention_bias import delay_doppler_attention_bias
from thesis_transformer_v1.physics.masks import (
    ensure_at_least_one_valid,
    reliability_attention_mask,
)
from thesis_transformer_v1.physics.priors import token_coordinate_slice


class EPGTHybridTransformer(nn.Module):
    """Hybrid EPGT-v1 model compatible with the existing LS recovery pipeline."""

    def __init__(
        self,
        config: TransformerConfig,
        guidance: EPGTGuidanceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.guidance = guidance or EPGTGuidanceConfig()
        self.observation_encoder = ObservationEncoder(config)
        self.query_encoder = FullGridQueryEncoder(config)
        self.global_head = GlobalParameterHead(config)
        self.path_head = EffectivePathHead(
            config,
            center_doppler_values=self.guidance.center_residual_doppler,
            predict_uncertainty=config.predict_path_uncertainty,
        )
        self.cross_attention = GuidedCrossAttention(config.d_model, config.nhead, config.dropout)
        self.refine_norm = nn.LayerNorm(config.d_model)

    def _context_coords(self, tokens: torch.Tensor) -> torch.Tensor:
        coord_slice = token_coordinate_slice(self.config.n_rx, self.config.n_tx)
        if tokens.shape[-1] < coord_slice.stop:
            raise ValueError(
                f"EPGT-v1 expects token coordinates at features {coord_slice}; "
                f"got input_dim={tokens.shape[-1]}"
            )
        return tokens[..., coord_slice]

    def _attention_bias(
        self,
        tokens: torch.Tensor,
        query_coords: torch.Tensor,
        path_params: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not self.guidance.use_cross_attention_bias:
            return None
        bias = delay_doppler_attention_bias(
            query_coords,
            self._context_coords(tokens),
            path_params["rel_delay_s"],
            path_params["doppler_hz"],
            path_params["path_gates"],
            n_sym=self.config.n_sym,
            n_sc=self.config.n_sc,
            symbol_period_s=self.guidance.symbol_period_s,
            subcarrier_spacing_hz=self.guidance.subcarrier_spacing_hz,
            bias_scale=self.guidance.bias_scale,
            eps=self.guidance.bias_eps,
        )
        if self.guidance.use_reliability_mask:
            reliability = tokens[..., -1]
            mask = reliability_attention_mask(
                reliability,
                min_reliability=self.guidance.min_reliability,
                mask_value=-1.0e4,
            )
            bias = bias + ensure_at_least_one_valid(mask)
        return bias

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        global_state, context = self.observation_encoder(tokens)
        initial_path_params = self.path_head(global_state)
        query, query_coords = self.query_encoder(tokens.shape[0], tokens.device)
        attn_bias = self._attention_bias(tokens, query_coords, initial_path_params)
        cross_state, _weights = self.cross_attention(query, context, attn_bias=attn_bias)
        refined_state = self.refine_norm(global_state + cross_state.mean(dim=1))
        outputs = {}
        outputs.update(self.global_head(refined_state))
        outputs.update(self.path_head(refined_state))
        return outputs
