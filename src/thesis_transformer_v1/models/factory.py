"""Model factory for selecting architecture versions."""

from __future__ import annotations

from torch import nn

from thesis_transformer_v1.models.original_v1_transformer import (
    DirectHTransformer as OriginalV1DirectHTransformer,
)
from thesis_transformer_v1.models.original_v1_transformer import (
    HybridTransformer as OriginalV1HybridTransformer,
)
from thesis_transformer_v1.models.pgt import EPGTGuidanceConfig, EPGTHybridTransformer
from thesis_transformer_v1.models.query_transformer import QueryHybridTransformer
from thesis_transformer_v1.models.transformer import (
    DirectHTransformer,
    HybridTransformer,
    TransformerConfig,
)
from thesis_transformer_v1.models.uncertainty_transformer import UncertaintyHybridTransformer

SUPPORTED_ARCHITECTURES = ("current", "original_v1", "epgt_v1", "uncertainty_v1", "query_v1")
SUPPORTED_DIRECT_H_ARCHITECTURES = ("current", "original_v1", "uncertainty_v1", "query_v1")


def build_hybrid_transformer(
    config: TransformerConfig,
    architecture: str = "current",
    guidance: EPGTGuidanceConfig | None = None,
) -> nn.Module:
    if architecture == "current":
        return HybridTransformer(config)
    if architecture == "original_v1":
        return OriginalV1HybridTransformer(config)
    if architecture == "epgt_v1":
        return EPGTHybridTransformer(config, guidance=guidance)
    if architecture == "uncertainty_v1":
        return UncertaintyHybridTransformer(config)
    if architecture == "query_v1":
        return QueryHybridTransformer(config)
    raise ValueError(f"Unsupported architecture {architecture!r}; choose {SUPPORTED_ARCHITECTURES}")


def build_direct_h_transformer(
    config: TransformerConfig,
    architecture: str = "current",
) -> nn.Module:
    if architecture == "current":
        return DirectHTransformer(config)
    if architecture == "original_v1":
        return OriginalV1DirectHTransformer(config)
    if architecture == "uncertainty_v1":
        return DirectHTransformer(config)
    if architecture == "query_v1":
        return DirectHTransformer(config)
    raise ValueError(
        f"Unsupported direct-H architecture {architecture!r}; "
        f"choose {SUPPORTED_DIRECT_H_ARCHITECTURES}"
    )
