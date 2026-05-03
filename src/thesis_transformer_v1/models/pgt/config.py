"""Configuration for EPGT-v1 model components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_transformer_v1.tdlc.config import deep_merge, load_mapping


@dataclass
class EPGTGuidanceConfig:
    """Physics-control knobs for EPGT-v1."""

    use_cross_attention_bias: bool = True  #是否在cross attention中使用bias
    use_reliability_mask: bool = False #是否根据 token reliability 做 attention mask
    center_residual_doppler: bool = True #是否将残差多普勒中心化到0 Hz
    bias_scale: float = 1.0 #缩放 cross attention bias 的系数，较大的值会更强地引导模型关注物理先验
    bias_eps: float = 1.0e-6
    min_reliability: float = 0.0  #可靠性掩码的最小值，较大的值会更强地抑制不可靠 token 的影响
    symbol_period_s: float = 1.0e-3
    subcarrier_spacing_hz: float = 15_000.0


def _load_extending_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    values = load_mapping(path)
    extends = values.get("extends")
    if not extends:
        return values
    base_values = _load_extending_mapping(path.parent / str(extends))
    overrides = {key: value for key, value in values.items() if key != "extends"}
    return deep_merge(base_values, overrides)


def guidance_config_from_dict(values: dict[str, Any] | None) -> EPGTGuidanceConfig:
    """Build `EPGTGuidanceConfig` from a YAML `physics` mapping."""
    values = values or {}
    allowed = EPGTGuidanceConfig.__dataclass_fields__
    kwargs = {key: values[key] for key in values if key in allowed}
    return EPGTGuidanceConfig(**kwargs)


def load_epgt_model_config(
    path: str | Path,
) -> tuple[dict[str, Any], EPGTGuidanceConfig, dict[str, Any]]:
    """Load a PGT model YAML file.

    Returns:
        model_overrides: Values that can be applied to `TransformerConfig`.
        guidance: Physics-guidance config consumed by `EPGTHybridTransformer`.
        raw: Fully merged YAML mapping for metrics/debugging.
    """
    raw = _load_extending_mapping(path)
    model_values = dict(raw.get("model", {}))
    model_values.pop("architecture", None)
    guidance = guidance_config_from_dict(raw.get("physics", {}))
    return model_values, guidance, raw
