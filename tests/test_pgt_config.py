from thesis_transformer_v1.models.pgt.config import EPGTGuidanceConfig, load_epgt_model_config


def test_load_epgt_model_config_with_extends() -> None:
    model_overrides, guidance, raw = load_epgt_model_config(
        "configs/model/pgt/epgt_v1_mask_only.yaml"
    )
    assert isinstance(guidance, EPGTGuidanceConfig)
    assert model_overrides["d_model"] == 128
    assert guidance.use_cross_attention_bias is False
    assert guidance.use_reliability_mask is True
    assert guidance.min_reliability == 0.5
    assert raw["model"]["architecture"] == "epgt_v1"

