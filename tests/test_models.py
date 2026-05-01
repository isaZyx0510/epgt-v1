import torch

from thesis_transformer_v1.models import (
    DirectHTransformer,
    HybridTransformer,
    TransformerConfig,
    build_direct_h_transformer,
    build_hybrid_transformer,
)


def test_transformer_forward_shapes() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    hybrid = HybridTransformer(cfg)
    outputs = hybrid(tokens)
    assert outputs["rel_delay_s"].shape == (2, 6)
    assert outputs["doppler_hz"].shape == (2, 6)
    assert outputs["rx_time_offsets_s"].shape == (2, 4)
    assert torch.allclose(outputs["rx_time_offsets_s"][:, 0], torch.zeros(2))

    direct = DirectHTransformer(cfg)
    assert direct(tokens).shape == (2, 4, 4, 8, 48, 2)


def test_original_v1_factory_forward_shapes() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    hybrid = build_hybrid_transformer(cfg, architecture="original_v1")
    direct = build_direct_h_transformer(cfg, architecture="original_v1")
    assert hybrid(tokens)["rel_delay_s"].shape == (2, 6)
    assert direct(tokens).shape == (2, 4, 4, 8, 48, 2)


def test_epgt_v1_is_supported_hybrid_architecture() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    hybrid = build_hybrid_transformer(cfg, architecture="epgt_v1")
    outputs = hybrid(tokens)
    assert outputs["rel_delay_s"].shape == (2, 6)
    assert outputs["path_gates"].shape == (2, 6)


def test_uncertainty_v1_forward_shapes() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    hybrid = build_hybrid_transformer(cfg, architecture="uncertainty_v1")
    outputs = hybrid(tokens)
    assert outputs["rel_delay_s"].shape == (2, 6)
    assert outputs["rel_delay_log_var"].shape == (2, 6)
    assert outputs["doppler_log_var"].shape == (2, 6)


def test_query_v1_forward_shapes() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    hybrid = build_hybrid_transformer(cfg, architecture="query_v1")
    direct = build_direct_h_transformer(cfg, architecture="query_v1")
    outputs = hybrid(tokens)
    assert outputs["rel_delay_s"].shape == (2, 6)
    assert outputs["doppler_hz"].shape == (2, 6)
    assert outputs["rx_time_offsets_s"].shape == (2, 4)
    assert outputs["rel_delay_log_var"].shape == (2, 6)
    assert outputs["doppler_log_var"].shape == (2, 6)
    assert direct(tokens).shape == (2, 4, 4, 8, 48, 2)
