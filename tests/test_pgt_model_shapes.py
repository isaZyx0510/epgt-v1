import torch

from thesis_transformer_v1.models import TransformerConfig, build_hybrid_transformer
from thesis_transformer_v1.models.pgt import EPGTHybridTransformer


def test_epgt_v1_factory_forward_shapes() -> None:
    tokens = torch.zeros((2, 96, 19), dtype=torch.float32)
    coord_start = 2 * 4 + 2 * 4
    tokens[..., coord_start] = torch.linspace(-1.0, 1.0, 96)
    tokens[..., coord_start + 1] = torch.repeat_interleave(torch.tensor([-1.0, 1.0]), 48)
    cfg = TransformerConfig(input_dim=19, l_eff=6, n_rx=4, n_tx=4, n_sym=8, n_sc=48)
    model = build_hybrid_transformer(cfg, architecture="epgt_v1")
    assert isinstance(model, EPGTHybridTransformer)
    outputs = model(tokens)
    assert outputs["total_delay_s"].shape == (2,)
    assert outputs["cfo_hz"].shape == (2,)
    assert outputs["rel_delay_s"].shape == (2, 6)
    assert outputs["doppler_hz"].shape == (2, 6)
    assert outputs["path_gates"].shape == (2, 6)
    assert outputs["rx_time_offsets_s"].shape == (2, 4)
    assert torch.allclose(outputs["rx_time_offsets_s"][:, 0], torch.zeros(2))
    assert torch.all(outputs["path_gates"] >= 0.0)
    assert torch.all(outputs["path_gates"] <= 1.0)

