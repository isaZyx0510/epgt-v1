import torch

from thesis_transformer_v1.physics.attention_bias import delay_doppler_attention_bias
from thesis_transformer_v1.physics.losses import center_doppler
from thesis_transformer_v1.physics.priors import normalized_ofdm_grid


def test_normalized_ofdm_grid_shape_and_bounds() -> None:
    coords = normalized_ofdm_grid(n_sym=8, n_sc=48)
    assert coords.shape == (8 * 48, 2)
    assert torch.all(coords >= -1.0)
    assert torch.all(coords <= 1.0)


def test_delay_doppler_attention_bias_shape() -> None:
    query = normalized_ofdm_grid(n_sym=2, n_sc=3)
    context = query[:4].unsqueeze(0).expand(2, -1, -1)
    delays = torch.full((2, 3), 1.0e-7)
    dopplers = torch.zeros((2, 3))
    gates = torch.ones((2, 3))
    bias = delay_doppler_attention_bias(
        query,
        context,
        delays,
        dopplers,
        gates,
        n_sym=2,
        n_sc=3,
        symbol_period_s=1.0e-3,
        subcarrier_spacing_hz=15_000.0,
    )
    assert bias.shape == (2, 6, 4)
    assert torch.isfinite(bias).all()


def test_center_doppler_weighted_mean_is_zero() -> None:
    doppler = torch.tensor([[10.0, 20.0, 30.0]])
    gates = torch.tensor([[1.0, 2.0, 1.0]])
    centered = center_doppler(doppler, gates)
    weighted_mean = torch.sum(centered * gates, dim=-1) / torch.sum(gates, dim=-1)
    assert torch.allclose(weighted_mean, torch.zeros_like(weighted_mean), atol=1.0e-6)

