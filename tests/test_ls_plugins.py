import torch

from tests.test_data_pipeline import small_config

from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.estimation.differentiable_ls import (
    complex_nmse_loss,
    reconstruct_via_differentiable_ls,
)
from thesis_transformer_v1.estimation.ls_plugins import recover_with_ls_plugin


def test_weighted_ls_plugin_returns_h_hat_with_uniform_fallback() -> None:
    cfg = small_config()
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    params = nonlinear_oracle_params(data["channel_labels"], l_eff=12)
    gains, h_hat, extra = recover_with_ls_plugin(
        "learnable_weighted_ls",
        data["rx_grid"],
        obs["tx_grid_observed"],
        obs["indices"],
        params,
        data["freq_hz"],
        data["time_s"],
        ridge=1e-6,
    )
    assert gains.shape == (2, 12, 4, 4)
    assert h_hat.shape == data["h_freq"].shape
    assert extra["weights_source"] == "uniform"


def test_differentiable_ls_backpropagates_to_parameters() -> None:
    cfg = small_config()
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    oracle = nonlinear_oracle_params(data["channel_labels"], l_eff=12)
    params = {
        key: torch.tensor(value, dtype=torch.float32, requires_grad=True)
        for key, value in oracle.items()
    }
    _gains, h_hat = reconstruct_via_differentiable_ls(
        data["rx_grid"],
        obs["tx_grid_observed"],
        obs["indices"],
        params,
        data["freq_hz"],
        data["time_s"],
        ridge=1e-8,
    )
    loss = complex_nmse_loss(h_hat, data["h_freq"])
    loss.backward()
    assert params["rel_delay_s"].grad is not None
    assert params["doppler_hz"].grad is not None
