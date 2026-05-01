import numpy as np

from tests.test_data_pipeline import small_config
from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.estimation.common_delay import reconstruct_channel, recover_path_gains_ls
from thesis_transformer_v1.metrics import nmse_db


def test_oracle_ls_near_zero_nmse_when_l_eff_equals_l_true() -> None:
    cfg = small_config()
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    params = nonlinear_oracle_params(data["channel_labels"], l_eff=12)
    gains = recover_path_gains_ls(
        data["rx_grid"],
        obs["tx_grid_observed"],
        obs["indices"],
        params,
        data["freq_hz"],
        data["time_s"],
    )
    h_hat = reconstruct_channel(gains, params, data["freq_hz"], data["time_s"])
    assert np.isfinite(h_hat).all()
    assert nmse_db(h_hat, data["h_freq"]) < -70.0


def test_recover_path_gains_can_return_ls_diagnostics() -> None:
    cfg = small_config()
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    params = nonlinear_oracle_params(data["channel_labels"], l_eff=12)
    gains, diagnostics = recover_path_gains_ls(
        data["rx_grid"],
        obs["tx_grid_observed"],
        obs["indices"],
        params,
        data["freq_hz"],
        data["time_s"],
        return_diagnostics=True,
    )
    assert gains.shape == (2, 12, 4, 4)
    assert diagnostics["cond_a"].shape == (2, 4)
    assert diagnostics["cond_gram"].shape == (2, 4)
    assert diagnostics["rank"].shape == (2, 4)
    assert int(diagnostics["num_columns"]) == 48
    assert np.isfinite(diagnostics["cond_a"]).all()
