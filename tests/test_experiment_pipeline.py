from pathlib import Path

from tests.test_data_pipeline import small_config
from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.experiments.evaluation import evaluate_hybrid_params
from thesis_transformer_v1.experiments.metrics_io import build_metrics, read_metrics, write_metrics
from thesis_transformer_v1.experiments.sweeps import run_method_set


def test_metrics_round_trip() -> None:
    metrics = build_metrics(
        experiment="demo",
        method="hybrid",
        history=[{"step": 0, "train_loss": 1.0}],
        final={"channel_nmse_db": -10.0},
    )
    path = Path("experiments/test_artifacts/metrics_round_trip.json")
    write_metrics(metrics, path)
    assert read_metrics(path)["final"]["channel_nmse_db"] == -10.0


def test_hybrid_evaluation_reaches_h_hat() -> None:
    cfg = small_config()
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    params = nonlinear_oracle_params(data["channel_labels"], l_eff=12)
    metrics = evaluate_hybrid_params(
        params,
        data,
        obs,
        observed_symbol_indices=(6, 7),
    )
    assert "channel_nmse_db" in metrics
    assert "observed_symbol_nmse_db" in metrics
    assert "ls_cond_a_mean" in metrics
    assert "ls_log10_cond_a_max_channel_nmse_db_corr" in metrics


def test_sweep_runner_returns_expected_rows() -> None:
    rows = run_method_set(
        small_config(),
        "demo",
        ["oracle_ls"],
        steps=1,
        lr=1e-3,
        eval_interval=1,
        extra={"sweep_name": "l_eff", "sweep_value": 12},
    )
    assert rows[0]["method"] == "oracle_ls"
    assert rows[0]["sweep_value"] == 12
    assert "channel_nmse_db" in rows[0]["final"]
