"""E0 oracle LS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.estimation.common_delay import recover_path_gains_ls, reconstruct_channel
from thesis_transformer_v1.metrics import nmse_db


def run_oracle_ls(config_path: str | Path, l_eff_values: list[int] | None = None) -> list[dict]:
    config = load_experiment_config(config_path)
    data = generate_thesis_dataset(config)
    token_data = build_observation_tokens(
        data,
        ObservationTokenConfig(
            symbol_indices=config.observation.input_symbol_indices,
            symbol_error_rate=config.observation.symbol_error_rate,
            include_reliability=config.observation.include_reliability,
        ),
        rng=np.random.default_rng(config.dataset.seed + 99),
    )
    if l_eff_values is None:
        l_eff_values = [config.l_eff]

    results = []
    for l_eff in l_eff_values:
        params = nonlinear_oracle_params(data["channel_labels"], l_eff=l_eff)
        gains = recover_path_gains_ls(
            data["rx_grid"],
            token_data["tx_grid_observed"],
            token_data["indices"],
            params,
            data["freq_hz"],
            data["time_s"],
            ridge=config.ridge,
        )
        h_hat = reconstruct_channel(gains, params, data["freq_hz"], data["time_s"])
        full_nmse = nmse_db(h_hat, data["h_freq"])
        observed_n = config.observation.input_symbol_indices
        observed_nmse = nmse_db(h_hat[:, :, :, observed_n, :], data["h_freq"][:, :, :, observed_n, :])
        results.append(
            {
                "l_eff": int(l_eff),
                "channel_nmse_db": full_nmse,
                "observed_symbol_nmse_db": observed_nmse,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E0 oracle LS sanity check.")
    parser.add_argument("--config", default="configs/data/e0_oracle_clean.yaml")
    parser.add_argument("--l-eff-values", nargs="*", type=int)
    parser.add_argument("--output", default="experiments/e0_oracle_clean/metrics.json")
    args = parser.parse_args()

    results = run_oracle_ls(args.config, args.l_eff_values)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

