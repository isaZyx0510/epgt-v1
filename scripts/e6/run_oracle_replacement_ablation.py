"""Diagnose EPGT-v1 bottlenecks by replacing predicted parameters with oracle ones."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.experiments.evaluation import (
    evaluate_hybrid_params,
    torch_params_to_numpy,
)
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.training import (
    model_config_from_tokens,
    normalized_param_loss,
    prepare_batches,
)
from thesis_transformer_v1.models import build_hybrid_transformer
from thesis_transformer_v1.models.pgt.config import EPGTGuidanceConfig, load_epgt_model_config

REPLACEMENT_GROUPS = {
    "predicted_all": (),
    "oracle_total_delay": ("total_delay_s",),
    "oracle_cfo": ("cfo_hz",),
    "oracle_rx_offsets": ("rx_time_offsets_s",),
    "oracle_rel_delay": ("rel_delay_s",),
    "oracle_doppler": ("doppler_hz",),
    "oracle_delay_stack": ("total_delay_s", "rx_time_offsets_s", "rel_delay_s"),
    "oracle_time_stack": ("cfo_hz", "doppler_hz"),
    "oracle_path_shape": ("rel_delay_s", "doppler_hz"),
    "oracle_all": (
        "total_delay_s",
        "cfo_hz",
        "rx_time_offsets_s",
        "rel_delay_s",
        "doppler_hz",
    ),
}


def apply_cli_model_overrides(
    model_overrides: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    merged = dict(model_overrides)
    cli_values = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    merged.update({key: value for key, value in cli_values.items() if value is not None})
    return merged


def replace_params(
    predicted: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    params = {key: value.copy() for key, value in predicted.items()}
    for key in keys:
        params[key] = oracle[key].copy()
    return params


def physical_param_errors(
    predicted: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key in ("total_delay_s", "cfo_hz", "rx_time_offsets_s", "rel_delay_s", "doppler_hz"):
        diff = predicted[key] - oracle[key]
        errors[f"{key}_mae"] = float(np.mean(np.abs(diff)))
        errors[f"{key}_rmse"] = float(np.sqrt(np.mean(np.square(diff))))
    return errors


def train_model(
    *,
    cfg: Any,
    architecture: str,
    guidance: EPGTGuidanceConfig | None,
    model_overrides: dict[str, Any],
    steps: int,
    lr: float,
    train_batches: int,
) -> tuple[torch.nn.Module, Any, list[dict[str, float]]]:
    train_data = prepare_batches(cfg, max(1, train_batches), seed_offset=0)
    _first_data, _first_obs, first_tokens = train_data[0]
    model_cfg = model_config_from_tokens(cfg, first_tokens, model_overrides=model_overrides)
    model = build_hybrid_transformer(model_cfg, architecture=architecture, guidance=guidance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for step in range(steps):
        data, _obs, tokens = train_data[step % len(train_data)]
        labels = nonlinear_oracle_params(data["channel_labels"], l_eff=cfg.l_eff)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(tokens)
        loss, parts = normalized_param_loss(outputs, labels, model_cfg)
        loss.backward()
        optimizer.step()
        history.append({"step": float(step), "train_loss": float(loss.detach()), **parts})
    return model, model_cfg, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle replacement diagnostics.")
    parser.add_argument("--data-config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--model-config", default="configs/model/pgt/epgt_v1_full.yaml")
    parser.add_argument("--architecture", default="epgt_v1")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--val-seed-offset", type=int, default=10_000)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--dim-feedforward", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--output",
        default="experiments/e6_physics_guided_attention/diagnostics/oracle_replacement.json",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.data_config)
    model_overrides, guidance, raw_model_config = load_epgt_model_config(args.model_config)
    model_overrides = apply_cli_model_overrides(model_overrides, args)
    model, model_cfg, history = train_model(
        cfg=cfg,
        architecture=args.architecture,
        guidance=guidance,
        model_overrides=model_overrides,
        steps=args.steps,
        lr=args.lr,
        train_batches=args.train_batches,
    )

    val_data = prepare_batches(cfg, 1, seed_offset=args.val_seed_offset)
    val_sample, val_obs, val_tokens = val_data[0]
    oracle = nonlinear_oracle_params(val_sample["channel_labels"], l_eff=cfg.l_eff)
    with torch.no_grad():
        predicted = torch_params_to_numpy(model(val_tokens))

    replacement_rows = []
    for variant, keys in REPLACEMENT_GROUPS.items():
        params = replace_params(predicted, oracle, keys)
        metrics = evaluate_hybrid_params(
            params,
            val_sample,
            val_obs,
            ridge=cfg.ridge,
            observed_symbol_indices=cfg.observation.input_symbol_indices,
        )
        replacement_rows.append(
            {
                "variant": variant,
                "oracle_keys": list(keys),
                "final": metrics,
            }
        )

    output = {
        "experiment": "e6_physics_guided_attention",
        "diagnostic": "oracle_replacement",
        "config": {
            "data_config": args.data_config,
            "model_config": args.model_config,
            "architecture": args.architecture,
            "steps": args.steps,
            "lr": args.lr,
            "train_batches": args.train_batches,
            "val_seed_offset": args.val_seed_offset,
            "model": model_overrides,
            "physics": raw_model_config.get("physics", {}),
            "l_eff": cfg.l_eff,
            "model_cfg": deepcopy(model_cfg.__dict__),
        },
        "history": history,
        "param_errors": physical_param_errors(predicted, oracle),
        "replacement_rows": replacement_rows,
    }
    write_metrics(output, args.output)
    print(f"wrote {args.output}")
    for row in replacement_rows:
        final = row["final"]
        print(
            row["variant"],
            {
                "channel_nmse_db": final["channel_nmse_db"],
                "observed_symbol_nmse_db": final["observed_symbol_nmse_db"],
            },
        )


if __name__ == "__main__":
    main()
