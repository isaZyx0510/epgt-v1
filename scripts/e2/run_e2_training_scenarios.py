"""Run focused E2 effective-path training scenarios.

This is the preferred E2 entry point for comparing reconstruction-trained
hybrid variants across L_eff values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.sweeps import clone_config, run_oracle_reference
from thesis_transformer_v1.experiments.training import train_hybrid_quick
from thesis_transformer_v1.models.pgt.config import load_epgt_model_config
from thesis_transformer_v1.tdlc.config import load_mapping


DEFAULT_VALUES = [2, 4, 6, 8, 12]
DEFAULT_PGT_CONFIGS = [
    "configs/model/pgt/epgt_v1_full.yaml",
    "configs/model/pgt/epgt_v1_uncertainty_ls.yaml",
]

SCENARIOS = {
    "oracle": {
        "kind": "oracle",
        "variant": "oracle_ls",
    },
    "hybrid": {
        "kind": "hybrid",
        "architecture": "original_v1",
        "variant": "hybrid_original_v1",
        "ls_mode": "traditional_ls",
    },
    "query": {
        "kind": "hybrid",
        "architecture": "query_v1",
        "variant": "query_v1",
        "ls_mode": "traditional_ls",
    },
    "query_weighted": {
        "kind": "hybrid",
        "architecture": "query_v1",
        "variant": "query_v1_weighted_ls",
        "ls_mode": "learnable_weighted_ls",
    },
    "uncertainty": {
        "kind": "hybrid",
        "architecture": "uncertainty_v1",
        "variant": "uncertainty_v1_weighted_ls",
        "ls_mode": "learnable_weighted_ls",
    },
    "epgt": {
        "kind": "epgt",
        "variant": "epgt_v1_full",
        "model_config": "configs/model/pgt/epgt_v1_full.yaml",
        "ls_mode": "traditional_ls",
    },
    "epgt_weighted": {
        "kind": "epgt",
        "variant": "epgt_v1_uncertainty_ls",
        "model_config": "configs/model/pgt/epgt_v1_uncertainty_ls.yaml",
        "ls_mode": "learnable_weighted_ls",
    },
}


def model_overrides_from_yaml(path: str | Path) -> dict[str, Any]:
    values = load_mapping(path)
    model_values = dict(values.get("model", {}))
    model_values.pop("architecture", None)
    return model_values


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


def add_row(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    variant: str,
    result: dict[str, Any],
    config: dict[str, Any],
    l_eff: int,
) -> None:
    rows.append(
        {
            "experiment": "e2_effective_paths",
            "scenario": scenario,
            "method": config.get("method", scenario),
            "variant": variant,
            "sweep_name": "l_eff",
            "sweep_value": l_eff,
            "config": config,
            "history": result["history"],
            "final": result["final"],
        }
    )


def run_hybrid_scenario(
    cfg: Any,
    args: argparse.Namespace,
    scenario_name: str,
    scenario: dict[str, Any],
    model_overrides: dict[str, Any],
) -> dict[str, Any]:
    return train_hybrid_quick(
        cfg,
        steps=args.steps,
        lr=args.lr,
        eval_interval=args.eval_interval,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        model_overrides=model_overrides,
        architecture=scenario["architecture"],
        ls_mode=scenario["ls_mode"],
        loss_mode=args.loss_mode,
        reconstruction_weight=args.reconstruction_weight,
        warmup_steps=args.warmup_steps,
        finetune_loss_mode=args.finetune_loss_mode,
        uncertainty_regularization_weight=(
            args.uncertainty_regularization_weight
            if "weighted" in scenario_name or scenario["architecture"] in {"query_v1", "uncertainty_v1"}
            else 0.0
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused E2 training scenario comparisons.")
    parser.add_argument("--config", default="configs/data/e2_effective_paths.yaml")
    parser.add_argument("--values", nargs="*", type=int, default=DEFAULT_VALUES)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=["oracle", "hybrid", "query", "query_weighted", "epgt", "epgt_weighted"],
    )
    parser.add_argument("--baseline-model-config", default="configs/model/hybrid_transformer.yaml")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--dim-feedforward", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--loss-mode",
        choices=["reconstruction", "param", "param_plus_reconstruction", "two_stage"],
        default="reconstruction",
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--finetune-loss-mode",
        choices=["reconstruction", "param_plus_reconstruction"],
        default="param_plus_reconstruction",
    )
    parser.add_argument("--uncertainty-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--output", default="experiments/e2_effective_paths/training_scenarios.json")
    args = parser.parse_args()

    base_cfg = load_experiment_config(args.config)
    baseline_overrides = apply_cli_model_overrides(
        model_overrides_from_yaml(args.baseline_model_config),
        args,
    )

    rows: list[dict[str, Any]] = []
    for l_eff in args.values:
        cfg = clone_config(base_cfg)
        cfg.l_eff = int(l_eff)
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            if scenario["kind"] == "oracle":
                add_row(
                    rows,
                    scenario=scenario_name,
                    variant=scenario["variant"],
                    result=run_oracle_reference(cfg),
                    config={
                        "method": "oracle_ls",
                        "data_config": args.config,
                        "l_eff": cfg.l_eff,
                    },
                    l_eff=cfg.l_eff,
                )
                continue

            if scenario["kind"] == "epgt":
                model_overrides, guidance, raw_model_config = load_epgt_model_config(
                    scenario["model_config"]
                )
                model_overrides = apply_cli_model_overrides(model_overrides, args)
                result = train_hybrid_quick(
                    cfg,
                    steps=args.steps,
                    lr=args.lr,
                    eval_interval=args.eval_interval,
                    train_batches=args.train_batches,
                    val_batches=args.val_batches,
                    model_overrides=model_overrides,
                    architecture="epgt_v1",
                    guidance=guidance,
                    ls_mode=scenario["ls_mode"],
                    loss_mode=args.loss_mode,
                    reconstruction_weight=args.reconstruction_weight,
                    warmup_steps=args.warmup_steps,
                    finetune_loss_mode=args.finetune_loss_mode,
                    uncertainty_regularization_weight=(
                        args.uncertainty_regularization_weight
                        if scenario["ls_mode"] == "learnable_weighted_ls"
                        else 0.0
                    ),
                )
                add_row(
                    rows,
                    scenario=scenario_name,
                    variant=scenario["variant"],
                    result=result,
                    config={
                        "method": "epgt_v1",
                        "data_config": args.config,
                        "model_config": scenario["model_config"],
                        "model": model_overrides,
                        "physics": raw_model_config.get("physics", {}),
                        "l_eff": cfg.l_eff,
                        "ls_mode": scenario["ls_mode"],
                        "loss_mode": args.loss_mode,
                    },
                    l_eff=cfg.l_eff,
                )
                continue

            result = run_hybrid_scenario(
                cfg,
                args,
                scenario_name,
                scenario,
                baseline_overrides,
            )
            add_row(
                rows,
                scenario=scenario_name,
                variant=scenario["variant"],
                result=result,
                config={
                    "method": "hybrid",
                    "data_config": args.config,
                    "model_config": args.baseline_model_config,
                    "model": baseline_overrides,
                    "architecture": scenario["architecture"],
                    "l_eff": cfg.l_eff,
                    "ls_mode": scenario["ls_mode"],
                    "loss_mode": args.loss_mode,
                },
                l_eff=cfg.l_eff,
            )

    output = write_metrics(rows, args.output)
    print(f"wrote {output}")
    for row in rows:
        final = row["final"]
        print(
            row["sweep_value"],
            row["variant"],
            final.get("channel_nmse_db"),
            final.get("observed_symbol_nmse_db"),
        )


if __name__ == "__main__":
    main()
