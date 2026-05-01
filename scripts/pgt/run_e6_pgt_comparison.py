"""Run first E6 comparison between baseline hybrid and EPGT-v1 variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import build_metrics, write_metrics
from thesis_transformer_v1.experiments.sweeps import run_oracle_reference
from thesis_transformer_v1.experiments.training import train_hybrid_quick
from thesis_transformer_v1.models.pgt.config import load_epgt_model_config
from thesis_transformer_v1.tdlc.config import load_mapping

DEFAULT_PGT_CONFIGS = [
    "configs/model/pgt/epgt_v1_bias_only.yaml",
    "configs/model/pgt/epgt_v1_mask_only.yaml",
    "configs/model/pgt/epgt_v1_loss_only.yaml",
    "configs/model/pgt/epgt_v1_full.yaml",
]


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


def row_from_result(
    *,
    experiment: str,
    method: str,
    variant: str,
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "method": method,
        "variant": variant,
        "config": config,
        "history": result["history"],
        "final": result["final"],
    }


def write_variant_metrics(output_root: Path, row: dict[str, Any]) -> None:
    metrics = build_metrics(
        experiment=row["experiment"],
        method=row["method"],
        config=row["config"],
        history=row["history"],
        final=row["final"],
        extra={"variant": row["variant"]},
    )
    write_metrics(metrics, output_root / row["variant"] / "metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E6 EPGT-v1 comparison.")
    parser.add_argument("--data-config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--baseline-model-config", default="configs/model/hybrid_transformer.yaml")
    parser.add_argument(
        "--baseline-architecture",
        choices=["current", "original_v1"],
        default="current",
    )
    parser.add_argument("--pgt-configs", nargs="*", default=DEFAULT_PGT_CONFIGS)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--eval-interval", type=int, default=5)
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
        default="reconstruction",
    )
    parser.add_argument("--output-root", default="experiments/e6_physics_guided_attention")
    parser.add_argument(
        "--output",
        default="experiments/e6_physics_guided_attention/comparison_metrics.json",
    )
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args()

    experiment = "e6_physics_guided_attention"
    output_root = Path(args.output_root)
    cfg = load_experiment_config(args.data_config)
    rows: list[dict[str, Any]] = []

    if not args.no_oracle:
        oracle_result = run_oracle_reference(cfg)
        rows.append(
            row_from_result(
                experiment=experiment,
                method="oracle_ls",
                variant="oracle_ls",
                result=oracle_result,
                config={
                    "data_config": args.data_config,
                    "l_eff": cfg.l_eff,
                },
            )
        )

    baseline_overrides = apply_cli_model_overrides(
        model_overrides_from_yaml(args.baseline_model_config),
        args,
    )
    baseline_result = train_hybrid_quick(
        cfg,
        steps=args.steps,
        lr=args.lr,
        eval_interval=args.eval_interval,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        model_overrides=baseline_overrides,
        architecture=args.baseline_architecture,
        loss_mode=args.loss_mode,
        reconstruction_weight=args.reconstruction_weight,
        warmup_steps=args.warmup_steps,
        finetune_loss_mode=args.finetune_loss_mode,
    )
    rows.append(
        row_from_result(
            experiment=experiment,
            method="hybrid",
            variant=f"baseline_{args.baseline_architecture}",
            result=baseline_result,
            config={
                "data_config": args.data_config,
                "model_config": args.baseline_model_config,
                "architecture": args.baseline_architecture,
                "model": baseline_overrides,
                "l_eff": cfg.l_eff,
                "loss_mode": args.loss_mode,
                "reconstruction_weight": args.reconstruction_weight,
                "warmup_steps": args.warmup_steps,
                "finetune_loss_mode": args.finetune_loss_mode,
            },
        )
    )

    for pgt_config in args.pgt_configs:
        model_overrides, guidance, raw_model_config = load_epgt_model_config(pgt_config)
        model_overrides = apply_cli_model_overrides(model_overrides, args)
        variant = Path(pgt_config).stem
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
            loss_mode=args.loss_mode,
            reconstruction_weight=args.reconstruction_weight,
            warmup_steps=args.warmup_steps,
            finetune_loss_mode=args.finetune_loss_mode,
        )
        rows.append(
            row_from_result(
                experiment=experiment,
                method="epgt_v1",
                variant=variant,
                result=result,
                config={
                    "data_config": args.data_config,
                    "model_config": pgt_config,
                    "architecture": "epgt_v1",
                    "model": model_overrides,
                    "physics": raw_model_config.get("physics", {}),
                    "l_eff": cfg.l_eff,
                    "loss_mode": args.loss_mode,
                    "reconstruction_weight": args.reconstruction_weight,
                    "warmup_steps": args.warmup_steps,
                    "finetune_loss_mode": args.finetune_loss_mode,
                },
            )
        )

    for row in rows:
        write_variant_metrics(output_root, row)
    write_metrics(rows, args.output)

    print(f"wrote {args.output}")
    for row in rows:
        print(row["variant"], row["final"])


if __name__ == "__main__":
    main()
