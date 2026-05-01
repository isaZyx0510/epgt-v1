"""Run EPGT-v1 physics-parameter ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import build_metrics, write_metrics
from thesis_transformer_v1.experiments.training import train_hybrid_quick
from thesis_transformer_v1.models.pgt.config import (
    guidance_config_from_dict,
    load_epgt_model_config,
)


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


def parse_float_values(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(part) for part in value.split(",") if part)
    return parsed


def variant_name(param: str, value: float) -> str:
    safe = str(value).replace("-", "m").replace(".", "p")
    return f"{param}_{safe}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-parameter EPGT-v1 ablation.")
    parser.add_argument("--data-config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--model-config", default="configs/model/pgt/epgt_v1_bias_only.yaml")
    parser.add_argument("--param", choices=["bias_scale", "min_reliability"], default="bias_scale")
    parser.add_argument("--values", nargs="+", default=["0.25", "0.5", "1.0", "2.0"])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=1)
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
    parser.add_argument(
        "--output-root",
        default="experiments/e6_physics_guided_attention/ablations",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_experiment_config(args.data_config)
    base_model_overrides, _base_guidance, raw_model_config = load_epgt_model_config(
        args.model_config
    )
    model_overrides = apply_cli_model_overrides(base_model_overrides, args)
    values = parse_float_values(args.values)
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []

    for value in values:
        raw_variant_config = deepcopy(raw_model_config)
        physics_config = dict(raw_variant_config.get("physics", {}))
        physics_config[args.param] = value
        raw_variant_config["physics"] = physics_config
        guidance = guidance_config_from_dict(physics_config)
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
        variant = variant_name(args.param, value)
        row = {
            "experiment": "e6_physics_guided_attention",
            "method": "epgt_v1",
            "variant": variant,
            "ablation_param": args.param,
            "ablation_value": value,
            "config": {
                "data_config": args.data_config,
                "model_config": args.model_config,
                "architecture": "epgt_v1",
                "model": model_overrides,
                "physics": physics_config,
                "l_eff": cfg.l_eff,
                "loss_mode": args.loss_mode,
                "reconstruction_weight": args.reconstruction_weight,
                "warmup_steps": args.warmup_steps,
                "finetune_loss_mode": args.finetune_loss_mode,
            },
            "history": result["history"],
            "final": result["final"],
        }
        rows.append(row)
        write_metrics(
            build_metrics(
                row["experiment"],
                row["method"],
                config=row["config"],
                history=row["history"],
                final=row["final"],
                extra={
                    "variant": row["variant"],
                    "ablation_param": args.param,
                    "ablation_value": value,
                },
            ),
            output_root / Path(args.model_config).stem / variant / "metrics.json",
        )

    output = args.output or output_root / f"{Path(args.model_config).stem}_{args.param}.json"
    write_metrics(rows, output)
    print(f"wrote {output}")
    for row in rows:
        print(row["variant"], row["final"])


if __name__ == "__main__":
    main()
