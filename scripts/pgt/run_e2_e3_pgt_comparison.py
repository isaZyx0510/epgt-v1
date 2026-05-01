"""Run E2/E3 sweeps comparing baseline hybrid against EPGT-v1 variants."""

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

DEFAULT_PGT_CONFIGS = [
    "configs/model/pgt/epgt_v1_mask_only.yaml",
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


def make_sweep_config(base_cfg: Any, experiment: str, value: float) -> Any:
    cfg = clone_config(base_cfg)
    if experiment == "e2":
        cfg.l_eff = int(value)
    elif experiment == "e3":
        cfg.thesis_impairment.awgn_snr_db = float(value)
    else:
        raise ValueError(f"Unsupported experiment {experiment!r}")
    return cfg


def sweep_metadata(experiment: str, value: float) -> tuple[str, float | int, str]:
    if experiment == "e2":
        return "l_eff", int(value), "e2_effective_paths"
    if experiment == "e3":
        return "snr_db", float(value), "e3_awgn"
    raise ValueError(f"Unsupported experiment {experiment!r}")


def append_row(
    rows: list[dict[str, Any]],
    *,
    experiment_name: str,
    method: str,
    variant: str,
    result: dict[str, Any],
    config: dict[str, Any],
    sweep_name: str,
    sweep_value: float | int,
) -> None:
    rows.append(
        {
            "experiment": experiment_name,
            "method": method,
            "variant": variant,
            "sweep_name": sweep_name,
            "sweep_value": sweep_value,
            "config": config,
            "history": result["history"],
            "final": result["final"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2/E3 baseline-vs-EPGT comparison.")
    parser.add_argument("--experiment", choices=["e2", "e3"], required=True)
    parser.add_argument("--data-config")
    parser.add_argument("--values", nargs="*", type=float)
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
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    data_config = args.data_config
    if data_config is None:
        data_config = (
            "configs/data/e2_effective_paths.yaml"
            if args.experiment == "e2"
            else "configs/data/e3_awgn.yaml"
        )
    values = args.values
    if values is None or len(values) == 0:
        values = [4, 8, 12] if args.experiment == "e2" else [30.0, 20.0, 10.0, 5.0]

    base_cfg = load_experiment_config(data_config)
    rows: list[dict[str, Any]] = []
    baseline_overrides = apply_cli_model_overrides(
        model_overrides_from_yaml(args.baseline_model_config),
        args,
    )

    pgt_configs = []
    for path in args.pgt_configs:
        model_overrides, guidance, raw_model_config = load_epgt_model_config(path)
        pgt_configs.append(
            (
                path,
                apply_cli_model_overrides(model_overrides, args),
                guidance,
                raw_model_config,
            )
        )

    for value in values:
        cfg = make_sweep_config(base_cfg, args.experiment, value)
        sweep_name, sweep_value, experiment_name = sweep_metadata(args.experiment, value)
        if not args.no_oracle:
            append_row(
                rows,
                experiment_name=experiment_name,
                method="oracle_ls",
                variant="oracle_ls",
                result=run_oracle_reference(cfg),
                config={"data_config": data_config, "l_eff": cfg.l_eff},
                sweep_name=sweep_name,
                sweep_value=sweep_value,
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
        append_row(
            rows,
            experiment_name=experiment_name,
            method="hybrid",
            variant=f"baseline_{args.baseline_architecture}",
            result=baseline_result,
            config={
                "data_config": data_config,
                "model_config": args.baseline_model_config,
                "architecture": args.baseline_architecture,
                "model": baseline_overrides,
                "l_eff": cfg.l_eff,
                "loss_mode": args.loss_mode,
                "reconstruction_weight": args.reconstruction_weight,
                "warmup_steps": args.warmup_steps,
                "finetune_loss_mode": args.finetune_loss_mode,
            },
            sweep_name=sweep_name,
            sweep_value=sweep_value,
        )

        for pgt_path, model_overrides, guidance, raw_model_config in pgt_configs:
            variant = Path(pgt_path).stem
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
            append_row(
                rows,
                experiment_name=experiment_name,
                method="epgt_v1",
                variant=variant,
                result=result,
                config={
                    "data_config": data_config,
                    "model_config": pgt_path,
                    "architecture": "epgt_v1",
                    "model": model_overrides,
                    "physics": raw_model_config.get("physics", {}),
                    "l_eff": cfg.l_eff,
                    "loss_mode": args.loss_mode,
                    "reconstruction_weight": args.reconstruction_weight,
                    "warmup_steps": args.warmup_steps,
                    "finetune_loss_mode": args.finetune_loss_mode,
                },
                sweep_name=sweep_name,
                sweep_value=sweep_value,
            )

    default_output = (
        "experiments/e2_effective_paths/pgt_comparison_quick.json"
        if args.experiment == "e2"
        else "experiments/e3_awgn/pgt_comparison_quick.json"
    )
    output = args.output or default_output
    write_metrics(rows, output)
    print(f"wrote {output}")
    for row in rows:
        print(row["sweep_name"], row["sweep_value"], row["variant"], row["final"])


if __name__ == "__main__":
    main()
