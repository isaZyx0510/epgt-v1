"""Run the complete E3-E5 robustness training/evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.plotting import plot_comparison, plot_sweep
from thesis_transformer_v1.experiments.sweeps import clone_config, run_method_set


def model_overrides_from_args(args: argparse.Namespace) -> dict:
    return {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def run_snr_stage(args: argparse.Namespace, model_overrides: dict) -> list[dict]:
    base = load_experiment_config(args.base_config)
    rows = []
    for value in args.snr_values:
        cfg = clone_config(base)
        cfg.thesis_impairment.awgn_snr_db = value
        rows.extend(
            run_method_set(
                cfg,
                "e3_awgn",
                args.methods,
                steps=args.steps,
                lr=args.lr,
                eval_interval=args.eval_interval,
                train_batches=args.train_batches,
                val_batches=args.val_batches,
                model_overrides=model_overrides,
                architecture=args.architecture,
                ls_mode=args.ls_mode,
                loss_mode=args.loss_mode,
                reconstruction_weight=args.reconstruction_weight,
                extra={"sweep_name": "snr_db", "sweep_value": value},
            )
        )
    return rows


def run_ser_stage(args: argparse.Namespace, model_overrides: dict) -> list[dict]:
    base = load_experiment_config(args.base_config)
    rows = []
    for value in args.ser_values:
        cfg = clone_config(base)
        cfg.observation.symbol_error_rate = value
        rows.extend(
            run_method_set(
                cfg,
                "e4_symbol_error",
                args.methods,
                steps=args.steps,
                lr=args.lr,
                eval_interval=args.eval_interval,
                train_batches=args.train_batches,
                val_batches=args.val_batches,
                model_overrides=model_overrides,
                architecture=args.architecture,
                ls_mode=args.ls_mode,
                loss_mode=args.loss_mode,
                reconstruction_weight=args.reconstruction_weight,
                extra={"sweep_name": "ser", "sweep_value": value},
            )
        )
    return rows


def run_full_stress_stage(args: argparse.Namespace, model_overrides: dict) -> list[dict]:
    cfg = load_experiment_config(args.full_stress_config)
    return run_method_set(
        cfg,
        "e5_full_stress",
        args.full_stress_methods,
        steps=args.steps,
        lr=args.lr,
        eval_interval=args.eval_interval,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        model_overrides=model_overrides,
        architecture=args.architecture,
        ls_mode=args.ls_mode,
        loss_mode=args.loss_mode,
        reconstruction_weight=args.reconstruction_weight,
    )


def summary_rows(metrics: list[dict]) -> list[dict]:
    return [
        {
            "experiment": row["experiment"],
            "method": row["method"],
            "sweep_name": row.get("sweep_name"),
            "sweep_value": row.get("sweep_value"),
            "channel_nmse": row["final"].get("channel_nmse"),
            "observed_symbol_nmse": row["final"].get("observed_symbol_nmse"),
        }
        for row in metrics
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3-E5 robustness pipeline.")
    parser.add_argument("--base-config", default="configs/data/base_common_delay.yaml")
    parser.add_argument("--full-stress-config", default="configs/data/e5_full_stress.yaml")
    parser.add_argument("--methods", nargs="*", default=["hybrid", "direct_h"])
    parser.add_argument(
        "--full-stress-methods",
        nargs="*",
        default=["oracle_ls", "hybrid", "direct_h"],
    )
    parser.add_argument("--snr-values", nargs="*", type=float, default=[30.0, 20.0, 10.0, 5.0])
    parser.add_argument(
        "--ser-values",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--architecture",
        choices=["current", "original_v1", "epgt_v1", "uncertainty_v1", "query_v1"],
        default="original_v1",
    )
    parser.add_argument(
        "--ls-mode",
        choices=["traditional_ls", "learnable_weighted_ls"],
        default="traditional_ls",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["param", "reconstruction", "param_plus_reconstruction"],
        default="reconstruction",
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--output-root", default="experiments/robustness_pipeline")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    model_overrides = model_overrides_from_args(args)

    e3_rows = run_snr_stage(args, model_overrides)
    e4_rows = run_ser_stage(args, model_overrides)
    e5_rows = run_full_stress_stage(args, model_overrides)

    e3_metrics = write_metrics(e3_rows, output_root / "e3_awgn_metrics.json")
    e4_metrics = write_metrics(e4_rows, output_root / "e4_symbol_error_metrics.json")
    e5_metrics = write_metrics(e5_rows, output_root / "e5_full_stress_metrics.json")
    summary_path = write_metrics(
        {
            "architecture": args.architecture,
            "ls_mode": args.ls_mode,
            "loss_mode": args.loss_mode,
            "reconstruction_weight": args.reconstruction_weight,
            "model": model_overrides,
            "steps": args.steps,
            "lr": args.lr,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "rows": summary_rows(e3_rows + e4_rows + e5_rows),
        },
        output_root / "summary.json",
    )

    e3_plot = plot_sweep(e3_rows, output_root / "e3_nmse_vs_snr.png", metric="channel_nmse")
    e4_plot = plot_sweep(e4_rows, output_root / "e4_nmse_vs_ser.png", metric="channel_nmse")
    e5_plot = plot_comparison(
        e5_rows,
        output_root / "e5_full_stress_comparison.png",
        metric="channel_nmse",
    )

    print(json.dumps({
        "e3_metrics": str(e3_metrics),
        "e4_metrics": str(e4_metrics),
        "e5_metrics": str(e5_metrics),
        "summary": str(summary_path),
        "plots": [str(e3_plot), str(e4_plot), str(e5_plot)],
    }, indent=2))


if __name__ == "__main__":
    main()
