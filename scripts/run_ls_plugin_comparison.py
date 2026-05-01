"""Compare traditional LS and uncertainty-weighted LS plugins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.plotting import plot_comparison
from thesis_transformer_v1.experiments.training import train_hybrid_quick


def model_overrides_from_args(args: argparse.Namespace) -> dict:
    return {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LS plugin modes.")
    parser.add_argument("--config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--output-root", default="experiments/ls_plugin_comparison")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    model_overrides = model_overrides_from_args(args)
    variants = [
        ("original_v1_traditional_ls", "original_v1", "traditional_ls"),
        ("uncertainty_v1_traditional_ls", "uncertainty_v1", "traditional_ls"),
        ("uncertainty_v1_weighted_ls", "uncertainty_v1", "learnable_weighted_ls"),
    ]

    rows = []
    for method_name, architecture, ls_mode in variants:
        result = train_hybrid_quick(
            cfg,
            steps=args.steps,
            lr=args.lr,
            eval_interval=args.eval_interval,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            model_overrides=model_overrides,
            architecture=architecture,
            ls_mode=ls_mode,
        )
        rows.append(
            {
                "experiment": "ls_plugin_comparison",
                "method": method_name,
                "history": result["history"],
                "final": result["final"],
                "architecture": architecture,
                "ls_mode": ls_mode,
            }
        )

    output_root = Path(args.output_root)
    metrics_path = write_metrics(rows, output_root / "metrics.json")
    plot_path = plot_comparison(rows, output_root / "comparison_nmse.png", metric="channel_nmse")
    summary = {
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "rows": [
            {
                "method": row["method"],
                "channel_nmse": row["final"].get("channel_nmse"),
                "observed_symbol_nmse": row["final"].get("observed_symbol_nmse"),
                "ls_mode": row["ls_mode"],
                "weights_source": row["final"].get("weights_source"),
            }
            for row in rows
        ],
    }
    summary_path = write_metrics(summary, output_root / "summary.json")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()

