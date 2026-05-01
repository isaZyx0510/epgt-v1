"""Run E5 full stress comparison."""

from __future__ import annotations

import argparse

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.sweeps import run_method_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E5 full stress comparison.")
    parser.add_argument("--config", default="configs/data/e5_full_stress.yaml")
    parser.add_argument("--methods", nargs="*", default=["oracle_ls", "hybrid", "direct_h"])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=2)
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
    parser.add_argument("--output", default="experiments/e5_full_stress/comparison_metrics.json")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    rows = run_method_set(
        cfg,
        "e5_full_stress",
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
    )
    write_metrics(rows, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
