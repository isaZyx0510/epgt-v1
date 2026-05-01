"""Run E3 AWGN robustness sweep."""

from __future__ import annotations

import argparse

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.sweeps import clone_config, run_method_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNR sweep.")
    parser.add_argument("--config", default="configs/data/base_common_delay.yaml")
    parser.add_argument("--values", nargs="*", type=float, default=[30.0, 20.0, 10.0, 5.0])
    parser.add_argument("--methods", nargs="*", default=["hybrid", "direct_h"])
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
    parser.add_argument("--output", default="experiments/e3_awgn/sweep_metrics.json")
    args = parser.parse_args()

    base = load_experiment_config(args.config)
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    rows = []
    for value in args.values:
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
    write_metrics(rows, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
