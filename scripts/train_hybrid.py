"""Train/evaluate HybridTransformer with unified metrics output."""

from __future__ import annotations

import argparse

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import (
    build_metrics,
    default_metrics_path,
    experiment_name_from_config,
    write_metrics,
)
from thesis_transformer_v1.experiments.training import train_hybrid_quick


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HybridTransformer on one generated batch.")
    parser.add_argument("--config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--steps", type=int, default=25)
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
        choices=["param", "reconstruction", "param_plus_reconstruction", "two_stage"],
        default="reconstruction",
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--finetune-loss-mode",
        choices=["reconstruction", "param_plus_reconstruction"],
        default="param_plus_reconstruction",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    experiment = experiment_name_from_config(args.config)
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    result = train_hybrid_quick(
        cfg,
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
        warmup_steps=args.warmup_steps,
        finetune_loss_mode=args.finetune_loss_mode,
    )
    metrics = build_metrics(
        experiment=experiment,
        method="hybrid",
        config={
            "config_path": args.config,
            "steps": args.steps,
            "lr": args.lr,
            "eval_interval": args.eval_interval,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "l_eff": cfg.l_eff,
            "model": model_overrides,
            "architecture": args.architecture,
            "ls_mode": args.ls_mode,
            "loss_mode": args.loss_mode,
            "reconstruction_weight": args.reconstruction_weight,
            "warmup_steps": args.warmup_steps,
            "finetune_loss_mode": args.finetune_loss_mode,
        },
        history=result["history"],
        final=result["final"],
    )
    output = args.output or default_metrics_path(experiment, "hybrid")
    write_metrics(metrics, output)
    print(f"wrote {output}")
    print(metrics["final"])


if __name__ == "__main__":
    main()
