"""Train/evaluate EPGT-v1 with full-grid H reconstruction loss by default."""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import (
    build_metrics,
    experiment_name_from_config,
    write_metrics,
)
from thesis_transformer_v1.experiments.training import train_hybrid_quick
from thesis_transformer_v1.models.pgt.config import load_epgt_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EPGT-v1 with H reconstruction loss.")
    parser.add_argument("--config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--model-config", default="configs/model/pgt/epgt_v1_full.yaml")
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
        "--ls-mode",
        choices=["traditional_ls", "learnable_weighted_ls"],
        default="traditional_ls",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["reconstruction", "param_plus_reconstruction", "two_stage"],
        default="reconstruction",
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--finetune-loss-mode",
        choices=["reconstruction", "param_plus_reconstruction"],
        default="reconstruction",
    )
    parser.add_argument(
        "--output",
        default="experiments/e6_physics_guided_attention/epgt_v1_full_h_loss/metrics.json",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    experiment = experiment_name_from_config(args.config)
    model_overrides, guidance, raw_model_config = load_epgt_model_config(args.model_config)
    cli_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model_overrides.update(
        {key: value for key, value in cli_overrides.items() if value is not None}
    )
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
        ls_mode=args.ls_mode,
        loss_mode=args.loss_mode,
        reconstruction_weight=args.reconstruction_weight,
        warmup_steps=args.warmup_steps,
        finetune_loss_mode=args.finetune_loss_mode,
        uncertainty_regularization_weight=args.uncertainty_regularization_weight,
    )
    metrics = build_metrics(
        experiment=experiment,
        method="epgt_v1_h_loss",
        config={
            "config_path": args.config,
            "model_config_path": args.model_config,
            "steps": args.steps,
            "lr": args.lr,
            "eval_interval": args.eval_interval,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "l_eff": cfg.l_eff,
            "model": model_overrides,
            "physics": raw_model_config.get("physics", {}),
            "architecture": "epgt_v1",
            "ls_mode": args.ls_mode,
            "loss_mode": args.loss_mode,
            "reconstruction_weight": args.reconstruction_weight,
            "uncertainty_regularization_weight": args.uncertainty_regularization_weight,
            "warmup_steps": args.warmup_steps,
            "finetune_loss_mode": args.finetune_loss_mode,
            "variant": f"{Path(args.model_config).stem}_h_loss",
        },
        history=result["history"],
        final=result["final"],
    )
    write_metrics(metrics, args.output)
    print(f"wrote {args.output}")
    print(metrics["final"])


if __name__ == "__main__":
    main()
