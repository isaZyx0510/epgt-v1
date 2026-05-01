"""Sweep reconstruction weight for query_v1 two-stage training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import build_metrics, write_metrics
from thesis_transformer_v1.experiments.training import train_hybrid_quick


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query_v1 two-stage reconstruction sweep.")
    parser.add_argument("--config", default="configs/data/e1_clean_transformer.yaml")
    parser.add_argument("--weights", nargs="*", type=float, default=[0.01, 0.05, 0.1, 0.5])
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--finetune-steps", type=int, default=80)
    parser.add_argument(
        "--finetune-loss-mode",
        choices=["reconstruction", "param_plus_reconstruction"],
        default="param_plus_reconstruction",
    )
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
        "--ls-mode",
        choices=["traditional_ls", "learnable_weighted_ls"],
        default="traditional_ls",
    )
    parser.add_argument("--output-root", default="experiments/two_stage_reconstruction_sweep")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    steps = args.warmup_steps + args.finetune_steps
    rows = []
    for weight in args.weights:
        result = train_hybrid_quick(
            cfg,
            steps=steps,
            lr=args.lr,
            eval_interval=args.eval_interval,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            model_overrides=model_overrides,
            architecture="query_v1",
            ls_mode=args.ls_mode,
            loss_mode="two_stage",
            reconstruction_weight=weight,
            warmup_steps=args.warmup_steps,
            finetune_loss_mode=args.finetune_loss_mode,
        )
        metrics = build_metrics(
            experiment="two_stage_reconstruction_sweep",
            method="query_v1_two_stage",
            config={
                "config_path": args.config,
                "steps": steps,
                "warmup_steps": args.warmup_steps,
                "finetune_steps": args.finetune_steps,
                "lr": args.lr,
                "eval_interval": args.eval_interval,
                "train_batches": args.train_batches,
                "val_batches": args.val_batches,
                "l_eff": cfg.l_eff,
                "model": model_overrides,
                "architecture": "query_v1",
                "ls_mode": args.ls_mode,
                "loss_mode": "two_stage",
                "finetune_loss_mode": args.finetune_loss_mode,
                "reconstruction_weight": weight,
            },
            history=result["history"],
            final=result["final"],
        )
        metrics_path = output_root / f"weight_{weight:g}_metrics.json"
        write_metrics(metrics, metrics_path)
        rows.append(
            {
                "weight": weight,
                "metrics_path": str(metrics_path),
                "channel_nmse": result["final"].get("channel_nmse"),
                "observed_symbol_nmse": result["final"].get("observed_symbol_nmse"),
                "train_loss": result["final"].get("train_loss"),
                "param_loss": result["final"].get("param_loss"),
                "reconstruction_loss": result["final"].get("reconstruction_loss"),
            }
        )
        print(json.dumps(rows[-1], indent=2))

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4.5))
    for row in rows:
        metrics = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
        xs = [item["step"] for item in metrics["history"] if "channel_nmse" in item]
        ys = [item["channel_nmse"] for item in metrics["history"] if "channel_nmse" in item]
        plt.plot(xs, ys, marker="o", label=f"w={row['weight']:g}")
    plt.axvline(args.warmup_steps - 1, color="0.4", linestyle="--", linewidth=1.0)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("channel NMSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_root / "weight_sweep_curve.png"
    plt.savefig(plot_path, dpi=180)
    print(json.dumps({"summary": str(summary_path), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
