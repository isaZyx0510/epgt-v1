"""Run the formal E2 effective-path comparison package.

The script compares model/LS combinations under the same E2 condition and writes
all report-ready artifacts into one result folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from thesis_transformer_v1.data.config import ExperimentConfig, load_experiment_config
from thesis_transformer_v1.experiments.metrics_io import write_metrics
from thesis_transformer_v1.experiments.sweeps import clone_config, run_oracle_reference
from thesis_transformer_v1.experiments.training import (
    prepare_batch,
    train_direct_h_quick,
    train_hybrid_quick,
)
from thesis_transformer_v1.models.pgt.config import load_epgt_model_config
from thesis_transformer_v1.tdlc.config import load_mapping


DEFAULT_VALUES = [2, 4, 6, 8, 12]

SCENARIOS = {
    "oracle": {
        "kind": "oracle",
        "variant": "oracle_ls",
        "ls_mode": "traditional_ls",
        "description": "Oracle physical parameters with traditional LS.",
    },
    "hybrid": {
        "kind": "hybrid",
        "architecture": "original_v1",
        "variant": "hybrid_original_v1_traditional_ls",
        "ls_mode": "traditional_ls",
        "description": "Mean-pooling hybrid baseline.",
    },
    "uncertainty_traditional": {
        "kind": "hybrid",
        "architecture": "uncertainty_v1",
        "variant": "uncertainty_v1_traditional_ls",
        "ls_mode": "traditional_ls",
        "description": "Uncertainty-head model evaluated with uniform LS.",
    },
    "uncertainty_weighted": {
        "kind": "hybrid",
        "architecture": "uncertainty_v1",
        "variant": "uncertainty_v1_weighted_ls",
        "ls_mode": "learnable_weighted_ls",
        "description": "Uncertainty-head model with uncertainty-derived LS weights.",
    },
    "query": {
        "kind": "hybrid",
        "architecture": "query_v1",
        "variant": "query_v1_traditional_ls",
        "ls_mode": "traditional_ls",
        "description": "Query decoder with traditional LS.",
    },
    "query_weighted": {
        "kind": "hybrid",
        "architecture": "query_v1",
        "variant": "query_v1_weighted_ls",
        "ls_mode": "learnable_weighted_ls",
        "description": "Query decoder with uncertainty-derived LS weights.",
    },
    "epgt": {
        "kind": "epgt",
        "variant": "epgt_v1_full_traditional_ls",
        "model_config": "configs/model/pgt/epgt_v1_full.yaml",
        "ls_mode": "traditional_ls",
        "description": "Full EPGT physics guidance with traditional LS.",
    },
    "epgt_weighted": {
        "kind": "epgt",
        "variant": "epgt_v1_uncertainty_weighted_ls",
        "model_config": "configs/model/pgt/epgt_v1_uncertainty_ls.yaml",
        "ls_mode": "learnable_weighted_ls",
        "description": "EPGT with path uncertainty and weighted LS.",
    },
    "direct_h": {
        "kind": "direct_h",
        "architecture": "original_v1",
        "variant": "direct_h_original_v1",
        "ls_mode": "none",
        "description": "Direct full-grid H regression baseline without LS.",
    },
}

DEFAULT_SCENARIOS = [
    "oracle",
    "hybrid",
    "uncertainty_traditional",
    "uncertainty_weighted",
    "query",
    "query_weighted",
    "epgt",
    "epgt_weighted",
    "direct_h",
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
        "max_rel_delay_s": args.max_rel_delay_s,
        "max_doppler_hz": args.max_doppler_hz,
        "max_total_delay_s": args.max_total_delay_s,
        "max_cfo_hz": args.max_cfo_hz,
        "max_rx_time_offset_s": args.max_rx_time_offset_s,
    }
    merged.update({key: value for key, value in cli_values.items() if value is not None})
    return merged


def param_loss_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "tau0": float(args.tau0_loss_weight),
        "cfo": float(args.cfo_loss_weight),
        "delay": float(args.delay_loss_weight),
        "doppler": float(args.doppler_loss_weight),
        "rx_offset": float(args.rx_offset_loss_weight),
    }


def serializable_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "batch_size": cfg.dataset.batch_size,
        "seed": cfg.dataset.seed,
        "l_eff": cfg.l_eff,
        "ridge": cfg.ridge,
        "ofdm": dict(cfg.dataset.ofdm.__dict__),
        "pilot": dict(cfg.dataset.pilot.__dict__),
        "channel": dict(cfg.dataset.channel.__dict__),
        "impairment": dict(cfg.dataset.impairment.__dict__),
        "thesis_impairment": dict(cfg.thesis_impairment.__dict__),
        "observation": dict(cfg.observation.__dict__),
    }


def dataset_info(cfg: ExperimentConfig, train_batches: int, val_batches: int) -> dict[str, Any]:
    sample_data, sample_obs, sample_tokens = prepare_batch(cfg)
    batch_size = int(cfg.dataset.batch_size)
    return {
        "train_batches": int(train_batches),
        "val_batches": int(val_batches),
        "batch_size": batch_size,
        "train_samples_per_l_eff_scenario": int(train_batches * batch_size),
        "validation_samples_per_l_eff_scenario": int(val_batches * batch_size),
        "token_shape": list(sample_tokens.shape),
        "token_feature_dim": int(sample_obs["token_feature_dim"]),
        "observation_indices_shape": list(sample_obs["indices"].shape),
        "tx_grid_observed_shape": list(sample_obs["tx_grid_observed"].shape),
        "rx_grid_shape": list(sample_data["rx_grid"].shape),
        "h_freq_shape": list(sample_data["h_freq"].shape),
        "freq_hz_shape": list(sample_data["freq_hz"].shape),
        "time_s_shape": list(sample_data["time_s"].shape),
        "input_symbol_indices": list(cfg.observation.input_symbol_indices),
        "symbol_error_rate": float(cfg.observation.symbol_error_rate),
        "include_reliability": bool(cfg.observation.include_reliability),
        "seeds": {
            "train_seed_start": int(cfg.dataset.seed),
            "val_seed_start": int(cfg.dataset.seed + 10_000),
            "observation_rng_offset": 99,
        },
    }


def sci(value: Any) -> str:
    if value is None:
        return ""
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return str(value_float)
    return f"{value_float:.6e}"


def add_row(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    scenario_cfg: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
    l_eff: int,
) -> None:
    rows.append(
        {
            "experiment": "e2_effective_paths",
            "scenario": scenario,
            "method": config.get("method", scenario),
            "variant": scenario_cfg["variant"],
            "description": scenario_cfg["description"],
            "sweep_name": "l_eff",
            "sweep_value": int(l_eff),
            "config": config,
            "model": result.get("model", {}),
            "history": result["history"],
            "final": result["final"],
        }
    )


def run_scenario(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    scenario_name: str,
    scenario: dict[str, Any],
    baseline_overrides: dict[str, Any],
) -> dict[str, Any]:
    if scenario["kind"] == "oracle":
        return run_oracle_reference(cfg)
    if scenario["kind"] == "direct_h":
        return train_direct_h_quick(
            cfg,
            steps=args.steps,
            lr=args.lr,
            eval_interval=args.eval_interval,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            model_overrides=baseline_overrides,
            architecture=scenario["architecture"],
            device=args.device,
        )
    if scenario["kind"] == "epgt":
        model_overrides, guidance, _raw_model_config = load_epgt_model_config(
            scenario["model_config"]
        )
        return train_hybrid_quick(
            cfg,
            steps=args.steps,
            lr=args.lr,
            eval_interval=args.eval_interval,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            model_overrides=apply_cli_model_overrides(model_overrides, args),
            architecture="epgt_v1",
            guidance=guidance,
            ls_mode=scenario["ls_mode"],
            loss_mode=args.loss_mode,
            reconstruction_weight=args.reconstruction_weight,
            warmup_steps=args.warmup_steps,
            finetune_loss_mode=args.finetune_loss_mode,
            uncertainty_regularization_weight=(
                args.uncertainty_regularization_weight
                if scenario["ls_mode"] == "learnable_weighted_ls"
                else 0.0
            ),
            device=args.device,
            param_loss_weights=param_loss_weights_from_args(args),
        )
    return train_hybrid_quick(
        cfg,
        steps=args.steps,
        lr=args.lr,
        eval_interval=args.eval_interval,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        model_overrides=baseline_overrides,
        architecture=scenario["architecture"],
        ls_mode=scenario["ls_mode"],
        loss_mode=args.loss_mode,
        reconstruction_weight=args.reconstruction_weight,
        warmup_steps=args.warmup_steps,
        finetune_loss_mode=args.finetune_loss_mode,
        uncertainty_regularization_weight=(
            args.uncertainty_regularization_weight
            if scenario["ls_mode"] == "learnable_weighted_ls"
            else 0.0
        ),
        device=args.device,
        param_loss_weights=param_loss_weights_from_args(args),
    )


def scenario_config_for_row(
    scenario_name: str,
    scenario: dict[str, Any],
    args: argparse.Namespace,
    l_eff: int,
    model_overrides: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "method": scenario["kind"],
        "scenario": scenario_name,
        "data_config": args.config,
        "l_eff": int(l_eff),
        "ls_mode": scenario["ls_mode"],
        "loss_mode": args.loss_mode if scenario["kind"] != "direct_h" else "direct_h_mse",
        "steps": args.steps,
        "lr": args.lr,
        "eval_interval": args.eval_interval,
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "model": model_overrides,
        "param_loss_weights": param_loss_weights_from_args(args),
    }
    if "architecture" in scenario:
        config["architecture"] = scenario["architecture"]
    if "model_config" in scenario:
        config["model_config"] = scenario["model_config"]
    return config


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    fields = [
        "l_eff",
        "scenario",
        "variant",
        "ls_mode",
        "channel_nmse",
        "channel_nmse_db",
        "observed_symbol_nmse",
        "observed_symbol_nmse_db",
        "train_loss_final",
        "train_param_loss_final",
        "train_normalized_mse_sum",
        "validation_loss_final",
        "validation_param_loss_final",
        "val_normalized_mse_sum",
        "val_tau0_loss",
        "val_cfo_loss",
        "val_delay_loss",
        "val_doppler_loss",
        "val_rx_offset_loss",
        "weights_source",
        "weights_min",
        "weights_max",
        "tau0_loss_weight",
        "cfo_loss_weight",
        "delay_loss_weight",
        "doppler_loss_weight",
        "rx_offset_loss_weight",
        "max_total_delay_s",
        "max_rel_delay_s",
        "max_rx_time_offset_s",
        "parameter_count",
    ]
    csv_path = output_root / "summary_scientific.csv"
    md_path = output_root / "summary_scientific.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            final = row["final"]
            history = row["history"]
            train_final = history[-1] if history else {}
            train_mse_sum = sum(
                float(train_final[key])
                for key in [
                    "tau0_loss",
                    "cfo_loss",
                    "delay_loss",
                    "doppler_loss",
                    "rx_offset_loss",
                ]
                if key in train_final
            )
            val_mse_sum = sum(
                float(final[key])
                for key in [
                    "val_tau0_loss",
                    "val_cfo_loss",
                    "val_delay_loss",
                    "val_doppler_loss",
                    "val_rx_offset_loss",
                ]
                if key in final
            )
            validation_loss = final.get("channel_nmse")
            writer.writerow(
                {
                    "l_eff": row["sweep_value"],
                    "scenario": row["scenario"],
                    "variant": row["variant"],
                    "ls_mode": row["config"].get("ls_mode"),
                    "channel_nmse": sci(final.get("channel_nmse")),
                    "channel_nmse_db": sci(final.get("channel_nmse_db")),
                    "observed_symbol_nmse": sci(final.get("observed_symbol_nmse")),
                    "observed_symbol_nmse_db": sci(final.get("observed_symbol_nmse_db")),
                    "train_loss_final": sci(history[-1].get("train_loss") if history else None),
                    "train_param_loss_final": sci(
                        history[-1].get("param_loss") if history else None
                    ),
                    "train_normalized_mse_sum": sci(train_mse_sum if history else None),
                    "validation_loss_final": sci(validation_loss),
                    "validation_param_loss_final": sci(final.get("val_param_loss")),
                    "val_normalized_mse_sum": sci(val_mse_sum if final else None),
                    "val_tau0_loss": sci(final.get("val_tau0_loss")),
                    "val_cfo_loss": sci(final.get("val_cfo_loss")),
                    "val_delay_loss": sci(final.get("val_delay_loss")),
                    "val_doppler_loss": sci(final.get("val_doppler_loss")),
                    "val_rx_offset_loss": sci(final.get("val_rx_offset_loss")),
                    "weights_source": final.get("weights_source", ""),
                    "weights_min": sci(final.get("weights_min")),
                    "weights_max": sci(final.get("weights_max")),
                    "tau0_loss_weight": sci(row["config"]["param_loss_weights"]["tau0"]),
                    "cfo_loss_weight": sci(row["config"]["param_loss_weights"]["cfo"]),
                    "delay_loss_weight": sci(row["config"]["param_loss_weights"]["delay"]),
                    "doppler_loss_weight": sci(row["config"]["param_loss_weights"]["doppler"]),
                    "rx_offset_loss_weight": sci(
                        row["config"]["param_loss_weights"]["rx_offset"]
                    ),
                    "max_total_delay_s": sci(
                        row.get("model", {}).get("model_cfg", {}).get("max_total_delay_s")
                    ),
                    "max_rel_delay_s": sci(
                        row.get("model", {}).get("model_cfg", {}).get("max_rel_delay_s")
                    ),
                    "max_rx_time_offset_s": sci(
                        row.get("model", {})
                        .get("model_cfg", {})
                        .get("max_rx_time_offset_s")
                    ),
                    "parameter_count": sci(row.get("model", {}).get("parameter_count")),
                }
            )
    lines = ["# E2 Summary", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            lines.append("| " + " | ".join(item[field] for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def history_points(row: dict[str, Any], metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for item in row["history"]:
        value = item.get(metric)
        if value is None and metric == "train_param_loss":
            value = item.get("param_loss")
        if value is None and metric == "validation_param_loss":
            value = item.get("val_param_loss")
        if value is None and metric == "train_normalized_mse_sum":
            keys = ["tau0_loss", "cfo_loss", "delay_loss", "doppler_loss", "rx_offset_loss"]
            if all(key in item for key in keys):
                value = sum(float(item[key]) for key in keys)
        if value is None and metric == "validation_normalized_mse_sum":
            keys = [
                "val_tau0_loss",
                "val_cfo_loss",
                "val_delay_loss",
                "val_doppler_loss",
                "val_rx_offset_loss",
            ]
            if all(key in item for key in keys):
                value = sum(float(item[key]) for key in keys)
        if value is None and metric == "validation_loss":
            value = item.get("val_channel_nmse", item.get("channel_nmse"))
        if value is None:
            continue
        xs.append(int(item["step"]))
        ys.append(float(value))
    return xs, ys


def apply_y_axis(metric: str, values: list[float]) -> None:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return
    log_metrics = {
        "train_normalized_mse_sum",
        "validation_normalized_mse_sum",
        "validation_loss",
        "final_validation_nmse",
        "final_validation_normalized_mse",
    }
    if metric in log_metrics and min(finite_values) > 0.0:
        plt.yscale("log")
        plt.ylim(min(finite_values) / 1.6, max(finite_values) * 1.6)
        return

    ymin = min(finite_values)
    ymax = max(finite_values)
    span = ymax - ymin
    if span <= 0.0:
        span = max(abs(ymax), 1.0)
    plt.ylim(ymin - 0.12 * span, ymax + 0.12 * span)


def plot_curves(rows: list[dict[str, Any]], output_root: Path) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    l_eff_values = sorted({int(row["sweep_value"]) for row in rows})
    for l_eff in l_eff_values:
        subset = [row for row in rows if int(row["sweep_value"]) == l_eff]
        for metric, ylabel, filename in [
            ("train_loss", "signed train loss", f"train_loss_l_eff_{l_eff}.png"),
            (
                "train_param_loss",
                "signed train param loss",
                f"train_param_loss_l_eff_{l_eff}.png",
            ),
            (
                "train_normalized_mse_sum",
                "train normalized MSE sum",
                f"train_normalized_mse_sum_l_eff_{l_eff}.png",
            ),
            (
                "validation_param_loss",
                "signed validation param loss",
                f"validation_param_loss_l_eff_{l_eff}.png",
            ),
            (
                "validation_normalized_mse_sum",
                "validation normalized MSE sum",
                f"validation_normalized_mse_sum_l_eff_{l_eff}.png",
            ),
            (
                "validation_loss",
                "validation channel NMSE",
                f"validation_channel_nmse_l_eff_{l_eff}.png",
            ),
        ]:
            plt.figure(figsize=(9, 5))
            all_values: list[float] = []
            for row in subset:
                xs, ys = history_points(row, metric)
                if xs:
                    plt.plot(xs, ys, label=row["variant"], linewidth=1.6)
                    all_values.extend(ys)
            apply_y_axis(metric, all_values)
            plt.xlabel("step")
            plt.ylabel(ylabel)
            plt.title(f"E2 {ylabel}, L_eff={l_eff}")
            plt.grid(True, alpha=0.3)
            if plt.gca().get_legend_handles_labels()[0]:
                plt.legend(fontsize=7)
            plt.savefig(plot_root / filename, dpi=180, bbox_inches="tight")
            plt.close()

    plt.figure(figsize=(9, 5))
    all_nmse_values: list[float] = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda item: int(item["sweep_value"]),
        )
        x = [int(row["sweep_value"]) for row in variant_rows]
        y = [float(row["final"].get("channel_nmse", np.nan)) for row in variant_rows]
        plt.plot(x, y, marker="o", label=variant)
        all_nmse_values.extend(y)
    apply_y_axis("final_validation_nmse", all_nmse_values)
    plt.xlabel("L_eff")
    plt.ylabel("final validation channel NMSE")
    plt.title("E2 final validation NMSE vs L_eff")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.savefig(plot_root / "final_validation_nmse_vs_l_eff.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    all_param_values: list[float] = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda item: int(item["sweep_value"]),
        )
        x = [int(row["sweep_value"]) for row in variant_rows]
        y = [float(row["final"].get("val_param_loss", np.nan)) for row in variant_rows]
        plt.plot(x, y, marker="o", label=variant)
        all_param_values.extend(y)
    apply_y_axis("final_validation_param_loss", all_param_values)
    plt.xlabel("L_eff")
    plt.ylabel("final signed validation param loss")
    plt.title("E2 final signed validation param loss vs L_eff")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.savefig(plot_root / "final_validation_param_loss_vs_l_eff.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    all_mse_values: list[float] = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda item: int(item["sweep_value"]),
        )
        x = [int(row["sweep_value"]) for row in variant_rows]
        y = []
        for row in variant_rows:
            final = row["final"]
            keys = [
                "val_tau0_loss",
                "val_cfo_loss",
                "val_delay_loss",
                "val_doppler_loss",
                "val_rx_offset_loss",
            ]
            y.append(sum(float(final.get(key, np.nan)) for key in keys))
        plt.plot(x, y, marker="o", label=variant)
        all_mse_values.extend(y)
    apply_y_axis("final_validation_normalized_mse", all_mse_values)
    plt.xlabel("L_eff")
    plt.ylabel("final validation normalized MSE sum")
    plt.title("E2 final validation normalized MSE vs L_eff")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.savefig(
        plot_root / "final_validation_normalized_mse_vs_l_eff.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def write_run_metadata(
    args: argparse.Namespace,
    base_cfg: ExperimentConfig,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "e2_effective_paths",
        "purpose": "Compare 9 model/LS combinations under identical E2 effective-path settings.",
        "scenario_order": args.scenarios,
        "l_eff_values": args.values,
        "training_hyperparameters": {
            "steps": args.steps,
            "lr": args.lr,
            "optimizer": "AdamW",
            "eval_interval": args.eval_interval,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "loss_mode": args.loss_mode,
            "reconstruction_weight": args.reconstruction_weight,
            "warmup_steps": args.warmup_steps,
            "finetune_loss_mode": args.finetune_loss_mode,
            "uncertainty_regularization_weight": args.uncertainty_regularization_weight,
            "param_loss_weights": param_loss_weights_from_args(args),
            "device": args.device,
        },
        "model_hyperparameters": {
            "baseline_model_config": args.baseline_model_config,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_rel_delay_s": args.max_rel_delay_s,
            "max_doppler_hz": args.max_doppler_hz,
            "max_total_delay_s": args.max_total_delay_s,
            "max_cfo_hz": args.max_cfo_hz,
            "max_rx_time_offset_s": args.max_rx_time_offset_s,
        },
        "base_config": serializable_config(base_cfg),
        "data_info": dataset_info(base_cfg, args.train_batches, args.val_batches),
        "scenarios": {name: SCENARIOS[name] for name in args.scenarios},
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal E2 model/LS comparison.")
    parser.add_argument("--config", default="configs/data/e2_effective_paths.yaml")
    parser.add_argument("--values", nargs="*", type=int, default=DEFAULT_VALUES)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=DEFAULT_SCENARIOS,
    )
    parser.add_argument("--baseline-model-config", default="configs/model/hybrid_transformer.yaml")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-rel-delay-s", type=float, default=None)
    parser.add_argument("--max-doppler-hz", type=float, default=None)
    parser.add_argument("--max-total-delay-s", type=float, default=None)
    parser.add_argument("--max-cfo-hz", type=float, default=None)
    parser.add_argument("--max-rx-time-offset-s", type=float, default=None)
    parser.add_argument("--tau0-loss-weight", type=float, default=1.0)
    parser.add_argument("--cfo-loss-weight", type=float, default=1.0)
    parser.add_argument("--delay-loss-weight", type=float, default=1.0)
    parser.add_argument("--doppler-loss-weight", type=float, default=1.0)
    parser.add_argument("--rx-offset-loss-weight", type=float, default=1.0)
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
        default="param_plus_reconstruction",
    )
    parser.add_argument("--uncertainty-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="experiments/e2_effective_paths/formal_9way")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    base_cfg = load_experiment_config(args.config)
    write_run_metadata(args, base_cfg, output_root)
    baseline_overrides = apply_cli_model_overrides(
        model_overrides_from_yaml(args.baseline_model_config),
        args,
    )

    rows: list[dict[str, Any]] = []
    for l_eff in args.values:
        cfg = clone_config(base_cfg)
        cfg.l_eff = int(l_eff)
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            print(f"running L_eff={l_eff} scenario={scenario_name} variant={scenario['variant']}")
            result = run_scenario(cfg, args, scenario_name, scenario, baseline_overrides)
            config = scenario_config_for_row(
                scenario_name,
                scenario,
                args,
                l_eff,
                baseline_overrides,
            )
            add_row(
                rows,
                scenario=scenario_name,
                scenario_cfg=scenario,
                result=result,
                config=config,
                l_eff=l_eff,
            )
            write_metrics(rows, output_root / "results_partial.json")

    write_metrics(rows, output_root / "results.json")
    write_summary(rows, output_root)
    plot_curves(rows, output_root)
    print(f"wrote result package to {output_root}")
    print(f"summary: {output_root / 'summary_scientific.csv'}")


if __name__ == "__main__":
    main()
