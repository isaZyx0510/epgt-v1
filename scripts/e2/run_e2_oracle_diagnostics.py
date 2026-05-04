"""Run E2 oracle perturbation and replace-one-parameter diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.estimation.ls_plugins import uncertainty_observation_weights
from thesis_transformer_v1.experiments.evaluation import (
    evaluate_hybrid_params,
    evaluate_oracle_ls,
    torch_params_to_numpy,
)
from thesis_transformer_v1.experiments.training import (
    average_metric_rows,
    train_hybrid_quick,
)


PHYSICAL_KEYS = [
    "total_delay_s",
    "cfo_hz",
    "rel_delay_s",
    "doppler_hz",
    "rx_time_offsets_s",
]


def param_loss_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "tau0": float(args.tau0_loss_weight),
        "cfo": float(args.cfo_loss_weight),
        "delay": float(args.delay_loss_weight),
        "doppler": float(args.doppler_loss_weight),
        "rx_offset": float(args.rx_offset_loss_weight),
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


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.array(value, copy=True)
        for key, value in params.items()
        if isinstance(value, np.ndarray)
    }


def perturb_params(
    oracle: dict[str, np.ndarray],
    key: str,
    std: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    params = clone_params(oracle)
    if std <= 0.0:
        return params
    noise = rng.normal(0.0, std, size=params[key].shape).astype(np.float32)
    params[key] = (params[key] + noise).astype(np.float32)
    return params


def replace_keys(
    model_params: dict[str, np.ndarray],
    oracle_params: dict[str, np.ndarray],
    keys: list[str],
) -> dict[str, np.ndarray]:
    params = clone_params(model_params)
    for key in keys:
        params[key] = np.array(oracle_params[key], copy=True)
    return params


def weight_stats(params: dict[str, np.ndarray], observation: dict[str, np.ndarray], data: dict[str, Any]) -> dict[str, float]:
    if "rel_delay_log_var" not in params or "doppler_log_var" not in params:
        return {}
    weights, _extra = uncertainty_observation_weights(
        params,
        observation["indices"],
        data["freq_hz"],
        data["time_s"],
    )
    weights = weights.detach().cpu().numpy()
    return {
        "weights_min": float(np.min(weights)),
        "weights_mean": float(np.mean(weights)),
        "weights_max": float(np.max(weights)),
    }


def evaluate_params_across_batches(
    rows: list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]],
    params_by_batch: list[dict[str, np.ndarray]],
    *,
    ridge: float,
    observed_symbol_indices: tuple[int, ...],
    ls_mode: str,
) -> dict[str, float]:
    metric_rows: list[dict[str, float]] = []
    for (data, observation, _tokens), params in zip(rows, params_by_batch, strict=True):
        metrics = evaluate_hybrid_params(
            params,
            data,
            observation,
            ridge=ridge,
            observed_symbol_indices=observed_symbol_indices,
            ls_mode=ls_mode,
        )
        metrics.update(weight_stats(params, observation, data))
        metric_rows.append(metrics)
    return average_metric_rows(metric_rows)


def model_predictions(
    model: torch.nn.Module,
    device: torch.device,
    val_data: list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]],
) -> list[dict[str, np.ndarray]]:
    predictions = []
    model.eval()
    for _data, _observation, tokens in val_data:
        with torch.no_grad():
            outputs = model(tokens.to(device))
        predictions.append(torch_params_to_numpy(outputs))
    return predictions


def oracle_params_for_batches(
    val_data: list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]],
    l_eff: int,
) -> list[dict[str, np.ndarray]]:
    return [
        nonlinear_oracle_params(data["channel_labels"], l_eff=l_eff)
        for data, _observation, _tokens in val_data
    ]


def run_perturbation(
    val_data: list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]],
    oracle_params: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.perturb_seed)
    perturb_grid = {
        "rel_delay_s": [0.0, 0.1e-9, 0.25e-9, 0.5e-9, 1.0e-9, 2.0e-9, 5.0e-9, 10.0e-9],
        "total_delay_s": [0.0, 0.1e-9, 0.25e-9, 0.5e-9, 1.0e-9, 2.0e-9, 5.0e-9, 10.0e-9],
        "rx_time_offsets_s": [0.0, 0.1e-9, 0.25e-9, 0.5e-9, 1.0e-9, 2.0e-9, 5.0e-9, 10.0e-9],
        "cfo_hz": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        "doppler_hz": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    }
    rows = []
    for key, std_values in perturb_grid.items():
        for std in std_values:
            perturbed = [
                perturb_params(params, key, float(std), rng)
                for params in oracle_params
            ]
            metrics = evaluate_params_across_batches(
                val_data,
                perturbed,
                ridge=args.ridge,
                observed_symbol_indices=tuple(args.observed_symbol_indices),
                ls_mode="traditional_ls",
            )
            rows.append(
                {
                    "experiment": "oracle_perturbation",
                    "perturbed_key": key,
                    "noise_std": float(std),
                    "noise_unit": "seconds" if key.endswith("_s") else "hz",
                    **metrics,
                }
            )
    return rows


def run_replacement_ablation(
    val_data: list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]],
    model_params: list[dict[str, np.ndarray]],
    oracle_params: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    variants = {
        "model_all": [],
        "oracle_total_delay": ["total_delay_s"],
        "oracle_rx_offsets": ["rx_time_offsets_s"],
        "oracle_rel_delay": ["rel_delay_s"],
        "oracle_cfo": ["cfo_hz"],
        "oracle_doppler": ["doppler_hz"],
        "oracle_shared_phase": ["total_delay_s", "cfo_hz", "rx_time_offsets_s"],
        "oracle_path_phase": ["rel_delay_s", "doppler_hz"],
        "oracle_all_physical_keep_model_uncertainty": PHYSICAL_KEYS,
    }
    rows = []
    for name, keys in variants.items():
        if keys:
            params_by_batch = [
                replace_keys(model_batch, oracle_batch, keys)
                for model_batch, oracle_batch in zip(model_params, oracle_params, strict=True)
            ]
        else:
            params_by_batch = [clone_params(params) for params in model_params]
        metrics = evaluate_params_across_batches(
            val_data,
            params_by_batch,
            ridge=args.ridge,
            observed_symbol_indices=tuple(args.observed_symbol_indices),
            ls_mode=args.ls_mode,
        )
        rows.append(
            {
                "experiment": "replace_one_parameter",
                "variant": name,
                "oracle_keys": ",".join(keys),
                "ls_mode": args.ls_mode,
                **metrics,
            }
        )

    oracle_metric_rows = []
    for data, observation, _tokens in val_data:
        oracle_metric_rows.append(
            evaluate_oracle_ls(
                data,
                observation,
                args.l_eff,
                ridge=args.ridge,
                observed_symbol_indices=tuple(args.observed_symbol_indices),
                ls_mode="traditional_ls",
            )
        )
    rows.append(
        {
            "experiment": "replace_one_parameter",
            "variant": "oracle_ls_no_ai",
            "oracle_keys": ",".join(PHYSICAL_KEYS),
            "ls_mode": "traditional_ls",
            **average_metric_rows(oracle_metric_rows),
        }
    )
    return rows


def parameter_error_rows(
    model_params: list[dict[str, np.ndarray]],
    oracle_params: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for key in PHYSICAL_KEYS:
        pred = np.concatenate([params[key].reshape(-1) for params in model_params])
        target = np.concatenate([params[key].reshape(-1) for params in oracle_params])
        err = pred - target
        rows.append(
            {
                "parameter": key,
                "target_min": float(np.min(target)),
                "target_mean": float(np.mean(target)),
                "target_max": float(np.max(target)),
                "pred_min": float(np.min(pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_max": float(np.max(pred)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "p95_abs_error": float(np.percentile(np.abs(err), 95.0)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: sci(row.get(key)) if isinstance(row.get(key), int | float | np.number) else row.get(key, "")
                    for key in fields
                }
            )


def plot_perturbation(rows: list[dict[str, Any]], output_root: Path) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for key in sorted({row["perturbed_key"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["perturbed_key"] == key],
            key=lambda row: float(row["noise_std"]),
        )
        x = [float(row["noise_std"]) for row in subset]
        y = [float(row["channel_nmse"]) for row in subset]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o")
        plt.yscale("log")
        if max(x) > 0.0:
            plt.xscale("symlog", linthresh=max(min(v for v in x if v > 0.0), 1e-12))
        plt.xlabel(f"{key} noise std")
        plt.ylabel("validation H NMSE")
        plt.title(f"Oracle perturbation sensitivity: {key}")
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_root / f"oracle_perturbation_{key}.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_replacement(rows: list[dict[str, Any]], output_root: Path) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["channel_nmse"]))
    labels = [row["variant"] for row in ordered]
    y = [float(row["channel_nmse"]) for row in ordered]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), y)
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("validation H NMSE")
    plt.title("Replace-one-parameter ablation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(plot_root / "replace_one_parameter_h_nmse.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_quick_readout(
    output_root: Path,
    perturb_rows: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    best_replacement = sorted(replacement_rows, key=lambda row: float(row["channel_nmse"]))
    model_all = next(row for row in replacement_rows if row["variant"] == "model_all")
    lines = [
        "# E2 Oracle Perturbation and Replace-One Diagnostics",
        "",
        "## Settings",
        "",
        "```text",
        f"L_eff = {args.l_eff}",
        f"architecture = {args.architecture}",
        f"ls_mode = {args.ls_mode}",
        f"max_rel_delay_s = {args.max_rel_delay_s:.6e}",
        f"max_total_delay_s = {args.max_total_delay_s}",
        f"max_rx_time_offset_s = {args.max_rx_time_offset_s}",
        f"param_loss_weights = {param_loss_weights_from_args(args)}",
        f"warmup_steps = {args.warmup_steps}",
        f"finetune_steps = {args.steps - args.warmup_steps}",
        f"reconstruction_weight = {args.reconstruction_weight}",
        f"train_batches = {args.train_batches}",
        f"val_batches = {args.val_batches}",
        "```",
        "",
        "## Replace-One Ranking",
        "",
        "| rank | variant | H NMSE | H NMSE dB | weights min | weights max |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(best_replacement, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row['variant']}`",
                    sci(row.get("channel_nmse")),
                    sci(row.get("channel_nmse_db")),
                    sci(row.get("weights_min")),
                    sci(row.get("weights_max")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Perturbation Takeaway",
        "",
        "The perturbation CSV records how quickly oracle LS degrades when one physical parameter is artificially noised.",
        "Compare the curves to the model-all H NMSE to estimate how precise the learned parameter must be.",
        "",
        "## Initial Interpretation",
        "",
        f"`model_all` H NMSE is `{sci(model_all.get('channel_nmse'))}`.",
        "If replacing one parameter with oracle produces a large drop in H NMSE, that parameter is the likely bottleneck.",
        "",
        "## Artifacts",
        "",
        "- `run_metadata.json`",
        "- `trained_result.json`",
        "- `oracle_perturbation.csv`",
        "- `replace_one_parameter.csv`",
        "- `parameter_errors.csv`",
        "- `plots/oracle_perturbation_*.png`",
        "- `plots/replace_one_parameter_h_nmse.png`",
    ]
    (output_root / "QUICK_READOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2 oracle diagnostics.")
    parser.add_argument("--config", default="configs/data/e2_effective_paths.yaml")
    parser.add_argument("--output-root", default="experiments/e2_effective_paths/oracle_parameter_diagnostics_l6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--l-eff", type=int, default=6)
    parser.add_argument("--architecture", default="uncertainty_v1")
    parser.add_argument("--ls-mode", default="learnable_weighted_ls")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=0.02)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--max-rel-delay-s", type=float, default=5.0e-7)
    parser.add_argument("--max-total-delay-s", type=float, default=None)
    parser.add_argument("--max-rx-time-offset-s", type=float, default=None)
    parser.add_argument("--max-cfo-hz", type=float, default=None)
    parser.add_argument("--max-doppler-hz", type=float, default=None)
    parser.add_argument("--tau0-loss-weight", type=float, default=1.0)
    parser.add_argument("--cfo-loss-weight", type=float, default=1.0)
    parser.add_argument("--delay-loss-weight", type=float, default=1.0)
    parser.add_argument("--doppler-loss-weight", type=float, default=1.0)
    parser.add_argument("--rx-offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--observed-symbol-indices", nargs="*", type=int, default=[6, 7])
    parser.add_argument("--perturb-seed", type=int, default=20260504)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(args.config)
    cfg.l_eff = args.l_eff
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_rel_delay_s": args.max_rel_delay_s,
    }
    for key, value in {
        "max_total_delay_s": args.max_total_delay_s,
        "max_rx_time_offset_s": args.max_rx_time_offset_s,
        "max_cfo_hz": args.max_cfo_hz,
        "max_doppler_hz": args.max_doppler_hz,
    }.items():
        if value is not None:
            model_overrides[key] = value

    trained = train_hybrid_quick(
        cfg,
        steps=args.steps,
        lr=args.lr,
        eval_interval=args.eval_interval,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        model_overrides=model_overrides,
        architecture=args.architecture,
        ls_mode=args.ls_mode,
        loss_mode="two_stage",
        reconstruction_weight=args.reconstruction_weight,
        warmup_steps=args.warmup_steps,
        finetune_loss_mode="param_plus_reconstruction",
        uncertainty_regularization_weight=1.0e-4,
        device=args.device,
        return_model_object=True,
        param_loss_weights=param_loss_weights_from_args(args),
    )

    val_data = trained.pop("val_data_object")
    model = trained.pop("model_object")
    device = trained.pop("device_object")
    trained.pop("model_cfg_object", None)
    (output_root / "trained_result.json").write_text(
        json.dumps(trained, indent=2),
        encoding="utf-8",
    )

    model_params = model_predictions(model, device, val_data)
    oracle_params = oracle_params_for_batches(val_data, args.l_eff)
    perturb_rows = run_perturbation(val_data, oracle_params, args)
    replacement_rows = run_replacement_ablation(val_data, model_params, oracle_params, args)
    error_rows = parameter_error_rows(model_params, oracle_params)

    write_csv(output_root / "oracle_perturbation.csv", perturb_rows)
    write_csv(output_root / "replace_one_parameter.csv", replacement_rows)
    write_csv(output_root / "parameter_errors.csv", error_rows)
    (output_root / "run_metadata.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    plot_perturbation(perturb_rows, output_root)
    plot_replacement(replacement_rows, output_root)
    write_quick_readout(output_root, perturb_rows, replacement_rows, args)
    print(f"wrote oracle diagnostics to {output_root}")


if __name__ == "__main__":
    main()
