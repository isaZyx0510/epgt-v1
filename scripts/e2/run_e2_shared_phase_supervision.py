"""Run L=6 shared-phase supervision diagnostics for E2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from run_e2_oracle_diagnostics import (
    model_predictions,
    oracle_params_for_batches,
    parameter_error_rows,
    param_loss_weights_from_args,
    run_replacement_ablation,
    sci,
    write_csv,
)
from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.experiments.training import train_hybrid_quick


SETTINGS = [
    {
        "name": "baseline_repeat",
        "max_total_delay_s": 1.0e-6,
        "max_rx_time_offset_s": 2.0e-7,
        "tau0_loss_weight": 1.0,
        "rx_offset_loss_weight": 1.0,
    },
    {
        "name": "normalization_only",
        "max_total_delay_s": 2.5e-7,
        "max_rx_time_offset_s": 6.0e-8,
        "tau0_loss_weight": 1.0,
        "rx_offset_loss_weight": 1.0,
    },
    {
        "name": "loss_weight_only",
        "max_total_delay_s": 1.0e-6,
        "max_rx_time_offset_s": 2.0e-7,
        "tau0_loss_weight": 4.0,
        "rx_offset_loss_weight": 4.0,
    },
    {
        "name": "combined",
        "max_total_delay_s": 2.5e-7,
        "max_rx_time_offset_s": 6.0e-8,
        "tau0_loss_weight": 4.0,
        "rx_offset_loss_weight": 4.0,
    },
]


REPLACEMENT_VARIANTS = [
    "model_all",
    "oracle_shared_phase",
    "oracle_total_delay",
    "oracle_rx_offsets",
    "oracle_all_physical_keep_model_uncertainty",
    "oracle_ls_no_ai",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def replacement_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant"]): row for row in rows}


def error_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["parameter"]): row for row in rows}


def plot_setting_curves(setting_root: Path, history: list[dict[str, Any]]) -> None:
    plot_root = setting_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("train_loss", "train loss"),
        ("param_loss", "param loss"),
        ("val_channel_nmse", "validation H NMSE"),
        ("val_param_loss", "validation param loss"),
    ]
    for metric, ylabel in metrics:
        xs = []
        ys = []
        for row in history:
            value = row.get(metric)
            if value is None and metric == "val_channel_nmse":
                value = row.get("channel_nmse")
            if value is None:
                continue
            xs.append(int(row["step"]))
            ys.append(float(value))
        if not xs:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, linewidth=1.7)
        if all(value > 0.0 for value in ys):
            plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(f"{setting_root.name} {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_root / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()


def run_setting(args: argparse.Namespace, setting: dict[str, Any], output_root: Path) -> dict[str, Any]:
    setting_root = output_root / setting["name"]
    setting_root.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(args.config)
    cfg.l_eff = args.l_eff

    setting_arg_values = dict(vars(args))
    setting_arg_values.update(
        {
            "tau0_loss_weight": setting["tau0_loss_weight"],
            "rx_offset_loss_weight": setting["rx_offset_loss_weight"],
            "cfo_loss_weight": args.cfo_loss_weight,
            "delay_loss_weight": args.delay_loss_weight,
            "doppler_loss_weight": args.doppler_loss_weight,
        }
    )
    setting_args = SimpleNamespace(**setting_arg_values)
    model_overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_rel_delay_s": args.max_rel_delay_s,
        "max_total_delay_s": setting["max_total_delay_s"],
        "max_rx_time_offset_s": setting["max_rx_time_offset_s"],
    }

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
        uncertainty_regularization_weight=args.uncertainty_regularization_weight,
        device=args.device,
        return_model_object=True,
        param_loss_weights=param_loss_weights_from_args(setting_args),
    )

    val_data = trained.pop("val_data_object")
    model = trained.pop("model_object")
    device = trained.pop("device_object")
    trained.pop("model_cfg_object", None)
    write_json(setting_root / "trained_result.json", trained)
    plot_setting_curves(setting_root, trained["history"])

    model_params = model_predictions(model, device, val_data)
    oracle_params = oracle_params_for_batches(val_data, args.l_eff)
    replacement_args = SimpleNamespace(
        ridge=args.ridge,
        observed_symbol_indices=args.observed_symbol_indices,
        l_eff=args.l_eff,
        ls_mode=args.ls_mode,
    )
    replacement_rows = run_replacement_ablation(
        val_data,
        model_params,
        oracle_params,
        replacement_args,
    )
    param_error_rows = parameter_error_rows(model_params, oracle_params)
    write_csv(setting_root / "replace_one_parameter.csv", replacement_rows)
    write_csv(setting_root / "parameter_errors.csv", param_error_rows)
    write_json(
        setting_root / "run_metadata.json",
        {
            "setting": setting,
            "args": vars(args),
            "model_overrides": model_overrides,
            "param_loss_weights": param_loss_weights_from_args(setting_args),
        },
    )

    replacement_by_name = replacement_lookup(replacement_rows)
    errors_by_name = error_lookup(param_error_rows)
    summary = {
        "setting": setting["name"],
        "max_total_delay_s": setting["max_total_delay_s"],
        "max_rx_time_offset_s": setting["max_rx_time_offset_s"],
        "tau0_loss_weight": setting["tau0_loss_weight"],
        "rx_offset_loss_weight": setting["rx_offset_loss_weight"],
        "model_all_channel_nmse": float(replacement_by_name["model_all"]["channel_nmse"]),
        "oracle_shared_phase_channel_nmse": float(
            replacement_by_name["oracle_shared_phase"]["channel_nmse"]
        ),
        "oracle_total_delay_channel_nmse": float(
            replacement_by_name["oracle_total_delay"]["channel_nmse"]
        ),
        "oracle_rx_offsets_channel_nmse": float(
            replacement_by_name["oracle_rx_offsets"]["channel_nmse"]
        ),
        "oracle_all_physical_channel_nmse": float(
            replacement_by_name["oracle_all_physical_keep_model_uncertainty"][
                "channel_nmse"
            ]
        ),
        "oracle_ls_no_ai_channel_nmse": float(
            replacement_by_name["oracle_ls_no_ai"]["channel_nmse"]
        ),
        "total_delay_mae": float(errors_by_name["total_delay_s"]["mae"]),
        "total_delay_rmse": float(errors_by_name["total_delay_s"]["rmse"]),
        "total_delay_pred_min": float(errors_by_name["total_delay_s"]["pred_min"]),
        "total_delay_pred_max": float(errors_by_name["total_delay_s"]["pred_max"]),
        "rx_offset_mae": float(errors_by_name["rx_time_offsets_s"]["mae"]),
        "rx_offset_rmse": float(errors_by_name["rx_time_offsets_s"]["rmse"]),
        "rx_offset_pred_min": float(errors_by_name["rx_time_offsets_s"]["pred_min"]),
        "rx_offset_pred_max": float(errors_by_name["rx_time_offsets_s"]["pred_max"]),
        "result_dir": str(setting_root),
    }
    write_quick_readout(setting_root, summary, replacement_rows, param_error_rows)
    return {
        "summary": summary,
        "replacement_rows": replacement_rows,
        "param_error_rows": param_error_rows,
    }


def write_quick_readout(
    setting_root: Path,
    summary: dict[str, Any],
    replacement_rows: list[dict[str, Any]],
    param_error_rows: list[dict[str, Any]],
) -> None:
    ranked = sorted(replacement_rows, key=lambda row: float(row["channel_nmse"]))
    errors = error_lookup(param_error_rows)
    lines = [
        f"# Shared Phase Setting: {summary['setting']}",
        "",
        "## Result",
        "",
        f"- `model_all` H NMSE: `{sci(summary['model_all_channel_nmse'])}`",
        f"- `oracle_shared_phase` H NMSE: `{sci(summary['oracle_shared_phase_channel_nmse'])}`",
        f"- total delay MAE: `{sci(summary['total_delay_mae'])}`",
        f"- RX offset MAE: `{sci(summary['rx_offset_mae'])}`",
        "",
        "## Replace-One Ranking",
        "",
        "| rank | variant | H NMSE |",
        "| ---: | --- | ---: |",
    ]
    for idx, row in enumerate(ranked, 1):
        lines.append(f"| {idx} | `{row['variant']}` | {sci(row['channel_nmse'])} |")
    lines += [
        "",
        "## Parameter Error",
        "",
        "| parameter | pred min | pred max | MAE | RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ["total_delay_s", "rx_time_offsets_s"]:
        row = errors[key]
        lines.append(
            f"| `{key}` | {sci(row['pred_min'])} | {sci(row['pred_max'])} | "
            f"{sci(row['mae'])} | {sci(row['rmse'])} |"
        )
    (setting_root / "QUICK_READOUT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_summary_tables(output_root: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "setting",
        "max_total_delay_s",
        "max_rx_time_offset_s",
        "tau0_loss_weight",
        "rx_offset_loss_weight",
        "model_all_channel_nmse",
        "oracle_shared_phase_channel_nmse",
        "oracle_total_delay_channel_nmse",
        "oracle_rx_offsets_channel_nmse",
        "oracle_all_physical_channel_nmse",
        "oracle_ls_no_ai_channel_nmse",
        "total_delay_mae",
        "total_delay_rmse",
        "total_delay_pred_min",
        "total_delay_pred_max",
        "rx_offset_mae",
        "rx_offset_rmse",
        "rx_offset_pred_min",
        "rx_offset_pred_max",
        "result_dir",
    ]
    csv_path = output_root / "summary_scientific.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: sci(record[key])
                    if isinstance(record.get(key), int | float | np.number)
                    else record.get(key, "")
                    for key in fields
                }
            )

    lines = [
        "# E2 L=6 Shared Phase Supervision Sweep",
        "",
        "| setting | model_all H NMSE | oracle_shared_phase H NMSE | total_delay MAE | rx_offset MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(records, key=lambda item: item["model_all_channel_nmse"]):
        lines.append(
            f"| `{record['setting']}` | {sci(record['model_all_channel_nmse'])} | "
            f"{sci(record['oracle_shared_phase_channel_nmse'])} | "
            f"{sci(record['total_delay_mae'])} | {sci(record['rx_offset_mae'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `model_all` is the learned model with no oracle replacement.",
        "- A smaller gap between `model_all` and `oracle_shared_phase` means the shared phase bottleneck is reduced.",
        "- Compare total delay and RX offset MAE to detect whether prediction collapse improved.",
        "",
        "## Artifacts",
        "",
        "- `summary_scientific.csv`",
        "- `all_variants_scientific.csv`",
        "- `plots/best_h_nmse_by_shared_phase_setting.png`",
        "- `plots/shared_phase_parameter_error.png`",
    ]
    (output_root / "QUICK_READOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_variants(
    output_root: Path,
    setting_outputs: list[dict[str, Any]],
) -> None:
    fields = [
        "setting",
        "variant",
        "channel_nmse",
        "channel_nmse_db",
        "weights_min",
        "weights_max",
    ]
    with (output_root / "all_variants_scientific.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting_output in setting_outputs:
            setting = setting_output["summary"]["setting"]
            for row in setting_output["replacement_rows"]:
                writer.writerow(
                    {
                        "setting": setting,
                        "variant": row["variant"],
                        "channel_nmse": sci(row.get("channel_nmse")),
                        "channel_nmse_db": sci(row.get("channel_nmse_db")),
                        "weights_min": sci(row.get("weights_min")),
                        "weights_max": sci(row.get("weights_max")),
                    }
                )


def plot_summary(output_root: Path, records: list[dict[str, Any]]) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    labels = [record["setting"] for record in records]

    plt.figure(figsize=(9, 5))
    x = np.arange(len(records))
    y_model = [record["model_all_channel_nmse"] for record in records]
    y_shared = [record["oracle_shared_phase_channel_nmse"] for record in records]
    plt.plot(x, y_model, marker="o", linewidth=2.0, label="model_all")
    plt.plot(x, y_shared, marker="o", linewidth=2.0, label="oracle_shared_phase")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("validation H NMSE")
    plt.title("E2 L=6 shared phase setting comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(plot_root / "best_h_nmse_by_shared_phase_setting.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    total_mae = [record["total_delay_mae"] for record in records]
    rx_mae = [record["rx_offset_mae"] for record in records]
    width = 0.38
    plt.bar(x - width / 2, total_mae, width, label="total_delay MAE")
    plt.bar(x + width / 2, rx_mae, width, label="rx_offset MAE")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("absolute error")
    plt.title("E2 L=6 shared phase parameter errors")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(plot_root / "shared_phase_parameter_error.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2 L=6 shared phase diagnostics.")
    parser.add_argument("--config", default="configs/data/e2_effective_paths.yaml")
    parser.add_argument(
        "--output-root",
        default="experiments/e2_effective_paths/diagnostic_l6_shared_phase_supervision",
    )
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
    parser.add_argument("--uncertainty-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--max-rel-delay-s", type=float, default=5.0e-7)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--observed-symbol-indices", nargs="*", type=int, default=[6, 7])
    parser.add_argument("--cfo-loss-weight", type=float, default=1.0)
    parser.add_argument("--delay-loss-weight", type=float, default=1.0)
    parser.add_argument("--doppler-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_metadata.json", {"args": vars(args), "settings": SETTINGS})

    setting_outputs = []
    for setting in SETTINGS:
        print(f"running shared-phase setting={setting['name']}")
        setting_outputs.append(run_setting(args, setting, output_root))

    summaries = [item["summary"] for item in setting_outputs]
    write_summary_tables(output_root, summaries)
    write_all_variants(output_root, setting_outputs)
    plot_summary(output_root, summaries)
    print(f"wrote shared phase diagnostics to {output_root}")


if __name__ == "__main__":
    main()
