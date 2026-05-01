"""Plotting helpers for experiment metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def rows_from_metrics(metrics):
    if isinstance(metrics, list):
        return metrics
    return [metrics]


def plot_comparison(rows: list[dict], output: str | Path, metric: str = "channel_nmse") -> Path:
    labels = [row["method"] for row in rows]
    values = [row["final"].get(metric, float("nan")) for row in rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)])
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    return output


def plot_sweep(rows: list[dict], output: str | Path, metric: str = "channel_nmse") -> Path:
    methods = sorted({row["method"] for row in rows})
    plt.figure(figsize=(6, 4))
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row.get("sweep_value", 0),
        )
        x = [row.get("sweep_value", 0) for row in method_rows]
        y = [row["final"].get(metric, float("nan")) for row in method_rows]
        plt.plot(x, y, marker="o", label=method)
    sweep_name = rows[0].get("sweep_name", "sweep")
    plt.xlabel(sweep_name)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    return output

