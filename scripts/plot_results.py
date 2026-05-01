"""Plot experiment metrics JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis_transformer_v1.experiments.metrics_io import read_metrics
from thesis_transformer_v1.experiments.plotting import plot_comparison, plot_sweep, rows_from_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison or sweep metrics.")
    parser.add_argument("--metrics", default="experiments/e1_clean_transformer/comparison_metrics.json")
    parser.add_argument("--output", default="experiments/e1_clean_transformer/comparison_nmse.png")
    parser.add_argument("--metric", default="channel_nmse")
    parser.add_argument("--kind", choices=["auto", "comparison", "sweep"], default="auto")
    args = parser.parse_args()

    rows = rows_from_metrics(read_metrics(args.metrics))
    kind = args.kind
    if kind == "auto":
        kind = "sweep" if rows and "sweep_name" in rows[0] else "comparison"
    if kind == "sweep":
        plot_sweep(rows, args.output, args.metric)
    else:
        plot_comparison(rows, args.output, args.metric)
    print(args.output)


if __name__ == "__main__":
    main()
