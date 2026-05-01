"""Unified metrics schema helpers for experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def experiment_name_from_config(config_path: str | Path) -> str:
    return Path(config_path).stem


def default_metrics_path(experiment: str, method: str) -> Path:
    return Path("experiments") / experiment / method / "metrics.json"


def build_metrics(
    experiment: str,
    method: str,
    config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    final: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "experiment": experiment,
        "method": method,
        "config": config or {},
        "history": history or [],
        "final": final or {},
    }
    if extra:
        metrics.update(extra)
    return metrics


def write_metrics(metrics: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output


def read_metrics(path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

