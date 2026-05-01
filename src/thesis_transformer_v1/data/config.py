"""Configuration helpers for thesis experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_transformer_v1.tdlc.config import DatasetConfig, config_from_dict, deep_merge, load_mapping


@dataclass
class ThesisImpairmentConfig:
    tau0_range_s: tuple[float, float] = (0.0, 0.0)
    cfo_range_hz: tuple[float, float] = (0.0, 0.0)
    rx_time_offset_range_s: tuple[float, float] = (0.0, 0.0)
    awgn_snr_db: float | None = None


@dataclass
class ObservationConfig:
    input_symbol_indices: tuple[int, ...] = (6, 7)
    symbol_error_rate: float = 0.0
    include_reliability: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thesis_impairment: ThesisImpairmentConfig = field(default_factory=ThesisImpairmentConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    l_eff: int = 12
    ridge: float = 0.0


def _as_tuple(values: Any, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    if len(values) != 2:
        raise ValueError(f"Expected two values, got {values!r}")
    return float(values[0]), float(values[1])


def experiment_config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    dataset_values = {
        key: value
        for key, value in values.items()
        if key in {"batch_size", "seed", "ofdm", "pilot", "channel", "impairment"}
    }
    dataset = config_from_dict(dataset_values)

    thesis_values = values.get("thesis_impairment", {})
    thesis_impairment = ThesisImpairmentConfig(
        tau0_range_s=_as_tuple(
            thesis_values.get("tau0_range_s"), ThesisImpairmentConfig.tau0_range_s
        ),
        cfo_range_hz=_as_tuple(
            thesis_values.get("cfo_range_hz"), ThesisImpairmentConfig.cfo_range_hz
        ),
        rx_time_offset_range_s=_as_tuple(
            thesis_values.get("rx_time_offset_range_s"),
            ThesisImpairmentConfig.rx_time_offset_range_s,
        ),
        awgn_snr_db=thesis_values.get("awgn_snr_db"),
    )
    if thesis_impairment.awgn_snr_db is not None:
        thesis_impairment.awgn_snr_db = float(thesis_impairment.awgn_snr_db)

    observation_values = values.get("observation", {})
    observation = ObservationConfig(
        input_symbol_indices=tuple(
            int(v) for v in observation_values.get("input_symbol_indices", (6, 7))
        ),
        symbol_error_rate=float(observation_values.get("symbol_error_rate", 0.0)),
        include_reliability=bool(observation_values.get("include_reliability", True)),
    )
    return ExperimentConfig(
        dataset=dataset,
        thesis_impairment=thesis_impairment,
        observation=observation,
        l_eff=int(values.get("l_eff", dataset.channel.n_paths)),
        ridge=float(values.get("ridge", 0.0)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    values = load_mapping(path)
    extends = values.get("extends")
    if extends:
        base_path = path.parent / str(extends)
        base_values = load_mapping(base_path)
        values = deep_merge(base_values, {k: v for k, v in values.items() if k != "extends"})
    return experiment_config_from_dict(values)
