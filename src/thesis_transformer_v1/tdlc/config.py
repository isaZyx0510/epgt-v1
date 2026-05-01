"""Configuration dataclasses and YAML/JSON loading."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML.
    yaml = None


@dataclass
class OFDMConfig:
    n_tx: int = 4
    n_rx: int = 4
    n_sc: int = 128
    n_sym: int = 14
    cp_len: int = 16
    subcarrier_spacing_hz: float = 15_000.0
    modulation: Literal["QPSK", "16QAM", "64QAM"] = "QPSK"


@dataclass
class PilotConfig:
    mode: Literal["block", "comb", "sparse2d"] = "sparse2d"
    symbol_spacing: int = 4
    subcarrier_spacing: int = 8
    symbol_offset: int = 0
    subcarrier_offset: int = 0
    orthogonal_axis: Literal["time", "frequency"] = "frequency"
    pilot_power: float = 1.0


@dataclass
class ChannelConfig:
    model: Literal["tdl_c", "custom"] = "tdl_c"
    mimo_mode: Literal["common_delay", "array_response"] = "array_response"
    n_paths: int = 8
    delay_spread_s: float = 300e-9
    normalize_power: bool = True
    dominant_path_decay_db: Optional[float] = 3.0
    max_doppler_hz: float = 0.0
    carrier_frequency_hz: float = 3.5e9
    antenna_spacing_wavelength: float = 0.5
    rx_offset_range_wavelength: Tuple[float, float] = (0.0, 0.0)
    angle_range_deg: Tuple[float, float] = (-60.0, 60.0)
    randomize_path_subset: bool = False


@dataclass
class ImpairmentConfig:
    snr_db: float = 20.0
    cfo_hz: float = 0.0
    random_cfo_hz: float = 0.0


@dataclass
class DatasetConfig:
    batch_size: int = 32
    seed: int = 1234
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    impairment: ImpairmentConfig = field(default_factory=ImpairmentConfig)


def update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def config_from_dict(values: Dict[str, Any]) -> DatasetConfig:
    cfg = DatasetConfig()
    return update_dataclass(cfg, values)


def load_mapping(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError(f"Unsupported config suffix {path.suffix!r}; use .yaml/.yml/.json")
    return loaded or {}


def load_config(path: str | Path) -> DatasetConfig:
    return config_from_dict(load_mapping(path))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
