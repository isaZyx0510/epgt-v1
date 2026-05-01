"""Build training samples and labels from raw generated outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .npz_io import json_default


DEFAULT_SAMPLE_KEYS = ["rx_grid", "pilot_mask", "pilot_symbols", "data_mask"]
DEFAULT_LABEL_KEYS = ["h_freq"]


def get_by_path(data: Dict[str, Any], key_path: str) -> Any:
    value: Any = data
    for part in key_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Cannot find key path '{key_path}'")
        value = value[part]
    return value


def add_selected_arrays(
    arrays: Dict[str, np.ndarray],
    data: Dict[str, Any],
    keys: Iterable[str],
    prefix: str,
) -> List[str]:
    saved_keys = []
    for key in keys:
        value = get_by_path(data, key)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Selected key '{key}' is not a numpy array")
        saved_key = f"{prefix}.{key}"
        arrays[saved_key] = value
        saved_keys.append(saved_key)
    return saved_keys


def add_full_arrays(arrays: Dict[str, np.ndarray], data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[f"full.{key}"] = value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, np.ndarray):
                    arrays[f"full.{key}.{nested_key}"] = nested_value


def save_sample_label_npz(
    path: str | Path,
    data: Dict[str, Any],
    sample_keys: List[str],
    label_keys: List[str],
    metadata: Dict[str, Any],
    save_full: bool = False,
) -> Tuple[List[str], List[str]]:
    arrays: Dict[str, np.ndarray] = {}
    saved_sample_keys = add_selected_arrays(arrays, data, sample_keys, "sample")
    saved_label_keys = add_selected_arrays(arrays, data, label_keys, "label")

    if save_full:
        add_full_arrays(arrays, data)

    arrays["metadata_json"] = np.array(json.dumps(metadata, default=json_default))
    np.savez_compressed(path, **arrays)
    return saved_sample_keys, saved_label_keys
