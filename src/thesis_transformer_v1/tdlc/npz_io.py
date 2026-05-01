"""NPZ serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_full_npz(path: str | Path, data: Dict[str, Any]) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, dict):
            nested_metadata = {}
            for nested_key, nested_value in value.items():
                full_key = f"{key}.{nested_key}"
                if isinstance(nested_value, np.ndarray):
                    arrays[full_key] = nested_value
                else:
                    nested_metadata[nested_key] = nested_value
            if nested_metadata:
                metadata[key] = nested_metadata
        else:
            metadata[key] = value

    arrays["metadata_json"] = np.array(json.dumps(metadata, default=json_default))
    np.savez_compressed(path, **arrays)
