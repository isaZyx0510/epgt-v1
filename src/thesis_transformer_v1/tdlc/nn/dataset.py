"""PyTorch dataset wrappers for sparse pilot tokens."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from dataset_io import NpzShardDataset
from .labels import PhysicalLabelConfig, build_physical_labels
from .tokenizer import SparsePilotTokenConfig, build_sparse_pilot_tokens


class SparsePilotNpzDataset(Dataset):
    def __init__(
        self,
        files: Iterable[str],
        token_config: SparsePilotTokenConfig | None = None,
        label_config: PhysicalLabelConfig | None = None,
    ) -> None:
        self.base = NpzShardDataset(files)
        self.token_config = token_config or SparsePilotTokenConfig()
        self.label_config = label_config or PhysicalLabelConfig()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        split: str | None = None,
        stage: str | None = None,
        token_config: SparsePilotTokenConfig | None = None,
        label_config: PhysicalLabelConfig | None = None,
    ) -> "SparsePilotNpzDataset":
        base = NpzShardDataset.from_manifest(manifest_path, split=split, stage=stage)
        obj = cls.__new__(cls)
        obj.base = base
        obj.token_config = token_config or SparsePilotTokenConfig()
        obj.label_config = label_config or PhysicalLabelConfig()
        return obj

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base[index]
        sample = item["sample"]
        token_data = build_sparse_pilot_tokens(
            sample["rx_grid"],
            sample["pilot_symbols"],
            sample["pilot_mask"],
            self.token_config,
        )
        label_data = build_physical_labels(item["label"], self.label_config)
        return {
            **token_data,
            **label_data,
            "meta": item["meta"],
        }

    def close(self) -> None:
        self.base.close()


def collate_sparse_pilot_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_tokens = max(item["tokens"].shape[0] for item in items)
    feat_dim = items[0]["tokens"].shape[-1]
    bsz = len(items)

    tokens = torch.zeros((bsz, max_tokens, feat_dim), dtype=torch.float32)
    padding_mask = torch.ones((bsz, max_tokens), dtype=torch.bool)
    for i, item in enumerate(items):
        count = item["tokens"].shape[0]
        tokens[i, :count] = item["tokens"]
        padding_mask[i, :count] = False

    return {
        "tokens": tokens,
        "token_padding_mask": padding_mask,
        "path_params": torch.stack([item["path_params"] for item in items], dim=0),
        "path_mask": torch.stack([item["path_mask"] for item in items], dim=0),
        "delta_r": torch.stack([item["delta_r"] for item in items], dim=0),
        "meta": [item["meta"] for item in items],
    }
