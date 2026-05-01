"""Sweep helpers for E1-E5 quick experiment pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from thesis_transformer_v1.data.config import ExperimentConfig, load_experiment_config
from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.experiments.evaluation import evaluate_oracle_ls
from thesis_transformer_v1.experiments.training import train_direct_h_quick, train_hybrid_quick
from thesis_transformer_v1.models import EPGTGuidanceConfig


def clone_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return deepcopy(cfg)


def run_oracle_reference(cfg: ExperimentConfig) -> dict[str, Any]:
    data = generate_thesis_dataset(cfg)
    obs = build_observation_tokens(
        data,
        ObservationTokenConfig(
            cfg.observation.input_symbol_indices,
            cfg.observation.symbol_error_rate,
            cfg.observation.include_reliability,
        ),
        rng=np.random.default_rng(cfg.dataset.seed + 99),
    )
    final = evaluate_oracle_ls(
        data,
        obs,
        cfg.l_eff,
        ridge=cfg.ridge,
        observed_symbol_indices=cfg.observation.input_symbol_indices,
    )
    return {"history": [], "final": final}


def run_method(
    cfg: ExperimentConfig,
    method: str,
    steps: int,
    lr: float,
    eval_interval: int,
    train_batches: int = 1,
    val_batches: int = 1,
    model_overrides: dict[str, Any] | None = None,
    architecture: str = "current",
    guidance: EPGTGuidanceConfig | None = None,
    ls_mode: str = "traditional_ls",
    loss_mode: str = "param",
    reconstruction_weight: float = 1.0,
    warmup_steps: int = 0,
    finetune_loss_mode: str = "param_plus_reconstruction",
) -> dict[str, Any]:
    if method == "oracle_ls":
        return run_oracle_reference(cfg)
    if method == "hybrid":
        return train_hybrid_quick(
            cfg,
            steps=steps,
            lr=lr,
            eval_interval=eval_interval,
            train_batches=train_batches,
            val_batches=val_batches,
            model_overrides=model_overrides,
            architecture=architecture,
            guidance=guidance,
            ls_mode=ls_mode,
            loss_mode=loss_mode,
            reconstruction_weight=reconstruction_weight,
            warmup_steps=warmup_steps,
            finetune_loss_mode=finetune_loss_mode,
        )
    if method == "direct_h":
        return train_direct_h_quick(
            cfg,
            steps=steps,
            lr=lr,
            eval_interval=eval_interval,
            train_batches=train_batches,
            val_batches=val_batches,
            model_overrides=model_overrides,
            architecture=architecture,
        )
    raise ValueError(f"Unknown method: {method}")


def run_method_set(
    cfg: ExperimentConfig,
    experiment: str,
    methods: list[str],
    steps: int,
    lr: float,
    eval_interval: int,
    train_batches: int = 1,
    val_batches: int = 1,
    model_overrides: dict[str, Any] | None = None,
    architecture: str = "current",
    guidance: EPGTGuidanceConfig | None = None,
    ls_mode: str = "traditional_ls",
    loss_mode: str = "param",
    reconstruction_weight: float = 1.0,
    warmup_steps: int = 0,
    finetune_loss_mode: str = "param_plus_reconstruction",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        result = run_method(
            cfg,
            method,
            steps=steps,
            lr=lr,
            eval_interval=eval_interval,
            train_batches=train_batches,
            val_batches=val_batches,
            model_overrides=model_overrides,
            architecture=architecture,
            guidance=guidance,
            ls_mode=ls_mode,
            loss_mode=loss_mode,
            reconstruction_weight=reconstruction_weight,
            warmup_steps=warmup_steps,
            finetune_loss_mode=finetune_loss_mode,
        )
        rows.append(
            {
                "experiment": experiment,
                "method": method,
                "history": result["history"],
                "final": result["final"],
                **(extra or {}),
            }
        )
    return rows


def load_base_config(path: str) -> ExperimentConfig:
    return load_experiment_config(path)
