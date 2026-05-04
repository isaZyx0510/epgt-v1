"""Reusable training helpers for thesis experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from thesis_transformer_v1.data.config import ExperimentConfig
from thesis_transformer_v1.data.generator import generate_thesis_dataset
from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.estimation.differentiable_ls import (
    complex_nmse_loss,
    reconstruct_via_differentiable_ls,
)
from thesis_transformer_v1.experiments.evaluation import (
    evaluate_direct_h_tensor,
    evaluate_hybrid_params,
    torch_params_to_numpy,
)
from thesis_transformer_v1.models import (
    EPGTGuidanceConfig,
    TransformerConfig,
    build_direct_h_transformer,
    build_hybrid_transformer,
)


def prepare_batch(
    cfg: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]:
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
    tokens = torch.from_numpy(obs["tokens"])
    return data, obs, tokens


def seeded_config(cfg: ExperimentConfig, seed: int) -> ExperimentConfig:
    cloned = deepcopy(cfg)
    cloned.dataset.seed = int(seed)
    return cloned


def model_config_from_tokens(
    cfg: ExperimentConfig,
    tokens: torch.Tensor,
    model_overrides: dict[str, Any] | None = None,
) -> TransformerConfig:
    cfo_bound = max(abs(v) for v in cfg.thesis_impairment.cfo_range_hz)
    tau_bound = max(abs(v) for v in cfg.thesis_impairment.tau0_range_s)
    rx_bound = max(abs(v) for v in cfg.thesis_impairment.rx_time_offset_range_s)
    doppler_bound = abs(float(cfg.dataset.channel.max_doppler_hz))
    model_cfg = TransformerConfig(
        input_dim=tokens.shape[-1],
        l_eff=cfg.l_eff,
        n_rx=cfg.dataset.ofdm.n_rx,
        n_tx=cfg.dataset.ofdm.n_tx,
        n_sym=cfg.dataset.ofdm.n_sym,
        n_sc=cfg.dataset.ofdm.n_sc,
        max_doppler_hz=max(doppler_bound, 1e-6),
        max_total_delay_s=max(tau_bound, 1e-12),
        max_cfo_hz=max(cfo_bound, 1e-6),
        max_rx_time_offset_s=max(rx_bound, 1e-12),
    )
    for key, value in (model_overrides or {}).items():
        if value is not None and hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
    return model_cfg


def prepare_batches(
    cfg: ExperimentConfig,
    count: int,
    seed_offset: int,
) -> list[tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]]:
    batches = []
    for i in range(count):
        batches.append(prepare_batch(seeded_config(cfg, cfg.dataset.seed + seed_offset + i)))
    return batches


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    averaged: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        if all(isinstance(value, int | float | np.number) for value in values):
            averaged[key] = float(np.mean(values))
        else:
            averaged[key] = values[0]
    return averaged


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _model_summary(
    model: torch.nn.Module,
    model_cfg: TransformerConfig,
    device: torch.device,
) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "model_cfg": dict(model_cfg.__dict__),
        "parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "device": str(device),
    }

#参数域损失
def normalized_param_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    model_cfg: TransformerConfig,
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["total_delay_s"].device
    weights = {
        "tau0": 1.0,
        "cfo": 1.0,
        "delay": 1.0,
        "doppler": 1.0,
        "rx_offset": 1.0,
    }
    weights.update(loss_weights or {})
    target_tau0 = torch.from_numpy(labels["total_delay_s"]).to(device)
    target_cfo = torch.from_numpy(labels["cfo_hz"]).to(device)
    target_delay = torch.from_numpy(labels["rel_delay_s"]).to(device)
    target_doppler = torch.from_numpy(labels["doppler_hz"]).to(device)
    target_rx = torch.from_numpy(labels["rx_time_offsets_s"]).to(device)

    loss_tau0 = F.mse_loss(
        outputs["total_delay_s"] / model_cfg.max_total_delay_s,
        target_tau0 / model_cfg.max_total_delay_s,
    )
    loss_cfo = F.mse_loss(
        outputs["cfo_hz"] / model_cfg.max_cfo_hz,
        target_cfo / model_cfg.max_cfo_hz,
    )
    loss_delay = F.mse_loss(
        outputs["rel_delay_s"] / model_cfg.max_rel_delay_s,
        target_delay / model_cfg.max_rel_delay_s,
    )
    loss_doppler = F.mse_loss(
        outputs["doppler_hz"] / model_cfg.max_doppler_hz,
        target_doppler / model_cfg.max_doppler_hz,
    )
    loss_rx = F.mse_loss(
        outputs["rx_time_offsets_s"] / model_cfg.max_rx_time_offset_s,
        target_rx / model_cfg.max_rx_time_offset_s,
    )
    total = (
        float(weights["tau0"]) * loss_tau0
        + float(weights["cfo"]) * loss_cfo
        + float(weights["delay"]) * loss_delay
        + float(weights["doppler"]) * loss_doppler
        + float(weights["rx_offset"]) * loss_rx
    )
    parts = {
        "tau0_loss": float(loss_tau0.detach()),
        "cfo_loss": float(loss_cfo.detach()),
        "delay_loss": float(loss_delay.detach()),
        "doppler_loss": float(loss_doppler.detach()),
        "rx_offset_loss": float(loss_rx.detach()),
        "tau0_loss_weight": float(weights["tau0"]),
        "cfo_loss_weight": float(weights["cfo"]),
        "delay_loss_weight": float(weights["delay"]),
        "doppler_loss_weight": float(weights["doppler"]),
        "rx_offset_loss_weight": float(weights["rx_offset"]),
    }
    if "rel_delay_log_var" in outputs and "doppler_log_var" in outputs:
        delay_err = (outputs["rel_delay_s"] - target_delay) / model_cfg.max_rel_delay_s
        doppler_err = (outputs["doppler_hz"] - target_doppler) / model_cfg.max_doppler_hz
        delay_var = torch.exp(outputs["rel_delay_log_var"]) / (model_cfg.max_rel_delay_s**2)
        doppler_var = torch.exp(outputs["doppler_log_var"]) / (model_cfg.max_doppler_hz**2)
        delay_var = delay_var.clamp(1e-8, 1e2)
        doppler_var = doppler_var.clamp(1e-8, 1e2)
        delay_nll = 0.5 * (delay_err.square() / delay_var + torch.log(delay_var)).mean()
        doppler_nll = 0.5 * (doppler_err.square() / doppler_var + torch.log(doppler_var)).mean()
        uncertainty_loss = delay_nll + doppler_nll
        total = total + 0.1 * uncertainty_loss
        parts.update(
            {
                "delay_uncertainty_nll": float(delay_nll.detach()),
                "doppler_uncertainty_nll": float(doppler_nll.detach()),
                "uncertainty_loss": float(uncertainty_loss.detach()),
            }
        )
    return total, parts


def path_uncertainty_regularization(
    outputs: dict[str, torch.Tensor],
    model_cfg: TransformerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Lightly discourage path uncertainty from becoming extreme under H-only loss."""
    device = outputs["rel_delay_s"].device
    zero = torch.zeros((), device=device)
    if "rel_delay_log_var" not in outputs or "doppler_log_var" not in outputs:
        return zero, {}

    delay_log_var = outputs["rel_delay_log_var"]
    doppler_log_var = outputs["doppler_log_var"]
    delay_log_scale = 2.0 * np.log(max(model_cfg.max_rel_delay_s, 1e-12))
    doppler_log_scale = 2.0 * np.log(max(model_cfg.max_doppler_hz, 1e-6))
    delay_norm_log_var = delay_log_var - float(delay_log_scale)
    doppler_norm_log_var = doppler_log_var - float(doppler_log_scale)
    loss = delay_norm_log_var.square().mean() + doppler_norm_log_var.square().mean()
    return loss, {
        "uncertainty_reg_loss": float(loss.detach()),
        "rel_delay_log_var_mean": float(delay_log_var.mean().detach()),
        "doppler_log_var_mean": float(doppler_log_var.mean().detach()),
    }


def reconstruction_loss_from_outputs(
    outputs: dict[str, torch.Tensor],
    data: dict[str, Any],
    observation: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    ls_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute differentiable full-grid H reconstruction loss."""
    _gains, h_hat = reconstruct_via_differentiable_ls(
        data["rx_grid"],
        observation["tx_grid_observed"],
        observation["indices"],
        outputs,
        data["freq_hz"],
        data["time_s"],
        ridge=max(cfg.ridge, 1e-8),
        use_uncertainty_weights=ls_mode == "learnable_weighted_ls",
    )
    full_loss = complex_nmse_loss(h_hat, data["h_freq"])
    obs_symbols = cfg.observation.input_symbol_indices
    observed_loss = complex_nmse_loss(
        h_hat[:, :, :, obs_symbols, :],
        data["h_freq"][:, :, :, obs_symbols, :],
    )
    return full_loss, {
        "reconstruction_loss": float(full_loss.detach()),
        "observed_reconstruction_loss": float(observed_loss.detach()),
    }  #full_loss (torch.Tensor): 全网格 NMSE。衡量整张 5D 时频网格上估计信道的准确性


def validation_param_loss_from_outputs(
    outputs: dict[str, torch.Tensor],
    data: dict[str, Any],
    cfg: ExperimentConfig,
    model_cfg: TransformerConfig,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    labels = nonlinear_oracle_params(data["channel_labels"], l_eff=cfg.l_eff)
    loss, parts = normalized_param_loss(outputs, labels, model_cfg, loss_weights)
    metrics = {"val_param_loss": float(loss.detach())}
    metrics.update({f"val_{key}": value for key, value in parts.items()})
    return metrics


def train_hybrid_quick(
    cfg: ExperimentConfig,
    steps: int = 25,
    lr: float = 1e-3,
    eval_interval: int = 5,
    train_batches: int = 1,
    val_batches: int = 1,
    model_overrides: dict[str, Any] | None = None,
    architecture: str = "current",
    guidance: EPGTGuidanceConfig | None = None,
    ls_mode: str = "traditional_ls",
    loss_mode: str = "reconstruction",
    reconstruction_weight: float = 1.0,
    warmup_steps: int = 0,
    finetune_loss_mode: str = "param_plus_reconstruction",
    uncertainty_regularization_weight: float = 0.0,
    device: str | torch.device | None = None,
    return_model_object: bool = False,
    param_loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if loss_mode not in {"param", "reconstruction", "param_plus_reconstruction", "two_stage"}:
        raise ValueError(
            "loss_mode must be one of 'param', 'reconstruction', or "
            f"'param_plus_reconstruction', or 'two_stage', got {loss_mode!r}"
        )
    if finetune_loss_mode not in {"reconstruction", "param_plus_reconstruction"}:
        raise ValueError(
            "finetune_loss_mode must be 'reconstruction' or "
            f"'param_plus_reconstruction', got {finetune_loss_mode!r}"
        )
    if loss_mode == "two_stage" and warmup_steps <= 0:
        raise ValueError("two_stage training requires warmup_steps > 0")
    train_data = prepare_batches(cfg, max(1, train_batches), seed_offset=0)
    val_data = prepare_batches(cfg, max(1, val_batches), seed_offset=10_000)
    _first_data, _first_obs, first_tokens = train_data[0]
    model_cfg = model_config_from_tokens(cfg, first_tokens, model_overrides=model_overrides)
    model = build_hybrid_transformer(model_cfg, architecture=architecture, guidance=guidance)
    train_device = _resolve_device(device)
    model.to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []

    for step in range(steps):
        data, _obs, tokens = train_data[step % len(train_data)]
        active_loss_mode = loss_mode
        stage = "single"
        if loss_mode == "two_stage":
            if step < warmup_steps:
                active_loss_mode = "param"
                stage = "warmup"
            else:
                active_loss_mode = finetune_loss_mode
                stage = "finetune"

        optimizer.zero_grad(set_to_none=True)
        tokens = tokens.to(train_device)
        outputs = model(tokens)
        param_loss: torch.Tensor | None = None
        param_parts: dict[str, float] = {}
        rec_parts: dict[str, float] = {}
        uncertainty_reg_parts: dict[str, float] = {}
        if active_loss_mode in {"param", "param_plus_reconstruction"}:
            labels = nonlinear_oracle_params(data["channel_labels"], l_eff=cfg.l_eff)
            param_loss, param_parts = normalized_param_loss(
                outputs,
                labels,
                model_cfg,
                param_loss_weights,
            )

        if active_loss_mode == "param":
            if param_loss is None:
                raise RuntimeError("param loss requires physical parameter labels")
            loss = param_loss
        else:
            rec_loss, rec_parts = reconstruction_loss_from_outputs(
                outputs,
                data,
                _obs,
                cfg,
                ls_mode,
            )
            if active_loss_mode == "reconstruction":
                loss = rec_loss
            else:
                if param_loss is None:
                    raise RuntimeError(
                        "param_plus_reconstruction requires physical parameter labels"
                    )
                loss = param_loss + float(reconstruction_weight) * rec_loss
        uncertainty_reg, uncertainty_reg_parts = path_uncertainty_regularization(
            outputs,
            model_cfg,
        )
        if float(uncertainty_regularization_weight) > 0.0 and uncertainty_reg_parts:
            loss = loss + float(uncertainty_regularization_weight) * uncertainty_reg
        loss.backward()
        optimizer.step()

        row: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.detach()),
            "loss_mode": loss_mode,
            "active_loss_mode": active_loss_mode,
            "stage": stage,
            "warmup_steps": warmup_steps,
            "reconstruction_weight": reconstruction_weight,
            "uncertainty_regularization_weight": uncertainty_regularization_weight,
            **param_parts,
            **rec_parts,
            **uncertainty_reg_parts,
        }
        if param_loss is not None:
            row["param_loss"] = float(param_loss.detach())
        should_eval = step == 0 or step == steps - 1
        should_eval = should_eval or (eval_interval > 0 and (step + 1) % eval_interval == 0)
        if should_eval:
            eval_rows = []
            param_eval_rows = []
            for val_batch in val_data:
                val_sample, val_obs, val_tokens = val_batch
                with torch.no_grad():
                    eval_outputs = model(val_tokens.to(train_device))
                    param_eval_rows.append(
                        validation_param_loss_from_outputs(
                            eval_outputs,
                            val_sample,
                            cfg,
                            model_cfg,
                            param_loss_weights,
                        )
                    )
                params = torch_params_to_numpy(eval_outputs)
                eval_rows.append(
                    evaluate_hybrid_params(
                        params,
                        val_sample,
                        val_obs,
                        ridge=cfg.ridge,
                        observed_symbol_indices=cfg.observation.input_symbol_indices,
                        ls_mode=ls_mode,
                    )
                )
            eval_metrics = average_metric_rows(eval_rows)
            row.update(eval_metrics)
            row.update({f"val_{key}": value for key, value in eval_metrics.items()})
            row.update(average_metric_rows(param_eval_rows))
        history.append(row)

    final = history[-1].copy()
    if "channel_nmse_db" not in final:
        eval_rows = []
        for val_sample, val_obs, val_tokens in val_data:
            with torch.no_grad():
                params = torch_params_to_numpy(model(val_tokens.to(train_device)))
            eval_rows.append(
                evaluate_hybrid_params(
                    params,
                    val_sample,
                    val_obs,
                    ridge=cfg.ridge,
                    observed_symbol_indices=cfg.observation.input_symbol_indices,
                    ls_mode=ls_mode,
                )
            )
        final.update(average_metric_rows(eval_rows))
    result: dict[str, Any] = {
        "history": history,
        "final": final,
        "model": _model_summary(model, model_cfg, train_device),
    }
    if return_model_object:
        result["model_object"] = model
        result["model_cfg_object"] = model_cfg
        result["device_object"] = train_device
        result["val_data_object"] = val_data
    return result


def train_direct_h_quick(
    cfg: ExperimentConfig,
    steps: int = 25,
    lr: float = 1e-3,
    eval_interval: int = 5,
    train_batches: int = 1,
    val_batches: int = 1,
    model_overrides: dict[str, Any] | None = None,
    architecture: str = "current",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    train_data = prepare_batches(cfg, max(1, train_batches), seed_offset=0)
    val_data = prepare_batches(cfg, max(1, val_batches), seed_offset=10_000)
    _first_data, _first_obs, first_tokens = train_data[0]
    model_cfg = model_config_from_tokens(cfg, first_tokens, model_overrides=model_overrides)
    model = build_direct_h_transformer(model_cfg, architecture=architecture)
    train_device = _resolve_device(device)
    model.to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []

    for step in range(steps):
        data, _obs, tokens = train_data[step % len(train_data)]
        target = torch.from_numpy(
            np.stack([data["h_freq"].real, data["h_freq"].imag], axis=-1).astype("float32")
        )
        optimizer.zero_grad(set_to_none=True)
        tokens = tokens.to(train_device)
        target = target.to(train_device)
        pred = model(tokens)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        row: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.detach()),
        }
        should_eval = step == 0 or step == steps - 1
        should_eval = should_eval or (eval_interval > 0 and (step + 1) % eval_interval == 0)
        if should_eval:
            eval_rows = []
            for val_sample, _val_obs, val_tokens in val_data:
                with torch.no_grad():
                    eval_rows.append(
                        evaluate_direct_h_tensor(
                            model(val_tokens.to(train_device)),
                            val_sample["h_freq"],
                            observed_symbol_indices=cfg.observation.input_symbol_indices,
                        )
                    )
            eval_metrics = average_metric_rows(eval_rows)
            row.update(eval_metrics)
            row.update({f"val_{key}": value for key, value in eval_metrics.items()})
        history.append(row)
    final = history[-1].copy()
    return {
        "history": history,
        "final": final,
        "model": _model_summary(model, model_cfg, train_device),
    }
