"""Evaluation helpers shared by training and sweep scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from thesis_transformer_v1.data.labels import nonlinear_oracle_params
from thesis_transformer_v1.estimation.common_delay import (
    reconstruct_channel,
    recover_path_gains_ls,
)
from thesis_transformer_v1.estimation.ls_plugins import recover_with_ls_plugin
from thesis_transformer_v1.metrics import nmse, nmse_db


def _nmse_per_sample(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    axes = tuple(range(1, prediction.ndim))
    numerator = np.sum(np.abs(prediction - target) ** 2, axis=axes)
    denominator = np.sum(np.abs(target) ** 2, axis=axes) + eps
    return numerator / denominator


def _finite_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    stats = {
        f"{prefix}_nonfinite_count": float(values.size - finite.size),
    }
    if finite.size == 0:
        stats.update(
            {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"),
                f"{prefix}_p95": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        )
        return stats
    stats.update(
        {
            f"{prefix}_mean": float(np.mean(finite)),
            f"{prefix}_median": float(np.median(finite)),
            f"{prefix}_p95": float(np.percentile(finite, 95.0)),
            f"{prefix}_max": float(np.max(finite)),
        }
    )
    return stats


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if np.allclose(x_valid, x_valid[0]) or np.allclose(y_valid, y_valid[0]):
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _ls_condition_metrics(
    diagnostics: dict[str, np.ndarray],
    h_hat: np.ndarray,
    target_h: np.ndarray,
    observed_symbol_indices: tuple[int, ...],
) -> dict[str, float]:
    cond_a = diagnostics["cond_a"]
    cond_gram = diagnostics["cond_gram"]
    ranks = diagnostics["rank"]
    num_columns = int(diagnostics["num_columns"])
    channel_nmse_by_sample = _nmse_per_sample(h_hat, target_h)
    channel_nmse_db_by_sample = 10.0 * np.log10(channel_nmse_by_sample + 1.0e-12)
    observed_nmse_by_sample = _nmse_per_sample(
        h_hat[:, :, :, observed_symbol_indices, :],
        target_h[:, :, :, observed_symbol_indices, :],
    )
    observed_nmse_db_by_sample = 10.0 * np.log10(observed_nmse_by_sample + 1.0e-12)
    sample_cond_a_max = np.max(cond_a, axis=1)
    sample_cond_a_mean = np.mean(cond_a, axis=1)
    sample_cond_gram_max = np.max(cond_gram, axis=1)
    log_sample_cond_a_max = np.log10(sample_cond_a_max)
    log_sample_cond_a_mean = np.log10(sample_cond_a_mean)
    log_sample_cond_gram_max = np.log10(sample_cond_gram_max)

    metrics = {
        **_finite_stats("ls_cond_a", cond_a),
        **_finite_stats("ls_cond_gram", cond_gram),
        "ls_rank_min": float(np.min(ranks)),
        "ls_rank_mean": float(np.mean(ranks)),
        "ls_rank_deficient_count": float(np.count_nonzero(ranks < num_columns)),
        "ls_log10_cond_a_max_channel_nmse_db_corr": _safe_corrcoef(
            log_sample_cond_a_max,
            channel_nmse_db_by_sample,
        ),
        "ls_log10_cond_a_mean_channel_nmse_db_corr": _safe_corrcoef(
            log_sample_cond_a_mean,
            channel_nmse_db_by_sample,
        ),
        "ls_log10_cond_gram_max_channel_nmse_db_corr": _safe_corrcoef(
            log_sample_cond_gram_max,
            channel_nmse_db_by_sample,
        ),
        "ls_log10_cond_a_max_observed_nmse_db_corr": _safe_corrcoef(
            log_sample_cond_a_max,
            observed_nmse_db_by_sample,
        ),
    }
    return metrics


def torch_params_to_numpy(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in outputs.items()
    }


def evaluate_hybrid_params(
    params: dict[str, np.ndarray],
    data: dict[str, Any],
    observation: dict[str, np.ndarray],
    ridge: float = 0.0,
    observed_symbol_indices: tuple[int, ...] = (6, 7),
    ls_mode: str = "traditional_ls",
) -> dict[str, float]:
    diagnostics = None
    if ls_mode == "traditional_ls":
        gains, diagnostics = recover_path_gains_ls(
            data["rx_grid"],
            observation["tx_grid_observed"],
            observation["indices"],
            params,
            data["freq_hz"],
            data["time_s"],
            ridge=ridge,
            return_diagnostics=True,
        )
        h_hat = reconstruct_channel(gains, params, data["freq_hz"], data["time_s"])
        ls_extra = {"ls_mode": ls_mode}
    else:
        _gains, h_hat, ls_extra = recover_with_ls_plugin(
            ls_mode,
            data["rx_grid"],
            observation["tx_grid_observed"],
            observation["indices"],
            params,
            data["freq_hz"],
            data["time_s"],
            ridge=ridge,
        )
    channel_nmse = nmse(h_hat, data["h_freq"])
    observed_nmse = nmse(
        h_hat[:, :, :, observed_symbol_indices, :],
        data["h_freq"][:, :, :, observed_symbol_indices, :],
    )
    metrics = {
        "channel_nmse": channel_nmse,
        "observed_symbol_nmse": observed_nmse,
        "channel_nmse_db": nmse_db(h_hat, data["h_freq"]),
        "observed_symbol_nmse_db": nmse_db(
            h_hat[:, :, :, observed_symbol_indices, :],
            data["h_freq"][:, :, :, observed_symbol_indices, :],
        ),
        **ls_extra,
    }
    if diagnostics is not None:
        metrics.update(
            _ls_condition_metrics(
                diagnostics,
                h_hat,
                data["h_freq"],
                observed_symbol_indices,
            )
        )
    return metrics


def evaluate_oracle_ls(
    data: dict[str, Any],
    observation: dict[str, np.ndarray],
    l_eff: int,
    ridge: float = 0.0,
    observed_symbol_indices: tuple[int, ...] = (6, 7),
    ls_mode: str = "traditional_ls",
) -> dict[str, float]:
    params = nonlinear_oracle_params(data["channel_labels"], l_eff=l_eff)
    return evaluate_hybrid_params(
        params,
        data,
        observation,
        ridge=ridge,
        observed_symbol_indices=observed_symbol_indices,
        ls_mode=ls_mode,
    )


def evaluate_direct_h_tensor(
    prediction_ri: torch.Tensor,
    target_h: np.ndarray,
    observed_symbol_indices: tuple[int, ...] = (6, 7),
) -> dict[str, float]:
    pred_np = prediction_ri.detach().cpu().numpy()
    h_hat = pred_np[..., 0] + 1j * pred_np[..., 1]
    channel_nmse = nmse(h_hat, target_h)
    observed_nmse = nmse(
        h_hat[:, :, :, observed_symbol_indices, :],
        target_h[:, :, :, observed_symbol_indices, :],
    )
    return {
        "channel_nmse": channel_nmse,
        "observed_symbol_nmse": observed_nmse,
        "channel_nmse_db": nmse_db(h_hat, target_h),
        "observed_symbol_nmse_db": nmse_db(
            h_hat[:, :, :, observed_symbol_indices, :],
            target_h[:, :, :, observed_symbol_indices, :],
        ),
    }
