"""Build last-two-symbol sparse observation tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thesis_transformer_v1.data.ser import corrupt_data_symbols


@dataclass(frozen=True)
class ObservationTokenConfig:
    symbol_indices: tuple[int, ...] = (6, 7)
    symbol_error_rate: float = 0.0
    include_reliability: bool = True


def normalized_coords(symbol_idx: np.ndarray, sc_idx: np.ndarray, n_sym: int, n_sc: int) -> np.ndarray:
    n_coord = 2.0 * symbol_idx.astype(np.float32) / float(max(n_sym - 1, 1)) - 1.0
    k_coord = 2.0 * sc_idx.astype(np.float32) / float(max(n_sc - 1, 1)) - 1.0
    return np.stack([k_coord, n_coord], axis=-1).astype(np.float32)


def build_observation_tokens(
    data: dict,
    config: ObservationTokenConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return batched tokens from selected OFDM symbols.

    Shapes:
        tokens: [B, N_obs, feature_dim]
        indices: [N_obs, 2], columns are [n, k]
        tx_grid_observed: [B, N_t, N_sym, N_sc]
    """
    rng = rng or np.random.default_rng(0)
    rx_grid = data["rx_grid"]
    tx_grid = data["tx_grid"]
    pilot_mask = data["pilot_mask"]
    data_mask = data["data_mask"]
    bsz, n_rx, n_sym, n_sc = rx_grid.shape
    _, n_tx, _, _ = tx_grid.shape

    observed_tx, changed_mask = corrupt_data_symbols(
        rng,
        tx_grid,
        data_mask,
        config.symbol_indices,
        config.symbol_error_rate,
    )
    symbol_idx = np.repeat(np.asarray(config.symbol_indices, dtype=np.int64), n_sc)
    sc_idx = np.tile(np.arange(n_sc, dtype=np.int64), len(config.symbol_indices))
    n_obs = symbol_idx.size

    y = rx_grid[:, :, symbol_idx, sc_idx].transpose(0, 2, 1)
    x = observed_tx[:, :, symbol_idx, sc_idx].transpose(0, 2, 1)
    coords = normalized_coords(symbol_idx, sc_idx, n_sym, n_sc)
    is_pilot = np.any(pilot_mask[:, symbol_idx, sc_idx], axis=0).astype(np.float32)
    reliability = 1.0 - np.any(changed_mask[:, :, symbol_idx, sc_idx], axis=1).astype(np.float32)

    features = [
        y.real.astype(np.float32),
        y.imag.astype(np.float32),
        x.real.astype(np.float32),
        x.imag.astype(np.float32),
        np.broadcast_to(coords[None, :, :], (bsz, n_obs, 2)),
        np.broadcast_to(is_pilot[None, :, None], (bsz, n_obs, 1)),
    ]
    if config.include_reliability:
        features.append(reliability[:, :, None].astype(np.float32))
    tokens = np.concatenate(features, axis=-1).astype(np.float32)
    return {
        "tokens": tokens,
        "indices": np.stack([symbol_idx, sc_idx], axis=-1),
        "tx_grid_observed": observed_tx,
        "changed_mask": changed_mask,
        "token_feature_dim": np.array(tokens.shape[-1], dtype=np.int64),
    }

