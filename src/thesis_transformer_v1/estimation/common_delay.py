"""Common-delay dictionary, LS recovery, and full-grid reconstruction."""

from __future__ import annotations

import numpy as np


def nonlinear_phase(
    rel_delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    delay_phase = np.exp(-1j * 2.0 * np.pi * rel_delay_s[:, :, None] * freq_hz[None, None, :])
    doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz[:, :, None] * time_s[None, None, :])
    return (doppler_phase[:, :, :, None] * delay_phase[:, :, None, :]).astype(np.complex64)


def global_phase(
    total_delay_s: np.ndarray,
    cfo_hz: np.ndarray,
    rx_time_offsets_s: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    delay = total_delay_s[:, None] + rx_time_offsets_s
    freq_phase = np.exp(-1j * 2.0 * np.pi * delay[:, :, None] * freq_hz[None, None, :])
    cfo_phase = np.exp(1j * 2.0 * np.pi * cfo_hz[:, None] * time_s[None, :])
    return (freq_phase[:, :, None, :] * cfo_phase[:, None, :, None]).astype(np.complex64)


def reconstruct_channel(
    path_gains: np.ndarray,
    params: dict[str, np.ndarray],
    freq_hz: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Reconstruct H with shape [B, N_r, N_t, N_sym, N_sc]."""
    path_phase = nonlinear_phase(params["rel_delay_s"], params["doppler_hz"], freq_hz, time_s)
    shared = global_phase(
        params["total_delay_s"],
        params["cfo_hz"],
        params["rx_time_offsets_s"],
        freq_hz,
        time_s,
    )
    h = np.einsum("blrt,blnk->brtnk", path_gains, path_phase, optimize=True)
    return (h * shared[:, :, None, :, :]).astype(np.complex64)


def _solve_complex_ls(a: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    if ridge <= 0.0:
        return np.linalg.lstsq(a, y, rcond=None)[0]
    lhs = a.conj().T @ a
    lhs = lhs + ridge * np.eye(lhs.shape[0], dtype=lhs.dtype)
    rhs = a.conj().T @ y
    return np.linalg.solve(lhs, rhs)


def recover_path_gains_ls(
    rx_grid: np.ndarray,
    tx_grid_observed: np.ndarray,
    observation_indices: np.ndarray,
    params: dict[str, np.ndarray],
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    ridge: float = 0.0,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recover path gains with complex LS.

    For each sample and RX antenna, solve all `(path, tx)` gains jointly because
    the received RE is the superposition of all TX streams.
    """
    bsz, n_rx, _, _ = rx_grid.shape
    _, n_tx, _, _ = tx_grid_observed.shape
    l_eff = params["rel_delay_s"].shape[1]
    gains = np.zeros((bsz, l_eff, n_rx, n_tx), dtype=np.complex64)
    cond_a = np.zeros((bsz, n_rx), dtype=np.float64)
    cond_gram = np.zeros((bsz, n_rx), dtype=np.float64)
    ranks = np.zeros((bsz, n_rx), dtype=np.int64)
    n_idx = observation_indices[:, 0]
    k_idx = observation_indices[:, 1]
    path_phase = nonlinear_phase(params["rel_delay_s"], params["doppler_hz"], freq_hz, time_s)
    shared = global_phase(
        params["total_delay_s"],
        params["cfo_hz"],
        params["rx_time_offsets_s"],
        freq_hz,
        time_s,
    )

    for b in range(bsz):
        x_obs = tx_grid_observed[b][:, n_idx, k_idx].T
        p_obs = path_phase[b][:, n_idx, k_idx].T
        for r in range(n_rx):
            shared_obs = shared[b, r, n_idx, k_idx]
            columns = []
            for path_idx in range(l_eff):
                for t in range(n_tx):
                    columns.append(shared_obs * p_obs[:, path_idx] * x_obs[:, t])
            a = np.stack(columns, axis=1)
            y = rx_grid[b, r, n_idx, k_idx]
            if return_diagnostics:
                cond_a[b, r] = np.linalg.cond(a)
                cond_gram[b, r] = np.linalg.cond(a.conj().T @ a)
                ranks[b, r] = np.linalg.matrix_rank(a)
            solution = _solve_complex_ls(a, y, ridge=ridge)
            gains[b, :, r, :] = solution.reshape(l_eff, n_tx)
    if not return_diagnostics:
        return gains
    diagnostics = {
        "cond_a": cond_a,
        "cond_gram": cond_gram,
        "rank": ranks,
        "num_columns": np.array(l_eff * n_tx, dtype=np.int64),
    }
    return gains, diagnostics
