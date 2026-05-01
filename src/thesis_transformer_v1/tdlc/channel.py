"""Parameterized TDL-C MIMO-OFDM channel generation."""

from __future__ import annotations

import numpy as np

from .config import ChannelConfig, DatasetConfig
from .constants import TDL_C_NORMALIZED_DELAYS, TDL_C_POWERS_DB


def select_profile_paths(
    rng: np.random.Generator, cfg: ChannelConfig
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.model == "tdl_c":
        delays = TDL_C_NORMALIZED_DELAYS * cfg.delay_spread_s
        powers_db = TDL_C_POWERS_DB.copy()
    elif cfg.model == "custom":
        delays = np.linspace(0.0, 3.0 * cfg.delay_spread_s, max(1, cfg.n_paths))
        powers_db = -np.arange(max(1, cfg.n_paths), dtype=np.float64) * 3.0
    else:
        raise ValueError(f"Unsupported channel model: {cfg.model}")

    n_paths = min(max(1, int(cfg.n_paths)), len(delays))
    if cfg.randomize_path_subset:
        probs = 10.0 ** (powers_db / 10.0)
        probs = probs / probs.sum()
        indices = rng.choice(len(delays), size=n_paths, replace=False, p=probs)
        indices = np.sort(indices)
    else:
        indices = np.argsort(powers_db)[::-1][:n_paths]
        indices = indices[np.argsort(delays[indices])]

    selected_delays = delays[indices]
    selected_powers_db = powers_db[indices]
    if cfg.dominant_path_decay_db is not None:
        rank = np.argsort(np.argsort(-selected_powers_db))
        selected_powers_db = selected_powers_db - rank * float(cfg.dominant_path_decay_db)
    return selected_delays.astype(np.float64), selected_powers_db.astype(np.float64)


def ula_response(
    n_ant: int,
    angle_rad: np.ndarray,
    spacing_wavelength: float,
    position_offsets_wavelength: np.ndarray | None = None,
) -> np.ndarray:
    positions = spacing_wavelength * np.arange(n_ant, dtype=np.float64)
    if position_offsets_wavelength is not None:
        offsets = np.asarray(position_offsets_wavelength, dtype=np.float64)
        if offsets.shape != (n_ant,):
            raise ValueError(f"position_offsets_wavelength must have shape {(n_ant,)}, got {offsets.shape}")
        positions = positions + offsets
    phase = 2.0 * np.pi * positions[:, None] * np.sin(angle_rad[None, :])
    return np.exp(1j * phase) / np.sqrt(n_ant)


def generate_rx_offsets(rng: np.random.Generator, cfg: DatasetConfig) -> np.ndarray:
    low, high = cfg.channel.rx_offset_range_wavelength
    offsets = np.zeros((cfg.batch_size, cfg.ofdm.n_rx), dtype=np.float64)
    if low == 0.0 and high == 0.0:
        return offsets
    if cfg.ofdm.n_rx > 1:
        offsets[:, 1:] = rng.uniform(low, high, size=(cfg.batch_size, cfg.ofdm.n_rx - 1))
    return offsets


def channel_axes(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    ofdm = cfg.ofdm
    subcarrier_index = np.arange(ofdm.n_sc, dtype=np.float64) - (ofdm.n_sc // 2)
    freq_hz = subcarrier_index * ofdm.subcarrier_spacing_hz
    time_s = np.arange(ofdm.n_sym, dtype=np.float64) * (
        (ofdm.n_sc + ofdm.cp_len) / (ofdm.n_sc * ofdm.subcarrier_spacing_hz)
    )
    return freq_hz, time_s


def common_delay_channel(
    rng: np.random.Generator,
    cfg: DatasetConfig,
    powers_lin: np.ndarray,
    doppler_phase: np.ndarray,
    delay_phase: np.ndarray,
    rx_offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    ofdm = cfg.ofdm
    bsz, n_paths = cfg.batch_size, len(powers_lin)
    pair_gains = (
        np.sqrt(powers_lin[None, :, None, None] / 2.0)
        * (
            rng.standard_normal((bsz, n_paths, ofdm.n_rx, ofdm.n_tx))
            + 1j * rng.standard_normal((bsz, n_paths, ofdm.n_rx, ofdm.n_tx))
        )
    )
    h_freq = np.einsum(
        "blrt,bln,lk->brtnk",
        pair_gains,
        doppler_phase,
        delay_phase,
        optimize=True,
    )
    angles = {
        "aoa_rad": np.full((bsz, n_paths), np.nan, dtype=np.float64),
        "aod_rad": np.full((bsz, n_paths), np.nan, dtype=np.float64),
    }
    return h_freq, angles, pair_gains


def array_response_channel(
    rng: np.random.Generator,
    cfg: DatasetConfig,
    powers_lin: np.ndarray,
    doppler_phase: np.ndarray,
    delay_phase: np.ndarray,
    rx_offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    ofdm, ch = cfg.ofdm, cfg.channel
    bsz, n_paths = cfg.batch_size, len(powers_lin)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(bsz, n_paths))
    scalar_gains = (
        np.sqrt(powers_lin[None, :] / 2.0)
        * (rng.standard_normal((bsz, n_paths)) + 1j * rng.standard_normal((bsz, n_paths)))
        * np.exp(1j * phases)
    )
    low, high = np.deg2rad(ch.angle_range_deg)
    aoa_rad = rng.uniform(low, high, size=(bsz, n_paths))
    aod_rad = rng.uniform(low, high, size=(bsz, n_paths))
    h_freq = np.zeros((bsz, ofdm.n_rx, ofdm.n_tx, ofdm.n_sym, ofdm.n_sc), dtype=np.complex128)

    for b in range(bsz):
        ar = ula_response(
            ofdm.n_rx,
            aoa_rad[b],
            ch.antenna_spacing_wavelength,
            rx_offsets[b],
        )
        at = ula_response(ofdm.n_tx, aod_rad[b], ch.antenna_spacing_wavelength)
        spatial = ar[:, None, :] * np.conj(at[None, :, :])
        h_freq[b] = np.einsum(
            "l,rtl,ln,lk->rtnk",
            scalar_gains[b],
            spatial,
            doppler_phase[b],
            delay_phase,
            optimize=True,
        )

    angles = {"aoa_rad": aoa_rad, "aod_rad": aod_rad}
    return h_freq, angles, scalar_gains


def normalize_channel_and_gains(
    h_freq: np.ndarray, path_gains: np.ndarray, mimo_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    mean_power = np.mean(np.abs(h_freq) ** 2, axis=(1, 2, 3, 4), keepdims=True)
    norm_scale = np.sqrt(mean_power + 1e-12)
    h_freq = h_freq / norm_scale
    if mimo_mode == "common_delay":
        path_gains = path_gains / norm_scale[:, :, :, :, 0]
    else:
        path_gains = path_gains / norm_scale[:, 0, 0, 0, 0][:, None]
    return h_freq, path_gains


def path_contributions(
    cfg: DatasetConfig,
    path_gains: np.ndarray,
    aoa_rad: np.ndarray,
    aod_rad: np.ndarray,
    rx_offsets: np.ndarray,
    doppler_phase: np.ndarray,
    delay_phase: np.ndarray,
) -> np.ndarray:
    ofdm, ch = cfg.ofdm, cfg.channel
    bsz, n_paths = cfg.batch_size, delay_phase.shape[0]
    contrib = np.zeros(
        (bsz, n_paths, ofdm.n_rx, ofdm.n_tx, ofdm.n_sym, ofdm.n_sc), dtype=np.complex64
    )
    if ch.mimo_mode == "common_delay":
        for l in range(n_paths):
            contrib[:, l] = (
                path_gains[:, l, :, :, None, None]
                * doppler_phase[:, l, None, None, :, None]
                * delay_phase[l, None, None, None, :]
            )
        return contrib

    for b in range(bsz):
        ar = ula_response(
            ofdm.n_rx,
            aoa_rad[b],
            ch.antenna_spacing_wavelength,
            rx_offsets[b],
        )
        at = ula_response(ofdm.n_tx, aod_rad[b], ch.antenna_spacing_wavelength)
        spatial = ar[:, None, :] * np.conj(at[None, :, :])
        for l in range(n_paths):
            contrib[b, l] = (
                path_gains[b, l]
                * spatial[:, :, l, None, None]
                * doppler_phase[b, l, None, None, :, None]
                * delay_phase[l, None, None, None, :]
            )
    return contrib


def generate_channel(
    rng: np.random.Generator, cfg: DatasetConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate H_freq [B, N_r, N_t, N_sym, N_sc] and physical labels."""
    ch = cfg.channel
    delays_s, powers_db = select_profile_paths(rng, ch)
    n_paths = len(delays_s)

    powers_lin = 10.0 ** (powers_db / 10.0)
    if ch.normalize_power:
        powers_lin = powers_lin / np.sum(powers_lin)

    doppler_hz = rng.uniform(-ch.max_doppler_hz, ch.max_doppler_hz, size=(cfg.batch_size, n_paths))
    if ch.max_doppler_hz == 0.0:
        doppler_hz.fill(0.0)

    freq_hz, time_s = channel_axes(cfg)
    delay_phase = np.exp(-1j * 2.0 * np.pi * delays_s[:, None] * freq_hz[None, :])
    doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz[:, :, None] * time_s[None, None, :])
    rx_offsets = generate_rx_offsets(rng, cfg)

    if ch.mimo_mode == "common_delay":
        h_freq, angles, path_gains = common_delay_channel(
            rng, cfg, powers_lin, doppler_phase, delay_phase, rx_offsets
        )
    elif ch.mimo_mode == "array_response":
        h_freq, angles, path_gains = array_response_channel(
            rng, cfg, powers_lin, doppler_phase, delay_phase, rx_offsets
        )
    else:
        raise ValueError(f"Unsupported MIMO mode: {ch.mimo_mode}")

    if ch.normalize_power:
        h_freq, path_gains = normalize_channel_and_gains(h_freq, path_gains, ch.mimo_mode)

    contrib = path_contributions(
        cfg,
        path_gains,
        angles["aoa_rad"],
        angles["aod_rad"],
        rx_offsets,
        doppler_phase,
        delay_phase,
    )
    labels = {
        "path_delays_s": np.broadcast_to(delays_s[None, :], (cfg.batch_size, n_paths)).copy(),
        "path_powers_db": np.broadcast_to(powers_db[None, :], (cfg.batch_size, n_paths)).copy(),
        "path_gains": np.asarray(path_gains).astype(np.complex64),
        "path_doppler_hz": doppler_hz.astype(np.float32),
        "path_aoa_rad": angles["aoa_rad"].astype(np.float32),
        "path_aod_rad": angles["aod_rad"].astype(np.float32),
        "rx_offsets": rx_offsets.astype(np.float32),
        "path_contrib_h_freq": contrib,
    }
    return h_freq.astype(np.complex64), labels


def apply_channel_frequency(tx_grid: np.ndarray, h_freq: np.ndarray) -> np.ndarray:
    """Apply frequency-domain MIMO channel."""
    return np.einsum("brtnk,btnk->brnk", h_freq, tx_grid, optimize=True).astype(np.complex64)
