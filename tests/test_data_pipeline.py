import numpy as np

from thesis_transformer_v1.data.config import ExperimentConfig, ThesisImpairmentConfig
from thesis_transformer_v1.data.generator import generate_thesis_dataset, timing_cfo_phase
from thesis_transformer_v1.data.tokenizer import ObservationTokenConfig, build_observation_tokens
from thesis_transformer_v1.tdlc.config import ChannelConfig, DatasetConfig, OFDMConfig, PilotConfig


def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            batch_size=2,
            seed=7,
            ofdm=OFDMConfig(n_tx=4, n_rx=4, n_sc=48, n_sym=8, modulation="16QAM"),
            pilot=PilotConfig(mode="sparse2d", symbol_spacing=2, subcarrier_spacing=4),
            channel=ChannelConfig(
                model="tdl_c",
                mimo_mode="common_delay",
                n_paths=12,
                delay_spread_s=3.0e-7,
                max_doppler_hz=0.0,
            ),
        ),
        thesis_impairment=ThesisImpairmentConfig(
            tau0_range_s=(10e-9, 20e-9),
            cfo_range_hz=(-30.0, 30.0),
            rx_time_offset_range_s=(-5e-9, 5e-9),
            awgn_snr_db=None,
        ),
        l_eff=12,
    )


def test_rx_time_offsets_shape_and_reference() -> None:
    data = generate_thesis_dataset(small_config())
    offsets = data["channel_labels"]["rx_time_offsets_s"]
    assert offsets.shape == (2, 4)
    assert np.allclose(offsets[:, 0], 0.0)


def test_timing_phase_ramp_matches_expected_subcarrier_ratio() -> None:
    freq_hz = np.array([0.0, 15_000.0], dtype=np.float64)
    time_s = np.array([0.0], dtype=np.float64)
    total_delay = np.array([100e-9], dtype=np.float32)
    cfo = np.array([0.0], dtype=np.float32)
    rx_offsets = np.array([[0.0, 20e-9]], dtype=np.float32)
    phase = timing_cfo_phase(freq_hz, time_s, total_delay, cfo, rx_offsets)
    ratio = phase[0, 1, 0, 1] / phase[0, 1, 0, 0]
    expected = np.exp(-1j * 2.0 * np.pi * (120e-9) * 15_000.0)
    assert np.allclose(ratio, expected, atol=1e-7)


def test_tokenization_uses_only_last_two_symbols() -> None:
    data = generate_thesis_dataset(small_config())
    obs = build_observation_tokens(data, ObservationTokenConfig(symbol_indices=(6, 7)))
    assert set(obs["indices"][:, 0].tolist()) == {6, 7}
    assert obs["tokens"].shape[1] == 2 * 48


def test_ser_corruption_does_not_change_pilots() -> None:
    data = generate_thesis_dataset(small_config())
    obs = build_observation_tokens(
        data,
        ObservationTokenConfig(symbol_indices=(6, 7), symbol_error_rate=1.0),
        rng=np.random.default_rng(123),
    )
    pilot_mask = data["pilot_mask"]
    changed = obs["changed_mask"]
    assert not np.any(changed & pilot_mask[None, :, :, :])
    selected = np.zeros(pilot_mask.shape[1:], dtype=bool)
    selected[[6, 7], :] = True
    assert np.any(changed & data["data_mask"][None, :, :, :] & selected[None, None, :, :])
