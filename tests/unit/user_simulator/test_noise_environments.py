"""Tests for PSD-template noise environments and their consumption by AudioPerturbator.

The integration tests write synthesized environments through the same
assets/noise contract (16 kHz mono 16-bit PCM WAV) that AudioPerturbator
loads, then verify the existing mix-at-SNR pipeline consumes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from eva.models.config import PerturbationConfig
from eva.user_simulator.noise_environments import (
    EnvironmentSpec,
    assign_to_slots,
    candidate_pool,
    coverage_radius,
    psd_band_features,
    select_coverage_prototypes,
    synthesize,
    write_noise_wav,
)
from eva.user_simulator.perturbation import _ASSETS_DIR, _FILE_NOISE_TYPES, _load_noise_wav, AudioPerturbator

_LOW_SPEC = EnvironmentSpec(name="low", band_gains_db=(12.0, 0.0, -12.0))
_HIGH_SPEC = EnvironmentSpec(name="high", band_gains_db=(-12.0, 0.0, 12.0))
_FLAT_SPEC = EnvironmentSpec(name="flat")
_MOD_SPEC = EnvironmentSpec(name="mod", modulation_rate_hz=0.5, modulation_depth=0.9)

_REAL_ASSETS_PRESENT = (_ASSETS_DIR / "coffee_shop.wav").exists()


def _make_pcm(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    n = int(duration_s * sample_rate)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate) * 16000).astype(np.int16)
    return samples.tobytes()


def _envelope_spread(samples: np.ndarray, frame_samples: int = 800) -> float:
    """Relative std of short-time RMS — how much the level fluctuates over time."""
    frames = samples[: len(samples) // frame_samples * frame_samples].reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return float(np.std(rms) / np.mean(rms))


class TestSynthesize:
    def test_deterministic_for_seeded_rng(self):
        first = synthesize(_FLAT_SPEC, 1.0, np.random.default_rng(7))
        second = synthesize(_FLAT_SPEC, 1.0, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_unit_rms(self):
        samples = synthesize(_LOW_SPEC, 1.0, np.random.default_rng(7))
        assert abs(float(np.sqrt(np.mean(samples**2))) - 1.0) < 1e-3

    def test_band_gains_shape_the_spectrum(self):
        low = psd_band_features(synthesize(_LOW_SPEC, 4.0, np.random.default_rng(11)))
        high = psd_band_features(synthesize(_HIGH_SPEC, 4.0, np.random.default_rng(11)))
        # Features are mean-centered log band energies: low half vs high half of the bands.
        assert low[:4].mean() > low[4:].mean()
        assert high[4:].mean() > high[:4].mean()

    def test_modulation_adds_envelope_dynamics(self):
        modulated = synthesize(_MOD_SPEC, 8.0, np.random.default_rng(3))
        stationary = synthesize(_FLAT_SPEC, 8.0, np.random.default_rng(3))
        assert _envelope_spread(modulated) > 2.0 * _envelope_spread(stationary)


class TestCoverageSelection:
    def test_pool_has_27_unique_specs(self):
        pool = candidate_pool()
        assert len(pool) == 27
        assert len({spec.name for spec in pool}) == 27

    def test_selects_k_distinct_prototypes_deterministically(self):
        rng = np.random.default_rng(5)
        features = [psd_band_features(synthesize(spec, 2.0, rng)) for spec in candidate_pool()]
        first = select_coverage_prototypes(features, 7)
        assert first == select_coverage_prototypes(features, 7)
        assert len(set(first)) == 7

    def test_adding_prototypes_shrinks_covered_radius(self):
        rng = np.random.default_rng(5)
        features = [psd_band_features(synthesize(spec, 2.0, rng)) for spec in candidate_pool()]
        selected = select_coverage_prototypes(features, 7)
        assert coverage_radius(features, selected) < coverage_radius(features, selected[:1])

    def test_rejects_invalid_k(self):
        features = [psd_band_features(synthesize(spec, 1.0, np.random.default_rng(2))) for spec in candidate_pool()[:3]]
        with pytest.raises(ValueError):
            select_coverage_prototypes(features, 0)
        with pytest.raises(ValueError):
            select_coverage_prototypes(features, 4)


class TestAssignToSlots:
    def test_assigns_each_slot_its_most_similar_prototype(self):
        rng = np.random.default_rng(9)
        prototypes = [psd_band_features(synthesize(spec, 2.0, rng)) for spec in candidate_pool()[:7]]
        slots = {"coffee_shop": prototypes[2], "road_noise": prototypes[5], "baby_crying": prototypes[0]}
        assignment = assign_to_slots(prototypes, slots)
        assert set(assignment) == set(slots)
        assert len(set(assignment.values())) == 3
        assert assignment["coffee_shop"] == 2
        assert assignment["road_noise"] == 5
        assert assignment["baby_crying"] == 0


class TestAudioPerturbatorIntegration:
    """Generated environments must satisfy the assets/noise consumption contract."""

    @pytest.fixture()
    def synthesized_assets(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(21)
        pool = candidate_pool()
        audio = [synthesize(spec, 2.0, rng) for spec in pool]
        features = [psd_band_features(samples) for samples in audio]
        selected = select_coverage_prototypes(features, 2)
        for noise_type, idx in zip(("coffee_shop", "road_noise"), selected, strict=True):
            write_noise_wav(tmp_path / f"{noise_type}.wav", audio[idx])
        monkeypatch.setattr("eva.user_simulator.perturbation._ASSETS_DIR", tmp_path)
        return tmp_path

    def test_perturbator_loads_and_mixes_synthesized_wav(self, synthesized_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        perturbator = AudioPerturbator(config)  # raises if the WAV violates the format contract
        pcm = _make_pcm()
        result = perturbator.apply(pcm)
        assert len(result) == len(pcm)
        assert result != pcm

    def test_synthesized_noise_mixes_at_target_snr(self, synthesized_assets):
        snr_db = 10.0
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=snr_db)
        perturbator = AudioPerturbator(config)
        pcm = _make_pcm()
        speech = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        mixed = np.frombuffer(perturbator.apply(pcm), dtype=np.int16).astype(np.float64)
        noise = mixed - speech
        measured_snr = 20.0 * np.log10(np.sqrt(np.mean(speech**2)) / np.sqrt(np.mean(noise**2)))
        assert abs(measured_snr - snr_db) < 1.0

    def test_ambient_chunk_from_synthesized_asset(self, synthesized_assets):
        config = PerturbationConfig(background_noise="road_noise", snr_db=10.0)
        perturbator = AudioPerturbator(config)
        chunk = perturbator.get_ambient_chunk(640)
        assert len(chunk) == 640
        assert chunk != b"\x00" * 640


class TestEndToEndSlotMatching:
    @pytest.mark.skipif(not _REAL_ASSETS_PRESENT, reason="noise assets not checked out")
    def test_selected_prototype_set_flows_through_perturbator(self, tmp_path, monkeypatch):
        """Full path: real assets → prototype selection and slot matching → consumable WAVs."""
        rng = np.random.default_rng(42)
        pool = candidate_pool()
        audio = [synthesize(spec, 2.0, rng) for spec in pool]
        features = [psd_band_features(samples) for samples in audio]

        slot_names = sorted(name for name in _FILE_NOISE_TYPES if (_ASSETS_DIR / f"{name}.wav").exists())
        slot_features = {name: psd_band_features(_load_noise_wav(name)) for name in slot_names}
        selected = select_coverage_prototypes(features, k=len(slot_names))
        assignment = assign_to_slots([features[i] for i in selected], slot_features)
        assert set(assignment) == set(slot_names)

        for slot, prototype_idx in assignment.items():
            write_noise_wav(tmp_path / f"{slot}.wav", audio[selected[prototype_idx]])
        monkeypatch.setattr("eva.user_simulator.perturbation._ASSETS_DIR", tmp_path)

        pcm = _make_pcm()
        for slot in slot_names:
            perturbator = AudioPerturbator(PerturbationConfig(background_noise=slot, snr_db=12.0))
            result = perturbator.apply(pcm)
            assert len(result) == len(pcm)
            assert result != pcm
