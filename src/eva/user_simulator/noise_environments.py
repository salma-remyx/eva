"""Background-noise environments synthesized from PSD templates.

Adapted from NOPE-HYPE (arXiv:2609.10058), which replaces hand-picked noise
recordings with a controllable environment simulator plus coverage-optimal
environment-set reduction over PSD templates. Here the simulator exposes a few
interpretable knobs (spectral tilt, band gains, envelope modulation), renders
noise whose average power spectrum follows the requested template, and emits
the 16 kHz mono 16-bit PCM WAVs that AudioPerturbator already loads from
assets/noise — so structured environments flow through the existing mix-at-SNR
pipeline without code or config changes.

The paper trains STT models on simulator output; this module instead supplies
evaluation noise assets for the perturbation analysis pipeline
(analysis/perturbations), whose bootstrap-CI statistical layer consumes the
resulting conditions without modification.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SAMPLE_RATE = 16000  # assets/noise contract enforced by perturbation._load_noise_wav
_DEFAULT_RMS = 3000.0  # int16-scale RMS, matching the reference speech level AudioPerturbator mixes against
_FREQ_FLOOR_HZ = 1.0  # keeps log2 finite for the DC bin
_TILT_REF_HZ = 200.0  # frequency the spectral tilt is anchored to
_PSD_MIN_HZ = 20.0
_PSD_MAX_HZ = 8000.0

# Piecewise-linear gain corners (Hz) for the low / mid / high shelves.
_GAIN_CORNERS_HZ = (20.0, 300.0, 600.0, 1600.0, 3200.0, 8000.0)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Interpretable simulator knobs defining one acoustic environment.

    The average power spectrum follows a piecewise-linear-in-log-frequency dB
    gain curve: low (<300 Hz), mid (300–2000 Hz) and high (>2000 Hz) shelf
    gains, plus an overall tilt in dB per octave. A slow sinusoidal envelope
    modulation adds the non-stationarity of fluctuating environments such as
    babble or street traffic.
    """

    name: str
    tilt_db_per_octave: float = 0.0
    band_gains_db: tuple[float, float, float] = (0.0, 0.0, 0.0)
    modulation_rate_hz: float = 0.0
    modulation_depth: float = 0.0


def candidate_pool() -> list[EnvironmentSpec]:
    """Build the 27-candidate grid over the simulator knobs.

    3 tilts x 3 band shapes x 3 modulation settings, mirroring the paper's
    structured 27-run sweep: small enough to inspect, broad enough to span
    stationary colored noise up to strongly modulated babble-like environments.
    """
    tilts = (-4.5, 0.0, 3.0)
    band_shapes = ((6.0, 0.0, -4.0), (0.0, 0.0, 0.0), (-4.0, 0.0, 6.0))
    modulations = ((0.0, 0.0), (0.8, 0.8), (4.0, 0.4))

    specs: list[EnvironmentSpec] = []
    for tilt in tilts:
        for bands in band_shapes:
            for rate, depth in modulations:
                shape = "low" if bands[0] > 0 else ("high" if bands[2] > 0 else "flat")
                mod = "stationary" if rate == 0.0 else f"mod{rate:g}hz"
                specs.append(
                    EnvironmentSpec(
                        name=f"tilt{tilt:+g}_{shape}_{mod}",
                        tilt_db_per_octave=tilt,
                        band_gains_db=bands,
                        modulation_rate_hz=rate,
                        modulation_depth=depth,
                    )
                )
    return specs


def _gain_amplitude(spec: EnvironmentSpec, freqs: np.ndarray) -> np.ndarray:
    """Amplitude gain implementing the spec's PSD template over rfft bins."""
    low, mid, high = spec.band_gains_db
    corner_db = np.asarray((low, low, mid, mid, high, high))
    log2f = np.log2(np.maximum(freqs, _FREQ_FLOOR_HZ))
    gain_db = np.interp(log2f, np.log2(np.asarray(_GAIN_CORNERS_HZ)), corner_db)
    gain_db = gain_db + spec.tilt_db_per_octave * (log2f - np.log2(_TILT_REF_HZ))
    amplitude = 10.0 ** (gain_db / 20.0)
    amplitude[0] = 0.0  # remove DC
    return amplitude


def synthesize(spec: EnvironmentSpec, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Render one noise realization of the spec at unit RMS.

    Gaussian noise is shaped in the frequency domain to the spec's PSD
    template, then multiplied by a slow sinusoidal envelope when modulation is
    requested. Deterministic for a given rng.

    Args:
        spec: Environment knobs.
        duration_s: Length of the rendered audio in seconds.
        rng: Random generator; seed it for reproducible assets.

    Returns:
        float32 samples at 16 kHz with unit RMS. Absolute level is irrelevant
        downstream because AudioPerturbator renormalizes the noise when mixing
        at a target SNR.
    """
    n = int(duration_s * _SAMPLE_RATE)
    freqs = np.fft.rfftfreq(n, d=1.0 / _SAMPLE_RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n)) * _gain_amplitude(spec, freqs)
    samples = np.fft.irfft(spectrum, n=n)

    if spec.modulation_depth > 0.0 and spec.modulation_rate_hz > 0.0:
        t = np.arange(n) / _SAMPLE_RATE
        phase = rng.uniform(0.0, 2.0 * np.pi)
        envelope = 1.0 - spec.modulation_depth * (0.5 + 0.5 * np.cos(2.0 * np.pi * spec.modulation_rate_hz * t + phase))
        samples = samples * envelope

    rms = np.sqrt(np.mean(samples**2))
    return (samples / (rms + 1e-12)).astype(np.float32)


def psd_band_features(samples: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """L2-normalized log band energies: the PSD-template descriptor.

    Periodogram power is pooled into ``n_bands`` log-spaced bands between
    20 Hz and 8 kHz; band log-energies are mean-centered and L2-normalized so
    the vector captures spectral shape only, independent of level and length.
    """
    power = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.maximum(np.fft.rfftfreq(len(samples), d=1.0 / _SAMPLE_RATE), _FREQ_FLOOR_HZ)
    edges = np.geomspace(_PSD_MIN_HZ, _PSD_MAX_HZ, n_bands + 1)

    band_energy = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (freqs >= lo) & (freqs < hi)
        band_energy.append(float(power[mask].mean()) if mask.any() else 0.0)

    log_energy = np.log10(np.asarray(band_energy) + 1e-12)
    centered = log_energy - log_energy.mean()
    return centered / (np.linalg.norm(centered) + 1e-12)


def select_coverage_prototypes(features: list[np.ndarray] | np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point selection of k prototypes (k-center, 2-approx).

    Coverage-optimal environment reduction in the paper's sense: each pick is
    the candidate farthest (Euclidean, in PSD feature space) from everything
    already selected, keeping the set's coverage radius — the worst-case
    distance from any candidate to its nearest prototype — small. The first
    prototype is the candidate farthest from the pool centroid, which makes
    the result deterministic without a seed.

    Args:
        features: One PSD feature vector per candidate.
        k: Number of prototypes to select (``1 <= k <= len(features)``).

    Returns:
        Indices of the selected candidates, in selection order.
    """
    feats = np.asarray(features, dtype=np.float64)
    if not 1 <= k <= len(feats):
        raise ValueError(f"k must be in [1, {len(feats)}], got {k}")

    first = int(np.argmax(np.linalg.norm(feats - feats.mean(axis=0), axis=1)))
    selected = [first]
    nearest = np.linalg.norm(feats - feats[first], axis=1)
    while len(selected) < k:
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(feats - feats[nxt], axis=1))
    return selected


def coverage_radius(features: list[np.ndarray] | np.ndarray, selected: list[int]) -> float:
    """Worst-case distance from any candidate to its nearest selected prototype."""
    feats = np.asarray(features, dtype=np.float64)
    distances = np.linalg.norm(feats[:, None, :] - feats[selected][None, :, :], axis=2)
    return float(distances.min(axis=1).max())


def assign_to_slots(prototypes: list[np.ndarray], slot_features: dict[str, np.ndarray]) -> dict[str, int]:
    """Greedily match noise-type slots to the most PSD-similar prototypes.

    All (slot, prototype) pairs are considered in order of descending cosine
    similarity (features are L2-normalized) and assigned while both sides are
    still free, so the globally strongest spectral matches win first. Ties
    break on slot name then prototype index, keeping the result deterministic.

    Returns:
        Mapping of slot name to index into ``prototypes``.
    """
    pairs = [
        (float(np.dot(slot_feat, proto_feat)), slot, idx)
        for idx, proto_feat in enumerate(prototypes)
        for slot, slot_feat in slot_features.items()
    ]
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    assignment: dict[str, int] = {}
    used: set[int] = set()
    for _, slot, idx in pairs:
        if slot not in assignment and idx not in used:
            assignment[slot] = idx
            used.add(idx)
    return assignment


def write_noise_wav(path: Path | str, samples: np.ndarray, rms: float = _DEFAULT_RMS) -> None:
    """Write samples as a 16 kHz mono 16-bit PCM WAV (the assets/noise contract).

    Args:
        path: Destination file, conventionally ``assets/noise/<noise_type>.wav``.
        samples: Float samples at any level; renormalized to ``rms`` here.
        rms: Target int16-scale RMS of the written file.
    """
    unit = samples / (np.sqrt(np.mean(samples**2)) + 1e-12)
    pcm = np.clip(unit * rms, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
