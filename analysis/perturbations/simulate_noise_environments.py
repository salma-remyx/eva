"""Synthesize coverage-optimal background-noise assets from PSD templates.

Builds the 27-candidate environment pool, selects coverage-optimal prototypes
in PSD space, matches each prototype to the background_noise slot whose
current asset it most resembles, and writes 16 kHz mono 16-bit PCM WAVs that
AudioPerturbator loads from assets/noise — the existing mix-at-SNR pipeline
consumes them with no code or config changes. Adapted from NOPE-HYPE
(arXiv:2609.10058).

Run from project root:
    uv run python analysis/perturbations/simulate_noise_environments.py            # preview in local/
    uv run python analysis/perturbations/simulate_noise_environments.py --install  # replace assets/noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eva.user_simulator.noise_environments import (
    assign_to_slots,
    candidate_pool,
    coverage_radius,
    psd_band_features,
    select_coverage_prototypes,
    synthesize,
    write_noise_wav,
)
from eva.user_simulator.perturbation import _ASSETS_DIR, _FILE_NOISE_TYPES, _load_noise_wav

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local" / "noise_environments"
DEFAULT_DURATION_S = 30.0


def main(argv: list[str] | None = None) -> None:
    """Build the candidate pool, select prototypes, match slots, and write assets."""
    parser = argparse.ArgumentParser(description="Synthesize coverage-optimal background-noise assets.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the environment simulator")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Seconds of noise per environment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write WAVs and manifest")
    parser.add_argument(
        "--install",
        action="store_true",
        help=f"Write into {_ASSETS_DIR} instead of --output-dir, replacing the checked-in noise assets",
    )
    args = parser.parse_args(argv)

    slot_names = sorted(name for name in _FILE_NOISE_TYPES if (_ASSETS_DIR / f"{name}.wav").exists())
    if not slot_names:
        raise FileNotFoundError(f"No existing noise assets found in {_ASSETS_DIR} to match prototypes against")

    rng = np.random.default_rng(args.seed)
    pool = candidate_pool()
    audio = [synthesize(spec, args.duration, rng) for spec in pool]
    features = [psd_band_features(samples) for samples in audio]

    selected = select_coverage_prototypes(features, k=len(slot_names))
    slot_features = {name: psd_band_features(_load_noise_wav(name)) for name in slot_names}
    assignment = assign_to_slots([features[i] for i in selected], slot_features)

    output_dir = _ASSETS_DIR if args.install else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "n_candidates": len(pool),
        "coverage_radius": coverage_radius(features, selected),
        "slots": {},
    }
    for slot, prototype_idx in assignment.items():
        idx = selected[prototype_idx]
        spec = pool[idx]
        write_noise_wav(output_dir / f"{slot}.wav", audio[idx])
        manifest["slots"][slot] = {
            "spec": spec.name,
            "tilt_db_per_octave": spec.tilt_db_per_octave,
            "band_gains_db": list(spec.band_gains_db),
            "modulation_rate_hz": spec.modulation_rate_hz,
            "modulation_depth": spec.modulation_depth,
            "psd_similarity_to_previous_asset": float(np.dot(slot_features[slot], features[idx])),
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    unassigned = [name for name in slot_names if name not in assignment]
    note = f" (slots without a prototype: {', '.join(unassigned)})" if unassigned else ""
    print(f"Wrote {len(assignment)} environment WAVs → {output_dir}{note}")
    print(f"Coverage radius of the selected set: {manifest['coverage_radius']:.3f}")
    print(f"Wrote manifest → {manifest_path}")
    if not args.install:
        print(f"Re-run with --install to replace {_ASSETS_DIR} with this set")


if __name__ == "__main__":
    main()
