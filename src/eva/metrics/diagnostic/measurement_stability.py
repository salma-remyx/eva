"""Measurement-stability statistics for LLM-as-judge auditing.

A judge sampled at non-zero temperature returns a *random* rating, so a single
evaluator-replacement shift ``|normA - normB|`` can reflect within-judge
sampling noise rather than a genuine disagreement between the two judges. These
parameter-free helpers separate the two signals from repeated samples of one
conversation's judges:

* ``measurement_variance`` -- the per-judge measurement variance
  ``E[(Y - mu)^2]`` over its normalized ratings.
* ``bootstrap_shift_ci`` / ``assess_stability`` -- a bootstrap confidence
  interval for the replacement shift, so a reported shift can be shown to
  exceed generation noise (CI excludes zero) instead of assumed significant.

Adapted from "When the Judge Changes, So Does the Measurement: Auditing
LLM-as-Judge Reliability" (arXiv:2607.08535), which argues measurement shifts
should carry error/variance estimates rather than be reported as point values.
The paper's full multi-model, multi-dataset bootstrap sweep is out of scope --
these operate on the repeated samples of a single conversation's two judges.
"""

import random
from collections.abc import Sequence
from typing import Any


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def measurement_variance(samples: Sequence[float]) -> float:
    """Population variance ``E[(Y - mu)^2]`` of a judge's normalized ratings.

    Uses the population (divide-by-n) form because we want the spread of the
    judge's own output distribution, not an estimate of a larger population's
    variance. Returns 0.0 for zero or one sample (no measurable spread).
    """
    n = len(samples)
    if n < 2:
        return 0.0
    mu = _mean(samples)
    return sum((y - mu) ** 2 for y in samples) / n


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated ``q`` quantile (``q`` in [0, 1]) of an ascending list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_shift_ci(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and ``(low, high)`` CI bounds for ``|mean(A) - mean(B)|``.

    Resamples each judge's ratings with replacement ``iterations`` times and
    reads the requested confidence interval off the bootstrap distribution of
    the shift. A fixed ``seed`` keeps the interval reproducible across runs.
    """
    point = abs(_mean(samples_a) - _mean(samples_b))
    if not samples_a or not samples_b:
        return round(point, 3), round(point, 3), round(point, 3)

    rng = random.Random(seed)
    n_a, n_b = len(samples_a), len(samples_b)
    shifts: list[float] = []
    for _ in range(iterations):
        ra = [samples_a[rng.randrange(n_a)] for _ in range(n_a)]
        rb = [samples_b[rng.randrange(n_b)] for _ in range(n_b)]
        shifts.append(abs(_mean(ra) - _mean(rb)))
    shifts.sort()

    tail = (1.0 - confidence) / 2.0
    low = _percentile(shifts, tail)
    high = _percentile(shifts, 1.0 - tail)
    return round(point, 3), round(low, 3), round(high, 3)


def assess_stability(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bundle per-judge variance and the bootstrap CI for the replacement shift.

    ``shift_exceeds_noise`` is True when the CI lower bound is above zero -- the
    audit-trail signal that the judge swap moved the score by more than either
    judge's own sampling noise.
    """
    point, low, high = bootstrap_shift_ci(samples_a, samples_b, iterations=iterations, confidence=confidence, seed=seed)
    return {
        "samples_per_judge": {"a": len(samples_a), "b": len(samples_b)},
        "judge_a_variance": round(measurement_variance(samples_a), 4),
        "judge_b_variance": round(measurement_variance(samples_b), 4),
        "shift_point": point,
        "shift_ci_low": low,
        "shift_ci_high": high,
        "shift_exceeds_noise": low > 0.0,
        "bootstrap_iterations": iterations,
        "confidence": confidence,
    }
