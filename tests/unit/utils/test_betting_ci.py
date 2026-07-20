"""Unit tests for src/eva/utils/betting_ci.py and its wiring into the CI helpers."""

from __future__ import annotations

import math

import numpy as np

# Import from the existing (non-new) helper module to exercise the integration wiring.
from eva.metrics.runner import MetricsRunner
from eva.models.results import MetricScore, RecordMetrics
from eva.utils.betting_ci import betting_ci
from eva.utils.bootstrap import mean_ci_fields, run_seed


def _hoeffding_width(values: np.ndarray, alpha: float = 0.05) -> float:
    n = len(values)
    return 2.0 * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


class TestBettingCI:
    def test_empty_returns_nones(self):
        assert betting_ci([]) == (None, None)

    def test_brackets_sample_mean(self):
        rng = np.random.default_rng(0)
        values = rng.uniform(0.3, 0.7, size=40)
        lower, upper = betting_ci(values)
        assert lower <= values.mean() <= upper

    def test_stays_within_bounds(self):
        rng = np.random.default_rng(1)
        values = rng.uniform(0.0, 1.0, size=25)
        lower, upper = betting_ci(values)
        assert 0.0 <= lower <= upper <= 1.0

    def test_tighter_than_hoeffding(self):
        # The paper's core claim: a coverage-valid interval much tighter than the
        # classical Hoeffding bound for low-variance bounded samples.
        rng = np.random.default_rng(2)
        values = rng.uniform(0.3, 0.7, size=30)
        lower, upper = betting_ci(values)
        assert (upper - lower) < _hoeffding_width(values)

    def test_smaller_alpha_widens(self):
        rng = np.random.default_rng(3)
        values = rng.uniform(0.0, 1.0, size=30)
        lo95, hi95 = betting_ci(values, alpha=0.05)
        lo90, hi90 = betting_ci(values, alpha=0.10)
        assert (hi95 - lo95) > (hi90 - lo90)

    def test_order_invariant(self):
        # Derandomization must make a sorted sample and a shuffled copy agree.
        values = np.array([1.0] * 7 + [0.0] * 13)
        sorted_ci = betting_ci(values)
        shuffled_ci = betting_ci(np.random.default_rng(9).permutation(values))
        assert abs(sorted_ci[0] - shuffled_ci[0]) < 0.02
        assert abs(sorted_ci[1] - shuffled_ci[1]) < 0.02

    def test_deterministic_for_fixed_seed(self):
        rng = np.random.default_rng(4)
        values = rng.uniform(0.0, 1.0, size=20)
        assert betting_ci(values, seed=7) == betting_ci(values, seed=7)

    def test_handles_values_above_unit_range(self):
        # scenario means can exceed the nominal [0, 1] range; bounds auto-widen.
        values = np.array([float(i) / 10 for i in range(20)])  # up to 1.9
        lower, upper = betting_ci(values)
        assert lower <= values.mean() <= upper

    def test_achieves_nominal_coverage(self):
        # Monte-Carlo coverage of the mean should meet the 1 - alpha target, unlike
        # the percentile bootstrap which under-covers at small n.
        rng = np.random.default_rng(2024)
        truth, n, trials, covered = 0.3, 30, 200, 0
        for t in range(trials):
            sample = (rng.random(n) < truth).astype(float)
            lower, upper = betting_ci(sample, seed=t)
            covered += lower <= truth <= upper
        assert covered / trials >= 0.95


class TestBettingCIWiring:
    """Betting bounds must flow through the existing production CI helpers."""

    def test_mean_ci_fields_emits_betting_bounds(self):
        seed = run_seed("wiring-check")
        values = [float(i) / 20.0 for i in range(20)]
        fields = mean_ci_fields(values, seed=seed)
        # Existing bootstrap fields remain untouched.
        assert "mean_ci_lower" in fields and "mean_ci_upper" in fields
        # New betting fields are populated and bracket the sample mean.
        assert fields["mean_betting_ci_lower"] is not None
        assert fields["mean_betting_ci_upper"] is not None
        mean = float(np.mean(values))
        assert fields["mean_betting_ci_lower"] <= mean <= fields["mean_betting_ci_upper"]

    def test_mean_ci_fields_empty_emits_null_betting_bounds(self):
        fields = mean_ci_fields([], seed=1)
        assert fields["mean_betting_ci_lower"] is None
        assert fields["mean_betting_ci_upper"] is None
        assert fields["mean_ci_n_scenarios"] == 0

    def test_betting_bounds_reach_per_metric_aggregates(self):
        # End-to-end: the betting interval surfaces in real per-metric aggregate output.
        records: dict[str, RecordMetrics] = {}
        for i in range(20):
            value = float(i) / 20.0
            score = MetricScore(name="task_completion", score=value, normalized_score=value)
            records[f"1.1.{i}"] = RecordMetrics(record_id=f"1.1.{i}", metrics={"task_completion": score})

        agg = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=run_seed("run-x")
        )
        entry = agg["task_completion"]
        assert entry["mean_betting_ci_lower"] <= entry["mean"] <= entry["mean_betting_ci_upper"]
