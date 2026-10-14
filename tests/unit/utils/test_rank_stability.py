"""Unit tests for src/eva/utils/rank_stability.py and its run-level aggregation tie-in."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.rank_stability.run_rank_stability import (
    build_stability_rows,
    load_run_metrics,
    per_scenario_composite_values,
)
from eva.metrics.aggregation import compute_record_aggregates, compute_run_level_aggregates
from eva.models.results import MetricScore, RecordMetrics
from eva.utils.rank_stability import (
    FIRM_RANK_HOLD,
    joint_cluster_bootstrap_ranks,
    rank_stability_table,
    stability_summary,
)

# Components of the EVA-overall_mean composite (eva.metrics.aggregation).
OVERALL_COMPONENTS = [
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "conciseness",
    "turn_taking",
]


def _stepped_table() -> dict[str, dict[str, float]]:
    """Four systems where only the top two are close enough to swap under resampling."""
    units = [f"s{u}" for u in range(20)]
    return {
        "top_a": {unit: (0.75 if u % 2 == 0 else 0.5) for u, unit in enumerate(units)},
        "top_b": {unit: (0.5 if u % 2 == 0 else 0.7) for u, unit in enumerate(units)},
        "mid": {unit: 0.45 for unit in units},
        "clear_worst": {unit: 0.1 for unit in units},
    }


class TestJointClusterBootstrapRanks:
    def test_separated_systems_hold_ranks(self):
        values = {"a": {f"s{u}": 0.9 for u in range(8)}, "b": {f"s{u}": 0.5 for u in range(8)}}
        ranks = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=1)
        np.testing.assert_array_equal(ranks["a"], np.ones(100))
        np.testing.assert_array_equal(ranks["b"], np.full(100, 2.0))

    def test_identical_systems_share_the_top_rank(self):
        # The joint draw gives both systems the same resampled mean, so every
        # replicate is a tie at rank 1.5 rather than an arbitrary winner.
        values = {"a": {f"s{u}": float(u % 3) for u in range(9)}, "b": {f"s{u}": float(u % 3) for u in range(9)}}
        ranks = joint_cluster_bootstrap_ranks(values, n_boot=50, seed=2)
        np.testing.assert_array_equal(ranks["a"], np.full(50, 1.5))
        np.testing.assert_array_equal(ranks["b"], np.full(50, 1.5))

    def test_deterministic_per_seed(self):
        values = _stepped_table()
        a = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=7)
        b = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=7)
        for system in values:
            np.testing.assert_array_equal(a[system], b[system])

    def test_empty_input(self):
        assert joint_cluster_bootstrap_ranks({}, n_boot=10, seed=0) == {}


class TestRankStabilityTable:
    def test_worst_row_firm_top_rows_flip(self):
        # The paper's headline shape: the bottom of the ranking is firm while the
        # top two swap often enough that the table cannot name the best system.
        table = rank_stability_table(_stepped_table(), seed=7, n_boot=400)

        assert list(table) == ["top_a", "top_b", "mid", "clear_worst"]
        assert table["top_a"]["rank"] == 1.0
        assert table["top_a"]["rank_hold_share"] < FIRM_RANK_HOLD
        assert table["top_b"]["rank_hold_share"] < FIRM_RANK_HOLD
        # Exactly one of the top two is rank 1 in each replicate they don't tie.
        assert table["top_a"]["share_rank_1"] + table["top_b"]["share_rank_1"] == 1.0
        assert 0.0 < table["top_b"]["share_rank_1"] < 1.0
        assert table["clear_worst"]["rank"] == 4.0
        assert table["clear_worst"]["rank_hold_share"] == 1.0
        assert table["clear_worst"]["firm"] is True

    def test_summary_identifies_worst_but_not_best(self):
        summary = stability_summary(rank_stability_table(_stepped_table(), seed=7, n_boot=400))

        assert summary["n_systems"] == 4
        assert summary["n_units"] == 20
        assert summary["n_firm_rows"] == 2
        assert summary["best_system"] == "top_a"
        assert summary["best_system_firm"] is False
        assert summary["worst_system"] == "clear_worst"
        assert summary["worst_system_firm"] is True

    def test_missing_units_reported_as_provenance(self):
        # A system that errored on some scenarios keeps those scenarios out of its
        # means and reports them as missing, instead of being silently padded.
        table = rank_stability_table(
            {"a": {"s1": 1.0, "s2": 0.5, "s3": 0.2}, "b": {"s1": 0.4}},
            seed=3,
            n_boot=200,
        )
        assert table["b"]["n_units"] == 1
        assert table["b"]["n_missing_units"] == 2
        assert table["b"]["mean"] == 0.4
        assert table["a"]["n_missing_units"] == 0

    def test_system_without_values_gets_no_rank(self):
        table = rank_stability_table({"a": {"s1": 1.0}, "b": {}}, seed=3, n_boot=50)
        assert table["b"]["rank"] is None
        assert table["b"]["rank_hold_share"] is None
        assert table["b"]["firm"] is False
        assert table["a"]["rank"] == 1.0

    def test_empty_table_summary(self):
        assert stability_summary({})["n_systems"] == 0


def _record(record_id: str, level: float) -> RecordMetrics:
    """A RecordMetrics whose every EVA-overall_mean component scores ``level``."""
    record = RecordMetrics(record_id=record_id)
    for name in OVERALL_COMPONENTS:
        record.metrics[name] = MetricScore(name=name, score=level, normalized_score=level)
    record.aggregate_metrics = compute_record_aggregates(record)
    return record


def _write_run(root: Path, label: str, scenario_levels: dict[str, float]) -> Path:
    """Persist a run's per-record metrics.json artifacts and return the run dir."""
    run_dir = root / label
    for scenario, level in scenario_levels.items():
        record_dir = run_dir / "records" / scenario
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / "metrics.json").write_text(_record(scenario, level).model_dump_json())
    return run_dir


class TestRunLevelAggregatesInterop:
    def test_table_agrees_with_run_level_means_and_cis(self, tmp_path: Path):
        # Feeds the same per-scenario composite means the run-level aggregates rank
        # and bootstrap into the rank-stability audit, and checks the two views of
        # the same runs agree.
        scenario_levels = {
            "strong": {f"s{u}": 0.9 for u in range(6)},
            "weak": {f"s{u}": 0.2 for u in range(6)},
        }
        runs = {label: _write_run(tmp_path, label, levels) for label, levels in scenario_levels.items()}
        loaded = {label: load_run_metrics(run_dir) for label, run_dir in runs.items()}

        run_level = {
            label: compute_run_level_aggregates(run_metrics, seed=11)["EVA-overall_mean"]
            for label, run_metrics in loaded.items()
        }
        per_unit = {
            label: per_scenario_composite_values(run_metrics, "EVA-overall_mean")
            for label, run_metrics in loaded.items()
        }

        table = rank_stability_table(per_unit, seed=11, n_boot=200)
        assert table["strong"]["mean"] == run_level["strong"]["mean"] == 0.9
        assert table["weak"]["mean"] == run_level["weak"]["mean"] == 0.2
        assert table["strong"]["rank"] == 1.0
        assert table["weak"]["rank"] == 2.0
        # Well-separated systems hold their ranks in every replicate.
        assert table["strong"]["rank_hold_share"] == 1.0
        assert table["weak"]["rank_hold_share"] == 1.0
        # The run-level CI the leaderboard already reports is present alongside.
        assert run_level["strong"]["mean_ci_lower"] == run_level["strong"]["mean_ci_upper"] == 0.9

    def test_build_stability_rows_end_to_end(self, tmp_path: Path):
        strong = _write_run(tmp_path, "strong", {f"s{u}": 0.9 for u in range(5)})
        weak = _write_run(tmp_path, "weak", {f"s{u}": 0.2 for u in range(5)})

        results, summary = build_stability_rows(
            {"strong": strong, "weak": weak},
            "EVA-overall_mean",
            n_boot=200,
            seed=13,
        )

        assert list(results["system"]) == ["strong", "weak"]
        assert results.loc[0, "mean_ci_lower"] == 0.9
        assert results.loc[0, "rank_hold_share"] == 1.0
        assert results.loc[0, "firm"]
        assert results.loc[1, "n_missing_units"] == 0
        assert {"mean_ci_upper", "measurement_date", "share_rank_1"}.issubset(results.columns)
        assert summary["best_system"] == "strong"
        assert summary["worst_system_firm"] is True
