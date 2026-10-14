"""Unit tests for analysis/rank_stability/run_rank_stability.py."""

from __future__ import annotations

from pathlib import Path

from analysis.rank_stability.run_rank_stability import (
    build_stability_rows,
    load_run_metrics,
    per_scenario_composite_values,
)
from eva.metrics.aggregation import compute_record_aggregates, compute_run_level_aggregates
from eva.models.results import MetricScore, RecordMetrics
from eva.utils.rank_stability import rank_stability_table

# Components of the EVA-overall_mean composite (eva.metrics.aggregation).
OVERALL_COMPONENTS = [
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "conciseness",
    "turn_taking",
]


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
