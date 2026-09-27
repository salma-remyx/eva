"""Unit tests for the cross-run rank-stability audit.

Exercises ``apps.rank_stability`` against the production contracts it builds
on: scenario clustering via ``parse_trial_record_id`` (the same unit the
run-level ``mean_ci_*`` bootstraps use), seeding via ``run_seed``, and
``RecordMetrics.get_score`` for the row values the analysis app collects.
"""

from __future__ import annotations

import pytest

from apps.rank_stability import rank_stability_from_matrix, rank_stability_from_rows
from eva.models.results import MetricScore, RecordMetrics
from eva.utils.bootstrap import run_seed
from eva.utils.pass_at_k import parse_trial_record_id


def _rows(values_by_system: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    """Build rows in the shape ``_collect_run_metrics`` emits (one row per trial)."""
    rows: list[dict[str, object]] = []
    for system, by_record in values_by_system.items():
        for record, value in by_record.items():
            rows.append({"system": system, "record": record, "faithfulness": value})
    return rows


def test_separated_systems_hold_rank() -> None:
    """Systems separated on every scenario keep their ranking in all replicates."""
    result = rank_stability_from_rows(
        _rows({"A": {f"r{i}": 0.9 for i in range(8)}, "B": {f"r{i}": 0.2 for i in range(8)}}),
        "faithfulness",
        seed=7,
    )
    assert result is not None
    assert result.observed_rank == {"A": 1, "B": 2}
    assert result.rank_hold_rate == {"A": 1.0, "B": 1.0}
    assert result.best_rate == {"A": 1.0, "B": 0.0}
    assert result.win_rate[("A", "B")] == 1.0


def test_overlapping_systems_do_not_hold_rank() -> None:
    """Near-tied systems with small per-scenario margins swap ranks across replicates."""
    result = rank_stability_from_rows(
        _rows(
            {
                "A": {f"r{i}": 0.53 if i < 8 else 0.44 for i in range(10)},
                "B": {f"r{i}": 0.50 if i < 8 else 0.51 for i in range(10)},
            }
        ),
        "faithfulness",
        seed=7,
    )
    assert result is not None
    assert result.rank_hold_rate["A"] < 0.95
    assert 0.5 < result.win_rate[("A", "B")] < 0.95
    # Exactly one system ranks first per replicate, and A leads it more often.
    assert result.best_rate["A"] + result.best_rate["B"] == pytest.approx(1.0)
    assert result.best_rate["A"] > result.best_rate["B"]


def test_clear_gap_is_firm_and_overlap_is_not() -> None:
    """A clearly-separated worst system holds rank while the top two do not."""
    result = rank_stability_from_rows(
        _rows(
            {
                "A": {f"r{i}": 0.56 if i % 2 == 0 else 0.45 for i in range(10)},
                "B": {f"r{i}": 0.45 if i % 2 == 0 else 0.55 for i in range(10)},
                "Worst": {f"r{i}": 0.1 for i in range(10)},
            }
        ),
        "faithfulness",
        seed=7,
    )
    assert result is not None
    assert result.observed_rank["Worst"] == 3
    assert result.rank_hold_rate["Worst"] == 1.0
    # The paper's headline shape: the bottom of the table is firm, the top is not.
    assert result.rank_hold_rate["A"] < 1.0 or result.rank_hold_rate["B"] < 1.0
    assert result.win_rate[("A", "Worst")] == 1.0
    assert result.win_rate[("B", "Worst")] == 1.0


def test_trials_collapse_to_scenario_means() -> None:
    """Rows sharing a scenario (trials) are averaged before ranking."""
    rows = [
        {"system": "A", "record": "rec1", "faithfulness": 1.0},
        {"system": "A", "record": "rec1", "faithfulness": 0.0},
        {"system": "A", "record": "rec2", "faithfulness": 0.4},
        {"system": "B", "record": "rec1", "faithfulness": 0.5},
        {"system": "B", "record": "rec2", "faithfulness": 0.5},
    ]
    result = rank_stability_from_rows(rows, "faithfulness", seed=3)
    assert result is not None
    assert result.observed_mean == {"A": 0.45, "B": 0.5}
    assert result.observed_rank == {"B": 1, "A": 2}


def test_flat_trial_record_ids_cluster_by_base_id() -> None:
    """Flat trial-suffixed record ids resolve to the same scenario."""
    # The clustering contract the run-level bootstrap already relies on.
    assert parse_trial_record_id("1.2.1_trial_0")[0] == "1.2.1"
    rows = [
        {"system": "A", "record": "1.2.1_trial_0", "faithfulness": 1.0},
        {"system": "A", "record": "1.2.1_trial_1", "faithfulness": 0.0},
        {"system": "B", "record": "1.2.1_trial_0", "faithfulness": 0.5},
        {"system": "B", "record": "1.2.1_trial_1", "faithfulness": 0.5},
        {"system": "A", "record": "2.1.0", "faithfulness": 0.5},
        {"system": "B", "record": "2.1.0", "faithfulness": 0.5},
    ]
    result = rank_stability_from_rows(rows, "faithfulness", seed=3)
    assert result is not None
    assert result.n_scenarios == 2
    assert result.observed_mean == {"A": 0.5, "B": 0.5}


def test_rows_carry_record_metrics_scores() -> None:
    """Row values come straight from the production metric-score accessor."""
    record = RecordMetrics(
        record_id="r1",
        metrics={
            "faithfulness": MetricScore(name="faithfulness", normalized_score=0.7),
            "errored": MetricScore(name="errored", score=0.9, error="judge failed"),
        },
    )
    assert record.get_score("faithfulness") == 0.7
    # Errored scores read as None, and None rows are dropped rather than ranked.
    assert record.get_score("errored") is None
    result = rank_stability_from_rows(
        [
            {"system": "A", "record": "r1", "faithfulness": record.get_score("faithfulness")},
            {"system": "A", "record": "r2", "faithfulness": 0.6},
            {"system": "A", "record": "r3", "faithfulness": 0.8},
            {"system": "B", "record": "r1", "faithfulness": 0.1},
            {"system": "B", "record": "r2", "faithfulness": None},
            {"system": "B", "record": "r3", "faithfulness": 0.2},
        ],
        "faithfulness",
        seed=1,
    )
    assert result is not None
    assert result.n_scenarios == 2  # r2 is missing for B and drops out
    assert result.n_dropped_scenarios == 1
    assert result.observed_rank["A"] == 1


def test_partial_overlap_uses_shared_scenarios_only() -> None:
    """Scenarios missing from any system are excluded and counted."""
    result = rank_stability_from_matrix(
        {"A": {"r1": 0.9, "r2": 0.8, "r3": 0.1}, "B": {"r1": 0.2, "r2": 0.1}},
        metric="faithfulness",
        seed=1,
    )
    assert result is not None
    assert result.n_scenarios == 2
    assert result.n_dropped_scenarios == 1
    assert result.n_systems == 2


def test_insufficient_data_returns_none() -> None:
    """A lone system or a single shared scenario carries nothing to audit."""
    assert rank_stability_from_matrix({"A": {"r1": 1.0, "r2": 0.5}}, metric="m") is None
    assert rank_stability_from_matrix({"A": {"r1": 1.0}, "B": {"r1": 0.5}}, metric="m") is None
    assert rank_stability_from_rows([{"system": "A", "record": "r1", "m": 1.0}], "m") is None


def test_seeded_and_default_bootstrap_are_deterministic() -> None:
    """Same inputs give the same audit, and the default seed is ``run_seed``-derived."""
    matrix = {"A": {"r1": 0.6, "r2": 0.4, "r3": 0.5}, "B": {"r1": 0.5, "r2": 0.5, "r3": 0.4}}
    first = rank_stability_from_matrix(matrix, metric="m", n_boot=200, seed=11)
    second = rank_stability_from_matrix(matrix, metric="m", n_boot=200, seed=11)
    assert first is not None and second is not None
    assert first.rank_hold_rate == second.rank_hold_rate
    assert first.win_rate == second.win_rate
    default = rank_stability_from_matrix(matrix, metric="m", n_boot=200)
    derived = rank_stability_from_matrix(matrix, metric="m", n_boot=200, seed=run_seed("m|A|B"))
    assert default is not None and derived is not None
    assert default.win_rate == derived.win_rate


def test_lower_is_better_flips_ranking() -> None:
    """Error-style metrics rank the smaller mean first."""
    result = rank_stability_from_matrix(
        {"A": {"r1": 0.1, "r2": 0.2}, "B": {"r1": 0.9, "r2": 0.8}},
        metric="stt_wer",
        higher_is_better=False,
        seed=5,
    )
    assert result is not None
    assert result.observed_rank == {"A": 1, "B": 2}
    assert result.win_rate[("A", "B")] == 1.0


def test_display_rows_follow_observed_rank() -> None:
    """Table helpers order rows best-first and pair adjacent ranks."""
    result = rank_stability_from_matrix(
        {"A": {"r1": 0.9, "r2": 0.8}, "B": {"r1": 0.5, "r2": 0.5}, "C": {"r1": 0.1, "r2": 0.1}},
        metric="m",
        seed=5,
    )
    assert result is not None
    assert [row["system"] for row in result.system_rows()] == ["A", "B", "C"]
    assert [row["rank"] for row in result.system_rows()] == [1, 2, 3]
    matchups = result.pairwise_rows()
    assert [(row["winner"], row["loser"]) for row in matchups] == [("A", "B"), ("B", "C")]
