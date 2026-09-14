"""Unit tests for reliability-inclusive composite scoring.

Exercises the inclusive aggregates against the exclusive ones from
``eva.metrics.aggregation`` (the module whose behavior they report on), and
checks that ``MetricsRunner._save_summary`` persists the report.
"""

import json

import pytest

from eva.metrics.aggregation import compute_record_aggregates
from eva.metrics.reliability import (
    classify_component,
    compute_reliability_report,
    reliability_inclusive_aggregates,
)
from eva.metrics.runner import MetricsRunner
from eva.models.results import MetricScore, RecordMetrics

from .conftest import make_record_metrics

_FULL_PASS = {
    "task_completion": 1.0,
    "faithfulness": 0.5,
    "agent_speech_fidelity": 0.95,
    "conversation_progression": 0.5,
    "turn_taking": 0.8,
    "conciseness": 0.5,
}


def _make_record(
    scores: dict[str, float],
    *,
    errors: list[str] | None = None,
    skipped: list[str] | None = None,
    record_id: str = "1.1.1",
) -> RecordMetrics:
    """Build a RecordMetrics with optional errored / skipped components."""
    metrics = {name: MetricScore(name=name, score=v, normalized_score=v) for name, v in scores.items()}
    for name in errors or []:
        metrics[name] = MetricScore(name=name, score=0.0, error="route failure")
    for name in skipped or []:
        metrics[name] = MetricScore(name=name, score=None, normalized_score=None, skipped=True)
    rm = RecordMetrics(record_id=record_id, metrics=metrics)
    rm.aggregate_metrics = compute_record_aggregates(rm)
    return rm


class TestClassifyComponent:
    def test_error_is_failure(self):
        assert classify_component(MetricScore(name="m", score=0.0, error="boom")) == "failed"

    def test_skipped_is_unsupported(self):
        assert classify_component(MetricScore(name="m", score=None, skipped=True)) == "unsupported"

    def test_absent_is_unsupported(self):
        assert classify_component(None) == "unsupported"

    def test_scored_is_supported(self):
        assert classify_component(MetricScore(name="m", score=0.4, normalized_score=0.4)) == "supported"


class TestInclusiveRecordAggregates:
    def test_errored_component_stays_in_score(self):
        """Exclusive mode drops the record (None); inclusive scores it as failure (0.0)."""
        rm = _make_record(_FULL_PASS, errors=["task_completion"])

        assert rm.aggregate_metrics["EVA-A_pass"] is None
        assert reliability_inclusive_aggregates(rm)["EVA-A_pass"] == 0.0

    def test_errored_component_counts_in_mean_denominator(self):
        """Failed components contribute 0.0 instead of leaving the mean's denominator."""
        rm = _make_record(_FULL_PASS, errors=["task_completion"])

        # Exclusive mean averages only the two non-errored components.
        assert rm.aggregate_metrics["EVA-A_mean"] == pytest.approx((0.5 + 0.95) / 2)
        # Inclusive mean keeps the failure: 3-component denominator.
        assert reliability_inclusive_aggregates(rm)["EVA-A_mean"] == pytest.approx((0.0 + 0.5 + 0.95) / 3)

    def test_skipped_component_stays_excluded(self):
        """Unsupported capability stays out: skipped components don't fail the composite."""
        rm = _make_record(_FULL_PASS, skipped=["agent_speech_fidelity"])

        assert reliability_inclusive_aggregates(rm)["EVA-A_pass"] == 1.0

    def test_all_components_unsupported_is_none(self):
        rm = _make_record({}, skipped=["task_completion", "faithfulness", "agent_speech_fidelity"])

        assert reliability_inclusive_aggregates(rm)["EVA-A_pass"] is None

    def test_matching_results_without_failures(self):
        """With no failures the inclusive aggregates equal the exclusive ones."""
        rm = _make_record(_FULL_PASS)
        inclusive = reliability_inclusive_aggregates(rm)

        for name, value in rm.aggregate_metrics.items():
            assert inclusive[name] == pytest.approx(value)


class TestReliabilityReport:
    def test_report_quantifies_silent_denominator_shrinkage(self):
        good = _make_record(_FULL_PASS, record_id="1.1.1")
        failed = _make_record(_FULL_PASS, errors=["task_completion"], record_id="1.1.2")

        report = compute_reliability_report({"1.1.1": good, "1.1.2": failed}, seed=42)
        entry = report["EVA-A_pass"]

        # Exclusive mode scores the run 1.0 by dropping the failed record;
        # inclusive mode keeps the failure in the denominator.
        assert entry["exclusive_mean"] == 1.0
        assert entry["inclusive_mean"] == 0.5
        assert entry["delta"] == -0.5
        assert entry["silently_dropped_failures"] == 1
        assert entry["failed_component_records"] == {"task_completion": 1}
        assert entry["mean_ci_lower"] <= entry["inclusive_mean"] <= entry["mean_ci_upper"]

    def test_reliability_inclusion_flips_ordering(self):
        """Miniature of IB2's headline result: dropping failures flips the ordering.

        System A: 3 records, 2 with route failures and 1 pass.
        System B: 3 records, 2 passes, no failures.
        Exclusive scoring ranks A above B; inclusive scoring ranks B above A.
        """
        system_a = {
            "a.1": _make_record(_FULL_PASS, record_id="a.1"),
            "a.2": _make_record(_FULL_PASS, errors=["task_completion"], record_id="a.2"),
            "a.3": _make_record(_FULL_PASS, errors=["task_completion"], record_id="a.3"),
        }
        system_b = {
            "b.1": _make_record(_FULL_PASS, record_id="b.1"),
            "b.2": _make_record(_FULL_PASS, record_id="b.2"),
            "b.3": _make_record(_FULL_PASS | {"task_completion": 0.0}, record_id="b.3"),
        }

        report_a = compute_reliability_report(system_a, seed=42)
        report_b = compute_reliability_report(system_b, seed=42)
        a, b = report_a["EVA-A_pass"], report_b["EVA-A_pass"]

        assert a["exclusive_mean"] == 1.0
        assert b["exclusive_mean"] == pytest.approx(2 / 3, abs=1e-4)
        assert a["inclusive_mean"] == pytest.approx(1 / 3, abs=1e-4)
        assert b["inclusive_mean"] == pytest.approx(2 / 3, abs=1e-4)
        assert a["exclusive_mean"] > b["exclusive_mean"]
        assert a["inclusive_mean"] < b["inclusive_mean"]

    def test_empty_metrics_returns_empty_component_counts(self):
        rm = make_record_metrics({"task_completion": 1.0}, record_id="1")
        rm.aggregate_metrics = compute_record_aggregates(rm)

        report = compute_reliability_report({"1": rm}, seed=42)

        # EVA-X components are absent -> unsupported, not failed.
        assert report["EVA-X_pass"]["failed_component_records"] == {}
        assert report["EVA-X_pass"]["unsupported_component_records"] == {
            "conciseness": 1,
            "conversation_progression": 1,
            "turn_taking": 1,
        }


class TestSummaryWiring:
    async def test_save_summary_persists_reliability_report(self, tmp_path):
        """MetricsRunner._save_summary writes the reliability block into metrics_summary.json."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text("id: agent_test\nrole: agent\ninstructions: hi\ntools: []\n")
        (run_dir / "config.json").write_text(json.dumps({"agent_config_path": str(agent_yaml)}))

        runner = MetricsRunner(run_dir=run_dir, dataset=[], metric_names=[], metric_configs={})

        good = _make_record(_FULL_PASS, record_id="1.1.1")
        failed = _make_record(_FULL_PASS, errors=["task_completion"], record_id="1.1.2")
        await runner._save_summary({"1.1.1": good, "1.1.2": failed})

        summary = json.loads((run_dir / "metrics_summary.json").read_text())
        assert "reliability" in summary
        entry = summary["reliability"]["EVA-A_pass"]
        assert entry["exclusive_mean"] == 1.0
        assert entry["inclusive_mean"] == 0.5
        assert entry["silently_dropped_failures"] == 1
