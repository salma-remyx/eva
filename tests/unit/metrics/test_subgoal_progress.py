"""Tests for SubgoalProgressMetric (TED-style per-turn subgoal progress).

These tests exercise the registry wiring (the import added to
``eva.metrics.accuracy.__init__``) and the integrated ``compute`` path against
the existing ``TextJudgeMetric`` LLM client — they import non-new modules
(``eva.metrics`` / the registry) and assert end-to-end behavior.
"""

import json

import pytest

import eva.metrics  # noqa: F401  (imports subpackages → registers metrics)
from eva.metrics.accuracy.subgoal_progress import (
    SubgoalProgressMetric,
    _derive_subgoals,
)
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


def _scenario_context(num_assistant_turns: int = 3, **overrides):
    """A context whose initial→expected DB diff yields one field-update subgoal."""
    trace = []
    for turn in range(1, num_assistant_turns + 1):
        trace.append({"turn_id": turn, "role": "user", "content": f"user turn {turn}"})
        trace.append({"turn_id": turn, "role": "assistant", "content": f"assistant turn {turn}"})
    return make_metric_context(
        num_assistant_turns=num_assistant_turns,
        conversation_trace=trace,
        initial_scenario_db={"users": {"alice": {"email": "old@example.com"}}},
        expected_scenario_db={"users": {"alice": {"email": "new@example.com"}}},
        **overrides,
    )


class TestSubgoalProgressRegistration:
    def test_metric_registered_via_accuracy_package(self):
        # The import line added to eva.metrics.accuracy.__init__ registers it.
        assert get_global_registry().get("subgoal_progress") is SubgoalProgressMetric

    def test_metric_attributes(self):
        metric = make_judge_metric(SubgoalProgressMetric)
        assert metric.name == "subgoal_progress"
        assert metric.category == "accuracy"
        assert metric.version == "v0.1"


class TestSubgoalDerivation:
    def test_subgoals_derived_from_scenario_db_diff(self):
        ctx = _scenario_context()
        subgoals = _derive_subgoals(ctx)
        assert len(subgoals) == 1
        assert subgoals[0]["id"] == "s1"
        # The changed field ("email") is surfaced in the subgoal description.
        assert "email" in subgoals[0]["description"]
        assert "alice" in subgoals[0]["description"]

    def test_empty_diff_falls_back_to_user_goal_subgoal(self):
        ctx = make_metric_context(
            user_goal="Book me a flight",
            initial_scenario_db={},
            expected_scenario_db={},
        )
        subgoals = _derive_subgoals(ctx)
        assert len(subgoals) == 1
        assert "Book me a flight" in subgoals[0]["description"]


class TestSubgoalProgressCompute:
    def setup_method(self):
        self.metric = make_judge_metric(SubgoalProgressMetric, mock_llm=True)

    @pytest.mark.asyncio
    async def test_compute_full_completion_early(self):
        # Subgoal s1 accomplished at turn 2 of 3.
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"subgoals": [{"id": "s1", "completion_turn": 2}], "overall_reasoning": "ok"}),
            None,
        )
        ctx = _scenario_context(num_assistant_turns=3)

        score = await self.metric.compute(ctx)

        assert score.error is None
        assert score.score == 1.0  # final_progress
        assert score.normalized_score == pytest.approx(0.6667)  # progress-curve AUC
        assert score.details["progress_curve"] == [0.0, 1.0, 1.0]
        assert score.details["final_progress"] == 1.0
        assert score.details["progress_per_turn"] == pytest.approx(0.3333)
        assert score.details["num_subgoals"] == 1
        assert score.details["error_analysis"]["incomplete"] is False
        assert score.details["error_analysis"]["stalled"] is False
        assert score.details["error_analysis"]["unachieved_count"] == 0
        # TED headline aggregates surface as sub-metrics for per-metric breakdowns.
        assert set(score.sub_metrics.keys()) == {"progress_auc", "final_progress", "progress_per_turn"}
        assert score.sub_metrics["progress_auc"].normalized_score == pytest.approx(0.6667)

    @pytest.mark.asyncio
    async def test_compute_unachieved_subgoal_is_diagnosed(self):
        # Subgoal never accomplished → flat zero curve, flagged incomplete + stalled.
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"subgoals": [{"id": "s1", "completion_turn": None}], "overall_reasoning": "no progress"}),
            None,
        )
        ctx = _scenario_context(num_assistant_turns=2)

        score = await self.metric.compute(ctx)

        assert score.score == 0.0
        assert score.normalized_score == 0.0
        assert score.details["progress_curve"] == [0.0, 0.0]
        diagnosis = score.details["error_analysis"]
        assert diagnosis["incomplete"] is True
        assert diagnosis["stalled"] is True
        assert diagnosis["unachieved_count"] == 1
        assert len(diagnosis["unachieved_subgoals"]) == 1

    @pytest.mark.asyncio
    async def test_compute_no_assistant_turns_returns_error(self):
        ctx = make_metric_context(num_assistant_turns=0, conversation_trace=[])
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None

    @pytest.mark.asyncio
    async def test_compute_unparseable_judge_response(self):
        self.metric.llm_client.generate_text.return_value = ("not json at all", None)
        ctx = _scenario_context()
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None
