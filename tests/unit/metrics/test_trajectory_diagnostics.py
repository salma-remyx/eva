"""Tests for TrajectoryDiagnosticsJudgeMetric."""

import json

import pytest

# Imports go through the eva.metrics package chain so these tests exercise the
# diagnostic-package wiring that registers the metric in the global registry.
from eva.metrics.diagnostic.trajectory_diagnostics import (
    TrajectoryDiagnosticsJudgeMetric,
    detect_failure_signals,
    serialize_trace_under_budget,
)
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


def _trace_with_tool_error() -> list[dict]:
    return [
        {"turn_id": 1, "role": "user", "content": "check my reservation"},
        {"turn_id": 1, "role": "assistant", "content": "one moment"},
        {"turn_id": 1, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"confirmation_number": "X"}},
        {
            "turn_id": 1,
            "type": "tool_response",
            "tool_name": "get_reservation",
            "tool_response": {"error_type": "invalid_confirmation_number_format", "message": "Invalid 'X'"},
        },
    ]


class TestTrajectoryDiagnosticsWiring:
    def test_registered_via_diagnostic_package(self):
        # Production resolves metrics by name from the global registry after the
        # eva.metrics import chain runs; assert that chain registers this metric.
        import eva.metrics  # noqa: F401 — imported for its registration side effect

        assert get_global_registry().get("trajectory_diagnostics") is TrajectoryDiagnosticsJudgeMetric

    def test_opt_in_not_in_default_metrics(self):
        assert "trajectory_diagnostics" not in get_global_registry().list_metrics()

    def test_metric_attributes(self):
        metric = make_judge_metric(TrajectoryDiagnosticsJudgeMetric)
        assert metric.name == "trajectory_diagnostics"
        assert metric.category == "diagnostic"
        assert metric.rating_scale == (1, 3)
        assert metric.version == "v0.1"


class TestDetectFailureSignals:
    def test_tool_call_format_error_flagged(self):
        ctx = make_metric_context(conversation_trace=_trace_with_tool_error())
        flags = [s["flag"] for s in detect_failure_signals(ctx)]
        assert "tool_call_format_error" in flags

    def test_business_logic_error_not_flagged(self):
        trace = _trace_with_tool_error()
        trace[-1]["tool_response"] = {"error_type": "not_found", "message": "no reservation"}
        ctx = make_metric_context(conversation_trace=trace)
        assert detect_failure_signals(ctx) == []

    def test_slow_and_interrupted_and_empty_turns_flagged(self):
        ctx = make_metric_context(
            conversation_trace=[
                {"turn_id": 1, "role": "assistant", "content": "hello"},
                {"turn_id": 2, "role": "assistant", "content": "   "},
            ],
            latency_assistant_turns={1: 12.0, 2: 0.5},
            assistant_interrupted_turns={1},
        )
        flags = sorted(s["flag"] for s in detect_failure_signals(ctx))
        assert flags == ["empty_assistant_turn", "interrupted_assistant_turn", "slow_assistant_turn"]

    def test_tool_call_only_turn_not_flagged_as_empty(self):
        ctx = make_metric_context(
            conversation_trace=[
                {"turn_id": 1, "role": "assistant", "content": ""},
                {"turn_id": 1, "type": "tool_call", "tool_name": "list_flights", "parameters": {}},
            ]
        )
        assert detect_failure_signals(ctx) == []

    def test_slow_turn_threshold_configurable(self):
        ctx = make_metric_context(
            conversation_trace=[{"turn_id": 1, "role": "assistant", "content": "hi"}],
            latency_assistant_turns={1: 9.0},
        )
        assert detect_failure_signals(ctx, slow_turn_seconds=10.0) == []


class TestSerializeTraceUnderBudget:
    def _long_trace(self, num_turns: int) -> list[dict]:
        trace = []
        for turn_id in range(1, num_turns + 1):
            trace.append({"turn_id": turn_id, "role": "user", "content": "question " * 40})
            trace.append({"turn_id": turn_id, "role": "assistant", "content": "answer " * 40})
        return trace

    def test_flagged_turn_kept_and_middle_omitted_under_budget(self):
        trace = self._long_trace(30)
        text, stats = serialize_trace_under_budget(trace, flagged_turn_ids={17}, budget_chars=1500)
        assert "Turn 17:" in text
        assert "unflagged turns omitted" in text
        assert stats["turns_total"] == 30
        assert stats["turns_kept"] < 30
        assert stats["turns_omitted"] > 0
        assert stats["serialized_chars"] <= 1500 + 2 * stats["turns_total"]  # marker slack only

    def test_no_omissions_when_trace_fits_budget(self):
        trace = [
            {"turn_id": 1, "role": "user", "content": "hi"},
            {"turn_id": 1, "role": "assistant", "content": "hello"},
        ]
        text, stats = serialize_trace_under_budget(trace, flagged_turn_ids=set(), budget_chars=2000)
        assert "omitted" not in text
        assert stats["turns_omitted"] == 0

    def test_long_entries_truncated_head_and_tail(self):
        trace = [
            {
                "turn_id": 1,
                "type": "tool_response",
                "tool_name": "list_flights",
                "tool_response": {"flights": "x" * 5000},
            }
        ]
        text, _ = serialize_trace_under_budget(trace, flagged_turn_ids=set(), budget_chars=10000)
        assert "truncated" in text
        assert len(text) < 1000


class TestTrajectoryDiagnosticsCompute:
    def setup_method(self):
        self.metric = make_judge_metric(TrajectoryDiagnosticsJudgeMetric, mock_llm=True)

    def test_get_prompt_variables_includes_rule_flags_and_trace(self):
        ctx = make_metric_context(conversation_trace=_trace_with_tool_error())
        variables = self.metric.get_prompt_variables(ctx, "unused")
        assert "tool_call_format_error" in variables["rule_flags"]
        assert "Turn 1" in variables["conversation_trace"]
        assert variables["agent_role"] == "Test role"

    @pytest.mark.asyncio
    async def test_compute_success_with_confirmed_signal(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 2,
                    "explanation": "Malformed confirmation number caused a failed lookup.",
                    "flags": {
                        "tool_call_format_error": {"confirmed": True, "evidence": "Turn 1: invalid format"},
                        "slow_assistant_turn": {"confirmed": False, "evidence": "user kept talking"},
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=_trace_with_tool_error(),
            latency_assistant_turns={1: 11.0},
        )
        score = await self.metric.compute(ctx)
        assert score.score == 2.0
        assert score.normalized_score == 0.5
        assert score.details["confirmed_signals"] == ["tool_call_format_error"]
        assert score.details["num_rule_signals"] == 2
        assert score.details["trace_budget"]["budget_chars"] == 12000
        assert score.sub_metrics["tool_call_format_error_rate"].score == 1.0
        assert score.sub_metrics["slow_assistant_turn_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_compute_empty_trace(self):
        ctx = make_metric_context(conversation_trace=[])
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "No transcript" in score.error

    def test_config_overrides_budget_and_threshold(self):
        metric = TrajectoryDiagnosticsJudgeMetric(config={"trace_budget_chars": 500, "slow_turn_seconds": 2.0})
        assert metric.trace_budget_chars == 500
        assert metric.slow_turn_seconds == 2.0
