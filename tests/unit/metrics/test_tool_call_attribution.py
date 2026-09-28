"""Tests for ToolCallAttributionMetric — serving stack vs. model attribution."""

import pytest

from eva.metrics.diagnostic import tool_call_attribution
from eva.metrics.diagnostic.tool_call_validity import ToolCallValidity
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context

GENERIC_ERROR = "I'm sorry, I encountered an error processing your request."


def _assistant(turn_id: int, content: str) -> dict:
    return {"role": "assistant", "content": content, "type": "intended", "turn_id": turn_id, "timestamp": 0}


def _call(turn_id: int, name: str = "get_reservation") -> dict:
    return {"tool_name": name, "parameters": {}, "type": "tool_call", "turn_id": turn_id, "timestamp": 0}


def _response(turn_id: int, error_type: str | None = None) -> dict:
    payload = {"status": "error", "error_type": error_type} if error_type else {"status": "success"}
    return {
        "tool_name": "get_reservation",
        "tool_response": payload,
        "type": "tool_response",
        "turn_id": turn_id,
        "timestamp": 0,
    }


@pytest.fixture
def metric():
    return tool_call_attribution.ToolCallAttributionMetric()


class TestMetricRegistration:
    def test_metric_registered_through_diagnostic_package(self):
        """Importing eva.metrics.diagnostic registers the metric in the global registry."""
        registry = get_global_registry()
        assert registry.get("tool_call_attribution") is tool_call_attribution.ToolCallAttributionMetric
        assert "tool_call_attribution" in registry.list_metrics()


class TestServingFailureConfound:
    @pytest.mark.asyncio
    async def test_serving_rejection_is_skipped_not_scored(self, metric):
        """Every assistant turn on the error path → fidelity unmeasurable, flagged as confounded."""
        trace = [
            _assistant(0, GENERIC_ERROR),
            _assistant(1, GENERIC_ERROR),
        ]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: GENERIC_ERROR, 1: GENERIC_ERROR},
            conversation_ended_reason="error",
        )
        result = await metric.compute(context)

        assert result.score is None
        assert result.normalized_score is None
        assert result.skipped is True
        assert result.details["confounded"] is True
        assert result.details["outcome_counts"]["serving_failure"] == 2
        assert result.details["outcome_counts"]["model_non_call"] == 0
        assert "not measurable" in result.details["note"]
        assert result.details["attribution"]["serving_layer"]["conversation_ended_in_error"] is True

    @pytest.mark.asyncio
    async def test_tool_call_validity_reads_same_record_as_perfect(self, metric):
        """The existing tool_call_validity metric scores a serving-rejected record 1.0.

        This is the hidden confound the attribution metric exists to surface: with
        zero tool_responses, ToolCallValidity has nothing to invalidate and reports
        perfect fidelity for a conversation the serving layer never let the model
        answer.
        """
        trace = [_assistant(0, GENERIC_ERROR)]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: GENERIC_ERROR},
            conversation_ended_reason="error",
        )

        validity = await ToolCallValidity().compute(context)
        attribution = await metric.compute(context)

        assert validity.score == 1.0
        assert validity.details["total_tool_calls"] == 0
        assert attribution.score is None
        assert attribution.skipped is True
        assert attribution.details["confounded"] is True


class TestDualGranularityFidelity:
    @pytest.mark.asyncio
    async def test_per_instance_and_turn_pooled_fidelity_diverge(self, metric):
        """Failures clustering in one turn make turn-pooled fidelity diverge from per-instance."""
        trace = [
            _assistant(0, "Checking your reservation…"),
            _call(0),
            _response(0),
            _call(0),
            _response(0),  # turn 0: both calls valid
            _assistant(1, "Rebooking…"),
            _call(1),
            _response(1),
            _call(1),
            _response(1, error_type="invalid_parameter"),  # turn 1: one bad
            _assistant(2, "You're all set!"),  # genuine model non-call
        ]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: "Checking…", 1: "Rebooking…", 2: "You're all set!"},
        )
        result = await metric.compute(context)

        assert result.score == pytest.approx(3 / 4, abs=0.001)
        assert result.details["per_instance_fidelity"] == pytest.approx(0.75, abs=0.001)
        assert result.details["turn_pooled_fidelity"] == pytest.approx(0.5, abs=0.001)
        assert result.details["aggregation_gap"] == pytest.approx(0.25, abs=0.001)
        assert result.details["outcome_counts"] == {
            "valid_call": 3,
            "parse_failure": 1,
            "serving_failure": 0,
            "model_non_call": 1,
        }
        assert result.details["confounded"] is False
        assert result.skipped is False

        subs = result.sub_metrics
        assert subs is not None
        assert subs["num_tool_calls"].score == 4.0
        assert subs["num_tool_calls"].normalized_score is None
        assert subs["valid_call_rate"].score == pytest.approx(0.75, abs=0.001)
        assert subs["parse_failure_rate"].score == pytest.approx(0.25, abs=0.001)
        assert subs["serving_failure_rate"].score == 0.0
        assert subs["turn_pooled_fidelity"].score == pytest.approx(0.5, abs=0.001)


class TestOutcomeTaxonomyReuse:
    @pytest.mark.asyncio
    async def test_parse_failures_reuse_call_error_taxonomy(self, metric):
        """Harness-side parse/validation errors (CALL_ERROR_TYPES) count as parse failures."""
        trace = [
            _assistant(0, "One moment…"),
            _call(0),
            _response(0, error_type="tool_not_found"),
            _call(0),
            _response(0, error_type="invalid_parameter"),
        ]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: "One moment…"},
        )
        result = await metric.compute(context)

        assert result.score == 0.0
        assert result.details["outcome_counts"]["parse_failure"] == 2
        assert result.details["outcome_counts"]["valid_call"] == 0
        # All failures are attributed to the model layer; none to the serving layer.
        assert result.details["attribution"]["serving_layer"]["failed_turns"] == 0


class TestNoCallEdgeCases:
    @pytest.mark.asyncio
    async def test_no_calls_without_serving_failures_is_skipped_as_model_choice(self, metric):
        """A clean conversation with no tool calls is a model non-call, not a confound."""
        trace = [_assistant(0, "Hi! How can I help?")]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: "Hi! How can I help?"},
        )
        result = await metric.compute(context)

        assert result.score is None
        assert result.skipped is True
        assert result.details["confounded"] is False
        assert result.details["outcome_counts"]["model_non_call"] == 1
        assert "no attempts" in result.details["note"]

    @pytest.mark.asyncio
    async def test_interrupted_call_without_response_is_ignored(self, metric):
        """A trailing tool_call with no response (interrupted) is dropped, not guessed at."""
        trace = [
            _assistant(0, "Checking…"),
            _call(0),
            _response(0),
            _call(0),  # interrupted before the tool responded
        ]
        context = make_metric_context(
            conversation_trace=trace,
            intended_assistant_turns={0: "Checking…"},
        )
        result = await metric.compute(context)

        assert result.details["total_tool_calls"] == 1
        assert result.score == 1.0
