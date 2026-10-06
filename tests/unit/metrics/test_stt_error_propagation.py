"""Tests for STTErrorPropagation metric.

Covers the CavaBench-style propagation signal: a user word the STT
mis-transcribed shows up (corrupted) in tool-call arguments, versus the
agent recovering the correct value.
"""

import asyncio

from eva.metrics.diagnostic.stt_error_propagation import STTErrorPropagation
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType

from .conftest import make_metric_context


def make_context(**overrides):
    """Build a MetricContext with a single Edinburgh/Edinburrow user turn."""
    defaults = {
        "intended_user_turns": {1: "I want to fly from Edinburgh to Toronto tomorrow."},
        "transcribed_user_turns": {1: "I want to fly from Edinburrow to Toronto tomorrow."},
        "tool_params": [
            {"tool_name": "search_flights", "tool_parameters": {"departure_city": "Edinburrow"}}
        ],
        "tool_responses": [
            {"tool_name": "search_flights", "tool_response": {"status": "success"}}
        ],
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


def compute(metric, context):
    """Run the async metric from a sync test (no pytest-asyncio needed)."""
    return asyncio.run(metric.compute(context))


def test_metric_is_registered_by_diagnostic_package():
    """Importing eva.metrics.diagnostic registers the metric for default runs."""
    import eva.metrics.diagnostic  # noqa: F401

    registry = get_global_registry()
    assert registry.get("stt_error_propagation") is STTErrorPropagation
    assert "stt_error_propagation" in registry.list_metrics()

    # Cascade-only diagnostic: skipped like stt_wer on audio-native pipelines,
    # and never contributes to pass@k.
    assert STTErrorPropagation.supported_pipeline_types == frozenset({PipelineType.CASCADE})
    assert STTErrorPropagation.exclude_from_pass_at_k is True


def test_mis_transcribed_word_propagates_into_tool_argument():
    """STT hears 'Edinburrow' and the agent books from 'Edinburrow'."""
    result = compute(STTErrorPropagation(), make_context())

    assert result.error is None
    assert result.score == 1.0  # every mis-transcribed word reached the arguments
    assert result.normalized_score == 0.0
    assert result.details["mis_transcribed_tokens"] == 1
    assert result.details["propagated"] == 1
    assert result.details["recovered"] == 0
    assert result.details["propagated_examples"] == [
        {
            "turn_id": 1,
            "intended_token": "edinburgh",
            "transcribed_as": "edinburrow",
            "tool_calls": ["search_flights"],
        }
    ]

    assert result.sub_metrics is not None
    propagation = result.sub_metrics["propagation_rate"]
    assert propagation.score == 1.0
    assert propagation.details == {
        "propagated": 1,
        "recovered": 0,
        "unattributed": 0,
        "mis_transcribed_tokens": 1,
    }
    assert result.sub_metrics["num_propagated"].score == 1.0
    assert result.sub_metrics["num_propagated"].normalized_score is None


def test_correct_argument_counts_as_recovered():
    """Same STT error, but the agent used the correct city: no propagation."""
    context = make_context(
        tool_params=[{"tool_name": "search_flights", "tool_parameters": {"departure_city": "Edinburgh"}}]
    )
    result = compute(STTErrorPropagation(), context)

    assert result.score == 0.0
    assert result.normalized_score == 1.0
    assert result.details["recovered"] == 1
    assert result.details["propagated"] == 0
    assert result.details["propagated_examples"] == []
    assert result.sub_metrics is not None
    assert result.sub_metrics["propagation_rate"].score == 0.0
    # The STT error itself is still visible in the mis-transcription rate.
    assert result.sub_metrics["mis_transcribed_token_rate"].score > 0


def test_clean_transcription_scores_zero():
    """No STT errors means nothing can propagate."""
    context = make_context(
        intended_user_turns={1: "I want to fly from Edinburgh to Toronto tomorrow."},
        transcribed_user_turns={1: "I want to fly from Edinburgh to Toronto tomorrow."},
    )
    result = compute(STTErrorPropagation(), context)

    assert result.score == 0.0
    assert result.normalized_score == 1.0
    assert result.details["propagated"] == 0
    assert "No transcription errors" in result.details["notes"][0]


def test_no_tool_calls_is_noted():
    """STT errors with no tool calls cannot corrupt anything."""
    context = make_context(tool_params=[], tool_responses=[])
    result = compute(STTErrorPropagation(), context)

    assert result.score == 0.0
    assert result.details["mis_transcribed_tokens"] == 1
    assert result.details["unattributed"] == 1
    assert "No tool-call arguments" in result.details["notes"][0]


def test_unrelated_tool_arguments_do_not_match():
    """Corrupted words that never appear in tool arguments stay unattributed."""
    context = make_context(
        tool_params=[{"tool_name": "search_flights", "tool_parameters": {"destination_city": "Tokyo"}}]
    )
    result = compute(STTErrorPropagation(), context)

    assert result.details["unattributed"] == 1
    assert result.details["propagated"] == 0
    # Rate keys stay stable (zero) so aggregated columns don't drop out.
    assert result.sub_metrics is not None
    assert set(result.sub_metrics) == {"num_propagated", "propagation_rate", "mis_transcribed_token_rate"}
