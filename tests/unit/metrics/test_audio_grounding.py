"""Tests for the audio_grounding diagnostic metric."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import eva.metrics  # noqa: F401  # importing the package registers every metric module
from eva.metrics.base import MetricContext
from eva.metrics.diagnostic.audio_grounding import AudioGroundingMetric
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType
from eva.models.results import MetricScore

from .conftest import make_judge_metric, make_metric_context


def make_judge_response(turns: list[dict]) -> str:
    """Create a JSON judge response with a ``turns`` wrapper."""
    return json.dumps({"turns": turns})


# User turn 0 is delivered sarcastically (cross-modal disagreement); user turn 2 is neutral.
TRACE = [
    {"role": "user", "content": "Oh great, wonderful, exactly what I needed", "type": "intended", "turn_id": 0},
    {"role": "assistant", "content": "Wonderful! Glad that worked out.", "type": "transcribed", "turn_id": 1},
    {"role": "user", "content": "Yeah, sure, go ahead and rebook it", "type": "intended", "turn_id": 2},
    {"role": "assistant", "content": "Done, you're all set!", "type": "transcribed", "turn_id": 3},
]

# Same conversation, but the user simulator also emitted a transcribed variant of turn 0.
TRACE_WITH_DUPLICATES = TRACE + [
    {"role": "user", "content": "Oh great, wonderful, exactly what I needed", "type": "transcribed", "turn_id": 0},
]


def _default_context(**overrides):
    """Context for audio grounding tests."""
    defaults = {
        "audio_user_path": "/fake/audio_user.wav",
        "pipeline_type": PipelineType.S2S,
        "conversation_trace": TRACE,
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


async def _run_compute(metric: AudioGroundingMetric, context: MetricContext) -> MetricScore:
    """Run compute() with the audio plumbing mocked out."""
    with patch.object(metric, "load_role_audio", return_value=MagicMock()):
        with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
            return await metric.compute(context)


@pytest.fixture
def metric():
    return make_judge_metric(
        AudioGroundingMetric,
        mock_llm=True,
        logger_name="test_audio_grounding",
    )


class TestRegistration:
    def test_registered_via_diagnostic_package(self):
        """The diagnostic package import wires the metric into the global registry."""
        assert get_global_registry().get("audio_grounding") is AudioGroundingMetric

    def test_opt_in_only(self, metric):
        """Excluded from default metric runs — must be requested via --metrics."""
        assert metric.exclude_from_default_metrics is True
        assert "audio_grounding" not in get_global_registry().list_metrics()

    def test_attributes(self, metric):
        assert metric.name == "audio_grounding"
        assert metric.category == "diagnostic"
        assert metric.version == "v0.1"
        assert PipelineType.S2S in metric.supported_pipeline_types
        assert PipelineType.AUDIO_LLM in metric.supported_pipeline_types
        assert PipelineType.CASCADE not in metric.supported_pipeline_types


class TestTraceFormatting:
    def test_user_and_assistant_lines_with_turn_ids(self, metric):
        formatted, user_turn_ids = metric._format_conversation_trace(_default_context())
        lines = formatted.split("\n")
        assert lines[0] == "Turn 0 - User: Oh great, wonderful, exactly what I needed"
        assert lines[1] == "Turn 1 - Assistant: Wonderful! Glad that worked out."
        assert user_turn_ids == [0, 2]

    def test_duplicate_role_turn_entries_deduplicated(self, metric):
        formatted, user_turn_ids = metric._format_conversation_trace(
            _default_context(conversation_trace=TRACE_WITH_DUPLICATES)
        )
        assert formatted.count("Turn 0 - User:") == 1
        assert user_turn_ids == [0, 2]

    def test_empty_trace(self, metric):
        formatted, user_turn_ids = metric._format_conversation_trace(_default_context(conversation_trace=[]))
        assert formatted == ""
        assert user_turn_ids == []


class TestCompute:
    def test_trapped_conflict_turn(self, metric):
        """One conflict turn taken as a trap -> zero grounding accuracy, full trap rate."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 0,
                        "has_disagreement": True,
                        "dimension": "emotion_state",
                        "surface_interpretation": "pleased",
                        "audio_interpretation": "bitter sarcasm",
                        "assistant_response": "transcript_trap",
                        "explanation": "Assistant responded cheerfully to a sarcastic turn.",
                    },
                    {
                        "turn_id": 2,
                        "has_disagreement": False,
                        "dimension": None,
                        "surface_interpretation": None,
                        "audio_interpretation": "neutral consent",
                        "assistant_response": "audio_grounded",
                        "explanation": "Assistant proceeded as asked.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert result.skipped is False
        assert result.error is None
        assert result.details["num_conflict"] == 1
        assert result.details["num_trapped"] == 1
        assert result.details["conflicts_by_dimension"] == {"emotion_state": [0]}
        assert result.sub_metrics is not None
        assert result.sub_metrics["trap_rate"].score == 1.0
        assert result.sub_metrics["disagreement_rate"].score == 0.5
        assert result.sub_metrics["consistent_accuracy"].score == 1.0

    def test_grounded_conflict_turn(self, metric):
        """Assistant acknowledges the incongruent affect -> perfect conflict accuracy."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 0,
                        "has_disagreement": True,
                        "dimension": "conversational_intent",
                        "surface_interpretation": "pleased",
                        "audio_interpretation": "bitter sarcasm",
                        "assistant_response": "audio_grounded",
                        "explanation": "Assistant acknowledged the frustration.",
                    },
                    {
                        "turn_id": 2,
                        "has_disagreement": False,
                        "dimension": None,
                        "surface_interpretation": None,
                        "audio_interpretation": "neutral consent",
                        "assistant_response": "unclear",
                        "explanation": "Purely transactional reply.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert result.score == 1.0
        assert result.sub_metrics is not None
        assert result.sub_metrics["trap_rate"].score == 0.0
        assert result.sub_metrics["consistent_accuracy"].score == 0.0

    def test_no_disagreement_skips(self, metric):
        """A conversation with no cross-modal disagreement has nothing to measure."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 0,
                        "has_disagreement": False,
                        "dimension": None,
                        "surface_interpretation": None,
                        "audio_interpretation": "genuine relief",
                        "assistant_response": "audio_grounded",
                        "explanation": "Appropriate response.",
                    },
                    {
                        "turn_id": 2,
                        "has_disagreement": False,
                        "dimension": None,
                        "surface_interpretation": None,
                        "audio_interpretation": "neutral consent",
                        "assistant_response": "audio_grounded",
                        "explanation": "Appropriate response.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert result.score is None
        assert result.normalized_score is None
        assert result.skipped is True
        assert result.error is None
        # No conflict turns -> no trap rate, but the conversation-level rates are still reported.
        assert result.sub_metrics is not None
        assert "trap_rate" not in result.sub_metrics
        assert result.sub_metrics["disagreement_rate"].score == 0.0
        assert result.sub_metrics["consistent_accuracy"].score == 1.0

    def test_unknown_dimension_not_tallied(self, metric):
        """Dimensions outside the paper's five stay visible in per-turn details but aren't tallied."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 0,
                        "has_disagreement": True,
                        "dimension": "sarcasm_level",
                        "surface_interpretation": "pleased",
                        "audio_interpretation": "bitter sarcasm",
                        "assistant_response": "audio_grounded",
                        "explanation": "Acknowledged the sarcasm.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert result.details["conflicts_by_dimension"] == {}
        assert result.details["per_turn"][0]["dimension"] == "sarcasm_level"

    def test_invalid_assistant_response_excluded(self, metric, caplog):
        """A conflict turn whose response cannot be classified is dropped, not coerced."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 0,
                        "has_disagreement": True,
                        "dimension": "emotion_state",
                        "assistant_response": "kind_of_grounded",
                        "explanation": "Malformed judge output.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert "Invalid assistant_response" in caplog.text
        assert result.skipped is True
        assert result.score is None
        assert result.details["num_rated"] == 0

    def test_turn_id_outside_trace_skipped(self, metric):
        """Judge entries for turns that were not sent are ignored."""
        metric.llm_client.generate_text.return_value = (
            make_judge_response(
                [
                    {
                        "turn_id": 7,
                        "has_disagreement": True,
                        "dimension": "emotion_state",
                        "assistant_response": "transcript_trap",
                        "explanation": "Not a real turn.",
                    },
                ]
            ),
            None,
        )
        result = asyncio.run(_run_compute(metric, _default_context()))

        assert result.details["num_rated"] == 0
        assert result.skipped is True


class TestErrorHandling:
    def test_no_user_audio_returns_error(self, metric):
        result = asyncio.run(metric.compute(_default_context(audio_user_path=None)))
        assert result.score == 0.0
        assert "No user audio" in result.error

    def test_no_user_turns_returns_error(self, metric):
        trace = [{"role": "assistant", "content": "Hello?", "type": "transcribed", "turn_id": 1}]
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            result = asyncio.run(metric.compute(_default_context(conversation_trace=trace)))
        assert result.score == 0.0
        assert "No user turns" in result.error

    def test_no_judge_response_returns_error(self, metric):
        metric.llm_client.generate_text.return_value = (None, None)
        result = asyncio.run(_run_compute(metric, _default_context()))
        assert result.score == 0.0
        assert result.error == "No response from judge"

    def test_unparsable_judge_response_returns_error(self, metric):
        metric.llm_client.generate_text.return_value = ("I could not evaluate this.", None)
        result = asyncio.run(_run_compute(metric, _default_context()))
        assert result.score == 0.0
        assert result.error == "No turns in judge response"

    def test_exception_returns_error_score(self, metric):
        with patch.object(metric, "load_role_audio", side_effect=RuntimeError("boom")):
            result = asyncio.run(metric.compute(_default_context()))
        assert result.score == 0.0
        assert "boom" in result.error
