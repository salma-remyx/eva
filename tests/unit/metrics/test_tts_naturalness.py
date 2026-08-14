"""Tests for the tts_naturalness diagnostic metric.

Adapted from "Beyond Naturalness: Probing Automated Text-To-Speech Evaluators
on Linguistically Grounded Dimensions" (arXiv:2608.09930). The metric reuses
the SpeechFidelityBaseMetric AudioJudge plumbing and surfaces the paper's 10
naturalness dimensions as per-dimension rate sub-metrics.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import eva.metrics.diagnostic  # noqa: F401  -- triggers @register_metric via the package import
from eva.metrics.diagnostic.tts_naturalness import TTSNaturalnessMetric, _NATURALNESS_DIMENSIONS
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType

from .conftest import make_judge_metric, make_metric_context

# Expected per-dimension sub-metric dict keys ("<dim>_rate"); each value's
# .name is namespaced under the parent metric (e.g. "tts_naturalness.intonation_rate").
_DIMENSION_RATE_KEYS = {f"{dim}_rate" for dim in _NATURALNESS_DIMENSIONS}


def make_judge_response(turns: list[dict]) -> str:
    """Create a JSON judge response with a ``turns`` wrapper."""
    return json.dumps({"turns": turns})


@pytest.fixture
def metric():
    return make_judge_metric(
        TTSNaturalnessMetric,
        mock_llm=True,
        logger_name="test_tts_naturalness",
    )


def _default_context(**overrides):
    """Context with two assistant turns for naturalness tests."""
    defaults = {
        "transcribed_user_turns": {0: "Hi", 1: "Help me"},
        "transcribed_assistant_turns": {0: "Hello", 1: "Sure"},
        "intended_user_turns": {0: "Hi", 1: "Help me"},
        "intended_assistant_turns": {0: "Hello there", 1: "Sure thing"},
        "audio_assistant_path": "/fake/audio_assistant.wav",
        "audio_user_path": "/fake/audio_user.wav",
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestRegistration:
    """Verify the metric is wired into the registry via the diagnostic package import."""

    def test_registered_by_name(self):
        """Importing eva.metrics.diagnostic registers tts_naturalness (the __init__ wiring)."""
        assert get_global_registry().get("tts_naturalness") is TTSNaturalnessMetric

    def test_opt_in_not_in_default_metrics(self):
        """Naturalness is diagnostic/opt-in: it must not appear in the default metric list."""
        assert "tts_naturalness" not in get_global_registry().list_metrics()


class TestClassAttributes:
    """Verify subclass metadata mirrors the tts_fidelity shape (complementary, diagnostic)."""

    def test_attributes(self, metric):
        assert metric.name == "tts_naturalness"
        assert metric.category == "diagnostic"
        assert metric.role == "assistant"
        assert metric.rating_scale == (0, 1)
        assert metric.exclude_from_pass_at_k is True
        assert metric.exclude_from_default_metrics is True
        assert PipelineType.CASCADE in metric.supported_pipeline_types
        assert PipelineType.AUDIO_LLM in metric.supported_pipeline_types


class TestPromptRubric:
    """The 10-dimension rubric loads from the dedicated naturalness_rubric namespace."""

    def test_prompt_contains_all_dimensions(self, metric):
        prompt = metric.get_judge_prompt(
            intended_turns_formatted="Turn 0: Hello there",
            expected_language="English",
        )
        # Every dimension key appears in the rubric instructions.
        for dim in _NATURALNESS_DIMENSIONS:
            assert dim in prompt
        # Rendered (not still a template placeholder).
        assert "{intended_turns_formatted}" not in prompt
        assert "Hello there" in prompt


class TestNoAudio:
    @pytest.mark.asyncio
    async def test_no_audio_returns_error(self, metric):
        context = _default_context(audio_assistant_path=None)
        result = await metric.compute(context)
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "No assistant audio" in result.error


class TestCompute:
    @pytest.mark.asyncio
    async def test_all_natural_perfect_score(self, metric):
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Natural"},
                {"turn_id": 1, "rating": 1, "explanation": "Natural"},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert result.score == 1.0
        assert result.normalized_score == 1.0
        assert result.details["num_evaluated"] == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_dimension_tags_produce_sub_metrics(self, metric):
        """One turn flagged for two prosodic dimensions -> 0.5 rate on each, 0.0 on the rest."""
        response = make_judge_response(
            [
                {
                    "turn_id": 0,
                    "rating": 0,
                    "explanation": "Flat intonation and uneven pace",
                    "failure_modes": ["intonation", "speech_rate"],
                },
                {"turn_id": 1, "rating": 1, "explanation": "Natural", "failure_modes": []},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert result.sub_metrics is not None
        # Exactly one sub-metric per naturalness dimension, named under the parent metric.
        assert set(result.sub_metrics.keys()) == _DIMENSION_RATE_KEYS
        assert result.sub_metrics["intonation_rate"].score == 0.5
        assert result.sub_metrics["speech_rate_rate"].score == 0.5
        assert result.sub_metrics["phonetic_accuracy_rate"].score == 0.0
        assert result.sub_metrics["human_plausibility_rate"].score == 0.0
        assert result.sub_metrics["intonation_rate"].name == "tts_naturalness.intonation_rate"
        assert result.sub_metrics["intonation_rate"].details == {
            "count": 1,
            "num_rated": 2,
            "turn_ids": [0],
        }
        assert result.details["per_turn_failure_modes"] == {
            0: ["intonation", "speech_rate"],
            1: [],
        }

    @pytest.mark.asyncio
    async def test_no_tags_all_dimensions_zero(self, metric):
        """Natural turns with no failure_modes -> all dimension rates 0.0 but still surfaced."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Natural"},
                {"turn_id": 1, "rating": 1, "explanation": "Natural"},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert set(result.sub_metrics.keys()) == _DIMENSION_RATE_KEYS
        for key in _DIMENSION_RATE_KEYS:
            assert result.sub_metrics[key].score == 0.0

    @pytest.mark.asyncio
    async def test_unknown_dimension_ignored(self, metric):
        """Dimensions the judge invents are stored in details but produce no extra sub-metric."""
        response = make_judge_response(
            [
                {
                    "turn_id": 0,
                    "rating": 0,
                    "explanation": "Made up",
                    "failure_modes": ["something_new", "intonation"],
                },
                {"turn_id": 1, "rating": 1, "explanation": "Natural", "failure_modes": []},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert "something_new" in result.details["per_turn_failure_modes"][0]
        assert set(result.sub_metrics.keys()) == _DIMENSION_RATE_KEYS
        assert result.sub_metrics["intonation_rate"].score == 0.5
