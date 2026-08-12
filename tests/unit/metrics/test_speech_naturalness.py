"""Integration tests for the speech_naturalness diagnostic metric.

These tests import the *existing* diagnostic package and global metric registry
(the wiring call site edited in ``src/eva/metrics/diagnostic/__init__.py``) to
prove the metric is discovered and registered — not just self-test the new file.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import eva.metrics.diagnostic as diagnostic_pkg
from eva.metrics.diagnostic.speech_naturalness import SpeechNaturalnessMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

# The ten linguistically grounded dimensions surfaced as sub-metrics, in order.
_EXPECTED_DIMENSIONS = (
    "mispronunciation",
    "unnatural_intonation",
    "inappropriate_stress",
    "unnatural_pacing",
    "pause_placement",
    "disfluency",
    "unnatural_voice_quality",
    "inexpressive_affect",
    "phrasing_boundary_error",
    "register_tone_mismatch",
)


def _make_judge_response(turns: list[dict]) -> str:
    """Wrap per-turn judge output in the ``{"turns": [...]}`` envelope compute() parses."""
    return json.dumps({"turns": turns})


def _default_context(**overrides):
    """Context with two assistant turns and an assistant audio path."""
    defaults = {
        "intended_assistant_turns": {0: "Hello there.", 1: "Sure thing, I can help."},
        "intended_user_turns": {0: "Hi", 1: "Help me"},
        "transcribed_assistant_turns": {0: "Hello there.", 1: "Sure thing."},
        "transcribed_user_turns": {0: "Hi", 1: "Help me"},
        "audio_assistant_path": "/fake/audio_assistant.wav",
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


@pytest.fixture
def metric():
    return make_judge_metric(SpeechNaturalnessMetric, mock_llm=True, logger_name="test_speech_naturalness")


class TestRegistration:
    """Prove the registry-add wiring in diagnostic/__init__.py exposes the metric."""

    def test_exported_from_diagnostic_package(self):
        # The import line added to the package __init__ makes the name resolvable + listed.
        assert "speech_naturalness" in diagnostic_pkg.__all__
        assert hasattr(diagnostic_pkg, "speech_naturalness")

    def test_registered_in_global_registry(self):
        # Importing the diagnostic package fires @register_metric; the global registry
        # (the non-new call-site module) must resolve the metric by name.
        assert get_global_registry().get("speech_naturalness") is SpeechNaturalnessMetric

    def test_metric_metadata(self, metric):
        assert metric.name == "speech_naturalness"
        assert metric.category == "diagnostic"
        assert metric.role == "assistant"
        assert metric.rating_scale == (0, 1)
        assert metric.exclude_from_default_metrics is True


class TestPromptRendering:
    """The in-module template must render with the variables compute() passes."""

    def test_prompt_renders_and_lists_all_dimensions(self, metric):
        prompt = metric.get_judge_prompt(
            intended_turns_formatted="Turn 0: Hello there.\nTurn 1: Sure thing.",
            expected_language="English",
        )
        # Every dimension key is named in the prompt so the judge returns scorable tags.
        for dimension in _EXPECTED_DIMENSIONS:
            assert dimension in prompt
        assert "English" in prompt


class TestNaturalnessSubMetrics:
    """compute() surfaces one binary rate sub-metric per dimension."""

    @pytest.mark.asyncio
    async def test_dimensional_breakdown(self, metric):
        """One unnatural turn (two dimensions flagged) + one natural turn."""
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "rating": 0,
                    "explanation": "Robotic pitch and rushed timing.",
                    "failure_modes": ["unnatural_intonation", "unnatural_pacing"],
                },
                {"turn_id": 1, "rating": 1, "explanation": "Natural.", "failure_modes": []},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        # Parent score = mean of per-turn ratings (0 + 1) / 2.
        assert result.score == 0.5
        assert result.normalized_score == 0.5
        assert result.error is None
        # Exactly ten sub-metrics, one per dimension.
        assert result.sub_metrics is not None
        assert set(result.sub_metrics.keys()) == {f"{d}_rate" for d in _EXPECTED_DIMENSIONS}
        # Flagged dimensions: 1 of 2 rated turns -> 0.5 rate.
        assert result.sub_metrics["unnatural_intonation_rate"].score == 0.5
        assert result.sub_metrics["unnatural_pacing_rate"].score == 0.5
        # Unflagged dimensions: 0.0.
        assert result.sub_metrics["mispronunciation_rate"].score == 0.0
        assert result.sub_metrics["register_tone_mismatch_rate"].score == 0.0
        # Sub-metric name carries the parent prefix.
        assert result.sub_metrics["unnatural_pacing_rate"].name == "speech_naturalness.unnatural_pacing_rate"

    @pytest.mark.asyncio
    async def test_all_natural_turns_yield_zero_rates(self, metric):
        """No defective dimensions -> every dimension rate is 0.0 but still surfaced."""
        response = _make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Natural."},
                {"turn_id": 1, "rating": 1, "explanation": "Natural."},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert result.score == 1.0
        assert result.sub_metrics is not None
        for dimension in _EXPECTED_DIMENSIONS:
            assert result.sub_metrics[f"{dimension}_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_unknown_dimension_ignored(self, metric):
        """A dimension the judge invents is preserved in details but creates no sub-metric."""
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "rating": 0,
                    "explanation": "Weird.",
                    "failure_modes": ["something_not_in_schema", "disfluency"],
                },
                {"turn_id": 1, "rating": 1, "explanation": "Natural.", "failure_modes": []},
            ]
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert set(result.sub_metrics.keys()) == {f"{d}_rate" for d in _EXPECTED_DIMENSIONS}
        assert "something_not_in_schema" in result.details["per_turn_failure_modes"][0]
        assert result.sub_metrics["disfluency_rate"].score == 0.5
