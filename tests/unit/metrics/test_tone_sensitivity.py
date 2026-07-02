"""Tests for the tone_sensitivity experience metric.

Exercises both the registration wiring (the metric is discoverable through the
shared global registry, proving the ``experience/__init__.py`` hook works) and
the compute() flow with a mocked audio judge.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType

from .conftest import make_judge_metric, make_metric_context

# A trace where the words are calm but delivery (in audio) carries distress.
DISTRESS_TRACE = [
    {"role": "user", "content": "No really, I'm fine, everything's fine.", "type": "intended", "turn_id": 0},
    {"role": "assistant", "content": "Great, I'll close your case then.", "type": "transcribed", "turn_id": 1},
]


def _default_context(**overrides):
    defaults = {
        "audio_user_path": "/fake/audio_user.wav",
        "pipeline_type": PipelineType.S2S,
        "conversation_trace": DISTRESS_TRACE,
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


@pytest.fixture
def metric():
    # Import lazily via the registry to prove the wiring registered it.
    metric_cls = get_global_registry().get("tone_sensitivity")
    assert metric_cls is not None
    return make_judge_metric(metric_cls, mock_llm=True, logger_name="test_tone_sensitivity")


def test_metric_registered_through_experience_package():
    """The experience/__init__.py hook must expose the metric in the global registry."""
    # Importing the metrics package (done transitively) triggers registration.
    import eva.metrics.experience  # noqa: F401

    registry = get_global_registry()
    assert "tone_sensitivity" in registry.get_all()
    cls = registry.get("tone_sensitivity")
    assert cls.category == "experience"


class TestClassAttributes:
    def test_attributes(self, metric):
        assert metric.name == "tone_sensitivity"
        assert metric.category == "experience"
        assert metric.rating_scale == (1, 3)


class TestSkip:
    @pytest.mark.asyncio
    async def test_no_user_audio_skips(self, metric):
        context = _default_context(audio_user_path=None)
        result = await metric.compute(context)
        assert result.skipped is True
        assert result.score is None

    @pytest.mark.asyncio
    async def test_no_delivery_signal_skips(self, metric):
        """When delivery adds nothing beyond the words, the record is excluded from scoring."""
        response = json.dumps({"delivery_conveys_signal": False, "rating": 3, "perceived_delivery": "neutral"})
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())
        assert result.skipped is True
        assert result.score is None
        assert result.error is None


class TestCompute:
    @pytest.mark.asyncio
    async def test_ignored_delivery_scores_low(self, metric):
        """Assistant acted on the words and ignored audible distress -> lowest rating + gap flag."""
        response = json.dumps(
            {
                "delivery_conveys_signal": True,
                "rating": 1,
                "perceived_delivery": "caller is crying",
                "explanation": "Assistant closed the case despite audible distress.",
                "dimensions": {
                    "emotional_intelligence_gap": {"flagged": True, "evidence": "closed case while user cried"}
                },
            }
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert result.score == 1.0
        assert result.normalized_score == 0.0
        assert result.details["emotional_intelligence_gap"] is True
        assert result.sub_metrics is not None
        assert result.sub_metrics["emotional_intelligence_gap_rate"].score == 1.0

    @pytest.mark.asyncio
    async def test_attended_delivery_scores_high(self, metric):
        response = json.dumps(
            {
                "delivery_conveys_signal": True,
                "rating": 3,
                "perceived_delivery": "caller sounds distressed",
                "explanation": "Assistant gently checked in on the caller's wellbeing.",
                "dimensions": {"emotional_intelligence_gap": {"flagged": False, "evidence": ""}},
            }
        )
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())

        assert result.score == 3.0
        assert result.normalized_score == 1.0
        assert result.details["emotional_intelligence_gap"] is False
        assert result.sub_metrics["emotional_intelligence_gap_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_no_judge_response_errors(self, metric):
        metric.llm_client.generate_text.return_value = (None, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())
        assert result.score == 0.0
        assert result.error == "No response from judge"

    @pytest.mark.asyncio
    async def test_invalid_rating_defaults_to_min(self, metric):
        response = json.dumps({"delivery_conveys_signal": True, "rating": 9, "perceived_delivery": "distress"})
        metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_default_context())
        # Out-of-range rating clamps to the minimum (1 -> normalized 0.0).
        assert result.score == 1.0
        assert result.normalized_score == 0.0
