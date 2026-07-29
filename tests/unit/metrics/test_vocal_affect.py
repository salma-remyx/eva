"""Tests for the vocal_affect experience metric (audio judge of vocal expressiveness).

Covers the registry wiring (the import in eva.metrics.experience.__init__) and the
compute() behaviour against a mocked audio judge.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import eva.metrics.experience  # noqa: F401  -- existing module; its __init__ wires the registration
from eva.metrics.experience.vocal_affect import VocalAffectMetric
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType

from .conftest import make_judge_metric, make_metric_context


def _judge_response(rating: int, *, transcript_driven: bool = False) -> str:
    """Build a vocal_affect judge JSON response."""
    return json.dumps(
        {
            "rating": rating,
            "explanation": "Mock explanation of vocal delivery.",
            "dimensions": {
                "flat_or_monotone": {"flagged": rating == 1, "evidence": "pitch variation"},
                "transcript_driven": {"flagged": transcript_driven, "evidence": "read-aloud feel"},
                "affect_mismatch": {"flagged": False, "evidence": "tone matches content"},
                "overacted": {"flagged": False, "evidence": "natural range"},
            },
        }
    )


def _context(**overrides):
    defaults = {
        "audio_assistant_path": "/fake/audio_assistant.wav",
        "pipeline_type": PipelineType.S2S,
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


@pytest.fixture
def metric():
    return make_judge_metric(VocalAffectMetric, mock_llm=True, logger_name="test_vocal_affect")


class TestRegistration:
    def test_registered_via_experience_package(self):
        """Importing the experience package registers the metric by name (wiring edit)."""
        registry = get_global_registry()
        assert registry.get("vocal_affect") is VocalAffectMetric

    def test_select_by_name_creates_instance(self):
        """The real select-by-name path the MetricsRunner uses returns our metric."""
        instance = get_global_registry().create("vocal_affect", config={})
        assert isinstance(instance, VocalAffectMetric)


class TestClassAttributes:
    def test_attributes(self, metric):
        assert metric.name == "vocal_affect"
        assert metric.category == "experience"
        assert metric.version == "v0.1"
        assert metric.rating_scale == (1, 3)
        assert PipelineType.S2S in metric.supported_pipeline_types
        assert PipelineType.AUDIO_LLM in metric.supported_pipeline_types
        assert PipelineType.CASCADE not in metric.supported_pipeline_types


class TestCompute:
    @pytest.mark.asyncio
    async def test_no_audio_returns_error(self, metric):
        context = _context(audio_assistant_path=None)
        result = await metric.compute(context)
        assert result.score == 0.0
        assert "No assistant audio" in result.error

    @pytest.mark.asyncio
    async def test_no_judge_response_returns_error(self, metric):
        metric.llm_client.generate_text.return_value = (None, None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_context())
        assert result.score == 0.0
        assert result.error == "No response from judge"

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_error(self, metric):
        metric.llm_client.generate_text.return_value = ("not json", None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_context())
        assert result.score == 0.0
        assert result.error == "Failed to parse judge response"

    @pytest.mark.asyncio
    async def test_expressive_rating_scores_full(self, metric):
        metric.llm_client.generate_text.return_value = (_judge_response(3), None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_context())
        assert result.error is None
        assert result.score == 3.0
        assert result.normalized_score == 1.0
        # No dimensions flagged -> every rate sub-metric is 0.0
        assert result.sub_metrics["transcript_driven_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_transcript_driven_flagged_as_sub_metric(self, metric):
        metric.llm_client.generate_text.return_value = (_judge_response(1, transcript_driven=True), None)
        with patch.object(metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
                result = await metric.compute(_context())
        assert result.score == 1.0
        assert result.normalized_score == 0.0
        # The paper's core failure mode ("transcript-driven") surfaces as a flagged sub-metric.
        assert result.sub_metrics["transcript_driven_rate"].score == 1.0
        assert result.sub_metrics["flat_or_monotone_rate"].score == 1.0
        assert result.sub_metrics["affect_mismatch_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_exception_returns_error_score(self, metric):
        with patch.object(metric, "load_role_audio", side_effect=RuntimeError("boom")):
            result = await metric.compute(_context())
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "boom" in result.error
