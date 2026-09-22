"""Tests for the prosody_expressiveness metric (decoupled dimension judging).

Covers the D-LPJ decoupling properties adapted into this metric: no overall
verdict, null-masking of uncertain dimensions, independent per-dimension
sub-metrics, and the dimension_collapse_rate verdict-coupling diagnostic.

Async compute() paths are driven with asyncio.run() inside sync tests so the
suite runs identically with or without pytest-asyncio installed.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from eva.metrics.diagnostic.prosody_expressiveness import ProsodyExpressivenessMetric
from eva.metrics.registry import get_global_registry
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric

from .conftest import make_judge_metric, make_metric_context

DIMENSIONS = ProsodyExpressivenessMetric.dimensions


def make_prosody_response(turns: list[dict]) -> str:
    """Create a JSON prosody judge response with a ``turns`` wrapper."""
    return json.dumps({"turns": turns})


def dimension_turn(turn_id: int, **ratings) -> dict:
    """Build one judge turn item from kwargs like ``emotion=3, intonation=None``."""
    return {
        "turn_id": turn_id,
        "dimensions": {
            name: (
                {"rating": value, "explanation": f"{name} was {value}"}
                if value is not None
                else {"rating": None, "explanation": "uncertain"}
            )
            for name, value in ratings.items()
        },
    }


@pytest.fixture
def metric():
    return make_judge_metric(
        ProsodyExpressivenessMetric,
        mock_llm=True,
        logger_name="test_prosody_expressiveness",
    )


def _default_context(**overrides):
    """Context with default intended turns for prosody tests."""
    defaults = {
        "intended_assistant_turns": {0: "Hello [warm]", 1: "Sure thing [firm]"},
        "audio_assistant_path": "/fake/audio_assistant.wav",
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


def _compute(metric, response_text, context):
    """Run compute() with the audio pipeline mocked, like the fidelity tests."""
    metric.llm_client.generate_text.return_value = (response_text, None)
    with patch.object(metric, "load_role_audio", return_value=MagicMock()):
        with patch.object(metric, "encode_audio_segment", return_value="base64audio"):
            return asyncio.run(metric.compute(context))


class TestRegistryWiring:
    """The module registers into the global registry the runner/scripts use."""

    def test_registered_in_global_registry(self):
        registry = get_global_registry()
        assert registry.get("prosody_expressiveness") is ProsodyExpressivenessMetric
        instance = registry.create("prosody_expressiveness")
        assert isinstance(instance, ProsodyExpressivenessMetric)

    def test_reuses_speech_fidelity_audio_pipeline(self):
        # The metric inherits the stock Gemini audio-judge machinery
        # (role audio loading, silence trimming, _call_and_parse retries).
        assert issubclass(ProsodyExpressivenessMetric, SpeechFidelityBaseMetric)
        assert hasattr(ProsodyExpressivenessMetric, "_call_and_parse")

    def test_opt_in_not_in_default_metrics(self):
        assert "prosody_expressiveness" not in get_global_registry().list_metrics()
        assert ProsodyExpressivenessMetric.exclude_from_default_metrics is True


class TestPrompt:
    """The judge prompt loads from configs/prompts/ and encodes the decoupling rules."""

    def test_prompt_loads_and_is_decoupled(self, metric):
        prompt = metric.get_judge_prompt(intended_turns_formatted="Turn 0: Hello [warm]", expected_language="English")
        for dimension in DIMENSIONS:
            assert dimension in prompt
        assert "Turn 0: Hello [warm]" in prompt
        assert "English" in prompt
        # No overall verdict target, and uncertainty masking is instructed.
        assert "There is no overall rating for a turn." in prompt
        assert "null" in prompt


class TestCompute:
    """compute() aggregates decoupled per-dimension ratings."""

    def test_decoupled_dimensions_and_sub_metrics(self, metric):
        response = make_prosody_response(
            [
                dimension_turn(0, emotion=3, intonation=2, energy=3),
                dimension_turn(1, emotion=2, intonation=2, energy=2),
            ]
        )
        result = _compute(metric, response, _default_context())

        assert result.error is None
        # Parent: mean over per-turn dimension means (0-3 scale normalized to 0-1).
        assert result.normalized_score == pytest.approx((8 / 9 + 2 / 3) / 2, abs=1e-3)
        assert result.score == pytest.approx(14 / 6, abs=1e-3)
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["num_masked_ratings"] == 0
        assert result.details["per_turn_ratings"][0] == {"emotion": 3, "intonation": 2, "energy": 3}

        sub_metrics = result.sub_metrics
        assert sub_metrics is not None
        assert set(sub_metrics.keys()) == {*DIMENSIONS, "dimension_collapse_rate"}
        # Each dimension keeps its own score instead of collapsing to one value.
        assert sub_metrics["emotion"].score == pytest.approx(2.5)
        assert sub_metrics["emotion"].normalized_score == pytest.approx(5 / 6, abs=1e-3)
        assert sub_metrics["intonation"].score == pytest.approx(2.0)
        assert sub_metrics["intonation"].normalized_score == pytest.approx(2 / 3, abs=1e-3)
        assert sub_metrics["energy"].normalized_score == pytest.approx(5 / 6, abs=1e-3)
        assert sub_metrics["emotion"].name == "prosody_expressiveness.emotion"

    def test_collapsed_turn_flags_verdict_coupling(self, metric):
        """Turn 1 has all dimensions identical (coupled); turn 0 does not."""
        response = make_prosody_response(
            [
                dimension_turn(0, emotion=3, intonation=1, energy=3),
                dimension_turn(1, emotion=2, intonation=2, energy=2),
            ]
        )
        result = _compute(metric, response, _default_context())

        collapse = result.sub_metrics["dimension_collapse_rate"]
        assert collapse.score == pytest.approx(0.5)
        assert collapse.details["collapsed_turns"] == [1]
        assert collapse.details["num_rated"] == 2

    def test_null_dimension_is_masked_not_guessed(self, metric):
        """Uncertain (null) dimensions drop out of their dimension's aggregation."""
        response = make_prosody_response(
            [
                dimension_turn(0, emotion=None, intonation=3, energy=2),
                dimension_turn(1, emotion=1, intonation=None, energy=None),
            ]
        )
        result = _compute(metric, response, _default_context())

        assert result.error is None
        assert result.details["num_masked_ratings"] == 3
        assert result.details["per_turn_ratings"][0]["emotion"] is None
        assert result.details["per_turn_explanations"][0]["emotion"] == "uncertain"

        # Emotion only has evidence for turn 1 (rating 1 -> normalized 1/3);
        # turn 0's null must not be averaged in as a guess.
        emotion = result.sub_metrics["emotion"]
        assert emotion.score == pytest.approx(1.0)
        assert emotion.normalized_score == pytest.approx(1 / 3, abs=1e-3)
        assert emotion.details == {"num_rated": 1, "num_masked": 1}

        # Turn 0's two rated dimensions differ, so no collapse is flagged.
        assert result.sub_metrics["dimension_collapse_rate"].score == pytest.approx(0.0)

    def test_invalid_rating_masked_like_null(self, metric):
        response = make_prosody_response(
            [
                dimension_turn(0, emotion=5, intonation=3, energy=3),
                dimension_turn(1, emotion=1, intonation=1, energy=1),
            ]
        )
        result = _compute(metric, response, _default_context())

        assert result.details["num_masked_ratings"] == 1
        assert result.details["per_turn_ratings"][0]["emotion"] is None
        emotion = result.sub_metrics["emotion"]
        assert emotion.score == pytest.approx(1.0)
        assert emotion.details["num_masked"] == 1

    def test_no_audio_returns_error(self, metric):
        context = _default_context(audio_assistant_path=None)
        result = asyncio.run(metric.compute(context))
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "No assistant audio" in result.error

    def test_unparseable_response_returns_error(self, metric):
        result = _compute(metric, "the audio sounded expressive", _default_context())
        assert result.score == 0.0
        assert result.error == "No prosody dimension ratings parsed from judge response"
        assert result.sub_metrics is None
