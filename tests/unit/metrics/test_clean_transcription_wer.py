"""Tests for CleanTranscriptionWERMetric.

Exercises the metric through the global metric registry (the same path the
production MetricsRunner uses: ``registry.create(name)`` -> ``compute``) rather
than calling the class directly, so the registry wiring is covered end to end.
"""

import pytest

from eva.metrics.diagnostic.clean_transcription_wer import (
    CleanTranscriptionWERMetric,
    clean_disfluencies,
)
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


@pytest.fixture
def metric():
    """Resolve the metric by name from the global registry, as the runner does."""
    instance = get_global_registry().create("clean_transcription_wer")
    assert isinstance(instance, CleanTranscriptionWERMetric)
    return instance


def test_registry_resolves_clean_transcription_wer():
    """The metric is registered under its capability name and is instantiable by name."""
    registry = get_global_registry()
    assert registry.get("clean_transcription_wer") is CleanTranscriptionWERMetric
    assert "clean_transcription_wer" in registry.list_metrics()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("um I uh want to book a flight", "I want to book a flight"),
        ("I I want the the flight", "I want the flight"),
        ("I want to--", "I want to"),
        ("uh um hmm yes", "yes"),
        ("Book a flight to London", "Book a flight to London"),
        ("", ""),
    ],
)
def test_clean_disfluencies_strips_fillers_and_repetitions(raw, expected):
    assert clean_disfluencies(raw) == expected


@pytest.mark.asyncio
async def test_metric_ignores_disfluency_inflation(metric):
    """Fillers/repetitions present only in the transcript must not count as errors."""
    context = make_metric_context(
        intended_user_turns={0: "I want to book a flight to London"},
        transcribed_user_turns={0: "um I I want to uh book a flight to London"},
    )
    result = await metric.compute(context)

    assert result.error is None
    # Cleaning collapses the transcript onto the reference -> no real error remains.
    assert result.score == 0.0
    assert result.normalized_score == 1.0
    # Verbatim WER still sees the stray filler/repetition tokens -> positive inflation.
    assert result.details["verbatim_wer"] > 0.0
    assert result.details["disfluency_inflation"] == result.details["verbatim_wer"]
    assert result.details["num_turns"] == 1


@pytest.mark.asyncio
async def test_metric_still_catches_real_substitution(metric):
    """Cleaning must not mask a genuine word-recognition error."""
    context = make_metric_context(
        intended_user_turns={0: "book a flight to London"},
        transcribed_user_turns={0: "book a flight to Paris"},
    )
    result = await metric.compute(context)

    assert result.error is None
    assert result.score > 0.0
    assert result.normalized_score < 1.0
    # No disfluency here, so clean and verbatim error rates agree.
    assert result.details["wer"] == result.details["verbatim_wer"]


@pytest.mark.asyncio
async def test_metric_scores_identical_clean_turns_as_zero(metric):
    context = make_metric_context(
        intended_user_turns={0: "hello world", 1: "goodbye"},
        transcribed_user_turns={0: "hello world", 1: "goodbye"},
    )
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == 0.0
    assert result.details["verbatim_wer"] == 0.0
    assert result.details["num_turns"] == 2
