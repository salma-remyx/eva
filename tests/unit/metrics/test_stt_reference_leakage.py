"""Tests for STT reference-leakage diagnostic metric.

Covers both probe families (directive leakage, truncation recovery) plus the
package wiring that registers the metric.
"""

import pytest

import eva.metrics.diagnostic  # noqa: F401  (import side effect: registers metrics)
from eva.metrics.diagnostic.stt_reference_leakage import STTReferenceLeakageMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


def test_metric_registered_via_diagnostic_package():
    """The diagnostic package import wires the metric into the global registry."""
    cls = get_global_registry().get("stt_reference_leakage")
    assert cls is STTReferenceLeakageMetric


@pytest.mark.asyncio
async def test_directive_leakage_flagged():
    """A non-speech directive recovered verbatim in the transcript is flagged."""
    context = make_metric_context(
        intended_user_turns={
            0: "Hi [slow] I need help with my flight",
            1: "It is flight four fifty to Boston",
        },
        transcribed_user_turns={
            0: "Hi slow I need help with my flight",
            1: "It is flight 450 to Boston",
        },
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is None
    assert score.score == 0.5  # 1 of 2 turns flagged
    assert score.details["flagged_turn_ids"] == [0]
    assert score.details["per_turn_evidence"][0]["leaked_directives"] == ["slow"]
    assert score.sub_metrics["directive_leakage_rate"].score == 0.5
    # No barge-in turns → no truncation sub-metric
    assert "truncation_recovery_rate" not in score.sub_metrics


@pytest.mark.asyncio
async def test_framework_labels_are_not_flagged():
    """Framework-inserted interruption labels appear on both sides and must not count."""
    context = make_metric_context(
        intended_user_turns={0: "[assistant interrupts] I need to change my flight"},
        transcribed_user_turns={0: "[assistant interrupts] I need to change my flight"},
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is None
    # No probe surface at all (no directives, no barge-in turns) → skipped, not "clean"
    assert score.skipped is True
    assert score.score is None


@pytest.mark.asyncio
async def test_truncation_recovery_flagged_on_barge_in_turn():
    """Full intended-utterance recovery on a barge-in turn is flagged, digit runs recorded."""
    context = make_metric_context(
        intended_user_turns={
            0: "My confirmation code is 4821 please rebook",
            1: "Thanks, that is all",
        },
        transcribed_user_turns={
            # STT wrote the digits as words; WER normalization converges both sides
            0: "My confirmation code is four eight two one please rebook",
            1: "Thanks, that is all",
        },
        assistant_interrupted_turns={0},
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is None
    assert score.score == 0.5  # 1 of 2 turns flagged
    evidence = score.details["per_turn_evidence"][0]
    assert evidence["interrupted"] is True
    assert evidence["coverage"] == 1.0
    assert evidence["digit_runs"] == ["4821"]
    assert score.sub_metrics["truncation_recovery_rate"].score == 1.0
    assert score.sub_metrics["truncation_recovery_rate"].details["num_eligible"] == 1


@pytest.mark.asyncio
async def test_partial_recovery_on_barge_in_turn_not_flagged():
    """A transcript that only heard the pre-barge-in prefix is legitimate."""
    context = make_metric_context(
        intended_user_turns={0: "My confirmation code is 4821 please rebook"},
        transcribed_user_turns={0: "My confirmation code"},
        assistant_interrupted_turns={0},
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is None
    assert score.score == 0.0
    assert score.details["flagged_turn_ids"] == []
    assert score.sub_metrics["truncation_recovery_rate"].score == 0.0


@pytest.mark.asyncio
async def test_clean_run_scores_zero_when_probe_surface_exists():
    """Directives present but never leaked → clean 0.0, not skipped."""
    context = make_metric_context(
        intended_user_turns={0: "Hi [slow] I need help with my flight"},
        transcribed_user_turns={0: "Hi I need help with my flight"},
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is None
    assert score.skipped is False
    assert score.score == 0.0
    assert score.details["num_directive_probe_turns"] == 1


@pytest.mark.asyncio
async def test_no_common_turns_returns_error():
    """No overlapping turns → error result, matching stt_wer's convention."""
    context = make_metric_context(
        intended_user_turns={0: "Hello"},
        transcribed_user_turns={1: "Hi"},
    )
    score = await STTReferenceLeakageMetric().compute(context)

    assert score.error is not None
    assert "No user turns" in score.error
