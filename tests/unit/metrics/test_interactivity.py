"""Tests for InteractivityMetric.

Interactivity (Mode 2 adapted port of arxiv:2407.06479) detects two micro-level
engagement features from the paper's taxonomy — backchannels and acknowledgment
openers — over transcript turns already in MetricContext, and aggregates them into
an agent-engagement-rate score.

Detector rules:
  backchannel     — <= MAX_BACKCHANNEL_TOKENS (3) tokens, all in the nod lexicon.
  acknowledgment  — opens with an acknowledgment phrase, or a safe single-word
                    lead ("sure", "great", "understood", ...) followed by content.
  substantive     — anything else.

Scoring:
  agent_engagement_rate = agent engagement turns / agent speakable turns
  interactivity.score   = min(1, agent_engagement_rate / ENGAGEMENT_TARGET_RATE)
  ENGAGEMENT_TARGET_RATE = 0.4  (so 40%+ agent engagement -> 1.0)

Turn 0 (greeting) and empty turns are excluded from the denominators.

Sub-metrics (flat): agent_engagement.rate, user_engagement.rate, backchannel.rate,
acknowledgment.rate.
"""

import logging

import pytest

from eva.metrics.experience.interactivity import InteractivityMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = InteractivityMetric()
    m.logger = logging.getLogger("test_interactivity")
    return m


# ---------- Integration: registration via the non-new experience package ----------


def test_metric_registered_via_experience_package():
    """Importing eva.metrics.experience wires the metric into the global registry."""
    import eva.metrics.experience  # noqa: F401  — existing (non-new) call-site module

    registry = get_global_registry()
    assert registry.get("interactivity") is InteractivityMetric
    assert registry.create("interactivity") is not None


# ---------- Detector unit tests ----------


class TestClassification:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Pure backchannels (short, all nod tokens).
            ("yeah", "backchannel"),
            ("ok", "backchannel"),
            ("Okay", "backchannel"),
            ("mhm mhm yeah", "backchannel"),
            ("sure", "backchannel"),
            ("right", "backchannel"),
            # Acknowledgment openers (lead a substantive turn).
            ("Got it, let me check that for you.", "acknowledgment"),
            ("I see, so the flight is at 5pm.", "acknowledgment"),
            ("Sure, here are your options.", "acknowledgment"),
            ("Understood, I will do that now.", "acknowledgment"),
            ("Makes sense.", "acknowledgment"),
            ("Of course!", "acknowledgment"),
            ("Great, let me help with that.", "acknowledgment"),
            ("Thanks for your patience.", "acknowledgment"),
            # Substantive (no engagement opener) — incl. the "right now" guard.
            ("The answer is 42.", "substantive"),
            ("Right now the price is five dollars.", "substantive"),
            ("I will book the flight for you.", "substantive"),
            ("", "empty"),
        ],
    )
    def test_classify_turn(self, metric, text, expected):
        assert metric._classify_turn(text) == expected


# ---------- End-to-end compute via MetricContext (the integration surface) ----------


@pytest.mark.asyncio
async def test_compute_engagement_drives_score(metric):
    """An engaging agent scores high; a one-sided agent scores 0 (exercises compute)."""
    ctx = make_metric_context(
        intended_assistant_turns={
            0: "Hello, how can I help you today?",  # greeting — excluded
            1: "Got it, let me look up flight 102 for you.",  # engagement
            2: "The flight arrives at 5pm.",  # substantive
            3: "Sure, I can book that seat for you.",  # engagement
        },
        intended_user_turns={
            0: "Hi",
            1: "I need flight info.",
            2: "Book seat 12A.",
            3: "Yeah, ok.",  # pure backchannel
        },
    )

    result = await metric.compute(ctx)

    assert result.name == "interactivity"
    assert result.score == pytest.approx(1.0, abs=1e-3)  # 2/3 engagement >> 0.4 target
    assert result.normalized_score == result.score
    # The greeting (turn 0) is excluded: 3 speakable agent turns, 2 engaging.
    assert result.details["counts"]["agent"]["speakable"] == 3
    assert result.details["counts"]["agent"]["engagement"] == 2
    assert result.details["counts"]["user"]["backchannel"] == 1
    # Sub-metrics present and well-formed.
    assert "agent_engagement.rate" in result.sub_metrics
    assert "backchannel.rate" in result.sub_metrics
    assert result.sub_metrics["agent_engagement.rate"].normalized_score == pytest.approx(2 / 3, abs=1e-3)


@pytest.mark.asyncio
async def test_compute_one_sided_agent_scores_zero(metric):
    """An agent that never acknowledges (all substantive) gets 0.0."""
    ctx = make_metric_context(
        intended_assistant_turns={
            0: "Hello.",
            1: "The flight is at 5pm.",
            2: "Seat 12A is booked.",
        },
        intended_user_turns={0: "Hi", 1: "When is the flight?", 2: "Book a seat."},
    )
    result = await metric.compute(ctx)
    assert result.score == 0.0
    assert result.details["agent_engagement_rate"] == 0.0


@pytest.mark.asyncio
async def test_compute_no_agent_turns_scores_zero(metric):
    """No agent turns to classify -> 0.0 (no interactivity signal), not an error."""
    ctx = make_metric_context(
        intended_assistant_turns={},
        intended_user_turns={0: "Hi", 1: "Are you there?"},
    )
    result = await metric.compute(ctx)
    assert result.score == 0.0
    assert result.error is None
