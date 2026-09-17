"""Tests for TurnDecisionSensitivityMetric.

ECHO-style matched-contrast scoring (arXiv:2609.17360v1) over EVA artifacts:
  - Barge-in transcripts classify expected action (backchannel → Keep, else Yield).
  - Observed action comes from how long agent speech sustained past the barge-in.
  - Pair accuracy = yield_decision_accuracy * keep_decision_accuracy, so a
    constant-yield policy scores 0.0 even though the base turn_taking metric
    grades its yields as near-perfect.
"""

import logging

import pytest

from eva.metrics.base import MetricContext
from eva.metrics.experience import turn_taking
from eva.metrics.experience.turn_decision_sensitivity import TurnDecisionSensitivityMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context

SUBSTANTIVE_TRANSCRIPT = "Actually, I need to change the flight to tomorrow instead."
BACKCHANNEL_TRANSCRIPT = "yeah"


def _barge_in_context(agent_timestamps: dict[int, list[tuple[float, float]]]) -> MetricContext:
    """Context with two user barge-ins: turn 1 substantive, turn 2 backchannel.

    The user barges in at 10.0s (speaks until 14.0) and at 20.0s (backchannel
    until 20.4). ``agent_timestamps`` controls the agent's floor-holding speech,
    which is what distinguishes a yielder from a keeper.
    """
    return make_metric_context(
        audio_timestamps_user_turns={1: [(10.0, 14.0)], 2: [(20.0, 20.4)]},
        audio_timestamps_assistant_turns=agent_timestamps,
        user_interrupted_turns={1, 2},
        transcribed_user_turns={1: SUBSTANTIVE_TRANSCRIPT, 2: BACKCHANNEL_TRANSCRIPT},
    )


# A constant-yield agent stops talking almost immediately on every barge-in.
CONSTANT_YIELD_AGENT = {
    0: [(2.0, 10.3)],  # cut-off turn: stops 300ms after the 10.0 barge-in
    1: [(16.0, 19.9)],  # stops just before the 20.0 barge-in (sustain = 0)
    2: [(21.0, 24.0)],  # settled response after the backchannel — not floor-holding
}

# A context-sensitive agent yields on the substantive interruption but keeps
# talking through the backchannel.
CONTEXT_SENSITIVE_AGENT = {
    0: [(2.0, 10.4)],  # yields 400ms after the substantive barge-in
    1: [(16.0, 34.0)],  # talks through the backchannel (sustains ~14s past it)
}


@pytest.fixture
def metric():
    m = TurnDecisionSensitivityMetric()
    m.logger = logging.getLogger("test_turn_decision_sensitivity")
    return m


# ---------- Wiring ----------


class TestWiring:
    def test_registered_through_experience_package(self):
        """The experience package wires the metric into the global registry (default metric set).

        Registration happens only because ``eva.metrics.experience`` imports the
        module — nothing else in the tree does — so this fails if the wiring is lost.
        """
        registry = get_global_registry()
        assert registry.get("turn_decision_sensitivity") is TurnDecisionSensitivityMetric
        assert "turn_decision_sensitivity" in registry.list_metrics()


# ---------- Pair accuracy over matched contrasts ----------


class TestPairAccuracy:
    async def test_constant_yield_policy_scores_zero_pair_accuracy(self, metric):
        """A constant-yield agent fails every backchannel member of the contrast."""
        score = await metric.compute(_barge_in_context(CONSTANT_YIELD_AGENT))

        assert score.error is None
        assert score.skipped is False
        assert score.score == 0.0
        assert score.sub_metrics["yield_decision_accuracy"].score == 1.0
        assert score.sub_metrics["keep_decision_accuracy"].score == 0.0

    async def test_constant_yield_policy_still_scores_well_on_base_turn_taking(self, metric):
        """The ECHO headline, target-native: interruption-only evaluation overestimates.

        On the exact same record the existing turn_taking metric grades the agent's
        yields as near-perfect (it always expects a fast yield), while
        turn_decision_sensitivity exposes the fixed action preference.
        """
        context = _barge_in_context(CONSTANT_YIELD_AGENT)

        base = turn_taking.TurnTakingMetric()
        base.logger = logging.getLogger("test_turn_decision_sensitivity")
        base_score = await base.compute(context)

        yield_score = base_score.sub_metrics["user_interruption.mean_yield_score"]
        assert yield_score.score >= 0.9  # looks like an excellent yielder...

        sensitivity = await TurnDecisionSensitivityMetric().compute(context)
        assert sensitivity.score == 0.0  # ...but decides context-insensitively

    async def test_context_sensitive_agent_scores_full_pair_accuracy(self, metric):
        """Yielding on substantive speech while keeping through backchannels scores 1.0."""
        score = await metric.compute(_barge_in_context(CONTEXT_SENSITIVE_AGENT))

        assert score.error is None
        assert score.score == 1.0
        assert score.normalized_score == 1.0
        assert score.sub_metrics["yield_decision_accuracy"].score == 1.0
        assert score.sub_metrics["keep_decision_accuracy"].score == 1.0
        assert score.details["num_evaluated"] == 2

    async def test_record_without_contrast_is_skipped(self, metric):
        """Only substantive barge-ins → no matched contrast → score None, not 0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 14.0)]},
            audio_timestamps_assistant_turns={0: [(2.0, 10.4)], 1: [(15.0, 18.0)]},
            user_interrupted_turns={1},
            transcribed_user_turns={1: SUBSTANTIVE_TRANSCRIPT},
        )
        score = await metric.compute(context)

        assert score.score is None
        assert score.skipped is True
        assert "contrast" in score.details["reason"]
        # The single-class accuracy is still surfaced for analysis.
        assert score.sub_metrics["yield_decision_accuracy"].score == 1.0

    async def test_non_english_record_is_skipped(self, metric):
        """The backchannel lexicon is English-only in v0.1."""
        context = make_metric_context(language="fr", user_interrupted_turns={1})
        score = await metric.compute(context)

        assert score.score is None
        assert score.skipped is True
        assert score.details["language"] == "fr"

    async def test_events_missing_signals_are_reported_not_scored(self, metric):
        """Barge-ins without a transcript or timestamps count as not-applicable, not errors."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 14.0)], 2: [(20.0, 20.4)]},
            audio_timestamps_assistant_turns={0: [(2.0, 10.4)], 1: [(16.0, 34.0)]},
            user_interrupted_turns={1, 2},
            transcribed_user_turns={2: BACKCHANNEL_TRANSCRIPT},  # turn 1 has no transcript
        )
        score = await metric.compute(context)

        assert score.skipped is True  # only the backchannel class is decidable
        assert score.details["num_not_applicable"] == 1
        assert score.details["per_event"][1]["skipped"] == "no_user_transcript"


# ---------- Backchannel proxy ----------


class TestBackchannelProxy:
    @pytest.mark.parametrize(
        ("transcript", "expected"),
        [
            ("yeah", "keep"),
            ("mm-hmm, right", "keep"),
            ("Okay, go ahead please", "keep"),
            ("[user interrupts] sure", "keep"),  # processor annotations are stripped
            ("Mm, okay okay", "keep"),
            (SUBSTANTIVE_TRANSCRIPT, "yield"),
            ("yes, but change it to row twelve", "yield"),
            ("no wait, that's the wrong confirmation code", "yield"),
        ],
    )
    def test_expected_action_backchannel_proxy(self, transcript: str, expected: str):
        assert TurnDecisionSensitivityMetric._expected_action(transcript) == expected

    def test_decision_window_matches_turn_taking_yield_deadline(self):
        """The keep/yield boundary shares turn_taking's notion of slowest acceptable yield."""
        assert TurnDecisionSensitivityMetric.DECISION_WINDOW_MS == turn_taking.TurnTakingMetric.YIELD_HARD_MS
