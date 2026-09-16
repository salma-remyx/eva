"""Tests for the BackchannelHandlingMetric."""

import pytest

import eva.metrics.experience  # noqa: F401
from eva.metrics.base import MetricContext
from eva.metrics.experience.backchannel_handling import BackchannelHandlingMetric, _is_backchannel_transcript
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


def _ctx(
    user_turns: dict[int, list[tuple[float, float]]],
    assistant_turns: dict[int, list[tuple[float, float]]],
    transcripts: dict[int, str],
    interrupted: set[int],
) -> MetricContext:
    """Build a MetricContext with only the signals backchannel_handling consumes."""
    return make_metric_context(
        audio_timestamps_user_turns=user_turns,
        audio_timestamps_assistant_turns=assistant_turns,
        transcribed_user_turns=transcripts,
        user_interrupted_turns=interrupted,
    )


class TestBackchannelHandlingWiring:
    def test_registered_via_experience_package_import(self):
        """Importing eva.metrics.experience (the package __init__) registers the metric."""
        registry = get_global_registry()
        assert registry.get("backchannel_handling") is BackchannelHandlingMetric

    def test_instantiable_by_name_and_opt_in(self):
        """The registry creates it by name, and it is excluded from default metrics."""
        registry = get_global_registry()
        metric = registry.create("backchannel_handling")
        assert isinstance(metric, BackchannelHandlingMetric)
        assert "backchannel_handling" not in registry.list_metrics()


class TestBackchannelHandlingMetric:
    @pytest.mark.asyncio
    async def test_talked_through_backchannel_scores_one(self):
        """Assistant speech spanning the backchannel's end → 1.0."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(12.0, 12.6)]},
            assistant_turns={1: [(10.0, 20.0)], 2: [(20.0, 25.0)]},
            transcripts={2: "Mm-hmm!"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        assert result.score == 1.0
        assert result.skipped is False
        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "talked_through"
        sub = result.sub_metrics or {}
        assert sub["talked_through_rate"].score == 1.0
        assert sub["num_backchannels"].score == 1.0

    @pytest.mark.asyncio
    async def test_paused_and_resumed_backchannel_scores_zero(self):
        """Assistant stops during the backchannel, restarts 900ms later → 0.0.

        The reference ground truth is "continues without pausing", so a 900ms pause
        is a failure regardless of how promptly the restart happens.
        """
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 11.5)]},
            assistant_turns={1: [(10.0, 11.4)], 2: [(12.4, 15.0)]},
            transcripts={2: "yeah"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "resumed"
        assert ev["resume_ms"] == pytest.approx(900.0)
        assert result.score == 0.0
        sub = result.sub_metrics or {}
        assert sub["mean_resume_ms"].score == pytest.approx(900.0)

    @pytest.mark.asyncio
    async def test_even_a_quick_resume_scores_zero(self):
        """A near-immediate restart still paused → 0.0; there is no resume tolerance band."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 11.5)]},
            assistant_turns={1: [(10.0, 11.4)], 2: [(11.6, 15.0)]},
            transcripts={2: "yeah"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "resumed"
        assert ev["resume_ms"] == pytest.approx(100.0)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_abandoned_backchannel_scores_zero(self):
        """Assistant stops mid-response during the backchannel and never resumes → 0.0."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 12.6)]},
            assistant_turns={1: [(10.0, 12.0)]},
            transcripts={2: "mm hmm yeah"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "abandoned"
        assert result.score == 0.0
        sub = result.sub_metrics or {}
        assert sub["abandoned_rate"].score == 1.0

    @pytest.mark.asyncio
    async def test_mixed_events_average(self):
        """One talked-through (1.0) + one abandoned (0.0) event → mean 0.5."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(12.0, 12.6)], 4: [(21.0, 22.6)]},
            assistant_turns={1: [(10.0, 20.0)], 3: [(20.0, 22.0)]},
            transcripts={2: "mm-hmm", 4: "right right"},
            interrupted={2, 4},
        )

        result = await metric.compute(ctx)

        assert result.score == 0.5
        sub = result.sub_metrics or {}
        assert sub["talked_through_rate"].score == 0.5
        assert sub["barge_in_backchannel_rate"].score == 1.0

    @pytest.mark.asyncio
    async def test_backchannel_on_stopping_response_scores_zero(self):
        """Assistant speech ends just after the barge-in and nothing follows → 0.0.

        The paper defines no natural-end grace window: the acceptance rule only
        requires embedding in the ongoing utterance, and stopping is stopping.
        """
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 11.6)]},
            assistant_turns={1: [(10.0, 11.2)]},
            transcripts={2: "okay"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "abandoned"
        assert result.score == 0.0
        sub = result.sub_metrics or {}
        assert sub["abandoned_rate"].score == 1.0

    @pytest.mark.asyncio
    async def test_real_interruption_is_not_a_backchannel(self):
        """A barge-in with content words stays an interruption for turn_taking → skipped."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 13.0)]},
            assistant_turns={1: [(10.0, 14.0)], 2: [(15.0, 18.0)]},
            transcripts={2: "wait, actually change it to Friday"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        assert result.score is None
        assert result.skipped is True
        assert result.details["num_backchannels"] == 0

    @pytest.mark.asyncio
    async def test_utterance_over_word_threshold_is_not_backchannel(self):
        """Seven continuer words exceed the paper's six-word threshold → not a backchannel."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 14.0)]},
            assistant_turns={1: [(10.0, 15.0)]},
            transcripts={2: "yeah yeah yeah yeah yeah yeah yeah"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        assert result.score is None
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_six_word_backchannel_is_detected(self):
        """A six-word continuer utterance is inside the paper's acceptance threshold."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={2: [(11.0, 13.0)]},
            assistant_turns={1: [(10.0, 15.0)]},
            transcripts={2: "mm hmm yeah right okay sure"},
            interrupted={2},
        )

        result = await metric.compute(ctx)

        assert result.score == 1.0
        ev = result.details["per_turn_evidence"][2]
        assert ev["reason"] == "talked_through"

    @pytest.mark.asyncio
    async def test_no_barge_ins_skips(self):
        """A conversation without user interruptions has nothing to score."""
        metric = BackchannelHandlingMetric()
        ctx = _ctx(
            user_turns={1: [(5.0, 6.0)]},
            assistant_turns={1: [(6.5, 9.0)]},
            transcripts={1: "hello"},
            interrupted=set(),
        )

        result = await metric.compute(ctx)

        assert result.score is None
        assert result.skipped is True
        assert result.error is None


class TestBackchannelLexicon:
    def test_pure_continuers_match(self):
        max_words = BackchannelHandlingMetric.BACKCHANNEL_MAX_WORDS
        assert max_words == 6  # the paper's English short-length threshold
        assert _is_backchannel_transcript("Mm-hmm!", max_words) is True
        assert _is_backchannel_transcript("yeah, right", max_words) is True
        assert _is_backchannel_transcript("Got it", max_words) is True

    def test_word_threshold_boundary(self):
        max_words = BackchannelHandlingMetric.BACKCHANNEL_MAX_WORDS
        assert _is_backchannel_transcript("mm hmm yeah right okay sure", max_words) is True
        assert _is_backchannel_transcript("mm hmm yeah right okay sure too", max_words) is False

    def test_content_fails(self):
        assert _is_backchannel_transcript("change it to Friday", 6) is False
        assert _is_backchannel_transcript("", 6) is False
        assert _is_backchannel_transcript(None, 6) is False
