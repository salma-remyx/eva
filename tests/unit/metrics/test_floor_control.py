"""Tests for FloorControlMetric.

Event detection (all from per-turn audio timestamps, greeting turn 0 excluded):
  Pause:           gap between consecutive user segments of one turn in
                   [PAUSE_MIN_MS=500, PAUSE_MAX_MS=3000]; takeover when an
                   assistant segment in the same turn intersects the gap.
  Backchannel:     user-interrupt turn with total user speech ≤ BACKCHANNEL_MAX_MS=1500;
                   resumed when an assistant segment in the turn ends after the
                   user's last speech.
  Interruption:    user-interrupt turn with user speech > 1500ms; responded under
                   the same speaks-after test, plus the delay to the first
                   assistant segment starting at/after the user's last speech
                   (0.0 when an assistant segment spans it — the agent never stopped).

Score = mean of the available components:
  pause_handling = 1 − takeover_rate, interruption_response, backchannel_resume.
No events at all ⇒ score=None with skipped=True.
"""

import logging

import pytest

from eva.metrics.experience.floor_control import FloorControlMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = FloorControlMetric()
    m.logger = logging.getLogger("test_floor_control")
    return m


# ---------- Wiring ----------


class TestRegistration:
    def test_registered_via_experience_package(self):
        """Importing eva.metrics.experience (whose __init__ wires floor_control) registers the metric."""
        import eva.metrics.experience  # noqa: F401
        from eva.metrics.registry import get_global_registry

        assert get_global_registry().get("floor_control") is FloorControlMetric

    def test_opt_in_not_in_default_metrics(self):
        """floor_control is excluded from the default run — default metric list is unchanged."""
        from eva.models.config import _get_all_metrics

        assert "floor_control" not in _get_all_metrics()


# ---------- Pause detection ----------


class TestPauseDetection:
    @pytest.mark.parametrize(
        "gap_ms, expected_events",
        [
            (400, 0),  # ordinary VAD splitting, not a deliberate pause
            (500, 1),  # inclusive lower bound
            (800, 1),
            (3000, 1),  # inclusive upper bound
            (3500, 0),  # user plausibly finished the utterance
        ],
    )
    async def test_pause_gap_boundaries(self, metric, gap_ms, expected_events):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0), (2.0 + gap_ms / 1000, 4.0)]},
            audio_timestamps_assistant_turns={1: [(4.5, 6.0)]},
        )
        pauses = metric._find_pause_events(context)
        assert len(pauses) == expected_events
        if expected_events:
            assert pauses[0]["pause_ms"] == pytest.approx(gap_ms, abs=1e-3)

    async def test_single_segment_turn_has_no_pauses(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 4.0)]},
            audio_timestamps_assistant_turns={1: [(4.5, 6.0)]},
        )
        assert metric._find_pause_events(context) == []

    async def test_greeting_turn_pause_ignored(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={0: [(0.0, 1.0), (1.8, 2.0)], 1: [(3.0, 5.0)]},
            audio_timestamps_assistant_turns={0: [(2.2, 2.8)], 1: [(5.3, 6.0)]},
        )
        assert metric._find_pause_events(context) == []

    async def test_takeover_requires_assistant_speech_inside_the_pause(self, metric):
        user = {1: [(0.0, 2.0), (2.8, 4.0)]}  # 800ms pause in [2.0, 2.8]
        # Agent waits until the user resumed → held the floor.
        held = make_metric_context(
            audio_timestamps_user_turns=user,
            audio_timestamps_assistant_turns={1: [(4.3, 6.0)]},
        )
        # Agent starts inside the pause window → takeover.
        took_over = make_metric_context(
            audio_timestamps_user_turns=user,
            audio_timestamps_assistant_turns={1: [(2.4, 3.5)]},  # starts inside the pause window
        )

        held_events = metric._find_pause_events(held)
        metric._classify_pause_events(held, held_events)
        assert [e["took_over"] for e in held_events] == [False]

        over_events = metric._find_pause_events(took_over)
        metric._classify_pause_events(took_over, over_events)
        assert [e["took_over"] for e in over_events] == [True]


# ---------- Backchannel / interruption split ----------


class TestInterruptTurns:
    @pytest.mark.parametrize(
        "speech_ms, expect_backchannel",
        [
            (600, True),
            (1500, True),  # inclusive bound
            (1600, False),  # substantive interruption
        ],
    )
    async def test_split_by_user_speech_duration(self, metric, speech_ms, expect_backchannel):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 10.0 + speech_ms / 1000)]},
            audio_timestamps_assistant_turns={1: [(9.0, 14.0)]},
            user_interrupted_turns={1},
        )
        backchannels, interruptions = metric._split_interrupt_turns(context)
        assert (len(backchannels), len(interruptions)) == ((1, 0) if expect_backchannel else (0, 1))

    async def test_interrupt_turn_without_user_audio_dropped(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={1: [(9.0, 14.0)]},
            user_interrupted_turns={1},
        )
        assert metric._split_interrupt_turns(context) == ([], [])

    async def test_resume_detected_when_agent_speaks_past_user_end(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 10.6)]},  # 600ms backchannel
            audio_timestamps_assistant_turns={1: [(9.0, 12.0)]},
            user_interrupted_turns={1},
        )
        backchannels, _ = metric._split_interrupt_turns(context)
        assert backchannels[0]["resumed"] is True

    async def test_no_resume_when_agent_went_silent(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 10.6)]},
            audio_timestamps_assistant_turns={1: [(9.0, 10.3)]},  # ends before the backchannel does
            user_interrupted_turns={1},
        )
        backchannels, _ = metric._split_interrupt_turns(context)
        assert backchannels[0]["resumed"] is False

    async def test_response_delay_measured_from_user_end(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(16.0, 18.5)]},  # 2500ms substantive interruption
            audio_timestamps_assistant_turns={1: [(15.0, 15.8), (19.0, 21.0)]},
            user_interrupted_turns={1},
        )
        _, interruptions = metric._split_interrupt_turns(context)
        assert interruptions[0]["responded"] is True
        assert interruptions[0]["response_delay_ms"] == pytest.approx(500.0, abs=1e-3)

    async def test_response_delay_zero_when_agent_never_stopped(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(16.0, 18.5)]},
            audio_timestamps_assistant_turns={1: [(15.0, 20.0)]},  # spans the user's end
            user_interrupted_turns={1},
        )
        _, interruptions = metric._split_interrupt_turns(context)
        assert interruptions[0]["responded"] is True
        assert interruptions[0]["response_delay_ms"] == pytest.approx(0.0, abs=1e-6)

    async def test_no_response_when_agent_stays_silent(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(16.0, 18.5)]},
            audio_timestamps_assistant_turns={1: [(15.0, 15.8)]},
            user_interrupted_turns={1},
        )
        _, interruptions = metric._split_interrupt_turns(context)
        assert interruptions[0]["responded"] is False
        assert interruptions[0]["response_delay_ms"] is None


# ---------- End-to-end compute ----------


class TestCompute:
    async def test_all_three_behaviours_compose(self, metric):
        """One held pause + one takeover + resumed backchannel + answered interruption."""
        context = make_metric_context(
            audio_timestamps_user_turns={
                1: [(0.0, 2.0), (2.8, 4.0)],  # 800ms pause, held
                2: [(7.0, 8.0), (9.0, 10.0)],  # 1000ms pause, taken over
                3: [(12.0, 12.6)],  # 600ms backchannel
                4: [(16.0, 18.5)],  # 2500ms interruption
            },
            audio_timestamps_assistant_turns={
                1: [(4.3, 6.0)],
                2: [(8.4, 10.5)],  # starts inside the pause window
                3: [(11.0, 14.0)],
                4: [(15.0, 15.8), (19.0, 21.0)],
            },
            user_interrupted_turns={3, 4},
        )
        result = await metric.compute(context)

        assert result.score == pytest.approx(0.8333, abs=1e-3)
        assert result.normalized_score == result.score
        assert result.skipped is False

        assert result.details["components"] == {
            "pause_handling": 0.5,
            "backchannel_resume": 1.0,
            "interruption_response": 1.0,
        }
        assert result.details["num_pause_events"] == 2
        assert result.details["num_interruption_events"] == 1
        assert result.details["num_backchannel_events"] == 1

        sub = result.sub_metrics
        assert sub["pause_handling.takeover_rate"].score == pytest.approx(0.5)
        assert sub["pause_handling.takeover_rate"].normalized_score == pytest.approx(0.5)
        assert sub["pause_handling.mean_pause_ms"].score == pytest.approx(900.0)
        assert sub["pause_handling.mean_pause_ms"].normalized_score is None
        assert sub["interruption.response_score"].score == pytest.approx(1.0)
        assert sub["interruption.mean_response_delay_ms"].score == pytest.approx(500.0)
        assert sub["backchannel.resume_score"].score == pytest.approx(1.0)
        assert sub["pause_handling.num_pauses"].score == 2.0
        assert sub["interruption.num_events"].score == 1.0
        assert sub["backchannel.num_events"].score == 1.0

    async def test_perfect_pause_handling_only(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0), (2.8, 4.0)]},
            audio_timestamps_assistant_turns={1: [(4.3, 6.0)]},
        )
        result = await metric.compute(context)
        assert result.score == pytest.approx(1.0)
        assert result.sub_metrics["pause_handling.takeover_rate"].score == pytest.approx(0.0)

    async def test_unanswered_backchannel_scores_zero(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(10.0, 10.6)]},
            audio_timestamps_assistant_turns={1: [(9.0, 10.3)]},
            user_interrupted_turns={1},
        )
        result = await metric.compute(context)
        assert result.score == pytest.approx(0.0)
        assert result.sub_metrics["backchannel.resume_score"].score == pytest.approx(0.0)

    async def test_skipped_when_no_events(self, metric):
        """Strictly alternating single-segment turns: nothing to judge, so skip instead of scoring."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(5.0, 7.0)]},
            audio_timestamps_assistant_turns={1: [(2.2, 4.0)], 2: [(7.3, 9.0)]},
        )
        result = await metric.compute(context)
        assert result.score is None
        assert result.normalized_score is None
        assert result.skipped is True
        assert result.error is None
        assert result.details["num_pause_events"] == 0

    async def test_empty_context_skips(self, metric):
        result = await metric.compute(make_metric_context())
        assert result.skipped is True
        assert result.score is None
