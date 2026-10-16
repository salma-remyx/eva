"""Tests for FloorControlMetric.

Event model (adapted from MultiPartyTurnBench, arXiv:2610.20172):
  - Agent overlaps split into floor takeovers (user speech ends within
    TAKEOVER_MAX_TAIL_MS = 500ms of the agent's entry) and non-taking
    overlaps (user talks on past it).
  - User barge-ins split into takeovers (≥ BACKCHANNEL_MAX_SPEECH_MS = 1000ms
    of user speech) and backchannels (< 1000ms).
  - Agent behavior per barge-in: yielded when its speech ended within
    YIELD_WINDOW_MS = 1000ms of the barge-in onset, held otherwise; barge-ins
    where the agent had already stopped (no floor to yield) are excluded.
  - selective_accuracy = balanced accuracy over the measurable classes and is
    the main score; fp_constrained_takeover_recall sweeps the yield threshold
    at FP_BUDGET = 0.2 on backchannels.
"""

import logging

import pytest

import eva.metrics.diagnostic  # noqa: F401  (exercises the registry wiring in the package __init__)
from eva.metrics.diagnostic.floor_control import FloorControlMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = FloorControlMetric()
    m.logger = logging.getLogger("test_floor_control")
    return m


# ---------- Registry wiring (call-site integration) ----------


class TestRegistryWiring:
    def test_registered_via_diagnostic_package(self):
        """The package __init__ import registers the metric in the global registry."""
        assert get_global_registry().get("floor_control") is FloorControlMetric

    def test_runs_by_default(self):
        """A diagnostic like response_speed: available in the default metric list."""
        assert "floor_control" in get_global_registry().list_metrics()

    def test_version_stamped(self, metric):
        """Concrete metric classes must carry a version (signature drift test enforces it)."""
        assert metric.version == "v0.1"


# ---------- Agent overlap classification ----------


class TestAgentOverlapClassification:
    @pytest.mark.asyncio
    async def test_takeover_vs_overlap_only(self, metric):
        """One overlap that takes the floor, one the user talks through."""
        context = make_metric_context(
            # Turn 1: agent enters at 11.8 while the user (10-12) is finishing — the user's
            # remaining speech after entry is 200ms ≤ 500ms → takeover.
            # Turn 2: agent enters at 21.0 but the user (20-24) continues for 3s → overlap_only.
            audio_timestamps_user_turns={1: [(10.0, 12.0)], 2: [(20.0, 24.0)]},
            audio_timestamps_assistant_turns={1: [(11.8, 15.0)], 2: [(21.0, 22.0)]},
            assistant_interrupted_turns={1, 2},
        )
        result = await metric.compute(context)
        assert result.error is None
        assert result.skipped is False
        sub = result.sub_metrics
        assert sub["agent_interruption.takeover_rate"].score == pytest.approx(0.5, abs=1e-3)
        assert sub["agent_interruption.takeover_count"].score == pytest.approx(1.0)
        # Signed entries: -200ms (entered 200ms before user end) and -3000ms.
        assert sub["agent_interruption.mean_signed_entry_ms"].score == pytest.approx(-1600.0, abs=1e-3)
        # No user barge-ins → no measurable selective behavior → no main score.
        assert result.score is None
        assert "user_interruption.yield_accuracy" not in sub
        events = result.details["agent_overlap_events"]
        assert events[1]["class"] == "takeover"
        assert events[2]["class"] == "overlap_only"
        assert events[1]["signed_entry_ms"] == pytest.approx(-200.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_no_agent_events_no_agent_sub_metrics(self, metric):
        """Clean turns produce no agent-overlap sub-metrics."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.5)], 2: [(7.0, 8.5)]},
        )
        result = await metric.compute(context)
        assert result.skipped is True
        assert result.score is None


# ---------- User barge-in classification and selective handling ----------


class TestUserBargeInClassification:
    @pytest.mark.asyncio
    async def test_selective_interruption_handling(self, metric):
        """Yield to a genuine takeover, hold through one backchannel, fail on another."""
        context = make_metric_context(
            audio_timestamps_user_turns={
                5: [(38.0, 39.5)],
                6: [(43.5, 45.5)],  # 2.0s of speech → takeover
                7: [(48.5, 49.0)],
                8: [(53.5, 53.9)],  # 0.4s of speech → backchannel
                9: [(58.0, 59.0)],
                10: [(61.0, 61.5)],  # 0.5s of speech → backchannel
            },
            audio_timestamps_assistant_turns={
                5: [(40.0, 44.0)],  # still speaking at 43.5, stops at 44 → 500ms overlap → yielded
                6: [(46.0, 48.0)],
                7: [(50.0, 54.0)],
                8: [(55.0, 56.0)],
                9: [(60.0, 64.0)],
                10: [(65.0, 66.0)],
            },
            user_interrupted_turns={6, 8, 10},
        )
        result = await metric.compute(context)
        assert result.error is None
        sub = result.sub_metrics
        # Turn 6 (takeover): agent stopped 500ms after the barge-in → yielded.
        assert sub["user_interruption.yield_accuracy"].score == pytest.approx(1.0, abs=1e-3)
        # Backchannels: turn 8 agent stopped after 500ms (wrongly yielded), turn 10 after
        # 3000ms (held) → hold_accuracy 0.5.
        assert sub["user_interruption.hold_accuracy"].score == pytest.approx(0.5, abs=1e-3)
        # Balanced accuracy over the two classes is the main score.
        assert sub["user_interruption.selective_accuracy"].score == pytest.approx(0.75, abs=1e-3)
        assert result.score == pytest.approx(0.75, abs=1e-3)
        assert result.normalized_score == pytest.approx(0.75, abs=1e-3)
        # The single takeover yields (500ms) exactly like the turn-8 backchannel does, so no
        # threshold separates them within the 20% FP budget → recall 0.
        assert sub["user_interruption.fp_constrained_takeover_recall"].score == pytest.approx(0.0, abs=1e-3)
        events = result.details["user_barge_in_events"]
        assert events[6]["class"] == "takeover"
        assert events[6]["agent_action"] == "yielded"
        assert events[8]["class"] == "backchannel"
        assert events[8]["agent_action"] == "yielded"
        assert events[10]["class"] == "backchannel"
        assert events[10]["agent_action"] == "held"

    @pytest.mark.asyncio
    async def test_barge_in_after_agent_already_stopped_is_excluded(self, metric):
        """A barge-in after the agent's speech ended has no floor to yield — excluded from accuracies."""
        context = make_metric_context(
            audio_timestamps_user_turns={
                1: [(0.0, 1.0)],
                2: [(20.0, 21.0)],  # 1.0s of speech → takeover class, but agent stopped at 12.0
            },
            audio_timestamps_assistant_turns={
                1: [(10.0, 12.0)],
                2: [(22.0, 23.0)],
            },
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        assert result.error is None
        event = result.details["user_barge_in_events"][2]
        assert event["class"] == "takeover"
        assert "agent_action" not in event
        # No measurable agent action → no accuracies and no main score, but not skipped.
        assert result.score is None
        assert result.skipped is False
        assert result.sub_metrics is None or "user_interruption.yield_accuracy" not in result.sub_metrics


# ---------- FP-constrained recall sweep ----------


class TestRecallAtFpBudget:
    def test_perfect_separation_full_recall(self, metric):
        """Takeover yields decisively, backchannels never yielded → recall 1.0 at zero FP."""
        recall = metric._recall_at_fp_budget([200.0, 400.0], [2500.0, 3000.0])
        assert recall == pytest.approx(1.0, abs=1e-3)

    def test_indistinguishable_yields_zero_recall(self, metric):
        """Event and non-event yields identical → any firing blows the budget."""
        recall = metric._recall_at_fp_budget([500.0], [500.0, 3000.0])
        assert recall == pytest.approx(0.0, abs=1e-3)

    def test_threshold_between_values(self, metric):
        """One backchannel fires identically to a takeover; recall still reaches 1.0 when the
        budget admits it (1 fire in 5 backchannels = 0.2 ≤ FP_BUDGET)."""
        recall = metric._recall_at_fp_budget([100.0, 900.0, 2000.0], [100.0, 3000.0, 4000.0, 5000.0, 6000.0])
        assert recall == pytest.approx(1.0, abs=1e-3)

    def test_undefined_without_both_classes(self, metric):
        assert metric._recall_at_fp_budget([], [100.0]) is None
        assert metric._recall_at_fp_budget([100.0], []) is None


# ---------- Empty records ----------


class TestEmptyRecords:
    @pytest.mark.asyncio
    async def test_no_evaluable_turns_skips(self, metric):
        result = await metric.compute(make_metric_context())
        assert result.skipped is True
        assert result.score is None
        assert result.error is None
