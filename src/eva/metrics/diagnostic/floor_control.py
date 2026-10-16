"""Floor-control diagnostic: genuine takeovers vs. confusable non-events.

Adapted from MultiPartyTurnBench (MPTB, arXiv:2610.20172), which evaluates floor
control as two distinct detection tasks — End-of-Turn (EOT) vs. temporary hold,
and Interruption (INT) vs. non-floor-taking speech — and argues that conflating
genuine events with confusable non-events hides the real failure mode: agents
enter turns fine but handle interruptions selectively at near-chance level.

This metric applies that lens to the dyadic conversations eva already records,
using only the per-turn channel-separated audio timestamps on ``MetricContext``
(the same signals ``turn_taking`` consumes — no processor contract change):

- Agent overlaps (``assistant_interrupted_turns``) are split into **takeovers**
  (the user stopped speaking shortly after the agent's entry — the floor
  actually transferred) and **non-taking overlaps** (the user talked on past
  the agent's entry — the agent's overlapped speech did not take the floor).
  ``turn_taking`` reports these as one binary flag.
- User barge-ins (``user_interrupted_turns``) are split into genuine
  **takeovers** (enough user speech to claim the floor) and **backchannels**
  (brief non-floor-taking vocalizations). The agent's response is scored per
  class: yield to takeovers, hold through backchannels — MPTB's "selective
  interruption handling", summarized as a balanced ``selective_accuracy``.
- A false-positive-constrained recall sweep over the agent's yield latency
  reports how much takeover recall survives a bounded rate of yielding to
  backchannels (MPTB's causal metric shape).
- Signed turn-entry latency on agent overlaps (negative = entry anticipation
  depth into the user's speech) replaces an unsigned magnitude.

MPTB conditions on exact event annotations; the parameter-free timestamp
heuristics below (tail-duration, speech-duration, yield-window thresholds) are
the target-native substitute for those annotations.

Debug metric for diagnosing model behavior — not directly used in final
evaluation scores.
"""

import statistics
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.experience.turn_taking import TurnTakingMetric
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class FloorControlMetric(CodeMetric):
    """Floor-control event classification and selective interruption handling.

    Splits the interruption flags already computed by the processor into
    MPTB-style event/non-event classes and reports, per direction:

    - ``agent_interruption.takeover_rate`` — fraction of agent-overlap turns
      that genuinely took the floor (the harmful subset; non-taking overlaps
      are the confusable non-event).
    - ``agent_interruption.mean_signed_entry_ms`` — mean signed latency of the
      agent's first overlapping entry relative to the user's last speech end
      (always ≤ 0; deeper anticipation = more negative).
    - ``user_interruption.yield_accuracy`` / ``hold_accuracy`` — correct agent
      behavior per event class (yield to genuine takeovers, hold through
      backchannels).
    - ``user_interruption.selective_accuracy`` — balanced accuracy across both
      classes; also the main score. Near-chance when the agent yields to
      everything or nothing, which is exactly the MPTB failure mode.
    - ``user_interruption.fp_constrained_takeover_recall`` — takeover recall
      surviving a false-positive budget on backchannels.
    """

    name = "floor_control"
    category = "diagnostic"
    description = (
        "Diagnostic metric: classifies interruptions into genuine floor takeovers vs. "
        "confusable non-events (non-taking overlaps, backchannels) and scores selective "
        "interruption handling, FP-constrained takeover recall, and signed entry latency"
    )
    exclude_from_pass_at_k = True
    version = "v0.1"

    # An agent overlap counts as a takeover when the user speaks at most this long
    # after the agent's first overlapping entry; a longer continuation means the
    # overlap failed to take the floor (non-floor-taking speech).
    TAKEOVER_MAX_TAIL_MS: float = 500.0

    # A user barge-in with less total speech than this is a backchannel
    # (non-floor-taking); at or above it, a genuine attempt to take the floor.
    BACKCHANNEL_MAX_SPEECH_MS: float = 1000.0

    # The agent "yielded" to a barge-in when its speech ended within this window of
    # the barge-in onset; it "held" when it kept speaking past the window.
    YIELD_WINDOW_MS: float = 1000.0

    # False-positive budget (fraction of backchannels wrongly yielded to) for the
    # FP-constrained takeover recall sweep.
    FP_BUDGET: float = 0.2

    @staticmethod
    def _first_overlapping_agent_start(
        u_segs: list[tuple[float, float]],
        a_segs: list[tuple[float, float]],
    ) -> float | None:
        """Start time of the first assistant segment that overlaps any user segment (s)."""
        entries = [
            a_start
            for a_start, a_end in a_segs
            if any(min(u_end, a_end) - max(u_start, a_start) > 0.001 for u_start, u_end in u_segs)
        ]
        return min(entries) if entries else None

    @staticmethod
    def _speech_after_ms(segs: list[tuple[float, float]], t_start: float) -> float:
        """Total speech duration (ms) in ``segs`` falling after ``t_start``."""
        return sum(max(0.0, end - max(start, t_start)) for start, end in segs) * 1000

    @staticmethod
    def _total_speech_ms(segs: list[tuple[float, float]]) -> float:
        """Total speech duration (ms) across all segments."""
        return sum(end - start for start, end in segs) * 1000

    @classmethod
    def _classify_agent_overlap(cls, context: MetricContext, turn_id: int) -> dict[str, Any] | None:
        """Classify one agent-overlap turn as a floor takeover or a non-taking overlap.

        Returns None when no assistant segment actually overlaps the user's speech in
        this turn (robustness against flags set on turns with degenerate timestamps).
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        a_segs = context.audio_timestamps_assistant_turns.get(turn_id)
        if not u_segs or not a_segs:
            return None
        entry = cls._first_overlapping_agent_start(u_segs, a_segs)
        if entry is None:
            return None
        user_tail_ms = cls._speech_after_ms(u_segs, entry)
        return {
            "class": "takeover" if user_tail_ms <= cls.TAKEOVER_MAX_TAIL_MS else "overlap_only",
            "signed_entry_ms": round((entry - u_segs[-1][1]) * 1000, 3),
            "user_tail_ms": round(user_tail_ms, 3),
        }

    @classmethod
    def _classify_user_barge_in(cls, context: MetricContext, turn_id: int) -> dict[str, Any] | None:
        """Classify one user barge-in turn as a takeover or a backchannel.

        ``agent_overlap_after_barge_in_ms`` mirrors turn_taking's yield signal (previous
        turn's last assistant end minus this turn's first user start) but keeps the
        sign: a value ≤ 0 means the agent had already stopped before the barge-in, so
        there was no floor to hold or yield — such turns are excluded from the
        accuracy denominators (``agent_action`` is unset).
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        if not u_segs:
            return None
        user_speech_ms = cls._total_speech_ms(u_segs)
        event: dict[str, Any] = {
            "class": "backchannel" if user_speech_ms < cls.BACKCHANNEL_MAX_SPEECH_MS else "takeover",
            "user_speech_ms": round(user_speech_ms, 3),
        }
        prev_a_segs = context.audio_timestamps_assistant_turns.get(turn_id - 1)
        if prev_a_segs:
            overlap_ms = (prev_a_segs[-1][1] - u_segs[0][0]) * 1000
            event["agent_overlap_after_barge_in_ms"] = round(overlap_ms, 3)
            if overlap_ms > 0:
                event["agent_action"] = "yielded" if overlap_ms <= cls.YIELD_WINDOW_MS else "held"
        return event

    @classmethod
    def _recall_at_fp_budget(cls, event_yields: list[float], non_event_yields: list[float]) -> float | None:
        """Max recall of events reachable while keeping the non-event fire rate within budget.

        "Fire" here means the agent treats a barge-in as an interruption to yield to,
        i.e. its speech overlap after the barge-in is ≤ the threshold τ. Sweeping τ over
        the observed overlap values traces the recall/FP trade-off; the never-yield
        corner (recall 0, FP 0) is always feasible, so the result is in [0, 1].
        """
        if not event_yields or not non_event_yields:
            return None
        best = 0.0
        for tau in sorted(set(event_yields + non_event_yields)):
            fp_rate = sum(1 for y in non_event_yields if y <= tau) / len(non_event_yields)
            if fp_rate <= cls.FP_BUDGET:
                best = max(best, sum(1 for y in event_yields if y <= tau) / len(event_yields))
        return best

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute floor-control classification sub-metrics from audio timestamps."""
        try:
            turn_keys = TurnTakingMetric._get_turn_ids_with_turn_taking(context)
            if not turn_keys:
                return MetricScore(name=self.name, score=None, normalized_score=None, skipped=True)

            agent_events: dict[int, dict[str, Any]] = {}
            for t in turn_keys:
                if t in context.assistant_interrupted_turns:
                    event = self._classify_agent_overlap(context, t)
                    if event is not None:
                        agent_events[t] = event

            user_events: dict[int, dict[str, Any]] = {}
            for t in turn_keys:
                if t in context.user_interrupted_turns:
                    event = self._classify_user_barge_in(context, t)
                    if event is not None:
                        user_events[t] = event

            if not agent_events and not user_events:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={"num_evaluated": len(turn_keys), "fp_budget": self.FP_BUDGET},
                )

            def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
                return MetricScore(
                    name=f"{self.name}.{key}",
                    score=value,
                    normalized_score=value if normalized else None,
                )

            sub: dict[str, MetricScore] = {}

            # --- Agent overlaps: takeover vs. non-taking overlap ---
            if agent_events:
                takeovers = [e for e in agent_events.values() if e["class"] == "takeover"]
                sub["agent_interruption.takeover_rate"] = _wrap(
                    "agent_interruption.takeover_rate", round(len(takeovers) / len(agent_events), 4), True
                )
                signed_entries = [e["signed_entry_ms"] for e in agent_events.values()]
                sub["agent_interruption.mean_signed_entry_ms"] = _wrap(
                    "agent_interruption.mean_signed_entry_ms", round(statistics.mean(signed_entries), 3), False
                )
                sub["agent_interruption.takeover_count"] = MetricScore(
                    name=f"{self.name}.agent_interruption.takeover_count",
                    score=float(len(takeovers)),
                    normalized_score=None,
                )

            # --- User barge-ins: selective interruption handling ---
            takeover_overlaps = [
                e["agent_overlap_after_barge_in_ms"]
                for e in user_events.values()
                if e["class"] == "takeover" and "agent_action" in e
            ]
            backchannel_overlaps = [
                e["agent_overlap_after_barge_in_ms"]
                for e in user_events.values()
                if e["class"] == "backchannel" and "agent_action" in e
            ]
            takeover_yields = [y for y in takeover_overlaps if y <= self.YIELD_WINDOW_MS]
            backchannel_yields = [y for y in backchannel_overlaps if y <= self.YIELD_WINDOW_MS]

            selective_parts: list[float] = []
            if takeover_overlaps:
                yield_accuracy = len(takeover_yields) / len(takeover_overlaps)
                sub["user_interruption.yield_accuracy"] = _wrap(
                    "user_interruption.yield_accuracy", round(yield_accuracy, 4), True
                )
                selective_parts.append(yield_accuracy)
            if backchannel_overlaps:
                hold_accuracy = 1.0 - len(backchannel_yields) / len(backchannel_overlaps)
                sub["user_interruption.hold_accuracy"] = _wrap(
                    "user_interruption.hold_accuracy", round(hold_accuracy, 4), True
                )
                selective_parts.append(hold_accuracy)

            recall = self._recall_at_fp_budget(takeover_overlaps, backchannel_overlaps)
            if recall is not None:
                sub["user_interruption.fp_constrained_takeover_recall"] = _wrap(
                    "user_interruption.fp_constrained_takeover_recall", round(recall, 4), True
                )

            # Balanced accuracy over the event classes we could measure; the main score.
            selective_accuracy = round(statistics.mean(selective_parts), 4) if selective_parts else None
            if selective_accuracy is not None:
                sub["user_interruption.selective_accuracy"] = _wrap(
                    "user_interruption.selective_accuracy", selective_accuracy, True
                )

            details: dict[str, Any] = {
                "agent_overlap_events": agent_events,
                "user_barge_in_events": user_events,
                "num_evaluated": len(turn_keys),
                "fp_budget": self.FP_BUDGET,
            }

            return MetricScore(
                name=self.name,
                score=selective_accuracy,
                normalized_score=selective_accuracy,
                details=details,
                sub_metrics=sub or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
