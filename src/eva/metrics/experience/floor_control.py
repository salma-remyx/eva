"""Floor-control metric for full-duplex conversational behaviour.

Full-duplex voice agents share the audio channel with the user, so the hardest
part of the interaction is often *when* they speak rather than what they say:

  * **Pause handling** — while the user pauses mid-utterance (a silence gap
    between two speech segments of the same turn), the agent should hold the
    floor and let the user finish.  Jumping in during the pause ("takeover")
    cuts the user off.
  * **Interruption response** — when the user barges in with a substantive
    interruption, the agent should stop, let the user finish, and speak again
    in response.
  * **Backchannel resume** — when the user emits a short backchannel
    ("mm-hmm", "right") while the agent is speaking, the agent should keep
    talking and finish its response rather than going silent.

These are the temporal behaviours that distinguish full-duplex
speech-to-speech systems from half-duplex pipelines: they are exactly the
signals Full-Duplex-Bench reports on (pause-handling takeover rate,
takeover + response quality after user interruptions, and backchannel
resume rate).  This metric scores them deterministically from the per-turn
audio timeline EVA already records on ``MetricContext`` — no judge model
required.

Adapted from "NemotronLabs VoiceChat: An Open Full-duplex Speech-to-Speech
Model with Tool Calling Capabilities" (arXiv:2609.21967), whose evaluation
frames these three floor-control behaviours as the headline real-time
properties of a full-duplex system.  The paper's benchmark harness and its
judged post-interruption response-quality panel are intentionally not
ported: the behaviours are measured on EVA's own conversation recordings,
so any pipeline EVA can run (cascade or S2S) can be compared on them.

Score composition — each component counts only when its events exist:

    pause handling         → 1 − takeover_rate           (higher = better)
    interruption response  → responded_rate
    backchannel resume     → resumed_rate
    floor_control.score    = mean(available components)

A conversation with none of these events (no mid-turn pauses, no user
barge-ins) is reported as ``skipped`` rather than scored, so half-duplex
conversations with strictly alternating turns don't inflate averages.
"""

import statistics
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class FloorControlMetric(CodeMetric):
    """Floor-control metric derived from per-turn audio timestamps.

    Measures how the agent handles the conversational floor during user
    mid-utterance pauses, substantive interruptions, and backchannels.
    """

    name = "floor_control"
    description = "Full-duplex floor control: pause takeover, interruption response, backchannel resume"
    category = "experience"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    version = "v0.1"

    # A silence gap between two user segments of the same turn counts as a
    # mid-utterance pause only when it lasts at least this long — shorter gaps
    # are ordinary VAD splitting inside one phrase, not a deliberate pause.
    PAUSE_MIN_MS: float = 500.0

    # …and at most this long — beyond it, the user has plausibly finished the
    # utterance and the floor legitimately changes hands.
    PAUSE_MAX_MS: float = 3000.0

    # User speech inside an interrupt turn at or below this duration is a
    # backchannel ("mm-hmm"); longer speech is a substantive interruption.
    BACKCHANNEL_MAX_MS: float = 1500.0

    @staticmethod
    def _speech_seconds(segments: list[tuple[float, float]]) -> float:
        """Total speech duration (seconds) across a turn's audio segments."""
        return sum(end - start for start, end in segments)

    @classmethod
    def _find_pause_events(cls, context: MetricContext) -> list[dict[str, Any]]:
        """Find user mid-utterance pauses: silence gaps between consecutive user segments of one turn.

        A gap qualifies when its duration falls in ``[PAUSE_MIN_MS, PAUSE_MAX_MS]``
        and the user resumes speaking afterwards — the later segment is what makes
        the pause *mid-utterance* rather than the end of the turn.  The greeting
        (turn 0) is excluded.
        """
        events: list[dict[str, Any]] = []
        for turn_id, user_segments in sorted(context.audio_timestamps_user_turns.items()):
            if turn_id == 0 or len(user_segments) < 2:
                continue
            segments = sorted(user_segments)
            for (_, prev_end), (next_start, _) in zip(segments, segments[1:]):
                pause_ms = (next_start - prev_end) * 1000
                if cls.PAUSE_MIN_MS <= pause_ms <= cls.PAUSE_MAX_MS:
                    events.append(
                        {
                            "turn_id": turn_id,
                            "pause_start_s": prev_end,
                            "pause_end_s": next_start,
                            "pause_ms": round(pause_ms, 3),
                            # filled in by _classify_pause_events
                            "took_over": False,
                        }
                    )
        return events

    @classmethod
    def _classify_pause_events(cls, context: MetricContext, pauses: list[dict[str, Any]]) -> None:
        """Mark each pause event with whether the agent took the floor inside it.

        Takeover = an assistant segment *in the same turn* intersecting the pause
        window.  Assistant speech that begins only after the user has resumed is
        overtalk (turn_taking's overlap signal), not pause takeover; assistant
        speech from the previous turn that merely spills across the boundary is
        likewise not attributed to this turn.
        """
        for event in pauses:
            assistant_segments = context.audio_timestamps_assistant_turns.get(event["turn_id"]) or []
            event["took_over"] = any(
                a_start < event["pause_end_s"] and a_end > event["pause_start_s"]
                for a_start, a_end in assistant_segments
            )

    @staticmethod
    def _agent_speaks_after(context: MetricContext, turn_id: int, t_end: float) -> tuple[bool, float | None]:
        """Did the agent speak past ``t_end`` in this turn, and how long after did speech (re)start?

        Returns ``(spoke, delay_s)`` where ``spoke`` is True when an assistant
        segment in the turn ends after ``t_end``.  ``delay_s`` is the gap from
        ``t_end`` to the first assistant segment starting at or after it —
        ``0.0`` when an assistant segment spans ``t_end`` (the agent never
        stopped), ``None`` when the agent produced no speech after ``t_end``.
        """
        assistant_segments = context.audio_timestamps_assistant_turns.get(turn_id) or []
        speaking_after = [(a_start, a_end) for a_start, a_end in assistant_segments if a_end > t_end]
        if not speaking_after:
            return False, None
        starts = [a_start for a_start, _ in speaking_after if a_start >= t_end]
        delay_s = max(0.0, min(starts) - t_end) if starts else 0.0
        return True, delay_s

    @classmethod
    def _split_interrupt_turns(cls, context: MetricContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split user-interrupt turns into backchannel and substantive-interruption events.

        Each event carries the user's speech duration in the turn and whether the
        agent was audibly speaking after the user's speech ended (resumed /
        responded), plus the response delay for substantive interruptions.  The
        greeting (turn 0) is excluded.  Turns without user audio are dropped —
        the flag alone can't size the barge-in without it.
        """
        backchannels: list[dict[str, Any]] = []
        interruptions: list[dict[str, Any]] = []
        for turn_id in sorted(context.user_interrupted_turns):
            if turn_id == 0:
                continue
            user_segments = context.audio_timestamps_user_turns.get(turn_id) or []
            if not user_segments:
                continue
            user_speech_ms = cls._speech_seconds(sorted(user_segments)) * 1000
            user_last_end = max(end for _, end in user_segments)
            spoke, delay_s = cls._agent_speaks_after(context, turn_id, user_last_end)
            event = {
                "turn_id": turn_id,
                "user_speech_ms": round(user_speech_ms, 3),
                "agent_spoke_after": spoke,
            }
            if user_speech_ms <= cls.BACKCHANNEL_MAX_MS:
                event["resumed"] = spoke
                backchannels.append(event)
            else:
                event["responded"] = spoke
                event["response_delay_ms"] = round(delay_s * 1000, 3) if delay_s is not None else None
                interruptions.append(event)
        return backchannels, interruptions

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the floor-control score and flat sub-metrics."""
        try:
            pauses = self._find_pause_events(context)
            self._classify_pause_events(context, pauses)
            backchannels, interruptions = self._split_interrupt_turns(context)

            details: dict[str, Any] = {
                "pause_events": pauses,
                "interruption_events": interruptions,
                "backchannel_events": backchannels,
                "num_pause_events": len(pauses),
                "num_interruption_events": len(interruptions),
                "num_backchannel_events": len(backchannels),
            }

            components: dict[str, float] = {}
            if pauses:
                takeovers = sum(1 for event in pauses if event["took_over"])
                components["pause_handling"] = 1.0 - takeovers / len(pauses)
            if interruptions:
                responded = sum(1 for event in interruptions if event["responded"])
                components["interruption_response"] = responded / len(interruptions)
            if backchannels:
                resumed = sum(1 for event in backchannels if event["resumed"])
                components["backchannel_resume"] = resumed / len(backchannels)

            if not components:
                # Strictly alternating turns with single-segment utterances:
                # no floor-control event to judge, so don't invent a score.
                self.logger.info(
                    f"[{context.record_id}] No pause/interruption/backchannel events; skipping floor_control."
                )
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details=details,
                )

            score = round(statistics.mean(components.values()), 4)
            details["components"] = {key: round(value, 4) for key, value in components.items()}

            def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
                return MetricScore(
                    name=f"{self.name}.{key}",
                    score=value,
                    normalized_score=value if normalized else None,
                )

            def _count(key: str, count: int) -> MetricScore:
                return MetricScore(
                    name=f"{self.name}.{key}",
                    score=float(count) if count else None,
                    normalized_score=None,
                )

            sub_metrics: dict[str, MetricScore] = {
                "pause_handling.num_pauses": _count("pause_handling.num_pauses", len(pauses)),
                "interruption.num_events": _count("interruption.num_events", len(interruptions)),
                "backchannel.num_events": _count("backchannel.num_events", len(backchannels)),
            }
            if pauses:
                # "_rate" suffix ⇒ lower is better (issue frequency): taking over
                # the user's mid-utterance pause is the failure mode.
                sub_metrics["pause_handling.takeover_rate"] = _wrap(
                    "pause_handling.takeover_rate", round(takeovers / len(pauses), 4), True
                )
                sub_metrics["pause_handling.mean_pause_ms"] = _wrap(
                    "pause_handling.mean_pause_ms", round(statistics.mean(e["pause_ms"] for e in pauses), 3), False
                )
            if interruptions:
                sub_metrics["interruption.response_score"] = _wrap(
                    "interruption.response_score",
                    round(components["interruption_response"], 4),
                    True,
                )
                delays = [e["response_delay_ms"] for e in interruptions if e["response_delay_ms"] is not None]
                if delays:
                    sub_metrics["interruption.mean_response_delay_ms"] = _wrap(
                        "interruption.mean_response_delay_ms", round(statistics.mean(delays), 3), False
                    )
            if backchannels:
                sub_metrics["backchannel.resume_score"] = _wrap(
                    "backchannel.resume_score", round(components["backchannel_resume"], 4), True
                )

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details=details,
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)
