"""Context-sensitive Yield/Keep decision metric for user barge-ins.

Adapted from ECHO (arXiv:2609.17360v1), a matched-contrast benchmark showing that
evaluating overlap events independently rewards fixed action preferences: most
full-duplex systems over-yield and look strong on interruptions while folding on
backchannels. This metric ports that diagnostic onto EVA's own artifacts:

  - Expected action: ECHO pairs identical overlap transcripts with contrasting
    dialogue contexts (one requiring Yield, one Keep). EVA has no curated pairs,
    so the expected action is derived from the interrupting user turn's
    transcript — a pure continuer ("yeah", "right", "go ahead") expects the agent
    to KEEP the floor (the backchannel / off-talk class), anything substantive
    expects a YIELD. The proxy is parameter-free and English-only in v0.1.
  - Observed action: inferred from the audio timestamps ``turn_taking`` already
    uses. The agent KEPT when its speech sustained past the barge-in for longer
    than ``DECISION_WINDOW_MS`` (it talked through the user); otherwise it
    YIELDED.
  - Pair accuracy: ECHO scores a pair only when both members are decided
    correctly, so constant-action policies earn no credit. With ``n`` substantive
    and ``m`` backchannel events, the fraction of (substantive, backchannel)
    contrast pairs decided correctly on both members is exactly
    ``yield_decision_accuracy * keep_decision_accuracy`` — that product is the
    headline score. It is None unless the record contains at least one event of
    each class (no within-record contrast to exploit).

The binary decision view is deliberately complementary to ``turn_taking``: that
metric grades HOW WELL the agent yields (every user interruption is scored
against a fast yield); this one grades WHETHER yielding was the contextually
right action. A constant-yield agent scores near-perfectly on
``turn_taking.user_interruption.mean_yield_score`` while its pair accuracy here
is 0.0.
"""

import re
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.experience.turn_taking import TurnTakingMetric
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore

# Bracketed processor annotations ("[user interrupts]", "[likely cut off by user]")
# and parentheticals carry no lexical signal for the backchannel proxy.
_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_WORD_RE = re.compile(r"[a-z']+")


@register_metric
class TurnDecisionSensitivityMetric(CodeMetric):
    """Matched-contrast scoring of agent Yield/Keep decisions on user barge-ins."""

    name = "turn_decision_sensitivity"
    description = (
        "Context-sensitivity of Yield/Keep decisions on user interruptions: pair accuracy "
        "over matched backchannel vs substantive contrast events (constant-yield policies score 0)"
    )
    category = "experience"
    pass_at_k_threshold = 0.8
    # Bias diagnostic: None on any record lacking both event classes, which pass@k
    # cannot consume. The class accuracies remain available as sub-metrics.
    exclude_from_pass_at_k = True
    version = "v0.1"

    # How long the agent's speech may sustain past a barge-in before the observed
    # decision counts as KEEP (talked through the user). Shares TurnTakingMetric's
    # notion of "slowest acceptable yield" so the two metrics stay consistent.
    DECISION_WINDOW_MS: float = TurnTakingMetric.YIELD_HARD_MS

    # A barge-in transcript classifies as a backchannel (Keep expected) when it is
    # at most this many words AND every word is in BACKCHANNEL_WORDS. Substantive
    # requests name content (dates, IDs, entities) that immediately leaves the
    # lexicon, which is what makes the closed-word-set rule hard to false-positive.
    BACKCHANNEL_MAX_WORDS: int = 4
    BACKCHANNEL_WORDS: frozenset[str] = frozenset(
        """
        absolutely ahead ah aha amazing aw aww carry certainly cool correct definitely
        exactly fantastic go going good got gotcha great ha haha heh hmm hm i it know
        lovely mm mmm mhm mmhmm nice oh okey ok okay ooh on perfect please right see
        sounds sure thank thanks true uh uhuh uhhuh wonderful whoa wow yeah yea yes yep
        yup you
        """.split()
    )

    @classmethod
    def _normalize_transcript(cls, transcript: str) -> list[str]:
        """Lowercase, drop annotations/punctuation, and return the word tokens."""
        cleaned = _ANNOTATION_RE.sub(" ", transcript.lower())
        return _WORD_RE.findall(cleaned)

    @classmethod
    def _expected_action(cls, transcript: str) -> str:
        """Return "keep" for a pure-backchannel barge-in transcript, else "yield"."""
        words = cls._normalize_transcript(transcript)
        is_backchannel = (
            bool(words)
            and len(words) <= cls.BACKCHANNEL_MAX_WORDS
            and all(word in cls.BACKCHANNEL_WORDS for word in words)
        )
        return "keep" if is_backchannel else "yield"

    @staticmethod
    def _sustain_ms(context: MetricContext, turn_id: int) -> float | None:
        """How long agent speech sustained past the barge-in at this turn (ms).

        Floor-holding speech is the cut-off turn's tail (turn N-1 segments) plus any
        agent segment that starts before the user finishes speaking (a re-barge into
        the user's turn). Segments starting after the user's turn ends are the
        settled next response, not floor-holding, and are excluded. Returns None
        when the timestamps needed to infer a decision are missing.
        """
        user_segs = context.audio_timestamps_user_turns.get(turn_id)
        prev_agent_segs = context.audio_timestamps_assistant_turns.get(turn_id - 1)
        if not user_segs or not prev_agent_segs:
            return None
        barge_in = user_segs[0][0]
        user_turn_end = user_segs[-1][1]
        agent_segments = [prev_agent_segs, context.audio_timestamps_assistant_turns.get(turn_id) or []]
        sustained_ends = [end for segs in agent_segments for start, end in segs if start < user_turn_end]
        if not sustained_ends:
            return None
        return max(0.0, max(sustained_ends) - barge_in) * 1000

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute pair accuracy and class accuracies over the record's barge-in decisions."""
        try:
            if context.language != "en":
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={
                        "reason": f"backchannel lexicon is English-only in {self.version}",
                        "language": context.language,
                    },
                )

            per_event: dict[int, dict[str, Any]] = {}
            for turn_id in sorted(context.user_interrupted_turns):
                transcript = (context.transcribed_user_turns or {}).get(turn_id, "").strip()
                if not transcript:
                    per_event[turn_id] = {"skipped": "no_user_transcript"}
                    continue
                sustain_ms = self._sustain_ms(context, turn_id)
                if sustain_ms is None:
                    per_event[turn_id] = {"skipped": "no_decision_signal", "transcript": transcript[:80]}
                    continue
                expected = self._expected_action(transcript)
                observed = "keep" if sustain_ms > self.DECISION_WINDOW_MS else "yield"
                per_event[turn_id] = {
                    "expected": expected,
                    "observed": observed,
                    "correct": expected == observed,
                    "sustain_ms": round(sustain_ms, 3),
                    "transcript": transcript[:80],
                }

            events = [event for event in per_event.values() if event.get("expected")]
            yield_events = [event for event in events if event["expected"] == "yield"]
            keep_events = [event for event in events if event["expected"] == "keep"]

            def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
                return MetricScore(
                    name=f"{self.name}.{key}",
                    score=value,
                    normalized_score=value if normalized else None,
                )

            def _accuracy(event_list: list[dict[str, Any]]) -> float | None:
                if not event_list:
                    return None
                return round(sum(1 for event in event_list if event["correct"]) / len(event_list), 4)

            yield_accuracy = _accuracy(yield_events)
            keep_accuracy = _accuracy(keep_events)

            sub_metrics: dict[str, MetricScore] = {
                "num_barge_in_events": _wrap("num_barge_in_events", float(len(events)), False),
                "num_substantive_events": _wrap("num_substantive_events", float(len(yield_events)), False),
                "num_backchannel_events": _wrap("num_backchannel_events", float(len(keep_events)), False),
            }
            if yield_accuracy is not None:
                sub_metrics["yield_decision_accuracy"] = _wrap("yield_decision_accuracy", yield_accuracy, True)
            if keep_accuracy is not None:
                sub_metrics["keep_decision_accuracy"] = _wrap("keep_decision_accuracy", keep_accuracy, True)

            details: dict[str, Any] = {
                "per_event": per_event,
                "num_turns": len(per_event),
                "num_evaluated": len(events),
                "num_not_applicable": len(per_event) - len(events),
                "num_substantive_events": len(yield_events),
                "num_backchannel_events": len(keep_events),
                "decision_window_ms": self.DECISION_WINDOW_MS,
            }

            if not events:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={**details, "reason": "no barge-in events with a decidable Yield/Keep action"},
                )

            # Pair accuracy over the implicit (substantive, backchannel) contrast pairs:
            # P(both members decided correctly) = P(yield-side correct) * P(keep-side correct).
            # None when only one event class is present — no within-record contrast exists.
            if yield_accuracy is None or keep_accuracy is None:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    sub_metrics=sub_metrics,
                    details={
                        **details,
                        "reason": "no matched contrast: record has only "
                        f"{'backchannel' if keep_events else 'substantive'} barge-in events",
                    },
                )

            pair_accuracy = round(yield_accuracy * keep_accuracy, 4)
            return MetricScore(
                name=self.name,
                score=pair_accuracy,
                normalized_score=pair_accuracy,
                sub_metrics=sub_metrics,
                details=details,
            )

        except Exception as e:
            return self._handle_error(e, context)
