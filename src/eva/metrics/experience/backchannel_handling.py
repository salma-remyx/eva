"""Backchannel handling metric using audio timestamps + user transcripts (no LLM judge).

A backchannel is a short listener vocalization ("mm-hmm", "yeah", "right") that lands on
top of the assistant's speech without requesting the floor. Systems designed for
full-duplex interaction are expected to sail through backchannels — keep talking —
rather than treating them like interruptions: yielding the floor to an acknowledgment,
pausing, abandoning the response, or freezing.

``turn_taking`` cannot see this distinction: it scores every user barge-in with a
yield curve, so a system that appropriately continues through "mm-hmm" and a system
that drops its response both look like "user interrupt" turns. This metric isolates
the backchannel subset of barge-ins and scores how the assistant handled each one.

Detection (per turn ``t`` in ``context.user_interrupted_turns``): the barge-in counts
as a backchannel when
  - the user transcript for the turn is *pure* backchannel vocabulary (every token a
    continuer/acknowledger, at most ``BACKCHANNEL_MAX_WORDS`` words), and
  - the user's speech is embedded in the assistant's ongoing utterance (the assistant
    was still speaking at the barge-in moment).
Barge-ins with real content ("wait, change it to Friday") fail the lexical filter and
remain ordinary interruptions for ``turn_taking``.

Handling score per detected backchannel (segments pooled from turns ``t-1`` and ``t``).
The reference definition of a supportive backchannel is that the assistant "retains the
floor and continues without pausing", so the outcome is binary — there is no tolerance
band for pausing and resuming:
  - talked_through — an assistant segment spans the backchannel's end (the assistant
    never stopped): 1.0
  - resumed        — assistant speech restarts ``resume_ms`` after the backchannel ends
    (the assistant paused): 0.0, with ``resume_ms`` retained as diagnostic evidence
  - abandoned      — the assistant stopped mid-response during the backchannel and no
    assistant speech follows: 0.0

Conversations with no detected backchannels are ``skipped`` (score None),
matching the response_speed / authentication_success skip idiom.

Fidelity to the reference ("Gander: A Native Duplex Interaction Model with an
Asynchronous Agent Loop", arXiv:2609.08977): the acceptance rule follows the paper's —
a turn is a backchannel only if it is embedded in the interlocutor's ongoing utterance
and falls below the paper's short-length threshold of six words in English, matched
against a lexicon of continuer expressions (the paper samples from a 170-expression
English lexicon organized into 11 intent categories; that lexicon is not released in
the reference code, so a conservative closed continuer set stands in — it also keeps
interrogative, requestive, and negative-stance cues out, per the paper's demotion
rule). The handling criterion implements the paper's ground truth that the assistant
continues without pausing. Scope limits, by design: this metric measures the user-side
robustness outcome only. It does not evaluate the reference's central agent-side
mechanism — the model emitting a ``<|backchannel|>`` control action predicted before
content — because model control decisions are not observable in EVA's transcripts and
audio timestamps, and it is not a port of the paper's Full-Duplex-Bench v3 protocol
(LLM-judged, per-unit duplex control); it is a deterministic, no-LLM-judge measure
over ``MetricContext``.

Main backchannel_handling.score = mean(per-event handling scores).
"""

import re
import statistics
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore

# Single-word continuers/acknowledgers. Tokens are compared after lowercasing and
# stripping to bare letters, so "Mm-hmm!" normalizes to the two tokens "mm", "hmm".
BACKCHANNEL_TOKENS: frozenset[str] = frozenset(
    {
        "absolutely",
        "ah",
        "aha",
        "ahh",
        "alright",
        "correct",
        "definitely",
        "exactly",
        "good",
        "great",
        "hm",
        "hmm",
        "huh",
        "indeed",
        "mhm",
        "mhmm",
        "mm",
        "mmm",
        "nice",
        "okey",
        "ok",
        "okay",
        "oh",
        "ooh",
        "right",
        "sure",
        "totally",
        "true",
        "uh",
        "uhhuh",
        "uhuh",
        "wow",
        "yep",
        "yes",
        "yeah",
        "yea",
        "yup",
    }
)

# Multi-word phrases that count as backchannels even though individual tokens
# ("got", "see", "sounds") are not continuers on their own.
BACKCHANNEL_PHRASES: frozenset[str] = frozenset(
    {
        "all right",
        "go ahead",
        "go on",
        "got it",
        "i know",
        "i see",
        "keep going",
        "keep talking",
        "makes sense",
        "of course",
        "sounds good",
        "sounds great",
        "thats right",
        "understood",
        "youre right",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split to bare-letter tokens so punctuation never blocks a match."""
    return re.findall(r"[a-z]+", text.lower().replace("'", ""))


def _is_backchannel_transcript(text: str | None, max_words: int) -> bool:
    """Return True when the utterance is nothing but backchannel vocabulary.

    Pure single-word continuers ("yeah") match token-wise; fixed phrases ("got it")
    match as a whole after joining. Anything containing content words fails, which is
    what keeps real interruptions out of this metric.
    """
    if not text:
        return False
    tokens = _tokenize(text)
    if not tokens or len(tokens) > max_words:
        return False
    if all(t in BACKCHANNEL_TOKENS for t in tokens):
        return True
    return " ".join(tokens) in BACKCHANNEL_PHRASES


@register_metric
class BackchannelHandlingMetric(CodeMetric):
    """Scores whether the assistant continues through user backchannels instead of derailing."""

    name = "backchannel_handling"
    description = (
        "Detects user backchannels among barge-in turns and scores whether the "
        "assistant kept talking through each one; pausing (even briefly) counts "
        "against it, per the reference's continues-without-pausing definition"
    )
    category = "experience"
    pass_at_k_threshold = 0.8
    version = "v0.2"
    # Opt-in (like tts_fidelity): enable via `--metrics backchannel_handling`. Most
    # conversations have no backchannels, so it should not inflate default runs.
    exclude_from_default_metrics = True

    # A turn only counts as a backchannel when its whole transcript is continuer
    # vocabulary within the paper's short-length threshold: six words in English.
    BACKCHANNEL_MAX_WORDS: int = 6

    @staticmethod
    def _total_speech_seconds(segments: list[tuple[float, float]]) -> float:
        """Total voiced duration across the segment list (gaps between segments excluded)."""
        return sum(end - start for start, end in segments)

    def _classify_event(self, context: MetricContext, turn_id: int) -> dict[str, Any] | None:
        """Classify one barge-in turn; return evidence dict, or None when not a backchannel.

        The returned dict carries ``reason`` ("talked_through" / "resumed" / "abandoned" /
        "unscored") plus the signals behind the classification. ``unscored`` events
        (backchannel detected but the overlap signal is missing) are counted but do
        not feed the score.
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        prev_a_segs = context.audio_timestamps_assistant_turns.get(turn_id - 1)
        if not u_segs or not prev_a_segs:
            return None

        transcript = context.transcribed_user_turns.get(turn_id)
        if not _is_backchannel_transcript(transcript, self.BACKCHANNEL_MAX_WORDS):
            return None

        evidence: dict[str, Any] = {"transcript": transcript, "turn_id": turn_id}
        bc_start = u_segs[0][0]
        bc_end = u_segs[-1][1]
        evidence["bc_duration_ms"] = round(self._total_speech_seconds(u_segs) * 1000, 3)

        # Require real overlap: the assistant's previous-turn speech must still be
        # running at the barge-in moment (the paper's "embedded within the
        # interlocutor's ongoing utterance"), otherwise the interrupt flag is
        # endpointing noise and there is nothing to score.
        if prev_a_segs[-1][1] <= bc_start:
            evidence["reason"] = "unscored"
            return evidence

        # Pool the assistant segments that could carry the continuation: the tail of
        # the interrupted response (turn t-1) plus the response to the backchannel
        # turn itself (turn t).
        pool = list(prev_a_segs) + list(context.audio_timestamps_assistant_turns.get(turn_id) or [])

        # Talked through: some assistant segment spans the backchannel's end. This is
        # the only passing outcome — the paper's ground truth is that the assistant
        # "continues without pausing", so there is no resume-tolerance band.
        if any(a_start <= bc_end < a_end for a_start, a_end in pool):
            evidence["reason"] = "talked_through"
            evidence["score"] = 1.0
            return evidence

        # Resumed: the first assistant speech after the backchannel ends. The pause
        # itself already violates "continues without pausing", so this scores 0;
        # resume_ms is kept as diagnostic evidence.
        settled_starts = [a_start for a_start, _ in pool if a_start > bc_end]
        if settled_starts:
            resume_ms = (min(settled_starts) - bc_end) * 1000
            evidence["reason"] = "resumed"
            evidence["resume_ms"] = round(resume_ms, 3)
            evidence["score"] = 0.0
            return evidence

        evidence["reason"] = "abandoned"
        evidence["score"] = 0.0
        return evidence

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute backchannel handling score and flat sub-metrics."""
        try:
            barge_in_turns = sorted(t for t in context.user_interrupted_turns if t != 0)
            events: dict[int, dict[str, Any]] = {}
            for turn_id in barge_in_turns:
                event = self._classify_event(context, turn_id)
                if event is not None:
                    events[turn_id] = event

            scored = {t: e for t, e in events.items() if "score" in e}
            details: dict[str, Any] = {
                "per_turn_evidence": events,
                "num_barge_in_turns": len(barge_in_turns),
                "num_backchannels": len(events),
                "num_scored": len(scored),
                "num_unscored": sum(1 for e in events.values() if e["reason"] == "unscored"),
            }

            if not events:
                self.logger.info(
                    f"[{context.record_id}] No backchannels among {len(barge_in_turns)} barge-in turns; skipping."
                )
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={**details, "reason": "No backchannel events detected"},
                )
            if not scored:
                self.logger.info(
                    f"[{context.record_id}] {len(events)} backchannels detected but none scoreable; skipping."
                )
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={**details, "reason": "No scoreable backchannel events"},
                )

            score = round(statistics.mean(e["score"] for e in scored.values()), 4)

            def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
                return MetricScore(
                    name=f"{self.name}.{key}",
                    score=value,
                    normalized_score=value if normalized else None,
                )

            sub: dict[str, MetricScore] = {
                "num_backchannels": _wrap("num_backchannels", float(len(events)), False),
                "num_scored": _wrap("num_scored", float(len(scored)), False),
                "talked_through_rate": _wrap(
                    "talked_through_rate",
                    round(sum(1 for e in scored.values() if e["reason"] == "talked_through") / len(scored), 4),
                    True,
                ),
                "abandoned_rate": _wrap(
                    "abandoned_rate",
                    round(sum(1 for e in scored.values() if e["reason"] == "abandoned") / len(scored), 4),
                    True,
                ),
                "barge_in_backchannel_rate": _wrap(
                    "barge_in_backchannel_rate", round(len(events) / len(barge_in_turns), 4), True
                ),
            }
            resumed_ms = [e["resume_ms"] for e in scored.values() if "resume_ms" in e]
            if resumed_ms:
                sub["mean_resume_ms"] = _wrap("mean_resume_ms", round(statistics.mean(resumed_ms), 3), False)

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details=details,
                sub_metrics=sub,
            )

        except Exception as e:
            return self._handle_error(e, context)
