"""Interactivity metric — backchannel / acknowledgment engagement from transcripts.

Adapted (Mode 2) from the micro-level span-feature taxonomy of:

    "Interaction Matters: An Evaluation Framework for Interactive Dialogue
     Assessment on English Second Language Conversations" (arxiv:2407.06479).

The paper models dialogue-level *interactivity* in two layers:
  1. deterministic micro-level span features (backchannels and a family of
     listener-engagement signals — 17 features total), and
  2. ML classifiers trained on a gated, human-annotated ESL corpus that map
     those features onto 4 dialogue-level interactivity labels (e.g. topic
     management).

This port keeps layer (1) — the DETECTOR taxonomy — at full fidelity, with
backchannels as the flagship detector (the feature named in the abstract), and
substitutes layer (2) with a transparent, parameter-free rate-based aggregate.
The substitution is justified by the paper's own finding that micro-feature
presence tracks interactivity quality, so exposing the rates directly gives eva
the same signal without hosting a trained classifier or the gated ESL dataset.
Audio-prosodic micro-features (pitch / F0) are intentionally out of scope: this
is a transcript CODE metric, consistent with the other text-based experience
metrics.

What is evaluated: the **agent** (system under test). The headline signal is the
agent's own acknowledgment / backchannel behaviour — does it register the user's
input ("Got it, let me check", "I see, so...") before responding — which is the
interactivity gap the experience category was missing (conciseness /
conversation_progression / turn_taking cover everything else). User-side rates
are reported alongside it for parity with the paper's two-party framing.

Turn text is read from ``intended_*_turns`` (the pipeline's own text — what each
side meant to say) with ``transcribed_*_turns`` (ASR output) as a fallback, so
the detector runs on data already present in ``MetricContext`` (no new plumbing).

Main ``interactivity.score`` = ``min(1, agent_engagement_rate / ENGAGEMENT_TARGET_RATE)`` —
saturating so that an agent acknowledging the user on ~40%+ of its turns is
considered fully interactive; zero acknowledgments scores 0.0 (one-sided).

Flat headline sub-metrics (one number each):
  agent_engagement.rate    — fraction of the agent's turns that engage (backchannel
                             or acknowledgment opener); the score driver.
  user_engagement.rate     — same, for the user (simulator) side.
  backchannel.rate         — overall fraction of turns that are pure backchannels.
  acknowledgment.rate      — overall fraction of turns that open with an
                             acknowledgment phrase.

Sub-metric keys use a dotted ``.rate`` suffix (mirroring ``turn_taking``'s
``agent_interruption.rate``) so they inherit the parent's higher-is-better
direction rather than the lower-is-better ``_rate`` convention reserved for issue
frequencies.
"""

import re
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore

# --- Tokenizer: lowercase and keep only alphabetic runs. ---
# "uh-huh" -> ["uh", "huh"]; "ok," -> ["ok"]; "I see" -> ["i", "see"].
_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into alphabetic tokens."""
    return _TOKEN_RE.findall((text or "").lower())


# --- Backchannel lexicon: listener "verbal nods". ---
# A turn is a backchannel when it is short (<= MAX_BACKCHANNEL_TOKENS) and every
# one of its tokens is in this set — i.e. the whole turn is a nod with no
# substantive content. Drawn from the conversational-analysis tradition the
# paper's taxonomy builds on (Yngve 1970; Schegloff 1982).
_BACKCHANNEL_TOKENS = frozenset(
    """
    yeah yea yes yep yup yah yas
    mhm mm mmm hmm uh huh
    ok okay okey k
    right exactly totally absolutely definitely completely
    sure cool awesome great nice wow whoa
    gotcha indeed correct true agreed
    roger acknowledged copy
    """.split()
)

# --- Acknowledgment openers: the turn registers the partner's prior input. ---
# These PHRASAL openers are matched as token-prefixes. They cover acknowledgments
# that lead a substantive turn ("Got it, let me look that up") as well as bare
# phrasal nods ("of course"). Kept separate from the single-word lead set below
# because their first token ("i", "of", "got", ...) is too common to match alone.
_ACK_OPENER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("got", "it"),
    ("i", "see"),
    ("i", "understand"),
    ("makes", "sense"),
    ("of", "course"),
    ("sure", "thing"),
    ("right", "so"),
    ("right", "let"),
    ("right", "i"),
    ("right", "i", "can"),
    ("right", "well"),
    ("ok", "so"),
    ("okay", "so"),
    ("ok", "let"),
    ("okay", "let"),
    ("ok", "i"),
    ("okay", "i"),
    ("ok", "i", "can"),
    ("okay", "i", "can"),
    ("good", "to", "know"),
    ("i", "appreciate"),
    ("sounds", "good"),
    ("sounds", "like"),
    ("no", "problem"),
    ("will", "do"),
    ("you", "got", "it"),
    ("i", "can", "do", "that"),
    ("i", "can", "help"),
    ("great", "question"),
    ("good", "question"),
    ("great", "to"),
    ("great", "thanks"),
    ("happy", "to"),
    ("thank", "you"),
)

# Single-word openers that are safe to match when followed by >=1 more token —
# i.e. words that, leading a longer turn, almost always signal acknowledgment.
# Ambiguous leads ("i", "of", "no", "right", "ok") are excluded here and only
# matched via the explicit phrases above to avoid false positives like
# "right now" or "ok the answer is".
_ACK_LEAD_WORDS = frozenset(
    """
    sure great perfect absolutely gotcha understood noted
    thanks exactly definitely totally cool awesome wow indeed roger
    """.split()
)


@register_metric
class InteractivityMetric(CodeMetric):
    """Backchannel / acknowledgment engagement detector (transcript-only, no LLM)."""

    name = "interactivity"
    description = "Agent interactivity from backchannel and acknowledgment engagement on transcripts"
    category = "experience"
    version = "v0.1"
    pass_at_k_threshold = 0.5

    # A turn is a backchannel if it has at most this many tokens, all in the
    # backchannel lexicon. Pure nods rarely exceed 3 tokens ("yeah yeah yeah").
    MAX_BACKCHANNEL_TOKENS: int = 3

    # Agent engagement rate at or above this saturates the score to 1.0 — i.e.
    # acknowledging the user on ~40%+ of turns reads as fully interactive.
    ENGAGEMENT_TARGET_RATE: float = 0.4

    @classmethod
    def _is_backchannel(cls, tokens: list[str]) -> bool:
        """True for a short turn made entirely of listener-nod tokens."""
        return (
            bool(tokens)
            and len(tokens) <= cls.MAX_BACKCHANNEL_TOKENS
            and all(tok in _BACKCHANNEL_TOKENS for tok in tokens)
        )

    @classmethod
    def _is_acknowledgment_opener(cls, tokens: list[str]) -> bool:
        """True when the turn opens with an acknowledgment phrase or lead word."""
        if not tokens:
            return False
        for opener in _ACK_OPENER_PHRASES:
            n = len(opener)
            if len(tokens) >= n and tuple(tokens[:n]) == opener:
                return True
        # A safe single-word lead followed by at least one more token.
        return len(tokens) >= 2 and tokens[0] in _ACK_LEAD_WORDS

    @classmethod
    def _classify_turn(cls, text: str) -> str:
        """Classify a single turn's text as backchannel / acknowledgment / substantive.

        Backchannel takes priority (a pure nod is the clearest engagement signal).
        """
        tokens = _tokenize(text)
        if not tokens:
            return "empty"
        if cls._is_backchannel(tokens):
            return "backchannel"
        if cls._is_acknowledgment_opener(tokens):
            return "acknowledgment"
        return "substantive"

    @staticmethod
    def _speaker_turns(context: MetricContext, side: str) -> dict[int, str]:
        """Return ``{turn_id: text}`` for one side, preferring intended over transcribed.

        ``side`` is ``"agent"`` or ``"user"``. Intended text is the pipeline's own
        string (cleanest); transcribed (ASR) fills any turn the intended map lacks.
        """
        if side == "agent":
            primary, fallback = context.intended_assistant_turns, context.transcribed_assistant_turns
        else:
            primary, fallback = context.intended_user_turns, context.transcribed_user_turns
        turns: dict[int, str] = {}
        for turn_id, text in fallback.items():
            turns[turn_id] = text
        for turn_id, text in primary.items():
            turns[turn_id] = text  # intended wins
        return turns

    @staticmethod
    def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
        return MetricScore(
            name=f"{InteractivityMetric.name}.{key}",
            score=value,
            normalized_score=value if normalized else None,
        )

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the interactivity score and flat engagement sub-metrics."""
        try:
            per_turn: dict[str, dict[int, str]] = {
                "agent": self._speaker_turns(context, "agent"),
                "user": self._speaker_turns(context, "user"),
            }
            classifications: dict[str, dict[int, str]] = {"agent": {}, "user": {}}
            counts: dict[str, dict[str, int]] = {
                side: {"speakable": 0, "engagement": 0, "backchannel": 0, "acknowledgment": 0}
                for side in ("agent", "user")
            }

            for side in ("agent", "user"):
                for turn_id, text in per_turn[side].items():
                    if turn_id == 0:
                        continue  # greeting — not an acknowledgment of prior user content
                    label = self._classify_turn(text)
                    classifications[side][turn_id] = label
                    if label == "empty":
                        continue
                    counts[side]["speakable"] += 1
                    if label in ("backchannel", "acknowledgment"):
                        counts[side]["engagement"] += 1
                    if label == "backchannel":
                        counts[side]["backchannel"] += 1
                    if label == "acknowledgment":
                        counts[side]["acknowledgment"] += 1

            agent_speakable = counts["agent"]["speakable"]
            user_speakable = counts["user"]["speakable"]
            agent_engagement_rate = counts["agent"]["engagement"] / agent_speakable if agent_speakable else 0.0
            user_engagement_rate = counts["user"]["engagement"] / user_speakable if user_speakable else 0.0

            # Headline score: agent engagement rate on a saturating curve.
            if agent_speakable == 0:
                score = 0.0
            else:
                score = round(min(1.0, agent_engagement_rate / self.ENGAGEMENT_TARGET_RATE), 4)

            total_speakable = agent_speakable + user_speakable
            total_backchannel = counts["agent"]["backchannel"] + counts["user"]["backchannel"]
            total_acknowledgment = counts["agent"]["acknowledgment"] + counts["user"]["acknowledgment"]

            sub_metrics: dict[str, MetricScore] = {
                "agent_engagement.rate": self._wrap("agent_engagement.rate", round(agent_engagement_rate, 4), True),
                "user_engagement.rate": self._wrap("user_engagement.rate", round(user_engagement_rate, 4), True),
            }
            if total_speakable:
                sub_metrics["backchannel.rate"] = self._wrap(
                    "backchannel.rate", round(total_backchannel / total_speakable, 4), True
                )
                sub_metrics["acknowledgment.rate"] = self._wrap(
                    "acknowledgment.rate", round(total_acknowledgment / total_speakable, 4), True
                )

            details: dict[str, Any] = {
                "agent_engagement_rate": round(agent_engagement_rate, 4),
                "user_engagement_rate": round(user_engagement_rate, 4),
                "engagement_target_rate": self.ENGAGEMENT_TARGET_RATE,
                "max_backchannel_tokens": self.MAX_BACKCHANNEL_TOKENS,
                "counts": counts,
                "classifications": classifications,
                "num_agent_turns": agent_speakable,
                "num_user_turns": user_speakable,
            }

            if agent_speakable == 0:
                self.logger.info(
                    f"[{context.record_id}] No agent turns to classify; scoring 0 (no interactivity signal)."
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
