"""STT error propagation into tool-call arguments metric.

Deterministic diagnostic that traces words the STT mis-transcribed in user
speech (the accent and noise failure surface) into the arguments of the tool
calls the agent made, separating errors that corrupted tool arguments from
errors the agent recovered from.

Adapted from "Mind the Accent Gap: British Accent Robustness in Speech-Driven
Financial Voice Assistants" (arXiv:2610.06587), which finds that accent-induced
ASR entity errors carry through to the LLM stage and corrupt tool-call
arguments. The paper's entity-error alignment is replaced here with the
parameter-free normalized token diff eva already computes for ``stt_wer`` —
no LLM judge is involved.

Debug metric for diagnosing model performance issues, not used in final
evaluation scores.
"""

import re
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[.*?\]")
_TOKEN_PATTERN = re.compile(r"\w+")

# Tokens shorter than this only count as argument matches on exact token
# equality; substring matches require more evidence to avoid flagging e.g.
# "ed" inside "edinburgh".
_MIN_SUBSTRING_TOKEN_LEN = 4

# Function words carry no entity meaning, so diffs on them are noise for
# argument-corruption purposes. English-only; in other languages unknown
# function words simply fall through as unattributed tokens.
_FUNCTION_WORDS = frozenset(
    {
        "about",
        "all",
        "also",
        "any",
        "are",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "may",
        "me",
        "might",
        "my",
        "no",
        "not",
        "now",
        "of",
        "off",
        "ok",
        "okay",
        "on",
        "or",
        "our",
        "out",
        "please",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "uh",
        "um",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "yes",
        "you",
        "your",
    }
)


def _content_tokens(text: str) -> list[str]:
    """Return entity-bearing tokens of normalized text.

    Function words and single characters are dropped: they rarely name a
    tool argument value, and single letters ("x j k") match too aggressively.
    """
    return [
        token
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) >= 2 and token not in _FUNCTION_WORDS
    ]


def _leaf_argument_values(value: Any) -> list[str]:
    """Flatten a (possibly nested) parameter value into leaf strings."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaf_argument_values(v)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaf_argument_values(item)]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    return []


@register_metric
class STTErrorPropagation(CodeMetric):
    """Rate at which mis-transcribed user words corrupt tool-call arguments.

    For every user turn, compares the intended text against the STT transcript
    (both normalized like ``stt_wer`` does). Content words present in the
    intended text but missing from the transcript are "mis-transcribed" — the
    words accents and noise corrupt first. Each is then classified against the
    values of every tool call the agent made:

    - propagated: the corrupted form (a word the STT inserted instead) appears
      in a tool argument, so the error reached the tool layer
    - recovered: the correct form appears in a tool argument anyway (agent
      confirmed, repeated, or inferred it)
    - unattributed: neither form appears in any tool argument

    Score = propagated / mis-transcribed words (lower is better);
    normalized_score = 1 - score (argument integrity, higher is better).

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "stt_error_propagation"
    version = "v0.1"
    description = (
        "Debug metric: fraction of mis-transcribed user words whose corrupted form "
        "appears in tool-call arguments (STT error propagation)"
    )
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the propagation rate for user turns with STT errors."""
        try:
            language = context.language or "en"
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())
            if not common_turn_ids:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            argument_values: list[tuple[str, str]] = []
            argument_tokens: set[str] = set()
            for call in context.tool_params or []:
                tool_name = call.get("tool_name") or ""
                for value in _leaf_argument_values(call.get("tool_parameters")):
                    argument_values.append((tool_name, value))
                    argument_tokens.update(_TOKEN_PATTERN.findall(value.lower()))

            def argument_hits(token: str) -> set[str]:
                """Tool names whose argument values reference ``token``."""
                hits: set[str] = set()
                for tool_name, value in argument_values:
                    lowered = value.lower()
                    exact = token in argument_tokens
                    substring = len(token) >= _MIN_SUBSTRING_TOKEN_LEN and token in lowered
                    if exact or substring:
                        hits.add(tool_name)
                return hits

            per_turn: dict[int, dict[str, Any]] = {}
            propagated_examples: list[dict[str, Any]] = []
            num_propagated = 0
            num_recovered = 0
            num_unattributed = 0
            num_dropped = 0
            num_content_tokens = 0

            for turn_id in common_turn_ids:
                reference = _BRACKET_PATTERN.sub("", context.intended_user_turns[turn_id]).strip()
                hypothesis = _BRACKET_PATTERN.sub("", context.transcribed_user_turns[turn_id]).strip()
                if not reference or not hypothesis:
                    continue
                ref_tokens = _content_tokens(normalize_text(reference, language))
                hyp_tokens = _content_tokens(normalize_text(hypothesis, language))
                dropped = [token for token in ref_tokens if token not in hyp_tokens]
                inserted = [token for token in hyp_tokens if token not in ref_tokens]

                num_content_tokens += len(ref_tokens)
                num_dropped += len(dropped)

                classifications: list[dict[str, Any]] = []
                for token in dropped:
                    if argument_hits(token):
                        num_recovered += 1
                        classifications.append({"token": token, "outcome": "recovered"})
                        continue
                    corrupted = next((ins for ins in inserted if argument_hits(ins)), None)
                    if corrupted is not None:
                        tools = sorted(argument_hits(corrupted))
                        num_propagated += 1
                        classifications.append(
                            {"token": token, "outcome": "propagated", "corrupted_as": corrupted}
                        )
                        propagated_examples.append(
                            {
                                "turn_id": turn_id,
                                "intended_token": token,
                                "transcribed_as": corrupted,
                                "tool_calls": tools,
                            }
                        )
                    else:
                        num_unattributed += 1
                        classifications.append({"token": token, "outcome": "unattributed"})

                if dropped or inserted:
                    per_turn[turn_id] = {
                        "dropped": dropped,
                        "inserted": inserted,
                        "classifications": classifications,
                    }

            propagation_rate = num_propagated / num_dropped if num_dropped > 0 else 0.0
            notes: list[str] = []
            if num_dropped == 0:
                notes.append("No transcription errors on content words detected")
            if not argument_values:
                notes.append("No tool-call arguments to check propagation against")

            sub_metrics: dict[str, MetricScore] = {
                "num_propagated": MetricScore(
                    name=f"{self.name}.num_propagated",
                    score=float(num_propagated),
                    normalized_score=None,
                    details={},
                ),
                "propagation_rate": make_rate_sub_metric(
                    parent_name=self.name,
                    key="propagation_rate",
                    numerator=num_propagated,
                    denominator=num_dropped,
                    details={
                        "propagated": num_propagated,
                        "recovered": num_recovered,
                        "unattributed": num_unattributed,
                        "mis_transcribed_tokens": num_dropped,
                    },
                ),
                "mis_transcribed_token_rate": make_rate_sub_metric(
                    parent_name=self.name,
                    key="mis_transcribed_token_rate",
                    numerator=num_dropped,
                    denominator=num_content_tokens,
                    details={
                        "mis_transcribed_tokens": num_dropped,
                        "content_tokens": num_content_tokens,
                    },
                ),
            }

            return MetricScore(
                name=self.name,
                score=round(propagation_rate, 3),
                normalized_score=round(1 - propagation_rate, 3),
                details={
                    "language": language,
                    "num_turns": len(common_turn_ids),
                    "mis_transcribed_tokens": num_dropped,
                    "content_tokens": num_content_tokens,
                    "propagated": num_propagated,
                    "recovered": num_recovered,
                    "unattributed": num_unattributed,
                    "propagated_examples": propagated_examples,
                    "per_turn": per_turn,
                    "notes": notes or None,
                },
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)
