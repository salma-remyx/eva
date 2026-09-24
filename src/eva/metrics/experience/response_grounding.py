"""Response grounding — did the agent speak and act on what the user actually said?

Adapted from the failure analysis in MSI-Bench (arXiv:2609.24812), which reports that
voice agents across the board "respond when no one has addressed them" (a lack of
conversational restraint) and fail speaker-scoped decision making. EVA conversations
are dyadic, so the multi-speaker framing is adapted rather than ported verbatim:

- "Speaker-scoped" becomes "user-utterance-scoped": the agent's only legitimate
  information sources are the user's speech and tool responses, so assistant content
  that overlaps neither is treated as unaddressed content.
- The paper's LLM-judged atomic rubrics are replaced by parameter-free lexical checks
  (content-word overlap and value traceability), matching the deterministic style of
  the turn_taking metric this sits alongside.
- A wrong-speaker binding is not directly observable in dyadic logs; it surfaces as a
  tool-call argument that traces to no source at all (user speech, tool output, the
  tool schema's enum vocabulary, or the run's current date).
- The converse failure (staying silent when addressed) is already covered by
  turn_taking's missed-turn signal and is out of scope here.

Scoring uses the paper's atomic-rubric framing: every content-bearing assistant turn
and every scalar tool-call argument is one pass/fail unit, and the headline score is
the fraction of units passed. Assistant turns that ask a question or carry fewer than
three content words after stopword removal ("Sure, anything else?") are interactional
rather than informational, so they are marked not applicable and excluded. The
stopword list is English; for other languages function words appear on both sides of
the conversation and simply ground each other, so the metric degrades gracefully
rather than failing.

Flat sub-metrics: turn_grounding_accuracy, unaddressed_turn_rate, mean_turn_grounding,
tool_argument_accuracy, fabricated_tool_argument_rate, num_unaddressed_turns.
"""

import re
import statistics
from collections.abc import Iterator
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore

# Words that carry no grounding signal — discourse glue the agent produces regardless
# of what the user said. Only assistant-turn content is filtered with this list.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with about into
    over after is am are was were be been being do does did have has had will would
    shall should can could may might must i you he she it we they me him her us them
    my your his its our their this that these those there here what which who whom
    whose how not no yes so too very just now up out off down all any both each few
    more most other some such only own same than also again once from
    sure okay ok alright thanks thank please sorry hi hello hey goodbye bye welcome
    glad nice great good fine well help assist need want like know think let lets
    going get got make day today ahead right away thing course moment minute second
    wait hold kindly definitely certainly absolutely wonderful awesome perfect quick
    really quite bit little lot happy delighted apologize apologies unfortunately
    """.split()
)

# Digit ↔ word equivalences so a tool-call value of "2" grounds against a spoken "two".
_NUMBER_CANON = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "fifty": "50", "hundred": "100",
}


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens ("AA123, 5:30" → ["aa123", "5", "30"])."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _canon(token: str) -> str:
    """Canonicalize a token so number words and digits compare equal."""
    return _NUMBER_CANON.get(token, token)


def _content_tokens(text: str) -> list[str]:
    """Salient tokens of an assistant turn: length ≥ 2 and not a stopword."""
    return [t for t in _tokens(text) if len(t) >= 2 and t not in _STOPWORDS]


def _has_digit(token: str) -> bool:
    """True when a token contains a digit (IDs, confirmation codes, amounts)."""
    return any(c.isdigit() for c in token)


def _scalar_to_str(value: Any) -> str:
    """Render a scalar argument for token matching, collapsing 2.0 → "2"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _iter_scalar_arguments(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (dotted_path, scalar_value) for every leaf of a parameters object."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_scalar_arguments(value, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _iter_scalar_arguments(item, f"{prefix}[{i}]")
    else:
        yield prefix, node


@register_metric
class ResponseGroundingMetric(CodeMetric):
    """Deterministic grounding of assistant content and tool calls in user speech."""

    name = "response_grounding"
    description = (
        "Grounding of assistant turns and tool-call arguments in user speech and tool "
        "responses (conversational restraint and speaker-scoped tool binding)"
    )
    category = "experience"
    pass_at_k_threshold = 0.8
    version = "v0.1"

    # A content-bearing assistant turn passes when this fraction of its content words
    # traces to the user's speech or tool responses...
    GROUNDING_PASS_RATIO: float = 0.3
    # ...or when at least one grounded token is entity-like (contains a digit or is at
    # least this long) — echoing the right confirmation code or flight number counts as
    # speaker-scoped grounding even if the surrounding phrasing is novel.
    HIGH_INFO_MIN_LEN: int = 5
    # Turns with fewer content words than this are interjections ("You're all set!")
    # rather than substantive responses, so the restraint rubric does not apply.
    MIN_CONTENT_WORDS: int = 3

    @classmethod
    def _static_vocab(cls, context: MetricContext) -> set[str]:
        """Fixed vocabulary the agent may legitimately use without user grounding.

        Combines enum values declared in the agent's tool schema (fixed vocabulary the
        agent is told about, e.g. fare classes) with tokens of the run's current date.
        """
        vocab = {_canon(t) for t in _tokens(context.current_date_time or "")}

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "enum" and isinstance(value, list):
                        vocab.update(_canon(t) for item in value for t in _tokens(_scalar_to_str(item)))
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        for tool in context.agent_tools or []:
            _walk(tool)
        return vocab

    @classmethod
    def _vocab_by_turn(cls, context: MetricContext, needed_turns: list[int]) -> dict[int, set[str]]:
        """Cumulative allowed vocabulary at each turn: user speech + tool responses.

        Args:
            context: MetricContext with transcripts and conversation trace.
            needed_turns: Sorted turn IDs to snapshot the vocabulary at.

        Returns:
            Mapping of turn ID to the set of canonical tokens observable at that turn.
        """
        user_by_turn: dict[int, set[str]] = {
            t: {_canon(tok) for tok in _tokens(text)} for t, text in (context.transcribed_user_turns or {}).items()
        }
        tool_resp_by_turn: dict[int, set[str]] = {}
        for entry in context.conversation_trace or []:
            if entry.get("type") != "tool_response":
                continue
            t = entry.get("turn_id") or 0
            toks = {_canon(tok) for tok in _tokens(str(entry.get("tool_response", "")))}
            tool_resp_by_turn.setdefault(t, set()).update(toks)

        acc = cls._static_vocab(context)
        user_ids = sorted(t for t in user_by_turn if user_by_turn[t])
        resp_ids = sorted(t for t in tool_resp_by_turn if tool_resp_by_turn[t])
        ui = ri = 0
        out: dict[int, set[str]] = {}
        for t in needed_turns:
            while ui < len(user_ids) and user_ids[ui] <= t:
                acc |= user_by_turn[user_ids[ui]]
                ui += 1
            while ri < len(resp_ids) and resp_ids[ri] <= t:
                acc |= tool_resp_by_turn[resp_ids[ri]]
                ri += 1
            out[t] = acc.copy()
        return out

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the fraction of grounding rubrics passed for one conversation."""
        try:
            assistant_turns = {
                t: text
                for t, text in (context.transcribed_assistant_turns or {}).items()
                if t >= 1 and text and text.strip()  # greeting (turn 0) is never user-grounded
            }
            tool_call_entries = sorted(
                (e for e in context.conversation_trace or [] if e.get("type") == "tool_call"),
                key=lambda e: e.get("turn_id") or 0,
            )
            needed = sorted(set(assistant_turns) | {e.get("turn_id") or 0 for e in tool_call_entries})
            vocab_by_turn = self._vocab_by_turn(context, needed)

            # --- Rubric 1: conversational restraint (content-bearing assistant turns). ---
            per_turn: dict[int, dict[str, Any]] = {}
            turn_passes = 0
            turn_ratios: list[float] = []
            num_not_applicable = 0
            for t, text in assistant_turns.items():
                content = _content_tokens(text)
                if len(content) < self.MIN_CONTENT_WORDS or "?" in text:
                    # Questions advance the dialog; interjection-length turns carry no
                    # substantive claim. Neither can be "responding when no one
                    # addressed the agent".
                    if len(content) < self.MIN_CONTENT_WORDS:
                        reason = "too_few_content_words"
                    else:
                        reason = "question_turn"
                    num_not_applicable += 1
                    per_turn[t] = {"not_applicable": True, "reason": reason}
                    continue
                allowed = vocab_by_turn.get(t, set())
                matched = [tok for tok in content if _canon(tok) in allowed]
                ratio = len(matched) / len(content)
                entity_echoed = any(_has_digit(tok) or len(tok) >= self.HIGH_INFO_MIN_LEN for tok in matched)
                passed = ratio >= self.GROUNDING_PASS_RATIO or entity_echoed
                turn_passes += passed
                turn_ratios.append(ratio)
                per_turn[t] = {
                    "ratio": round(ratio, 4),
                    "num_content_words": len(content),
                    "num_grounded": len(matched),
                    "passed": passed,
                }

            # --- Rubric 2: speaker-scoped tool binding (scalar argument values). ---
            per_argument: dict[str, dict[str, Any]] = {}
            arg_passes = 0
            arg_total = 0
            for entry in tool_call_entries:
                t = entry.get("turn_id") or 0
                tool_name = entry.get("tool_name") or "tool"
                allowed = vocab_by_turn.get(t, set())
                for path, value in _iter_scalar_arguments(entry.get("parameters") or {}):
                    if value is None or isinstance(value, bool):
                        continue
                    tokens = [_canon(tok) for tok in _tokens(_scalar_to_str(value))]
                    if not tokens:
                        continue
                    arg_total += 1
                    passed = all(tok in allowed for tok in tokens)
                    arg_passes += passed
                    per_argument[f"turn{t}:{tool_name}.{path}"] = {"value": _scalar_to_str(value), "passed": passed}

            total_units = len(turn_ratios) + arg_total
            passed_units = turn_passes + arg_passes
            details: dict[str, Any] = {
                "per_turn_grounding": per_turn,
                "per_tool_argument": per_argument,
                "num_turns": len(assistant_turns),
                "num_evaluated": len(turn_ratios),
                "num_not_applicable": num_not_applicable,
                "total_tool_arguments": arg_total,
                "grounded_tool_arguments": arg_passes,
                "total_rubrics": total_units,
                "passed_rubrics": passed_units,
            }
            if total_units == 0:
                details["note"] = "No grounding rubrics applicable (no content-bearing turns or tool arguments)"
                return MetricScore(name=self.name, score=1.0, normalized_score=1.0, details=details)

            score = round(passed_units / total_units, 4)

            num_unaddressed = len(turn_ratios) - turn_passes
            sub: dict[str, MetricScore] = {
                "num_unaddressed_turns": MetricScore(
                    name=f"{self.name}.num_unaddressed_turns",
                    score=float(num_unaddressed) if num_unaddressed else None,
                    normalized_score=None,
                )
            }
            if turn_ratios:
                sub["turn_grounding_accuracy"] = make_rate_sub_metric(
                    self.name, "turn_grounding_accuracy", turn_passes, len(turn_ratios), {}, 4
                )
                sub["unaddressed_turn_rate"] = make_rate_sub_metric(
                    self.name, "unaddressed_turn_rate", len(turn_ratios) - turn_passes, len(turn_ratios), {}, 4
                )
                mean_ratio = round(statistics.mean(turn_ratios), 4)
                sub["mean_turn_grounding"] = MetricScore(
                    name=f"{self.name}.mean_turn_grounding", score=mean_ratio, normalized_score=mean_ratio
                )
            if arg_total:
                sub["tool_argument_accuracy"] = make_rate_sub_metric(
                    self.name, "tool_argument_accuracy", arg_passes, arg_total, {}, 4
                )
                sub["fabricated_tool_argument_rate"] = make_rate_sub_metric(
                    self.name, "fabricated_tool_argument_rate", arg_total - arg_passes, arg_total, {}, 4
                )

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details=details,
                sub_metrics=sub,
            )

        except Exception as e:
            return self._handle_error(e, context)
