"""Disclosure sharding: condition variants for incremental user-goal disclosure.

Adapted from SCB (SpeechConversationBench, arXiv:2609.40198), which compares
speech systems on a task delivered in one turn (``full``), its information
shards delivered together (``concat``), and the same shards disclosed
incrementally across turns (``sharded``). Here the "problem" is an evaluation
record's user goal: shards are the clause-level information units of
``user_goal.high_level_user_goal``, and disclosure is staged by rewriting the
goal text the user simulator reads. No simulator, runner, or metric changes
are needed — the variants are ordinary ``EvaluationRecord``s.

The two single-turn conditions separate reformulation sensitivity
(``full`` vs ``concat``) from the cost of incremental spoken interaction
(``concat`` vs ``sharded``), mirroring the paper's protocol.
"""

from __future__ import annotations

import re

from eva.models.record import EvaluationRecord

CONDITIONS = ("full", "concat", "sharded")
"""Disclosure conditions, named as in the SCB protocol."""

DEFAULT_MAX_SHARDS = 4
"""Upper bound on shards per goal; finer clauses are merged into the tail."""

CONDITION_SEP = "::"
"""Separator between the base record id and the condition in variant ids."""

_MIN_CLAUSE_WORDS = 3
_MIN_AND_SPLIT_WORDS = 4

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"[;,]\s+")
_LEADING_CONJ_RE = re.compile(r"^(?:and|or|but|so|also|then|plus|while)\s+", re.IGNORECASE)
_AND_SPLIT_RE = re.compile(r"\s+and\s+")


def split_into_shards(text: str, *, max_shards: int = DEFAULT_MAX_SHARDS) -> list[str]:
    """Split a goal statement into clause-level information shards.

    Deterministic and parameter-free: sentences are split on terminal
    punctuation, then on comma/semicolon boundaries, leading conjunctions are
    dropped, and fragments too short to stand alone are merged back. When no
    clause boundary is found, a guarded ``and``-split is attempted. The first
    shard is the opener (core request); later shards are additional
    requirements.

    Args:
        text: The ``high_level_user_goal`` string to shard.
        max_shards: Merge the finest tail clauses until at most this many
            shards remain.

    Returns:
        The shards, each stripped; a list with a single element when the goal
        has no internal boundaries.
    """
    text = " ".join(text.split())
    if not text:
        return []

    shards: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        clauses: list[str] = []
        for part in _CLAUSE_SPLIT_RE.split(sentence):
            part = _LEADING_CONJ_RE.sub("", part) if clauses else part
            if clauses and len(part.split()) < _MIN_CLAUSE_WORDS:
                # Too small to disclose on its own; fold back into the previous clause.
                clauses[-1] = f"{clauses[-1].rstrip(' ,;')}, {part}"
            else:
                clauses.append(part)
        shards.extend(clause.strip() for clause in clauses if clause.strip())

    if len(shards) == 1:
        pieces = _AND_SPLIT_RE.split(shards[0])
        if len(pieces) > 1 and all(len(piece.split()) >= _MIN_AND_SPLIT_WORDS for piece in pieces):
            shards = [piece.strip() for piece in pieces]

    while len(shards) > max_shards:
        shards[-2] = f"{shards[-2].rstrip(' ,;.')}, and {shards[-1].rstrip(' ,;.')}"
        shards.pop()

    return [shard.strip(" ,;") for shard in shards if shard.strip(" ,;")]


def variant_id(record_id: str, condition: str) -> str:
    """Return the variant record id ``<record_id>::<condition>``."""
    return f"{record_id}{CONDITION_SEP}{condition}"


def parse_condition(record_id: str) -> tuple[str, str | None]:
    """Split a variant id into ``(base_id, condition)``.

    Args:
        record_id: A record id, possibly carrying a ``::condition`` suffix.

    Returns:
        The base id and the condition name, or ``None`` when the id is not a
        disclosure variant (or the suffix is not a known condition).
    """
    if CONDITION_SEP not in record_id:
        return record_id, None
    base, _, condition = record_id.rpartition(CONDITION_SEP)
    if condition in CONDITIONS:
        return base, condition
    return record_id, None


def _numbered(shards: list[str]) -> str:
    """Format shards as a numbered requirement list."""
    return "\n".join(f"{i}. {shard}" for i, shard in enumerate(shards, start=1))


def _concat_goal(shards: list[str]) -> str:
    """Build the single-turn goal that discloses every shard up front."""
    return (
        "Communicate all of the following requirements to the agent in your very first "
        f"message, then work toward them:\n{_numbered(shards)}"
    )


def _sharded_goal(shards: list[str]) -> str:
    """Build the staged goal that discloses one shard per conversational turn."""
    return (
        "Disclose your requirements to the agent ONE PER TURN, in this exact order:\n"
        f"{_numbered(shards)}\n"
        "Open with requirement 1 only. Reveal requirement k only after the agent has "
        "responded to the requirements disclosed so far. Never state the full list up "
        "front and never mention later requirements early. Every requirement still "
        "applies to the final outcome."
    )


def goal_shards(record: EvaluationRecord, *, max_shards: int = DEFAULT_MAX_SHARDS) -> list[str]:
    """Shard a record's ``high_level_user_goal`` into disclosure units.

    Args:
        record: The source record.
        max_shards: Passed to :func:`split_into_shards`.

    Returns:
        The shards of the goal statement.

    Raises:
        TypeError: If ``user_goal`` is not a dict with a string
            ``high_level_user_goal``.
    """
    goal = record.user_goal
    if not isinstance(goal, dict) or not isinstance(goal.get("high_level_user_goal"), str):
        raise TypeError(
            f"Record {record.id!r}: user_goal must be a dict with a string "
            "'high_level_user_goal' to build disclosure variants"
        )
    return split_into_shards(goal["high_level_user_goal"], max_shards=max_shards)


def variant_record(
    record: EvaluationRecord,
    condition: str,
    *,
    shards: list[str] | None = None,
    max_shards: int = DEFAULT_MAX_SHARDS,
) -> EvaluationRecord:
    """Build one disclosure-condition variant of an evaluation record.

    Only the record id and ``user_goal.high_level_user_goal`` change; the
    persona, decision tree, information requirements, ground truth, and
    per-language starting utterances are carried over untouched so every
    condition is graded against the same final outcome.

    Args:
        record: The source record.
        condition: One of :data:`CONDITIONS`.
        shards: Precomputed shards; derived from the record when omitted.
        max_shards: Passed to :func:`split_into_shards` when shards are derived.

    Returns:
        A deep copy of ``record`` rewritten for the condition.

    Raises:
        ValueError: If ``condition`` is unknown.
        TypeError: If ``user_goal`` is not a dict with a string
            ``high_level_user_goal``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown disclosure condition {condition!r}; expected one of {CONDITIONS}")
    if shards is None:
        shards = goal_shards(record, max_shards=max_shards)

    variant = record.model_copy(deep=True)
    variant.id = variant_id(record.id, condition)
    if condition == "concat":
        variant.user_goal["high_level_user_goal"] = _concat_goal(shards)
    elif condition == "sharded":
        variant.user_goal["high_level_user_goal"] = _sharded_goal(shards)
    return variant


def condition_variants(
    record: EvaluationRecord,
    *,
    conditions: tuple[str, ...] = CONDITIONS,
    max_shards: int = DEFAULT_MAX_SHARDS,
) -> list[EvaluationRecord]:
    """Build the disclosure-condition variants of an evaluation record.

    Args:
        record: The source record.
        conditions: Which conditions to emit, in order.
        max_shards: Passed to :func:`split_into_shards`.

    Returns:
        One variant per requested condition. The incremental conditions
        (``concat``/``sharded``) are skipped when the goal yields a single
        shard, since there is nothing to deliver incrementally; ``full`` is
        always emitted.
    """
    shards = goal_shards(record, max_shards=max_shards)

    variants: list[EvaluationRecord] = []
    for condition in conditions:
        if condition in ("concat", "sharded") and len(shards) < 2:
            continue
        variants.append(variant_record(record, condition, shards=shards))
    return variants
