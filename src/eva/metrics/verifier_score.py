"""Continuous verifier scoring for LLM judges.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). Standard LM judges emit a single discrete integer rating,
which quantizes the judge's belief into a handful of buckets and throws away
the separation the model actually encodes. Instead, this module recovers the
judge's *distribution* over the scoring token and returns the expectation

    E[rating] = sum_r  r * P(r)

where P is the softmax-normalized probability mass the model assigned to each
valid rating token at the position where the score was emitted. The result is a
continuous score with the same [min, max] range as the discrete rating, so it
is a drop-in replacement for the ``normalized_score`` contract that downstream
aggregation and bootstrap confidence intervals already consume — it simply
resolves finer differences between records than a 1/2/3 bucket can.

Only the paper's core probabilistic-scoring mechanism is implemented here. The
paper's other scaling axes (repeated evaluation, criteria decomposition), its
candidate-ranking algorithm, and its RL / Claude-Code extensions are out of
scope for a per-record metric scorer.
"""

import math
from typing import Any

# Type alias for readability: a per-token logprob entry as returned by the LLM
# API. Each entry exposes a sampled ``token`` and its ``top_logprobs`` (the
# alternative tokens considered at that position with their log-probabilities).
# Entries may be plain dicts or SDK objects, so access goes through ``_get``.
TokenLogprob = Any


def _get(entry: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a logprob entry that may be a dict or an SDK object."""
    if entry is None:
        return default
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _parse_rating_token(token: Any) -> int | None:
    """Parse a raw token into the integer rating it represents, or None.

    Judges emit ratings as JSON number tokens, which a tokenizer may surface
    with leading whitespace or wrapping quotes (e.g. ``" 2"`` or ``'"2"'``).
    Anything that is not a bare run of digits (punctuation, words, decimals) is
    not a rating token and returns None.
    """
    if not isinstance(token, str):
        return None
    stripped = token.strip().strip('"').strip()
    if stripped.isdigit():
        return int(stripped)
    return None


def _distribution_at(entry: TokenLogprob, valid: set[int]) -> dict[int, float]:
    """Return {rating: probability} from one token's top-logprob alternatives.

    Probabilities are exponentiated from the log-probs and renormalized over the
    valid rating tokens only, so mass the model spent on non-rating tokens
    (quotes, whitespace) does not distort the expectation. Repeated tokenizations
    of the same rating are summed before renormalization.
    """
    weights: dict[int, float] = {}
    for alt in _get(entry, "top_logprobs") or []:
        rating = _parse_rating_token(_get(alt, "token"))
        if rating is None or rating not in valid:
            continue
        logprob = _get(alt, "logprob")
        if logprob is None:
            continue
        weights[rating] = weights.get(rating, 0.0) + math.exp(float(logprob))

    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {rating: mass / total for rating, mass in weights.items()}


def expected_rating(
    logprob_tokens: list[TokenLogprob] | None,
    min_rating: int,
    max_rating: int,
    anchor: str = "rating",
) -> float | None:
    """Continuous expected rating from a judge's scoring-token logprobs.

    Walks the token stream, and at the first rating token emitted after the
    ``anchor`` key has appeared (the top-level ``"rating"`` field of the judge's
    JSON), computes the expectation over the model's distribution at that
    position. Returns None when logprobs are missing or no rating token can be
    located, so callers can fall back to the discrete rating.

    Args:
        logprob_tokens: Per-token logprob entries for the judge's completion.
        min_rating: Minimum valid rating (inclusive).
        max_rating: Maximum valid rating (inclusive).
        anchor: Substring that must appear before the scoring token is read.
            Pass an empty string to disable anchoring.

    Returns:
        The probability-weighted expected rating, or None.
    """
    if not logprob_tokens:
        return None

    valid = set(range(min_rating, max_rating + 1))
    seen_text = ""
    anchor_seen = not anchor  # no anchor -> immediately eligible

    for entry in logprob_tokens:
        token = _get(entry, "token")
        if isinstance(token, str):
            seen_text += token
            if not anchor_seen and anchor in seen_text.lower():
                anchor_seen = True

        if not anchor_seen:
            continue

        rating = _parse_rating_token(token)
        if rating is None or rating not in valid:
            continue

        # Found the scoring position. Prefer the full distribution; if the API
        # did not return alternatives, fall back to a point mass at the sampled
        # rating (equivalent to the discrete score, but still valid).
        distribution = _distribution_at(entry, valid)
        if not distribution:
            return float(rating)
        return sum(r * p for r, p in distribution.items())

    return None


def normalize_expected_rating(expected: float, min_val: int, max_val: int) -> float:
    """Normalize a continuous expected rating to the 0.0-1.0 range.

    Mirrors ``eva.metrics.utils.normalize_rating`` but accepts a float and
    clamps to the valid range, so an expectation that drifts a hair outside
    [min, max] (numerical noise) still yields a well-formed normalized score.
    """
    if max_val == min_val:
        return 1.0
    clamped = min(max(expected, min_val), max_val)
    return (clamped - min_val) / (max_val - min_val)
