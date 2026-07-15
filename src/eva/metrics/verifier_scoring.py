"""LLM-as-a-Verifier scoring primitive.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper's core mechanism scores a candidate by the
*expectation over the distribution of scoring-token logits* rather than by the
single discrete integer an LM judge would otherwise emit. This module exposes
that mechanism as a reusable primitive for EVA's judge metrics:

- :func:`expectation_score_from_logprobs` turns the logprob structure returned by
  a completion call into a continuous score (the paper's "score granularity"
  scaling axis), with no extra training.
- :func:`call_judge_with_logprobs` runs a judge completion with ``logprobs``
  enabled on the shared LiteLLM Router the rest of EVA uses.

Mode-2 adaptation note (honesty): the paper reads scoring-token logits straight
from the verifier model. EVA's :class:`~eva.utils.llm_client.LLMClient.generate_text`
only returns generated text, so rather than modify that shared client this module
calls the shared LiteLLM Router directly with ``logprobs=True`` / ``top_logprobs``.
The core scoring mechanism is implemented at full fidelity; the router's retry /
backoff loop (auxiliary transport) is deliberately omitted — callers run under
the metric's own error handling. The paper's "criteria decomposition" and
candidate-ranking axes are intentionally out of scope here.
"""

import math
from dataclasses import dataclass
from typing import Any

from eva.utils import router

# How far into the generated tokens to scan for the first rating/scoring token.
# Models occasionally emit a leading space, quote, or newline before the digit;
# this covers that without reading the explanatory prose that follows.
_MAX_SCORING_TOKEN_SCAN = 4


@dataclass(frozen=True)
class VerifierDistribution:
    """Continuous score derived from a scoring token's logprob distribution.

    Attributes:
        probabilities: ``{rating_value: probability}`` over the rating scale,
            renormalized to sum to 1.0 across the scale tokens observed.
        expectation: ``sum(rating * probability)`` — the continuous score on the
            rating scale (the paper's per-candidate verifier score).
        scoring_token: The generated token whose position the distribution was
            read from.
        from_top_logprobs: True if the distribution came from the model's
            ``top_logprobs``; False for the discrete fallback.
    """

    probabilities: dict[int, float]
    expectation: float
    scoring_token: str
    from_top_logprobs: bool


@dataclass(frozen=True)
class JudgeLogprobResponse:
    """Result of a judge completion run with logprobs enabled.

    Attributes:
        text: The generated message content.
        logprobs: The raw ``choices[0].logprobs`` structure (provider-shaped).
        usage: Token-usage dict compatible with EVA's usage logging, or None.
    """

    text: str
    logprobs: Any
    usage: dict[str, Any] | None


def _content_entries(logprobs: Any) -> list[Any]:
    """Return the per-token logprob entries from a provider-shaped logprobs object."""
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")
    if not content:
        return []
    return list(content)


def _field(entry: Any, name: str, default: Any = None) -> Any:
    """Read a field from a logprob entry that may be an object or a dict."""
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _scale_token_ratings(rating_scale: tuple[int, int]) -> dict[str, int]:
    """Map each rating-scale integer (as a token string) to its numeric value."""
    return {str(value): value for value in range(rating_scale[0], rating_scale[1] + 1)}


def _collect_scale_logprobs(entry: Any, scale_tokens: dict[str, int]) -> dict[int, float]:
    """Collect ``{rating: logprob}`` for scale tokens at one generated position.

    Uses the model's ``top_logprobs`` (its distribution over alternatives at this
    position) when present, and also folds in the chosen token's own logprob if it
    is a scale token. Duplicate ratings keep the higher-probability (less-negative)
    logprob.
    """
    collected: dict[int, float] = {}

    for alt in _field(entry, "top_logprobs", []) or []:
        token = str(_field(alt, "token", "") or "").strip()
        if token in scale_tokens:
            rating = scale_tokens[token]
            logprob = float(_field(alt, "logprob", 0.0) or 0.0)
            collected[rating] = max(collected.get(rating, float("-inf")), logprob)

    chosen_token = str(_field(entry, "token", "") or "").strip()
    if chosen_token in scale_tokens:
        rating = scale_tokens[chosen_token]
        logprob = float(_field(entry, "logprob", 0.0) or 0.0)
        collected[rating] = max(collected.get(rating, float("-inf")), logprob)

    return collected


def _softmax_over_ratings(rating_logprobs: dict[int, float]) -> dict[int, float]:
    """Convert ``{rating: logprob}`` to a normalized ``{rating: probability}`` map."""
    if not rating_logprobs:
        return {}
    highest = max(rating_logprobs.values())
    exps = {rating: math.exp(logprob - highest) for rating, logprob in rating_logprobs.items()}
    total = sum(exps.values())
    if total <= 0.0:
        return {}
    return {rating: value / total for rating, value in exps.items()}


def expectation_score_from_logprobs(
    logprobs: Any,
    rating_scale: tuple[int, int],
) -> VerifierDistribution | None:
    """Compute the LLM-as-a-Verifier continuous score from a completion's logprobs.

    Scans the first few generated tokens for the first position whose token is a
    rating-scale integer (the "scoring token"), reads that position's distribution
    over the scale, softmaxes it, and returns the expected rating plus the
    per-rating probabilities. This is the paper's core training-free mechanism: a
    continuous, calibrated score in place of a single integer.

    Args:
        logprobs: ``choices[0].logprobs`` from a completion run with
            ``logprobs=True`` (and ``top_logprobs`` populated). Object- or
            dict-shaped provider responses are both accepted.
        rating_scale: Inclusive ``(min, max)`` rating scale, e.g. ``(1, 3)``.

    Returns:
        The continuous score, or ``None`` if no scale token was found among the
        first generated tokens (the caller should fall back to a discrete rating).
    """
    entries = _content_entries(logprobs)
    if not entries:
        return None

    scale_tokens = _scale_token_ratings(rating_scale)
    for entry in entries[:_MAX_SCORING_TOKEN_SCAN]:
        chosen_token = str(_field(entry, "token", "") or "").strip()
        if chosen_token not in scale_tokens:
            continue
        probabilities = _softmax_over_ratings(_collect_scale_logprobs(entry, scale_tokens))
        if not probabilities:
            continue
        expectation = sum(rating * prob for rating, prob in probabilities.items())
        return VerifierDistribution(
            probabilities=dict(probabilities),
            expectation=expectation,
            scoring_token=chosen_token,
            from_top_logprobs=bool(_field(entry, "top_logprobs", None)),
        )
    return None


async def call_judge_with_logprobs(
    *,
    model: str,
    prompt: str,
    params: dict[str, Any],
    timeout: int,
    top_logprobs: int,
) -> JudgeLogprobResponse:
    """Run a judge completion with ``logprobs`` enabled on the shared LiteLLM Router.

    EVA's :class:`~eva.utils.llm_client.LLMClient.generate_text` returns only the
    generated text, so this calls the shared Router directly to surface the logprob
    structure the verifier score needs. ``top_logprobs`` controls the "score
    granularity" axis — more alternatives means a richer distribution over the
    rating scale.

    Args:
        model: Model name matching a deployment in ``EVA_MODEL_LIST``.
        prompt: The rendered judge prompt. Instruct the model to emit the rating
            token first so its logprobs land at a known position.
        params: Model parameters (``temperature``, ``max_tokens``, ...) merged into
            the completion call.
        timeout: Per-request timeout in seconds.
        top_logprobs: Number of alternative tokens to return logprobs for at each
            position.

    Returns:
        The generated text, the raw logprob structure, and token usage.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    kwargs.update(params)

    response = await router.get().acompletion(**kwargs)
    choice = response.choices[0]
    text = choice.message.content or ""
    logprobs = getattr(choice, "logprobs", None)

    usage: dict[str, Any] | None = None
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "model_name": getattr(response, "model", None),
        }
    return JudgeLogprobResponse(text=text, logprobs=logprobs, usage=usage)
