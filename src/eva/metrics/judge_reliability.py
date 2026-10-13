"""Reliability audit for EVA's LLM judges.

EVA already quantifies sampling noise (bootstrap CIs) and prompt drift
(``prompt_hash``), but still treats each judge verdict as a deterministic point
estimate. This module adds the missing axis: how much of a verdict is an
artifact of the judge rather than of the conversation.

Adapted from *All Verdicts are Not Equal: Rethinking LLM Judge Reliability*
(arXiv:2610.12083), which stress-tests LLM judges across repetitions and
presentation orders and collapses the failure modes into the *trustworthy
verdict rate* (T): the joint probability that a verdict is reproducible under
identical re-sampling, invariant to presentation order, and accurate.

EVA's judges score a single conversation against a rubric rather than compare
a pair, so the paper's A/B position swap maps to reversing the order in which
the transcript turns are presented to the judge. The paper's accuracy term
needs a ground-truth verdict per item, which EVA's rubric judges do not
produce, so it is reported only when one is supplied and is dropped from T
otherwise.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from eva.metrics.base import MetricContext
from eva.models.results import MetricScore

FORWARD = "forward"
REVERSED = "reversed"

Compute = Callable[[MetricContext], Awaitable[MetricScore]]


@dataclass(frozen=True)
class VerdictSample:
    """One judge verdict for a single item under one presentation order."""

    order: str
    verdict: float | None


def reverse_context_order(context: MetricContext) -> MetricContext:
    """Copy a context with its transcript turns presented in reverse order.

    The paper swaps the position of the two candidates in a pairwise prompt;
    EVA's judges read a single transcript, so the equivalent perturbation is to
    hand them the same turns back-to-front.
    """
    swapped = copy.copy(context)
    swapped.conversation_trace = list(reversed(context.conversation_trace))
    return swapped


def summarize_verdicts(samples: Sequence[VerdictSample], *, ground_truth: float | None = None) -> dict[str, Any]:
    """Summarise the reliability of one item's verdict samples.

    A component is ``None`` when it could not be estimated — reproducibility
    needs at least two forward samples, order invariance needs at least one
    reversed sample, accuracy needs a ground truth — and ``trustworthy`` is the
    conjunction of whichever components were estimable.
    """
    forward = [s.verdict for s in samples if s.order == FORWARD and s.verdict is not None]
    swapped = [s.verdict for s in samples if s.order == REVERSED and s.verdict is not None]
    modal = _modal(forward)

    reproducible = all(v == modal for v in forward) if len(forward) >= 2 else None
    order_invariant = modal == _modal(swapped) if swapped else None
    accurate = modal == ground_truth if modal is not None and ground_truth is not None else None
    known = [c for c in (reproducible, order_invariant, accurate) if c is not None]

    return {
        "num_samples": len(samples),
        "unparsed_verdicts": sum(1 for s in samples if s.verdict is None),
        "verdicts": {FORWARD: forward, REVERSED: swapped},
        "modal_verdict": modal,
        "reproducible": reproducible,
        "verdict_agreement_rate": _mean([v == modal for v in forward]) if forward else None,
        "order_invariant": order_invariant,
        "accurate": accurate,
        "trustworthy": all(known) if known else None,
    }


def trustworthy_verdict_rate(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item summaries into population reliability rates.

    ``trustworthy_verdict_rate`` is the paper's T. ``position_bias_accuracy_bound``
    is the paper's ceiling on accuracy under position bias: a verdict that flips
    under an arbitrary presentation order is correct in at most one of the two
    orders, so flipped items average at most half correct.
    """

    def rate(key: str) -> float | None:
        return _mean([summary[key] for summary in summaries])

    order_invariance = rate("order_invariant")
    flip_rate = _rounded(1 - order_invariance) if order_invariance is not None else None
    accuracy_bound = _rounded((1 + order_invariance) / 2) if order_invariance is not None else None

    return {
        "num_items": len(summaries),
        "reproducibility_rate": _rounded(rate("verdict_agreement_rate")),
        "order_invariance_rate": _rounded(order_invariance),
        "accuracy_rate": _rounded(rate("accurate")),
        "trustworthy_verdict_rate": _rounded(rate("trustworthy")),
        "flip_rate": flip_rate,
        "position_bias_accuracy_bound": accuracy_bound,
    }


async def audit_metric_verdict(
    compute: Compute,
    context: MetricContext,
    *,
    primary_verdict: float | None,
    reps: int = 0,
    order_swap: bool = False,
) -> dict[str, Any]:
    """Re-run one judge metric and summarise how far its verdict can be trusted.

    ``reps`` extra identical runs estimate reproducibility; ``order_swap`` adds a
    run on the reversed-order transcript. With neither enabled no extra judge
    calls are made and an empty dict is returned, so the audit costs nothing
    while turned off.
    """
    if reps <= 0 and not order_swap:
        return {}

    samples = [VerdictSample(order=FORWARD, verdict=primary_verdict)]
    for _ in range(reps):
        samples.append(VerdictSample(order=FORWARD, verdict=verdict_of(await compute(context))))

    if order_swap:
        # The reversed condition has no primary run of its own, so it needs one more.
        swapped_context = reverse_context_order(context)
        for _ in range(reps + 1):
            samples.append(VerdictSample(order=REVERSED, verdict=verdict_of(await compute(swapped_context))))

    return summarize_verdicts(samples)


def verdict_of(score: MetricScore) -> float | None:
    """Read the verdict out of a judge metric score.

    Conversation judges report an integer ``rating``; per-turn judges only report
    the aggregate, which is just as valid a verdict for comparing across runs.
    """
    rating = score.details.get("rating")
    if _is_numeric(rating):
        return float(rating)
    return float(score.score) if _is_numeric(score.score) else None


def _is_numeric(value: Any) -> bool:
    """True for an int or float, but not a bool — a ``rating`` of true means nothing."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _modal(verdicts: Sequence[float]) -> float | None:
    """Return the most common verdict, breaking ties toward the lower verdict."""
    if not verdicts:
        return None
    counts = Counter(verdicts)
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _mean(values: Sequence[bool | float | None]) -> float | None:
    """Return the mean of the non-None entries, or None when there are none."""
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _rounded(value: float | None, decimals: int = 4) -> float | None:
    """Round to ``decimals`` places, passing None through."""
    return round(value, decimals) if value is not None else None
