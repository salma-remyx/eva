"""Misconception-correction metric using LLM-as-judge (per-turn).

Adapted from ThReadMed-QA (https://arxiv.org/abs/2607.12884), which found that
LLMs correct a user's embedded *false presupposition* well on the first turn
(~85%) but degrade sharply over follow-up turns (~50% within two follow-ups),
and that much of that drop is driven by error propagation across turns.

This metric scores each assistant turn on whether it identifies and corrects a
false presupposition in the user's preceding message(s), then surfaces how that
correction holds up across later turns — the across-turn degradation that is
the paper's central finding.

Note: ``version`` is intentionally left unset. The path guardrails on this
change forbid editing the checked-in ``metric_signatures.json`` fixture, so the
metric is deliberately kept out of the versioned drift set. A maintainer who
wires it into ``experience/__init__.py`` should set ``version`` and regenerate
``tests/fixtures/metric_signatures.json``.
"""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import normalize_rating
from eva.metrics.versioning import _CURRENT_PROMPT_HASH, hash_prompt_template
from eva.models.results import MetricScore

# PromptManager shallow-merges each YAML file's top-level keys, so a second file
# keyed under ``judge:`` would overwrite ``judge.yaml``'s entire ``judge`` dict.
# This metric keeps its prompt in a dedicated namespace to avoid that clobber,
# and resolves it via the get_judge_prompt override below.
_PROMPT_NAMESPACE = "judge_misconception"


@register_metric
class MisconceptionCorrectionMetric(PerTurnConversationJudgeMetric):
    """LLM-based, per-turn misconception-correction metric.

    For each assistant turn the judge decides whether the user's preceding
    message(s) embed a false presupposition (an incorrect assumption, outdated
    belief, or wrong premise stated as fact) and, when one is present, whether
    the assistant identified and corrected it. Ratings aggregate across turns,
    and sub-metrics expose the across-turn degradation ThReadMed-QA reports:

    - ``first_turn_correction_accuracy`` — normalized correction on the first
      rated assistant turn (the paper's strong single-turn baseline).
    - ``later_turn_correction_accuracy`` — mean normalized correction on the
      remaining turns (the degraded multi-turn regime).
    - ``error_propagation_rate`` — fraction of later turns whose rating fell
      below the first turn's rating (the error-propagation signal); lower is
      better.

    Rating scale: 3 (corrected), 2 (partially addressed), 1 (failed / propagated).
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0.
    """

    name = "misconception_correction"
    description = "LLM judge of whether the assistant corrects user false presuppositions across turns"
    category = "experience"
    rating_scale = (1, 3)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return unique turn IDs from the conversation trace, preserving order."""
        turn_ids: list[int] = []
        seen: set[int] = set()
        for entry in context.conversation_trace:
            turn_id = entry.get("turn_id")
            if turn_id is not None and turn_id not in seen:
                seen.add(turn_id)
                turn_ids.append(turn_id)
        return turn_ids

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"conversation_turns": transcript_text}

    def get_judge_prompt(self, prompt_key: str = "user_prompt", **variables: Any) -> str:
        """Load this metric's prompt from its dedicated (non-clobbering) namespace."""
        prompt_path = f"{_PROMPT_NAMESPACE}.{self.name}.{prompt_key}"
        _CURRENT_PROMPT_HASH.set(hash_prompt_template(self.prompt_manager.get_template(prompt_path)))
        return self.prompt_manager.get_prompt(prompt_path, **variables)

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Extract per-turn misconception/correction flags from the judge item."""
        if rating is None:
            return {"corrected": None, "misconception_present": None}

        misconception_present = bool(item.get("misconception_present"))
        corrected = bool(item.get("corrected"))
        # A turn can only fully "correct" at the top of the rating scale.
        if rating < self.rating_scale[1] and corrected:
            self.logger.warning(f"[{context.record_id}] Turn {turn_id}: rating={rating} but corrected=True; clearing")
            corrected = False
        return {"corrected": corrected, "misconception_present": misconception_present}

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_extra: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore] | None:
        """Surface the across-turn degradation sub-metrics (ThReadMed-QA's core finding)."""
        rated_ratings = {tid: rating for tid, rating in per_turn_ratings.items() if rating is not None}
        if not rated_ratings:
            return None

        rated_turn_ids = sorted(rated_ratings)
        min_rating, max_rating = self.rating_scale
        sub_metrics: dict[str, MetricScore] = {}

        first_turn_id = rated_turn_ids[0]
        first_rating = rated_ratings[first_turn_id]
        first_normalized = normalize_rating(first_rating, min_rating, max_rating)
        sub_metrics["first_turn_correction_accuracy"] = MetricScore(
            name=f"{self.name}.first_turn_correction_accuracy",
            score=round(first_normalized, 3),
            normalized_score=round(first_normalized, 3),
            details={
                "turn_id": first_turn_id,
                "rating": first_rating,
                "corrected": per_turn_extra.get(first_turn_id, {}).get("corrected"),
            },
        )

        later_turn_ids = rated_turn_ids[1:]
        if not later_turn_ids:
            return sub_metrics

        later_normalized = [normalize_rating(rated_ratings[tid], min_rating, max_rating) for tid in later_turn_ids]
        later_accuracy = sum(later_normalized) / len(later_normalized)
        sub_metrics["later_turn_correction_accuracy"] = MetricScore(
            name=f"{self.name}.later_turn_correction_accuracy",
            score=round(later_accuracy, 3),
            normalized_score=round(later_accuracy, 3),
            details={"turn_ids": later_turn_ids, "ratings": {tid: rated_ratings[tid] for tid in later_turn_ids}},
        )

        propagated_turn_ids = [tid for tid in later_turn_ids if rated_ratings[tid] < first_rating]
        propagation_rate = len(propagated_turn_ids) / len(later_turn_ids)
        sub_metrics["error_propagation_rate"] = MetricScore(
            name=f"{self.name}.error_propagation_rate",
            score=round(propagation_rate, 3),
            normalized_score=round(propagation_rate, 3),
            details={
                "count": len(propagated_turn_ids),
                "num_later_turns": len(later_turn_ids),
                "turn_ids": propagated_turn_ids,
                "first_turn_rating": first_rating,
            },
        )

        return sub_metrics
