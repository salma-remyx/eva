"""Clarification-policy metric using LLM-as-judge (whole conversation).

Adapted from RegretBench (https://arxiv.org/abs/2607.21143v1), which frames
clarification as a sequential *policy* decision — whether to ask, what to ask,
when to stop asking, and when to answer — and scores an assistant by the
*regret* (value lost) relative to an optimal clarification policy.

This is a Mode-2 (adapted) port: the paper's regret-based objective and its
clarification signals (intent resolution, ineffective clarification, and the
stopping/interaction-cost decision) are preserved as the judge rubric in
``configs/prompts/judge.yaml``. The auxiliary components that eva does not host
— the paper's free-form multi-turn user simulator and its semantic-state-tracking
reference policy — are substituted with eva's native single-conversation judge
shape (``ConversationTextJudgeMetric``), using the record's ground-truth
``user_goal`` as the intent-resolution reference that RegretBench derives from a
hidden-intent formulation.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import (
    get_assistant_turns_disclaimer,
    get_user_turns_disclaimer,
)
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# Clarification-policy dimensions, mapped from RegretBench's regret signals.
_CLARIFICATION_DIMENSION_KEYS = (
    # Intent resolution: the assistant should have asked but proceeded on an assumption.
    "missed_clarification",
    # The assistant asked, but the question did not advance intent resolution.
    "ineffective_clarification",
    # Stopping / interaction cost: answered before intent was clear, or kept asking after it was.
    "poor_stopping_decision",
)


@register_metric
class ClarificationPolicyJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based clarification-policy metric (whole conversation).

    Evaluates the assistant's clarification behavior as a *policy* — whether it
    asked the right question at the right time and stopped once the user's intent
    was clear — rather than scoring isolated question quality. The overall rating
    reflects *regret*: how much value the chosen clarification policy lost relative
    to an optimal path toward the ground-truth ``user_goal``.

    Rating scale: 3 (efficient policy, low regret), 2 (minor inefficiency), 1 (clear policy failure, high regret)
    Normalized: 3→1.0, 2→0.5, 1→0.0
    """

    name = "clarification_policy"
    version = "v0.1"
    description = (
        "LLM judge evaluation of clarification as a policy behavior "
        "(whether to ask, what to ask, when to stop) relative to an optimal path"
    )
    category = "experience"
    # Opt-in metric: select by name via the `metrics` config field / --metrics flag.
    exclude_from_default_metrics = True
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "conversation_trace": transcript_text,
            "user_goal": context.user_goal,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "assistant_turns_disclaimer": get_assistant_turns_disclaimer(context.is_audio_native),
        }

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with analysis details and per-dimension issue-flag sub-metrics."""
        dimensions = response.get("dimensions", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_CLARIFICATION_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {"dimensions": dimensions}
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": analysis,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
