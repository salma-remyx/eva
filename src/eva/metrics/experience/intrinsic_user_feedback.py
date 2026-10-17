"""Intrinsic user feedback metric — turn-level satisfaction judged in the user's own voice."""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore

# Ratings at or below this count as latent dissatisfaction for the sub-metric.
DEFAULT_DISSATISFACTION_THRESHOLD = 2


@register_metric
class IntrinsicUserFeedbackMetric(PerTurnConversationJudgeMetric):
    """Replay judge that elicits the simulated user's latent evaluation states.

    Adapted from "Training User Simulators with Latent Intrinsic Feedback"
    (arXiv:2610.13617): the judge does not score the assistant as a third-party
    evaluator — it role-plays the user and reports, for every assistant turn, the
    internal thought and reaction the user was having at that moment plus a scalar
    satisfaction rating. Elicitation is post-hoc (a replay of the finished
    conversation) because eva's user simulators generate speech inside hosted voice
    models that cannot surface latent state mid-call; the judge is instead grounded
    in the same persona and goal the simulator was given (see ``_render_user_brief``).

    Rating scale: 5 (fully satisfied) to 1 (frustrated), normalized to 0.0-1.0.
    Opt-in: run with ``--metrics intrinsic_user_feedback``.
    """

    name = "intrinsic_user_feedback"
    version = "v0.1"
    description = "Turn-level satisfaction judged from the simulated user's first-person perspective"
    category = "experience"
    rating_scale = (1, 5)
    exclude_from_default_metrics = True

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize, reading the optional ``dissatisfaction_threshold`` config override."""
        super().__init__(config)
        self.dissatisfaction_threshold = int(
            self.config.get("dissatisfaction_threshold", DEFAULT_DISSATISFACTION_THRESHOLD)
        )

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return turn IDs containing assistant content — the user reacts to what the assistant said."""
        return list(
            dict.fromkeys(
                entry.get("turn_id")
                for entry in context.conversation_trace
                if entry.get("role") == "assistant" and entry.get("turn_id") is not None
            )
        )

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "user_brief": _render_user_brief(context),
            "conversation_turns": transcript_text,
        }

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Keep the latent states (internal thought, reaction) the judge reported for this turn."""
        if rating is None:
            return {"internal_thoughts": "", "reactions": ""}
        return {
            "internal_thoughts": str(item.get("internal_thought") or ""),
            "reactions": str(item.get("reaction") or ""),
        }

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_extra: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore] | None:
        """Surface the share of turns where the user was silently annoyed or worse."""
        rated_turn_ids = [tid for tid, rating in per_turn_ratings.items() if rating is not None]
        if not rated_turn_ids:
            return None
        dissatisfied_ids = [
            tid
            for tid in rated_turn_ids
            if per_turn_ratings[tid] is not None and per_turn_ratings[tid] <= self.dissatisfaction_threshold
        ]
        return {
            "dissatisfaction_rate": make_rate_sub_metric(
                parent_name=self.name,
                key="dissatisfaction_rate",
                numerator=len(dissatisfied_ids),
                denominator=len(rated_turn_ids),
                details={
                    "count": len(dissatisfied_ids),
                    "num_rated": len(rated_turn_ids),
                    "turn_ids": dissatisfied_ids,
                },
            )
        }


def _render_user_brief(context: MetricContext) -> str:
    """Render the first-person brief the judge replays the conversation from.

    Grounds the judge in the same persona and goal the user simulator was given, so
    latent states are judged against what *this* user wanted rather than a generic
    quality rubric. Handles both structured goal dicts and plain-string goals.
    """
    lines = [f"Who you are: {context.user_persona}"]
    goal = context.user_goal
    if isinstance(goal, dict):
        lines.append(f"What you want: {goal.get('high_level_user_goal', '')}")
        decision_tree = goal.get("decision_tree") or {}
        must_have = decision_tree.get("must_have_criteria") or []
        if must_have:
            lines.append("What must happen for you to be satisfied with the outcome:")
            lines.extend(f"- {criterion}" for criterion in must_have)
        resolution = decision_tree.get("resolution_condition")
        if resolution:
            lines.append(f"When you consider your issue resolved: {resolution}")
    else:
        lines.append(f"What you want: {goal}")
    return "\n".join(str(line) for line in lines if line)
