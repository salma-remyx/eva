"""Proactive dialogue metric using LLM-as-judge (whole conversation).

Adapted from ProactiveEval (ProactiveEval: A Unified Evaluation Framework for
Proactive Dialogue Agents, arxiv:2508.20973), which decomposes proactive
dialogue into two per-conversation judge dimensions. The upstream artefact
carries no licence, so the rubric is reimplemented here in EVA's idiom rather
than copied from the paper's code.

This is an *initiative* axis that complements EVA's existing experience
metrics: ``conversation_progression`` scores whether the assistant *avoided
regressing* (wasted tool calls, lost information, redundant turns), whereas
``proactive_dialogue`` scores whether the assistant *actively advanced and
guided* the exchange toward the user's goal. The two are orthogonal — a
passive agent can avoid every progression failure yet never take initiative.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_assistant_turns_disclaimer, get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# Proactive-dialogue dimensions, adapted from ProactiveEval's decomposition of
# proactive dialogue into a target-planning axis and a dialogue-guidance axis.
_PROACTIVE_DIMENSION_KEYS = (
    "target_planning",
    "dialogue_guidance",
)


@register_metric
class ProactiveDialogueJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based proactive-dialogue metric (conversation-level).

    Evaluates how proactively the assistant drove the conversation along two
    dimensions adapted from ProactiveEval — ``target_planning`` (working toward
    the user's goal) and ``dialogue_guidance`` (steering the dialogue itself).
    Where ``conversation_progression`` measures avoiding regressions, this
    measures taking initiative.

    Rating scale: 3 (strongly proactive), 2 (adequate), 1 (passive/reactive)
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0
    """

    name = "proactive_dialogue"
    version = "v0.1"
    description = "LLM judge of how proactively the assistant drove the conversation toward the user's goal"
    category = "experience"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "agent_role": context.agent_role,
            "user_goal": context.user_goal,
            "conversation_trace": transcript_text,
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
        """Build MetricScore with per-dimension detail and proactiveness-deficiency sub-metrics."""
        dimensions = response.get("dimensions", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_PROACTIVE_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {
            "dimensions": dimensions,
            "explanation": response.get("explanation", ""),
        }
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
