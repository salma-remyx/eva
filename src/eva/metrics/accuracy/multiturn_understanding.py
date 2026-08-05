"""Deep multi-turn understanding metric using LLM-as-judge (whole conversation).

Adapted from *Hy-MultiTurn: A Six-Dimensional Benchmark for Deep Multi-Turn
Dialogue Understanding* (arXiv:2607.29196). The paper derives six recurring
failure mechanisms from real chatbot failures and uses them to define six
controlled evaluation modes for long multi-turn dialogues. This metric ports
the **language-agnostic capability rubric** (the six failure mechanisms) onto
EVA's existing conversation-trace judge, deliberately dropping the paper's
Chinese benchmark dataset — evaluation here runs over EVA's own transcribed
voice-agent traces.

Each of the six dimensions is surfaced as a binary failure-flag sub-metric
(lower-is-better ``_rate``), following the same shape as ``faithfulness`` and
``conversation_progression``: the judge flags whether each failure occurred in
the conversation, and the parent rating summarizes overall multi-turn
understanding.
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

# The six deep multi-turn understanding failure modes, mapped 1:1 from
# Hy-MultiTurn's evaluation modes. Each is a binary issue flag (true = the
# failure occurred in this conversation) aggregated as a lower-is-better rate.
_MULTITURN_UNDERSTANDING_DIMENSION_KEYS = (
    "constraint_memory_failure",  # paper mode: constraint memory
    "precise_execution_error",  # paper mode: precise execution
    "constraint_synthesis_failure",  # paper mode: constraint synthesis
    "object_localization_error",  # paper mode: object localization
    "action_suppression_failure",  # paper mode: action suppression
    "reference_resolution_error",  # paper mode: reference resolution
)


@register_metric
class MultiTurnUnderstandingJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based deep multi-turn understanding metric (whole conversation).

    Evaluates whether the assistant correctly tracked and acted on information
    across a long, multi-turn conversation, across six capability dimensions
    adapted from Hy-MultiTurn. Each dimension is a binary failure flag.

    Rating scale: 3 (no understanding failures), 2 (minor), 1 (clear failure)
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0
    """

    name = "multiturn_understanding"
    version = "v0.1"
    description = (
        "LLM judge evaluation of deep multi-turn understanding across six "
        "capability dimensions: constraint memory, precise execution, "
        "constraint synthesis, object localization, action suppression, "
        "and reference resolution"
    )
    category = "accuracy"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "agent_role": context.agent_role,
            "agent_instructions": context.agent_instructions,
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
        """Build MetricScore with analysis details and per-dimension failure-flag sub-metrics."""
        dimensions = response.get("dimensions", {}) if isinstance(response, dict) else {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_MULTITURN_UNDERSTANDING_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {
            "dimensions": dimensions,
            "flags_count": response.get("flags_count", ""),
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
