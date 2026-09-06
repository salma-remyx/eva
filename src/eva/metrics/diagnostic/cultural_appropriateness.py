"""Cultural appropriateness judge metric (opt-in diagnostic)."""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore
from eva.utils.cultural_grounding import build_agent_cultural_brief, cultural_aspect_keys


@register_metric
class CulturalAppropriatenessMetric(ConversationTextJudgeMetric):
    """Judge whether the agent inferred and respected the user's implicit cultural constraints.

    The user simulator on the multilingual path carries per-language implicit cultural
    expectations (formality register, date/time/number formats, greeting norms — see
    ``eva.utils.cultural_grounding``) that it never states aloud. This judge scores
    whether the agent noticed and accommodated them, mirroring the inference-based
    cultural evaluation of CultureConverse (arXiv:2608.28405) on eva's languages.

    **Opt-in** — excluded from the default run; enable via ``--metrics cultural_appropriateness``.
    Records whose language has no grounding profile (e.g. English) get an error score
    rather than a pass.

    Rating scale: 3 (accommodated all applicable aspects), 2 (minor slips), 1 (clear violations)
    """

    name = "cultural_appropriateness"
    version = "v0.1"
    description = "Opt-in diagnostic: agent respected the user's implicit cultural constraints"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    rating_scale = (1, 3)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Run the judge, or return an error score when the language has no grounding profile."""
        if build_agent_cultural_brief(context.language) is None:
            return MetricScore(
                name=self.name,
                score=0.0,
                normalized_score=0.0,
                error=f"No cultural grounding profile for language {context.language!r}; "
                f"add one under configs/cultural_grounding.yaml or run on a grounded language.",
                details={"language": context.language},
            )
        return await super().compute(context)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "language_display_name": context.language_display_name,
            "conversation": transcript_text,
            "cultural_brief": build_agent_cultural_brief(context.language),
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
        """Build MetricScore with the per-aspect analysis and violation-rate sub-metrics."""
        aspect_analysis = response.get("aspect_analysis", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=aspect_analysis,
            entry_keys=cultural_aspect_keys(context.language),
            flag_field="violated",
            detail_fields=("analysis",),
        )

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "aspect_analysis": aspect_analysis,
                "language": context.language,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
