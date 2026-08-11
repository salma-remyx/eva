"""Benchmark-quality audit metric using a reference-free LLM judge.

Diagnostic metric that audits the quality of a single benchmark scenario's
*definition* — not the conversation it produced. For each record it asks an
LLM judge to rate three reference-free axes of benchmark quality:

  - consistency: is the expected outcome coherent with the task description
    and free of internal contradictions?
  - complexity: is the scenario non-trivial (multi-step, requires reasoning
    or tool use) rather than solvable in a single shallow turn?
  - policy_coverage: does the scenario exercise the agent's stated
    instructions / policy (rules, constraints, allowed actions)?

The judge also returns actionable diagnostics (concrete weaknesses), giving a
meta-evaluation layer over eva's own benchmark scenarios.

The scenario-definition fields the source paper calls description / policy /
expected_behavior map directly onto MetricContext's ``user_goal``,
``agent_instructions`` and ``expected_scenario_db``, so ``compute(context)``
reads fields the context already carries — no new data plumbing.

Adapted (Mode 2) from "Benchmarking the Benchmarks: Evaluating Benchmarks for
Conversational Agents" (arXiv:2608.06329). The paper's standalone
benchmark-evaluation framework is substituted with eva's existing
``TextJudgeMetric`` infrastructure (LLMClient, judge-response parsing, prompt
hashing, token-usage logging). The paper's human-annotation agreement study
and controlled-perturbation validation are intentionally out of scope here:
they validate the auditor rather than ship it, and evaluation of the auditor
belongs in a downstream PR.
"""

import json
from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import normalize_rating
from eva.models.results import MetricScore

# Per-dimension rating scale: 1 = poor, 2 = acceptable, 3 = strong.
_RATING_SCALE = (1, 3)
_DIMENSIONS = ("consistency", "complexity", "policy_coverage")
# Cap the (potentially large) expected-scenario DB dump that goes into the prompt.
_MAX_SCENARIO_DB_CHARS = 2000
_MAX_INSTRUCTIONS_CHARS = 4000


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + " …[truncated]"


@register_metric
class BenchmarkAuditMetric(TextJudgeMetric):
    """Reference-free LLM-judge audit of a single benchmark scenario's quality.

    Rates each scenario on consistency, complexity, and policy coverage (1-3),
    emits one sub-metric per dimension, and surfaces actionable diagnostics.
    Opt-in diagnostic: it measures the benchmark scenario, not the agent's
    conversation, so it is excluded from default metrics and from pass@k.
    """

    name = "benchmark_audit"
    version = "v0.1"
    description = "Diagnostic: reference-free LLM-judge audit of benchmark scenario quality"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    rating_scale = _RATING_SCALE

    def _build_prompt(self, context: MetricContext) -> str:
        """Render the audit prompt from the scenario-definition fields.

        The template lives at ``judge.benchmark_audit.user_prompt`` in
        configs/prompts/judge.yaml; ``get_judge_prompt`` stamps its hash into the
        per-run versioning trail, so prompt edits are detectable without a manual
        version bump.
        """
        scenario_db = json.dumps(context.expected_scenario_db, ensure_ascii=False, default=str)
        variables = {
            "user_goal": context.user_goal.strip() or "(not provided)",
            "agent_instructions": _truncate(context.agent_instructions.strip(), _MAX_INSTRUCTIONS_CHARS)
            or "(not provided)",
            "expected_scenario_db": _truncate(scenario_db, _MAX_SCENARIO_DB_CHARS) or "(not provided)",
        }
        return self.get_judge_prompt("user_prompt", **variables)

    def _coerce_rating(self, rating: Any, label: str, context: MetricContext) -> int | None:
        """Coerce a judge rating to an int within the rating scale, else None."""
        min_r, max_r = self.rating_scale
        try:
            rating_int = int(rating)
        except (TypeError, ValueError):
            self.logger.warning(f"[{context.record_id}] Non-numeric rating for {label}: {rating!r}")
            return None
        if not min_r <= rating_int <= max_r:
            self.logger.warning(f"[{context.record_id}] Out-of-range rating for {label}: {rating_int}")
            return None
        return rating_int

    async def compute(self, context: MetricContext) -> MetricScore:
        """Audit the scenario definition and return per-dimension + overall scores."""
        try:
            if not (context.user_goal and context.user_goal.strip()):
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    skipped=True,
                    error="No scenario description (user_goal) to audit",
                )

            prompt = self._build_prompt(context)
            response, raw_response = await self.call_judge(prompt, context)
            if response is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Failed to parse judge response",
                    details={"judge_prompt": prompt, "judge_raw_response": raw_response},
                )

            min_r, max_r = self.rating_scale
            sub_metrics: dict[str, MetricScore] = {}
            dim_ratings: dict[str, int | None] = {}
            dim_explanations: dict[str, str] = {}
            normalized_scores: list[float] = []
            raw_ratings: list[int] = []

            for dim in _DIMENSIONS:
                raw_entry = response.get(dim)
                entry: dict[str, Any] = raw_entry if isinstance(raw_entry, dict) else {}
                explanation = entry.get("explanation", "")
                dim_explanations[dim] = explanation
                rating_int = self._coerce_rating(entry.get("rating"), dim, context)
                dim_ratings[dim] = rating_int

                if rating_int is None:
                    sub_metrics[dim] = MetricScore(
                        name=f"{self.name}.{dim}",
                        score=None,
                        normalized_score=None,
                        details={"explanation": explanation, "invalid_rating": entry.get("rating")},
                    )
                    continue

                normalized = normalize_rating(rating_int, min_r, max_r)
                normalized_scores.append(normalized)
                raw_ratings.append(rating_int)
                sub_metrics[dim] = MetricScore(
                    name=f"{self.name}.{dim}",
                    score=float(rating_int),
                    normalized_score=round(normalized, 4),
                    details={"rating": rating_int, "explanation": explanation},
                )

            if not normalized_scores:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="All dimensions failed to evaluate",
                    details={
                        "judge_prompt": prompt,
                        "judge_raw_response": raw_response,
                        "dimension_ratings": dim_ratings,
                        "dimension_explanations": dim_explanations,
                    },
                )

            overall_raw = round(sum(raw_ratings) / len(raw_ratings), 3)
            overall_normalized = round(sum(normalized_scores) / len(normalized_scores), 4)

            diagnostics = response.get("diagnostics", [])
            if not isinstance(diagnostics, list):
                diagnostics = []

            # Judge's holistic rating, kept for transparency only (not used to score).
            holistic_rating = self._coerce_rating(response.get("rating"), "overall", context)

            details: dict[str, Any] = {
                "dimension_ratings": dim_ratings,
                "dimension_explanations": dim_explanations,
                "overall_rating": holistic_rating,
                "diagnostics": diagnostics,
                "explanation": response.get("explanation", ""),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            }

            return MetricScore(
                name=self.name,
                score=overall_raw,
                normalized_score=overall_normalized,
                details=details,
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)
