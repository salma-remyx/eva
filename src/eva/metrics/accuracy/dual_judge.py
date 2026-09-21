"""Citation-grounded dual-judge metrics for the LM inside cascaded voice agents.

Adapted from the MTVA-Bench scoring design (arXiv:2609.20152): the language
model of a cascaded agent is graded by two independent judges — a task judge
that checks the scenario's rules against ground truth, and a task-blind judge
that grades pure conversation quality — because a call can complete its task
and still go badly for the caller. Both judges must back every verdict with a
citation of a specific transcript message; here citations are structured
(turn, role, verbatim quote) and validated programmatically against the
transcript, so ungrounded verdicts are surfaced instead of silently trusted
(existing judge prompts only collect free-form "evidence" strings).

The two scores are meant to be aggregated with equal weight: include both
metric names in the same mean composite (e.g. EVA-overall_mean via
--accuracy-metrics/--experience-metrics in scripts/run_text_only.py).

Both metrics are opt-in (``exclude_from_default_metrics``) so the default
voice-pipeline metric set is unchanged; name them explicitly via ``--metrics``.
"""

import json
import re
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

QUALITY_DIMENSION_KEYS = (
    "responsiveness",
    "clarity",
    "coherence",
    "caller_experience",
)


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace and case so quoted spans match despite formatting."""
    return re.sub(r"\s+", " ", text).casefold().strip()


def validate_citations(citations: list[dict[str, Any]], conversation_trace: list[dict] | None) -> list[dict[str, Any]]:
    """Validate judge citations against the transcript they claim to quote.

    Each citation is ``{"turn": int, "role": "user"|"assistant", "quote": str}``.
    A citation is grounded (``valid=True``) when its normalized quote appears
    verbatim in the transcript — at the cited turn and role when both resolve,
    anywhere in the transcript as a fallback (flagged via ``location_match``).
    """
    messages = [
        (entry.get("turn_id"), entry.get("role"), entry.get("content", "") or "")
        for entry in conversation_trace or []
        if entry.get("role") in ("user", "assistant")
    ]

    validated: list[dict[str, Any]] = []
    for citation in citations:
        quote = citation.get("quote") or ""
        turn = citation.get("turn")
        role = citation.get("role")
        result: dict[str, Any] = {
            "turn": turn,
            "role": role,
            "quote": quote,
            "valid": False,
            "location_match": False,
        }
        normalized = _normalize_for_match(quote)
        if not normalized:
            result["reason"] = "missing or empty quote"
        elif any(
            turn == t and role == r and normalized in _normalize_for_match(content) for t, r, content in messages
        ):
            result["valid"] = True
            result["location_match"] = True
        elif any(normalized in _normalize_for_match(content) for _, _, content in messages):
            # Real quote, wrong pointer: still grounded, but not where the judge said.
            result["valid"] = True
            result["reason"] = "quote found in transcript but not at the cited turn/role"
        else:
            result["reason"] = "quote not found in transcript"
        validated.append(result)
    return validated


def collect_citations(entries: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Pair each judged rule/dimension with its citation dicts."""
    pairs: list[tuple[str, dict[str, Any]]] = []
    for key, entry in entries.items():
        for citation in entry.get("citations") or []:
            if isinstance(citation, dict):
                pairs.append((key, citation))
    return pairs


def scenario_rules_payload(user_goal: Any) -> str:
    """Render the scenario's ground-truth rules for the judge prompt."""
    if isinstance(user_goal, dict):
        # starting_utterance is the simulator's opener, not a rule — leave it out.
        rules = {key: value for key, value in user_goal.items() if key != "starting_utterance"}
    else:
        rules = {"high_level_user_goal": user_goal}
    return json.dumps(rules, indent=2, ensure_ascii=False, default=str)


class CitedConversationJudgeMetric(ConversationTextJudgeMetric):
    """Conversation judge whose evidence must be grounded in cited transcript messages.

    Subclasses ask the judge for a ``citations`` list on every verdict;
    ``attach_grounding`` validates those citations and appends a
    ``citation_grounding`` sub-metric (fraction of citations that could
    actually be found in the transcript).
    """

    exclude_from_default_metrics = True

    def attach_grounding(
        self, cited: list[tuple[str, dict[str, Any]]], context: MetricContext
    ) -> tuple[dict[str, Any], MetricScore | None]:
        """Validate cited evidence; return (details block, grounding sub-metric)."""
        validated = validate_citations([citation for _, citation in cited], context.conversation_trace)
        for (source, _), result in zip(cited, validated, strict=True):
            result["source"] = source

        details: dict[str, Any] = {"citations": validated}
        if not validated:
            return details, None

        grounded = [result for result in validated if result["valid"]]
        fraction = round(len(grounded) / len(validated), 4)
        details["citation_grounding"] = fraction
        details["ungrounded_sources"] = sorted(
            {result["source"] for result in validated} - {result["source"] for result in grounded}
        )
        sub_metric = MetricScore(
            name=f"{self.name}.citation_grounding",
            score=fraction,
            normalized_score=fraction,
            details={
                "grounded": len(grounded),
                "total": len(validated),
                "invalid": [result for result in validated if not result["valid"]],
            },
        )
        return details, sub_metric


@register_metric
class ScenarioRuleComplianceMetric(CitedConversationJudgeMetric):
    """Task judge: did the agent satisfy the scenario's ground-truth rules?

    Judges the conversation against the scenario rules (the user goal's
    decision tree) with the agent's instructions in scope — unlike the
    task-blind conversation_quality judge. Every rule verdict must cite the
    transcript messages that justify it.

    Rating scale: 3 (rules followed), 2 (minor deviation), 1 (clear violation)
    Normalized: 3→1.0, 2→0.5, 1→0.0
    """

    name = "scenario_rule_compliance"
    version = "v0.1"
    description = "LLM judge of compliance with the scenario's task rules, with citation-grounded evidence"
    category = "accuracy"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "agent_role": context.agent_role,
            "agent_instructions": context.agent_instructions,
            "scenario_rules": scenario_rules_payload(context.user_goal),
            "conversation_trace": transcript_text,
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
        """Build MetricScore with per-rule compliance and citation-grounding sub-metrics."""
        rules = [rule for rule in (response.get("rules") or []) if isinstance(rule, dict)]
        rule_entries = {rule.get("rule") or f"rule_{i}": rule for i, rule in enumerate(rules)}
        grounding_details, grounding_sub = self.attach_grounding(collect_citations(rule_entries), context)

        num_followed = sum(1 for rule in rules if rule.get("followed"))
        sub_metrics: dict[str, MetricScore] = {}
        if rules:
            violation_rate = round(1.0 - num_followed / len(rules), 4)
            sub_metrics["rule_violation_rate"] = MetricScore(
                name=f"{self.name}.rule_violation_rate",
                score=violation_rate,
                normalized_score=violation_rate,
                details={"num_rules": len(rules), "num_followed": num_followed},
            )
        if grounding_sub is not None:
            sub_metrics["citation_grounding"] = grounding_sub

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": {"rules": rules},
                "num_rules": len(rules),
                "num_rules_followed": num_followed,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
                **grounding_details,
            },
            sub_metrics=sub_metrics or None,
        )


@register_metric
class ConversationQualityMetric(CitedConversationJudgeMetric):
    """Task-blind judge: how was the conversation, independent of the task?

    Sees only the transcript — no user goal, scenario rules, agent role or
    instructions — so it cannot reward task completion; it grades how the
    assistant handled the conversation itself (MTVA-Bench's task-blind
    conversation-quality judge).

    Rating scale: 3 (good quality), 2 (mixed), 1 (poor)
    Normalized: 3→1.0, 2→0.5, 1→0.0
    """

    name = "conversation_quality"
    version = "v0.1"
    description = "Task-blind LLM judge of conversation quality, with citation-grounded evidence"
    category = "experience"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return only the transcript — the judge is deliberately task-blind."""
        return {"conversation_trace": transcript_text}

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with per-dimension issue-flag and citation-grounding sub-metrics."""
        dimensions = {
            key: entry for key, entry in (response.get("dimensions") or {}).items() if isinstance(entry, dict)
        }
        grounding_details, grounding_sub = self.attach_grounding(collect_citations(dimensions), context)

        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=QUALITY_DIMENSION_KEYS,
            flag_field="quality_issue",
            detail_fields=("rating", "citations"),
        )
        if grounding_sub is not None:
            sub_metrics["citation_grounding"] = grounding_sub

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": {"dimensions": dimensions},
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
                **grounding_details,
            },
            sub_metrics=sub_metrics or None,
        )
