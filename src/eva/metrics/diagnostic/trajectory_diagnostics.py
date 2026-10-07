"""Budget-bounded trajectory diagnostics: rule signals first, one rubric-guided judge second.

Adapted from LiteTrajEval (arXiv:2610.03315), "Lightweight, Rubric-Guided
Trajectory Evaluation for Production AI Agents". The paper's three-stage
recipe is kept at full fidelity — a compact domain-specific rule profile
marks heuristic failure signals, the trajectory is serialized under a fixed
global character budget, and a single rubric-guided LLM judge produces a
structured diagnostic report — while the auxiliary components are
target-native: the paper's offline-mined rule profiles are replaced by a
hand-curated profile over eva's audit-log fields (tool-call error taxonomy,
per-turn latency, interruptions), and the judge reuses the standard
ConversationTextJudgeMetric harness and prompt store.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.diagnostic.tool_call_validity import CALL_ERROR_TYPES
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# Fixed key set for the rule profile — stable keys keep judge verdicts and
# sub-metrics comparable across records.
RULE_SIGNAL_KEYS = (
    "tool_call_format_error",
    "slow_assistant_turn",
    "interrupted_assistant_turn",
    "empty_assistant_turn",
)

DEFAULT_TRACE_BUDGET_CHARS = 12000
DEFAULT_SLOW_TURN_SECONDS = 8.0
# Per-entry cap applied while rendering, before the global budget assembly.
_MAX_ENTRY_CHARS = 600

_OMISSION_MARKER = "[... {count} unflagged turns omitted to fit the {budget}-char trace budget ...]"


def _turn_groups(trace: list[dict]) -> list[tuple[int, list[dict]]]:
    """Group trace entries by turn_id, preserving first-seen order."""
    groups: dict[int, list[dict]] = {}
    for entry in trace:
        groups.setdefault(entry.get("turn_id", 0), []).append(entry)
    return list(groups.items())


def detect_failure_signals(context: MetricContext, slow_turn_seconds: float = DEFAULT_SLOW_TURN_SECONDS) -> list[dict]:
    """Run the rule profile over a record's trajectory and mark heuristic failure signals.

    Signals are proposals, not verdicts: business-logic failures and benign
    interruptions are exactly the cases the judge stage disambiguates. Each
    signal is ``{"flag", "turn_id", "evidence"}`` with a short evidence string.
    """
    signals: list[dict] = []

    # Rule 1 — malformed tool calls, reusing the tool_call_validity taxonomy
    # (wrong tool name, missing/malformed parameters, invalid enums/types).
    for entry in context.conversation_trace:
        if entry.get("type") != "tool_response":
            continue
        tool_response = entry.get("tool_response")
        if not isinstance(tool_response, dict) or tool_response.get("error_type") not in CALL_ERROR_TYPES:
            continue
        error_type = tool_response.get("error_type", "")
        evidence = f"{entry.get('tool_name', '?')} -> {error_type}: {tool_response.get('message', '')}".strip()
        signals.append(
            {
                "flag": "tool_call_format_error",
                "turn_id": entry.get("turn_id"),
                "evidence": evidence,
            }
        )

    # Rule 2 — unusually slow assistant turns (pipeline stalls the user can hear).
    for turn_id, latency in sorted((context.latency_assistant_turns or {}).items()):
        if latency > slow_turn_seconds:
            signals.append(
                {
                    "flag": "slow_assistant_turn",
                    "turn_id": turn_id,
                    "evidence": f"assistant response took {latency:.1f}s (> {slow_turn_seconds:.0f}s threshold)",
                }
            )

    # Rule 3 — assistant speech overlapped/cut off by the user; often benign,
    # but a recurring marker of turn-taking failures.
    for turn_id in sorted(context.assistant_interrupted_turns or set()):
        signals.append(
            {
                "flag": "interrupted_assistant_turn",
                "turn_id": turn_id,
                "evidence": "assistant turn interrupted by user speech",
            }
        )

    # Rule 4 — assistant turns that produced neither speech nor a tool call.
    for turn_id, entries in _turn_groups(context.conversation_trace):
        has_tool_call = any(entry.get("type") == "tool_call" for entry in entries)
        if has_tool_call:
            continue
        if any(entry.get("role") == "assistant" and not (entry.get("content") or "").strip() for entry in entries):
            signals.append(
                {
                    "flag": "empty_assistant_turn",
                    "turn_id": turn_id,
                    "evidence": "assistant turn has no spoken content and no tool call",
                }
            )

    return signals


def _truncate_middle(text: str, limit: int = _MAX_ENTRY_CHARS) -> str:
    """Cap a rendered entry, keeping head and tail (diagnostically useful ends)."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 40) // 2)
    return f"{text[:half]} [...truncated {len(text) - 2 * half} chars...] {text[-half:]}"


def _render_turn(entries: list[dict]) -> str:
    """Render one turn group, capping each entry (mirrors format_transcript_with_tools)."""
    lines: list[str] = []
    for entry in entries:
        role = entry.get("role")
        if role in ("user", "assistant"):
            lines.append(f"  {role}: {_truncate_middle(entry.get('content', ''))}")
        elif entry.get("type") == "tool_call":
            params = _truncate_middle(entry.get("parameters", {}))
            lines.append(f"  tool_call: {entry.get('tool_name', '')}({params})")
        elif entry.get("type") == "tool_response":
            response = _truncate_middle(entry.get("tool_response", ""))
            lines.append(f"  tool_response: {entry.get('tool_name', '')}: {response}")
    return "\n".join(lines)


def serialize_trace_under_budget(
    trace: list[dict],
    flagged_turn_ids: set[int],
    budget_chars: int = DEFAULT_TRACE_BUDGET_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Serialize the trajectory under a fixed global character budget.

    Flagged turns are always kept (they carry the failure signals the judge
    must verdict); remaining budget is filled from the outside in, since the
    opening turns carry the goal setup and the closing turns the outcome.
    Dropped spans collapse into a single omission marker. Not all raw tokens
    are equally useful for diagnosis — the middle of an uneventful trajectory
    is the cheapest to lose.
    """
    blocks = {turn_id: _render_turn(entries) for turn_id, entries in _turn_groups(trace)}
    order = list(blocks)
    unbounded_chars = sum(len(block) + 2 for block in blocks.values())

    kept = [turn_id for turn_id in order if turn_id in flagged_turn_ids]
    used = sum(len(blocks[turn_id]) + 2 for turn_id in kept)

    # Fill leftover budget from the outside in: head, tail, head+1, tail-1, ...
    remaining = [turn_id for turn_id in order if turn_id not in flagged_turn_ids]
    candidates: list[int] = []
    while remaining:
        candidates.append(remaining.pop(0))
        if remaining:
            candidates.append(remaining.pop())
    marker_len = len(_OMISSION_MARKER.format(count=99, budget=budget_chars))
    for turn_id in candidates:
        if used + len(blocks[turn_id]) + 2 + marker_len > budget_chars:
            continue
        kept.append(turn_id)
        used += len(blocks[turn_id]) + 2
    kept_set = set(kept)

    parts: list[str] = []
    omitted_run: list[int] = []
    for turn_id in order:
        if turn_id in kept_set:
            if omitted_run:
                parts.append(_OMISSION_MARKER.format(count=len(omitted_run), budget=budget_chars))
                omitted_run = []
            parts.append(f"Turn {turn_id}:\n{blocks[turn_id]}")
        else:
            omitted_run.append(turn_id)
    if omitted_run:
        parts.append(_OMISSION_MARKER.format(count=len(omitted_run), budget=budget_chars))

    text = "\n\n".join(parts)
    stats = {
        "budget_chars": budget_chars,
        "serialized_chars": len(text),
        "unbounded_chars": unbounded_chars,
        "turns_total": len(order),
        "turns_kept": len(kept_set),
        "turns_omitted": len(order) - len(kept_set),
    }
    return text, stats


def _render_signals(signals: list[dict]) -> str:
    """Render rule signals as compact bullet lines for the judge prompt."""
    if not signals:
        return "(no rule signals fired)"
    return "\n".join(f"- {s['flag']} (turn {s['turn_id']}): {s['evidence']}" for s in signals)


@register_metric
class TrajectoryDiagnosticsJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based trajectory diagnostic (rule signals + budget-bounded trace, one judge call).

    Stage 1 (rules): a compact rule profile marks heuristic failure signals —
    malformed tool calls, slow assistant turns, interrupted assistant turns,
    empty assistant turns. Stage 2 (judge): a single rubric-guided call sees
    only the signals plus a trace serialized under a fixed character budget,
    and verdicts each signal (confirmed failure vs benign) with an overall
    rating. Per-signal sub-metrics aggregate into "fraction of records where
    the judge confirmed this failure signal".

    Opt-in judge metric: enable via ``--metrics trajectory_diagnostics``.

    Rating scale: 1 (major confirmed failures), 3 (none)
    Normalized: 1→0.0, 2→0.5, 3→1.0
    """

    name = "trajectory_diagnostics"
    version = "v0.1"
    description = (
        "Debug metric: budget-bounded LLM diagnostic that verdicts rule-flagged failure signals "
        "across the tool-call trajectory"
    )
    category = "diagnostic"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True  # Opt-in judge — excluded from the default run.
    rating_scale = (1, 3)

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.trace_budget_chars = int(self.config.get("trace_budget_chars", DEFAULT_TRACE_BUDGET_CHARS))
        self.slow_turn_seconds = float(self.config.get("slow_turn_seconds", DEFAULT_SLOW_TURN_SECONDS))

    def _build_rule_report(self, context: MetricContext) -> dict[str, Any]:
        """Run the rule stage and the budgeted serialization in one pass."""
        signals = detect_failure_signals(context, self.slow_turn_seconds)
        flagged_turn_ids = {s["turn_id"] for s in signals if s["turn_id"] is not None}
        trace_text, budget_stats = serialize_trace_under_budget(
            context.conversation_trace, flagged_turn_ids, self.trace_budget_chars
        )
        return {"signals": signals, "trace_text": trace_text, "budget": budget_stats}

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting.

        ``transcript_text`` (the inherited plain rendering) is unused: the judge
        sees the budgeted serialization built from ``context`` instead.
        """
        report = self._build_rule_report(context)
        return {
            "agent_role": context.agent_role,
            "user_goal": context.user_goal,
            "rule_flags": _render_signals(report["signals"]),
            "conversation_trace": report["trace_text"],
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
        """Build MetricScore with rule-signal details and per-signal confirmed-rate sub-metrics."""
        report = self._build_rule_report(context)
        flags_response = response.get("flags", {}) if isinstance(response, dict) else {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=flags_response,
            entry_keys=RULE_SIGNAL_KEYS,
            flag_field="confirmed",
            detail_fields=("evidence",),
        )
        confirmed = [key for key in RULE_SIGNAL_KEYS if flags_response.get(key, {}).get("confirmed")]
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": response.get("explanation", "") if isinstance(response, dict) else "",
                "rule_signals": report["signals"],
                "num_rule_signals": len(report["signals"]),
                "confirmed_signals": confirmed,
                "trace_budget": report["budget"],
                "judge_flags": flags_response if isinstance(flags_response, dict) else {},
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
