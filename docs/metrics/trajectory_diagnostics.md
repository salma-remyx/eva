# Trajectory Diagnostics

> **Diagnostic Metric**: Budget-bounded LLM diagnostic that verdicts rule-flagged failure signals across the tool-call trajectory — not scored directly since it overlaps with the accuracy and experience categories.

## Overview

Two-stage metric that localizes *why* a conversation went wrong without sending the full raw trajectory to the judge:

1. **Rule stage (deterministic)** — a compact rule profile scans the trajectory and marks heuristic failure signals.
2. **Judge stage (single call)** — one rubric-guided LLM call sees only the signals plus a trace serialized under a fixed character budget, and verdicts each signal (confirmed failure vs benign) with an overall rating.

Adapted from LiteTrajEval ([arXiv:2610.03315](https://arxiv.org/abs/2610.03315)), "Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents". The paper's offline-mined rule profiles are replaced by a hand-curated profile over EVA's audit-log fields; the judge reuses the standard `ConversationTextJudgeMetric` harness.

### Capabilities Measured

- **Language Model**: Well-formed tool calls and recoverable failure handling.
- **Pipeline**: Response stalls and turn-taking breakdowns with user-visible impact.

## How It Works

### Rule Profile

| Signal | Rule |
|--------|------|
| `tool_call_format_error` | Tool response carries an `error_type` in the `tool_call_validity` taxonomy (wrong tool name, malformed parameters, invalid enums/types). Business-logic errors (e.g. `not_found`) are left to the judge. |
| `slow_assistant_turn` | Assistant turn latency exceeds `slow_turn_seconds` (default 8.0s). |
| `interrupted_assistant_turn` | Turn appears in `assistant_interrupted_turns`. |
| `empty_assistant_turn` | Assistant turn produced neither spoken content nor a tool call. |

Signals are proposals, not verdicts — the judge confirms or dismisses each one using conversation context.

### Budgeted Serialization

The trace is rendered with per-entry head/tail truncation, then assembled under a global character budget (`trace_budget_chars`, default 12000). Flagged turns are always kept; remaining budget is filled from the outside in (opening turns carry the goal setup, closing turns the outcome); dropped spans collapse into a single `[... N unflagged turns omitted ...]` marker. Budget statistics are recorded in `details.trace_budget` so cost-efficiency is measurable per record.

### Scoring

- **Scale**: 1-3 (1 = major confirmed failures, 2 = minor, 3 = none confirmed)
- **Normalization**: 1→0.0, 2→0.5, 3→1.0
- **Sub-metrics**: `<signal>_rate` per rule key — 1.0 when the judge confirmed that signal type; aggregates as "fraction of records where this failure was confirmed".

## Example Output

```json
{
  "name": "trajectory_diagnostics",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "explanation": "Malformed confirmation number caused a failed lookup; the retry recovered.",
    "rule_signals": [
      {
        "flag": "tool_call_format_error",
        "turn_id": 1,
        "evidence": "get_reservation -> invalid_confirmation_number_format: Invalid 'X'"
      }
    ],
    "confirmed_signals": ["tool_call_format_error"],
    "trace_budget": {
      "budget_chars": 12000,
      "serialized_chars": 147,
      "unbounded_chars": 141,
      "turns_total": 1,
      "turns_kept": 1,
      "turns_omitted": 0
    }
  },
  "sub_metrics": {
    "tool_call_format_error_rate": {"score": 1.0}
  }
}
```

## Related Metrics

- [tool_call_validity.md](tool_call_validity.md) - Deterministic fraction of well-formed tool calls (rule taxonomy shared with this metric's first stage)
- [response_speed.md](response_speed.md) - Latency measurement underlying the slow-turn rule

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/trajectory_diagnostics.py`
- **Class**: `TrajectoryDiagnosticsJudgeMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Configuration**: `trace_budget_chars` (default 12000), `slow_turn_seconds` (default 8.0)
- **Opt-in**: excluded from the default run; enable via `--metrics trajectory_diagnostics`
