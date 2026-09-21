# Scenario Rule Compliance

> **Accuracy Metric**: A call that violates the scenario's ground-truth rules is not a valid outcome, even if it felt smooth — the caller's actual requirements were not honored.

Part of the citation-grounded dual judges (with [conversation_quality.md](conversation_quality.md)), adapted from the MTVA-Bench scoring design: the language model inside a cascaded voice agent is graded by a **task judge** that checks the scenario's rules against ground truth, while the separate task-blind judge grades pure conversation quality — because a call can complete its task and still go badly for the caller.

## Overview

LLM-based metric that evaluates whether the assistant complied with the scenario's ground-truth rules: the caller's goal and its decision tree (`must_have_criteria` hard requirements plus negotiation, escalation, resolution, and edge-case expectations). Complementary to [task_completion.md](task_completion.md), which checks the final database state: scenario rule compliance checks the conversation itself — an action taken against the caller's stated constraints is a violation even when the end state happens to match.

Unlike the task-blind conversation quality judge, this judge sees the agent's role and instructions alongside the scenario rules, so it can evaluate rule adherence — not just how the call felt.

Every rule verdict must cite the transcript message(s) that justify it (turn, role, verbatim quote). Citations are validated programmatically against the transcript, so ungrounded verdicts are surfaced instead of silently trusted.

### Capabilities Measured

- **Language Model**: Did the model honor the scenario's rules — hard requirements, stated constraints, escalation and failure conditions — end to end?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2 (default text judge model)
- **Granularity**: Conversation-level (one rating per rule, plus an overall rating)

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: Full conversation with tool calls (via `format_transcript`)
- `user_goal`: The scenario's ground-truth rules — rendered as JSON for the judge, minus `starting_utterance` (the simulator's opener, not a rule)
- `agent_instructions`, `agent_role`: Agent configuration, so rule adherence is judged with the agent's mandate in scope

### Evaluation Methodology

The judge emits one entry per must-have criterion plus one per behavioral field the conversation actually exercised. Each entry carries a `followed` verdict, a short explanation, and citations of the transcript messages that justify it. Tool calls and their parameters count as evidence of what the assistant actually did. A rule the conversation never exercised counts as followed as long as nothing in the transcript violates it.

### Citation Grounding

Each citation is `{"turn": int, "role": "user"|"assistant", "quote": str}`. A citation is **grounded** when its quote appears verbatim in the transcript (whitespace/case tolerant) — at the cited turn and role when both resolve, anywhere in the transcript as a fallback (flagged via `location_match`). Quotes that cannot be found are reported in `ungrounded_sources` rather than trusted.

The `citation_grounding` sub-metric is the fraction of citations that were actually found in the transcript (1.0 = every verdict backed by a real quote).

### Scoring

- **Scale**: 1-3 (overall rating)
  - 3: All must-have requirements honored; no rule violations
  - 2: A minor deviation — a violated nice-to-have, edge-case, or negotiation expectation that did not defeat the caller's core requirement
  - 1: A clear violation — a must-have requirement broken, an action taken against the caller's stated constraints, or escalation/failure conditions mishandled
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Sub-metrics**:
  - `scenario_rule_compliance.rule_violation_rate`: fraction of judged rules not followed (lower is better)
  - `scenario_rule_compliance.citation_grounding`: fraction of citations grounded in the transcript (higher is better)

## Example Output

```json
{
  "name": "scenario_rule_compliance",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "explanation": {
      "rules": [
        {
          "rule": "New travel date is 2026-03-25.",
          "followed": true,
          "explanation": "Agent rebooked to March 25 as asked.",
          "citations": [{"turn": 2, "role": "assistant", "quote": "Your flight is now moved to March 25"}]
        },
        {
          "rule": "Escalate only if the rebooking fails.",
          "followed": false,
          "explanation": "Agent escalated although the rebooking succeeded.",
          "citations": [{"turn": 1, "role": "user", "quote": "a quote that never happened"}]
        }
      ]
    },
    "num_rules": 2,
    "num_rules_followed": 1,
    "num_turns": 14,
    "citation_grounding": 0.5,
    "ungrounded_sources": ["Escalate only if the rebooking fails."]
  },
  "sub_metrics": {
    "scenario_rule_compliance.rule_violation_rate": {"score": 0.5},
    "scenario_rule_compliance.citation_grounding": {"score": 0.5}
  }
}
```

## Running

**Opt-in** — excluded from the default voice-pipeline metric set; enable by naming it explicitly via `--metrics`:

```bash
# Voice-pipeline runs (python main.py)
python main.py --run-id <existing_run_id> --metrics scenario_rule_compliance

# Text-only runs (scripts/run_text_only.py): included in the default --metrics
# list (TEXT_COMPATIBLE_METRICS); add it to a composite for equal-weight
# aggregation with the task-blind judge:
python scripts/run_text_only.py \
    --accuracy-metrics task_completion,faithfulness,scenario_rule_compliance \
    --experience-metrics conversation_progression,conciseness,conversation_quality
```

The two dual-judge scores are meant to be aggregated with equal weight: include both metric names in the same mean composite (e.g. `EVA-overall_mean`).

## Related Metrics

- [conversation_quality.md](conversation_quality.md) - The task-blind half of the dual judges: grades the conversation itself, with no access to the task.
- [task_completion.md](task_completion.md) - Binary pass/fail via scenario DB state; this metric grades the conversation against the scenario's rules instead.
- [faithfulness.md](faithfulness.md) - Faithfulness to information, policies, and instructions during the conversation.

## Implementation Details

- **File**: `src/eva/metrics/accuracy/dual_judge.py`
- **Class**: `ScenarioRuleComplianceMetric`
- **Base Class**: `CitedConversationJudgeMetric` (extends `ConversationTextJudgeMetric`)
- **Prompt**: `configs/prompts/judge.yaml` under `judge.scenario_rule_compliance`
- **Configuration**: `judge_model` (default: "gpt-5.2")
