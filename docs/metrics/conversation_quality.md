# Conversation Quality

> **Experience Metric**: A call may complete its task perfectly and still have been a bad call for the caller — dismissive tone, rambling answers, or lost context degrade the experience regardless of correctness.

Part of the citation-grounded dual judges (with [scenario_rule_compliance.md](scenario_rule_compliance.md)), adapted from the MTVA-Bench scoring design: this is the **task-blind judge** — it sees only the transcript, with no user goal, scenario rules, agent role, or instructions, so it cannot reward task completion. It grades how the assistant handled the conversation itself, the way any listener would hear it.

## Overview

LLM-based metric that evaluates conversation quality along four dimensions, each scored independently and each verdict backed by a citation of a specific transcript message. Because the judge is task-blind, a high score here says nothing about whether the task was achieved — pair it with the task judge ([scenario_rule_compliance.md](scenario_rule_compliance.md)) for the dual-judge picture.

Every dimension verdict must cite the transcript message(s) that justify it (turn, role, verbatim quote). Citations are validated programmatically against the transcript, so ungrounded verdicts are surfaced instead of silently trusted.

### Capabilities Measured

- **Language Model**: Did the model conduct the conversation well — responsive, clear, self-consistent, and decent to the caller — independent of the task?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2 (default text judge model)
- **Granularity**: Conversation-level (one rating per dimension, plus an overall rating)

### Input Data

Uses only `conversation_trace` from MetricContext (via `format_transcript`). The prompt variables deliberately exclude `user_goal`, scenario rules, `agent_role`, and `agent_instructions` — that is what makes the judge task-blind.

`[assistant interrupts]`-style markers and `tool_call:` / `tool_response:` lines are treated as non-spoken metadata and backend traffic — context about what happened, never words the caller heard. Speech-recognition noise is not penalized itself; the assistant's handling of it is what gets judged.

### Evaluation Methodology

The judge scores four dimensions, each as a binary flag (true = quality issue present) plus a rating:

1. **responsiveness** — Did the assistant address what the caller actually said? (talking past questions, leaving requests unhandled)
2. **clarity** — Were responses understandable and appropriately concise for speech? (rambling turns, unexplained jargon, key information buried)
3. **coherence** — Was the assistant consistent with itself across the conversation? (contradicting earlier statements, re-deciding settled points)
4. **caller_experience** — Did the caller get a decent experience? (dismissive tone, steamrolling, unacknowledged frustration, fumbled openings/closings)

### Citation Grounding

Each citation is `{"turn": int, "role": "user"|"assistant", "quote": str}`. A citation is **grounded** when its quote appears verbatim in the transcript (whitespace/case tolerant) — at the cited turn and role when both resolve, anywhere in the transcript as a fallback (flagged via `location_match`). Quotes that cannot be found are reported in `ungrounded_sources` rather than trusted.

The `citation_grounding` sub-metric is the fraction of citations that were actually found in the transcript (1.0 = every verdict backed by a real quote).

### Scoring

- **Scale**: 1-3 (overall rating)
  - 3: No dimension flagged
  - 2: One or two minor (rating-2) flags and nothing worse
  - 1: Any dimension rated 1, or three or more dimensions flagged
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Sub-metrics**:
  - `conversation_quality.<dimension>_rate` (one per dimension): 1.0 when that quality issue was flagged, 0.0 otherwise — aggregated across records this reads as the fraction of records where the issue occurred (lower is better)
  - `conversation_quality.citation_grounding`: fraction of citations grounded in the transcript (higher is better)

## Example Output

```json
{
  "name": "conversation_quality",
  "score": 3.0,
  "normalized_score": 1.0,
  "details": {
    "rating": 3,
    "explanation": {
      "dimensions": {
        "responsiveness": {"quality_issue": false, "rating": 3, "explanation": "Addressed every request.", "citations": [{"turn": 1, "role": "assistant", "quote": "I can help with that"}]},
        "clarity": {"quality_issue": false, "rating": 3, "explanation": "Responses were brief and clear.", "citations": [{"turn": 2, "role": "assistant", "quote": "Your flight is now moved to March 25"}]},
        "coherence": {"quality_issue": false, "rating": 3, "explanation": "Consistent throughout.", "citations": [{"turn": 2, "role": "assistant", "quote": "keeping your window seat"}]},
        "caller_experience": {"quality_issue": false, "rating": 3, "explanation": "Courteous start to finish.", "citations": [{"turn": 1, "role": "assistant", "quote": "I can help with that"}]}
      }
    },
    "num_turns": 14,
    "citation_grounding": 1.0
  },
  "sub_metrics": {
    "conversation_quality.responsiveness_rate": {"score": 0.0},
    "conversation_quality.clarity_rate": {"score": 0.0},
    "conversation_quality.coherence_rate": {"score": 0.0},
    "conversation_quality.caller_experience_rate": {"score": 0.0},
    "conversation_quality.citation_grounding": {"score": 1.0}
  }
}
```

## Running

**Opt-in** — excluded from the default voice-pipeline metric set; enable by naming it explicitly via `--metrics`:

```bash
# Voice-pipeline runs (python main.py)
python main.py --run-id <existing_run_id> --metrics conversation_quality

# Text-only runs (scripts/run_text_only.py): included in the default --metrics
# list (TEXT_COMPATIBLE_METRICS); add it to a composite for equal-weight
# aggregation with the task judge:
python scripts/run_text_only.py \
    --accuracy-metrics task_completion,faithfulness,scenario_rule_compliance \
    --experience-metrics conversation_progression,conciseness,conversation_quality
```

The two dual-judge scores are meant to be aggregated with equal weight: include both metric names in the same mean composite (e.g. `EVA-overall_mean`).

## Related Metrics

- [scenario_rule_compliance.md](scenario_rule_compliance.md) - The task judge half of the dual judges: checks the scenario's ground-truth rules with the agent's mandate in scope.
- [conciseness.md](conciseness.md) - Per-turn conciseness for voice; the clarity dimension here is conversation-level.
- [conversation_progression.md](conversation_progression.md) - Whether the assistant moves the conversation forward without repetition.

## Implementation Details

- **File**: `src/eva/metrics/accuracy/dual_judge.py`
- **Class**: `ConversationQualityMetric`
- **Base Class**: `CitedConversationJudgeMetric` (extends `ConversationTextJudgeMetric`)
- **Category**: experience
- **Prompt**: `configs/prompts/judge.yaml` under `judge.conversation_quality`
- **Configuration**: `judge_model` (default: "gpt-5.2")
