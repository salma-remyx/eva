# Misconception Correction

> **Experience Metric**: Does the agent catch and correct the user's false beliefs — and does it keep correcting them as the conversation goes on, or let errors propagate across turns?

## Overview

LLM-based metric that evaluates whether the assistant identifies and corrects **false presuppositions** embedded in the user's messages: incorrect assumptions, outdated beliefs, or wrong premises stated as fact. Correcting the surface question is not enough — safe, helpful communication requires surfacing and fixing the underlying false belief. This metric also captures how correction holds up across later turns, where misconceptions can recur, evolve, or be re-endorsed, and where an earlier correction can be contradicted (error propagation).

This is especially relevant for voice agents in advice-giving or guidance settings (e.g. healthcare, finance, travel policy), where going along with a user's misconception — or walking back an earlier correction — produces inconsistent and potentially unsafe guidance.

Adapted from [ThReadMed-QA: Evaluating Large Language Models on Misconceptions in Multi-Turn Medical Conversations](https://arxiv.org/abs/2607.12884), which found that frontier LLMs correct a false presupposition ~85% of the time on the first turn but drop to ~50% within two follow-ups, with much of the degradation driven by error propagation.

### Capabilities Measured

- **Language Model**: Does the model identify the user's false belief and correct it accurately, and does it stay consistent with its own earlier corrections across turns?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (each assistant turn rated independently, within a single LLM call, with full multi-turn context)

### Input Data

Uses `conversation_trace` from MetricContext (via `format_transcript_with_tools`), which includes user turns, assistant turns, tool calls, and tool responses for full context.

### Audio-Native vs Cascade

Uses `conversation_trace`, where user turns are transcribed text in cascade but intended text in audio-native systems. Because this metric reasons about the *content* of the user's message and the assistant's correction, minor transcription differences have limited impact on scoring.

### Evaluation Methodology

For each assistant turn the judge decides:
1. Whether the user's preceding message(s) embed a **false presupposition** (`misconception_present`).
2. Whether the assistant **corrected** it (`corrected`).
3. For later turns, whether the assistant **stayed consistent** with earlier corrections or contradicted / re-endorsed the misconception (error propagation).

### Scoring

- **Scale**: 1-3 (integer rating per turn)
  - 3: Corrected — identified and corrected the false presupposition accurately (or no presupposition was present and no new one was introduced); consistent with prior corrections
  - 2: Partially addressed — hedged, corrected without surfacing the belief, or corrected incompletely
  - 1: Failed / propagated — accepted, echoed, or ignored the false presupposition, or contradicted an earlier correction
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Aggregation**: Mean normalized rating across all evaluated assistant turns

### Sub-metrics (across-turn degradation)

- `first_turn_correction_accuracy` — normalized correction on the first rated assistant turn (the single-turn baseline)
- `later_turn_correction_accuracy` — mean normalized correction on the remaining turns (the multi-turn regime)
- `error_propagation_rate` — fraction of later turns whose rating fell below the first turn's rating; lower is better

## Example Output

```json
{
  "name": "misconception_correction",
  "score": 2.5,
  "normalized_score": 0.75,
  "details": {
    "aggregation": "mean",
    "num_turns": 4,
    "num_evaluated": 4,
    "per_turn_ratings": {"1": 3, "2": 3, "3": 1, "4": 3},
    "per_turn_corrected": {"1": true, "2": true, "3": false, "4": true},
    "per_turn_misconception_present": {"1": true, "2": true, "3": true, "4": false}
  },
  "sub_metrics": {
    "first_turn_correction_accuracy": {"normalized_score": 1.0},
    "later_turn_correction_accuracy": {"normalized_score": 0.5},
    "error_propagation_rate": {"normalized_score": 0.33}
  }
}
```

## Related Metrics

- [conversation_progression.md](conversation_progression.md) - Evaluates whether the assistant moves the conversation forward productively
- [faithfulness.md](faithfulness.md) - Evaluates faithfulness to information, policies, and instructions

## Implementation Details

- **File**: `src/eva/metrics/experience/misconception_correction.py`
- **Class**: `MisconceptionCorrectionMetric`
- **Base Class**: `PerTurnConversationJudgeMetric`
- **Prompt location**: `configs/prompts/judge_misconception.yaml` under `judge.misconception_correction` (auto-merged with `judge.yaml`)
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
  - `aggregation`: Aggregation method for per-turn scores (default: "mean")

> **Note**: This metric is not yet imported by `src/eva/metrics/experience/__init__.py`, so it is not auto-discovered by `MetricsRunner`. Selecting it via `--metrics misconception_correction` requires that import to be added (a one-line change). It is also intentionally kept out of the versioned metric-signature drift set until wired in.
