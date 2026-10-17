# Intrinsic User Feedback

> **Experience Metric**: Accuracy and rubric-based judges tell you what the assistant did wrong by external standards. This metric reports what the *user* was privately thinking while it happened — turn-level satisfaction from the caller's own point of view.

## Overview

LLM-based metric that replays a finished conversation from inside the simulated user's head. Instead of scoring the assistant as a third-party evaluator, the judge role-plays the user — grounded in the same persona and goal the user simulator was given — and reports, after every assistant turn, the latent evaluation states the user was having at that moment:

- **internal_thought** — what the user was actually thinking about how the call was going (first person)
- **reaction** — the felt reaction (reassured, impatient, confused, annoyed) and why
- **rating** — a 1-5 scalar satisfaction rating for the assistant's response in that turn

This is an adaptation of ["Training User Simulators with Latent Intrinsic Feedback"](https://arxiv.org/abs/2610.13617), which trains user simulators that emit internal thoughts, reactions, and scalar ratings alongside their utterances. EVA's user simulators generate speech inside hosted voice models (ElevenLabs Agents, OpenAI Realtime) that cannot surface latent state mid-call, so elicitation happens post-hoc: a replay judge given the same first-person inputs (persona, high-level goal, must-have criteria) the simulator had. Two properties carry over from the paper:

1. **Turn-level signals, not one conversation score** — dissatisfaction is localized to the turn that caused it.
2. **No hindsight** — each turn is rated as the user experienced it at that point; a mistake fixed two turns later still counts as an earlier dissatisfaction.

### Capabilities Measured

- **Language Model**: Did the assistant's response satisfy this user at this moment — understand them the first time, avoid making them repeat themselves, and move the call toward what they came for?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per assistant turn (all turns rated within a single LLM call)
- **Opt-in**: Excluded from the default run. Enable with `--metrics intrinsic_user_feedback`.

### Input Data

Uses `conversation_trace`, `user_persona`, and `user_goal` from MetricContext. Tool calls and responses are shown to the judge as background context (the user never heard them, but they explain the assistant's behavior).

### Scoring

- **Scale**: 1-5 (integer rating per assistant turn)
  - 5 — Genuinely satisfied
  - 4 — Satisfied, minor imperfection
  - 3 — Tolerable, had to work around it
  - 2 — Annoyed (missed the point, made the user repeat themselves, hard to follow by ear)
  - 1 — Frustrated (wrong, dismissive, contradictory, or derailing)
- **Normalization**: `(rating - 1) / 4` → 5→1.0, 3→0.5, 1→0.0
- **Aggregation**: Mean across rated assistant turns (configurable via `aggregation`)

### Sub-metrics

- `intrinsic_user_feedback.dissatisfaction_rate` — fraction of rated turns at or below the dissatisfaction threshold (default 2), i.e. turns where the user was silently annoyed or worse. Lower is better.

## Example Output

```json
{
  "name": "intrinsic_user_feedback",
  "score": 4.0,
  "normalized_score": 0.75,
  "details": {
    "aggregation": "mean",
    "num_turns": 3,
    "num_evaluated": 3,
    "per_turn_ratings": {"1": 5, "2": 2, "3": 5},
    "per_turn_internal_thoughts": {
      "2": "I already told them the date, why are they asking again?"
    },
    "per_turn_reactions": {"2": "annoyed — repeating myself"},
    "per_turn_normalized": {"1": 1.0, "2": 0.25, "3": 1.0}
  },
  "sub_metrics": {
    "intrinsic_user_feedback.dissatisfaction_rate": {"score": 0.333, "normalized_score": 0.333}
  }
}
```

## Related Metrics

- [conversation_progression.md](conversation_progression.md) - Third-person view of whether the assistant moved the conversation forward
- [conciseness.md](conciseness.md) - Third-person view of cognitive load per response
- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Whether the simulated user can be trusted in the first place

## Implementation Details

- **File**: `src/eva/metrics/experience/intrinsic_user_feedback.py`
- **Class**: `IntrinsicUserFeedbackMetric`
- **Base Class**: `PerTurnConversationJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.intrinsic_user_feedback`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
  - `aggregation`: Aggregation method for per-turn scores (default: "mean")
  - `dissatisfaction_threshold`: Rating at or below which a turn counts toward `dissatisfaction_rate` (default: 2)
