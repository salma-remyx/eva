# Simulator Instruction Following

> **Validation Metric**: If the simulated user drifted from its prescribed scenario — skipping goals, completing them out of order, or going off-script — the conversation no longer tests the agent the dataset designed it to test, and downstream scores (task completion, faithfulness, progression) are measured against the wrong scenario.

## Overview

LLM-as-judge validation metric that grades how faithfully the simulated user **followed its prescribed scenario**. It answers two questions the binary `user_behavioral_fidelity` verdict does not: *how much* of the scenario did the simulator complete, and *in what order*?

Adapted from SimIF-Bench ([SteerablePlex: Can We Steer Full-Duplex Models?](https://arxiv.org/abs/2610.12201)), which measures whether a conversational model stays within a prescribed scenario and completes multiple goals in the required order — and finds that full-duplex simulators (including GPT-Realtime) lose that control as the conversation history grows. The prescribed scenario here is the record's `user_goal` decision tree, with the must-have criteria as the ordered goal sequence. This makes the metric a graded companion to `user_behavioral_fidelity`: that metric flags whether the simulation corrupted database state; this one quantifies scenario drift even when no write tool was affected, and reports the early-vs-late goal completion gap that surfaces late-conversation control loss.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:

- `conversation_trace`: Full conversation transcript including tool calls and responses
- `user_goal`: The prescribed scenario — high-level goal plus decision tree (must-have criteria are the ordered goals)
- `user_persona`: User's personality traits and behavior description
- `language`: Language the user was instructed to speak

### Evaluation Methodology

The judge is given the conversation trace, the prescribed scenario (persona, decision-tree behavior rules), and the ordered goal list, and returns:

1. **Per-goal analysis** — for each must-have criterion, whether the user's behavior satisfied it, at which turn, and the evidence.
2. **Scenario adherence** — the user turns where the simulator stepped outside its scenario (invented requests, contradicted constraints, abandoned the scenario).
3. **An overall rating** (0–3).

Deterministic post-processing turns the judge's structured output into sub-metrics:

| Sub-metric | Direction | Description |
|---|---|---|
| `goal_completion_accuracy` | higher is better | Fraction of prescribed goals the simulator completed |
| `ordered_goal_completion_accuracy` | higher is better | Longest prefix of goals completed in the prescribed order, over all goals |
| `early_goal_completion_accuracy` | higher is better | Completion of goals in the first half of the prescribed order |
| `late_goal_completion_accuracy` | higher is better | Completion of goals in the second half |
| `scenario_violation_rate` | lower is better | Distinct violating user turns over total user turns |

The details field also carries `early_late_completion_gap` (early minus late completion): a positive gap means goals late in the scenario were missed more often — the per-record proxy for SimIF-Bench's finding that simulator control degrades as the conversation history grows. Aggregated across records, the gap indicates whether a simulator (e.g., the OpenAI Realtime beta) starts conversations on-script and loses the plot later, which is exactly the failure mode that invalidates long evaluations.

### Audio-Native vs Cascade

The prompt uses the pipeline-aware user-turns disclaimer (`pipeline_prompts.get_user_turns_disclaimer`): in cascade mode the judge is told user turns are the agent's STT transcripts; in audio-native mode (S2S / AUDIO_LLM, e.g. the OpenAI Realtime simulator) that the turns are the simulator's intended or transcribed text. In both cases the judge is instructed not to count obvious speech-recognition artifacts (misheard names, digits, codes) as failed goals — with the Realtime simulator only a transcript is available, and transcription errors must not be mistaken for behavioral drift.

### Scoring

- **Scale**: 0–3
  - 3 (Faithful): all goals completed in the prescribed order, no scenario violations
  - 2 (Minor drift): all goals completed but out of order, or in order with one minor violation
  - 1 (Significant drift): some goals missed, far out of order, or repeated violations
  - 0 (Lost control): the simulator abandoned the scenario
- **Normalization**: rating / 3 (0.0–1.0)

## Example Output

**Example: late-conversation drift (1)**
```json
{
  "name": "simulator_instruction_following",
  "score": 1.0,
  "normalized_score": 0.33,
  "details": {
    "rating": 1,
    "explanation": "The user followed the scenario early on but invented a hotel request mid-call and never confirmed the cost constraint.",
    "goal_analysis": [
      {"goal_id": 1, "description": "New travel date is 2026-03-25 for AUS-LAX.", "completed": true, "turn_index": 2, "evidence": "User stated the new date unprompted."},
      {"goal_id": 2, "description": "Arrival in LAX by 4:00 PM Pacific.", "completed": true, "turn_index": 2, "evidence": "User stated the arrival constraint with the date."},
      {"goal_id": 3, "description": "Total rebooking cost is $120 or less.", "completed": false, "turn_index": null, "evidence": "Agent quoted $95 but the user never confirmed acceptance."}
    ],
    "scenario_adherence": {
      "violations": [{"turn_index": 7, "description": "User requested a hotel booking, which is outside the rebooking goal."}],
      "analysis": "Mostly on scenario; one invented request late in the call."
    },
    "goal_progression": {
      "num_goals": 3, "num_goals_completed": 2, "in_order_prefix": 2,
      "early_goals_total": 2, "early_goals_completed": 2,
      "late_goals_total": 1, "late_goals_completed": 0
    },
    "early_late_completion_gap": 1.0
  },
  "sub_metrics": {
    "simulator_instruction_following.goal_completion_accuracy": {"score": 0.667},
    "simulator_instruction_following.ordered_goal_completion_accuracy": {"score": 0.667},
    "simulator_instruction_following.early_goal_completion_accuracy": {"score": 1.0},
    "simulator_instruction_following.late_goal_completion_accuracy": {"score": 0.0},
    "simulator_instruction_following.scenario_violation_rate": {"score": 0.25}
  }
}
```

## Related Metrics

- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Binary corruption verdict (database-affecting user misbehavior)
- [conversation_valid_end.md](conversation_valid_end.md) - Validates conversation completed

## Implementation Details

- **File**: `src/eva/metrics/validation/simulator_instruction_following.py`
- **Class**: `SimulatorInstructionFollowingMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.simulator_instruction_following`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2-medium")
  - `judge_params`: LLM generation parameters
