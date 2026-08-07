# Conversational Moves

> **Experience Metric**: Even when an outcome is correct, the assistant may pick the wrong *kind* of response for the moment — asking for a detail it already has, or explaining when the user just wants the result.

## Overview

LLM-based metric that evaluates whether each assistant turn realized an appropriate **conversational move** for the dialogue context. Classical dialogue systems separated dialogue management (selecting the next dialogue action, or "move") from response realization; in LLM-driven voice agents that selection is hidden inside the model. This metric surfaces it again as an evaluation dimension: for every assistant turn it (1) classifies the move the assistant actually made and (2) rates whether that move was the right selection given what the user needed and where the conversation was in the task.

The move taxonomy and the framing of dialogue management as move selection are adapted from [Latent-IM: Latent Interaction Management for Speech LLMs](https://arxiv.org/abs/2607.26928). Latent-IM recovers move selection and realization inside an LLM and steers generation toward a chosen move; EVA cannot steer a model it is benchmarking, so this metric instead *measures* whether the assistant's chosen moves align with what a reference selection policy would pick, using the existing LLM-judge path as that reference policy.

### Capabilities Measured

- **Language Model**: Does the model select the right kind of conversational move for the context — querying for needed details, acknowledging before acting, replying with a result once the facts are in — rather than, say, re-asking for information already provided or explaining when a result is expected?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (all assistant turns judged in a single call, then aggregated)

### Input Data

Uses `conversation_trace` from MetricContext (via `format_transcript`), which includes user turns, assistant turns, tool calls, and tool responses.

### Audio-Native vs Cascade

Operates on `conversation_trace` text, so it behaves identically for cascade and audio-native (S2S) systems. As with other text judges, STT errors visible in a cascade trace may make a turn look slightly different from the audio-native intended text; the prompt asks the judge to evaluate the move the assistant was clearly making and not to assume an unfinished (interrupted) move was inappropriate.

### Conversational Moves

Each assistant turn is classified into exactly one move:

- **acknowledging** — confirms receipt/understanding of prior information without advancing new substance
- **checking** — verifies or clarifies information before acting (e.g., to avoid an error)
- **querying** — asks the user for information needed to proceed
- **explaining** — provides information, rationale, policy, or instruction
- **replying** — delivers the outcome, result, or direct answer the user was seeking
- **other** — catch-all for turns that do not fit the taxonomy (e.g., pure small talk)

### Scoring

- **Scale**: 1-3 (integer rating per turn)
  - 3: The move was the right choice for the context and advanced the dialogue effectively
  - 2: The move was reasonable but slightly suboptimal
  - 1: The move was wrong for the context (e.g., querying for information already provided, replying before gathering the needed details)
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Aggregation**: Mean of per-turn normalized scores (configurable via `aggregation`)

The parent score reflects move **appropriateness**. The judge's move classifications also surface as sub-metrics — one per move type (`<move>_rate`) — giving the **distribution** of moves across rated turns (share of rated turns classified as each move). This is distinct from conciseness (brevity), conversation_progression (forward movement / repetition), and turn_taking (timing).

## Example Output

```json
{
  "name": "conversational_moves",
  "score": 2.667,
  "normalized_score": 0.833,
  "details": {
    "per_turn_ratings": {"1": 3, "2": 2, "3": 3},
    "per_turn_move": {"1": "querying", "2": "querying", "3": "replying"},
    "aggregation": "mean",
    "num_turns": 3,
    "num_evaluated": 3
  },
  "sub_metrics": {
    "querying_rate": {"normalized_score": 0.667},
    "replying_rate": {"normalized_score": 0.333},
    "acknowledging_rate": {"normalized_score": 0.0},
    "checking_rate": {"normalized_score": 0.0},
    "explaining_rate": {"normalized_score": 0.0},
    "other_rate": {"normalized_score": 0.0}
  }
}
```

## Related Metrics

- [conversation_progression.md](conversation_progression.md) - Whether the assistant moved the conversation forward without repetition (conversation-level)
- [conciseness.md](conciseness.md) - Whether responses are appropriately brief (per-turn)

## Implementation Details

- **File**: `src/eva/metrics/experience/conversational_moves.py`
- **Class**: `ConversationalMovesJudgeMetric`
- **Base Class**: `PerTurnConversationJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.conversational_moves`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
  - `aggregation`: per-turn aggregation method (default: "mean")
