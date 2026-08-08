# Proactive Dialogue

> **Experience Metric**: An assistant that obeys every policy and wastes no tool call can still leave a user doing all the work. This metric scores whether the agent took the initiative to drive and guide the conversation.

## Overview

LLM-based metric that evaluates how **proactively** the assistant drove the conversation toward the user's goal, along two per-conversation dimensions adapted from [ProactiveEval](https://arxiv.org/abs/2508.20973) (ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents). It measures an *initiative* axis that is orthogonal to `conversation_progression`: where progression scores whether the assistant **avoided regressing** (wasted tool calls, lost information, redundant turns), proactive dialogue scores whether the assistant **actively advanced and guided** the exchange.

> Adapted from ProactiveEval (arXiv:2508.20973). The upstream artefact carries no licence, so the rubric is reimplemented here in EVA's idiom rather than ported.

### Capabilities Measured

- **Language Model**: Does the model take initiative — working toward the user's goal, anticipating needs, and steering the dialogue — rather than only reacting to the last user turn?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Conversation-level (single rating for the whole conversation)

### Input Data

Uses `conversation_trace` from MetricContext (via `format_transcript`), together with `agent_role` and `user_goal` so the judge can assess goal-directed planning. The trace includes user turns, assistant turns, tool calls, and tool responses.

### Evaluation Methodology

The judge evaluates two dimensions adapted from ProactiveEval's decomposition of proactive dialogue, each rated 1-3 and flagged when the assistant fell short:

1. **target_planning** — Did the assistant proactively work toward the user's goal, pursuing the needed sub-steps and making concrete progress rather than waiting for the user to direct every step?
2. **dialogue_guidance** — Did the assistant proactively guide the dialogue itself — anticipating needs, volunteering relevant information or suggestions, and asking forward-looking clarifying questions?

This metric evaluates **initiative only**. It does NOT evaluate policy compliance (faithfulness) or conversational efficiency (conversation_progression). A perfectly compliant, efficient assistant can still score low here if it never takes the initiative.

### Scoring

- **Scale**: 1-3 (integer rating)
  - 3: Strongly proactive — the assistant drove the conversation on both dimensions
  - 2: Adequate — proactive on one axis but fell short on the other
  - 1: Passive/reactive — deficits on both dimensions, or clearly passive on either
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Sub-metrics**: `target_planning_rate` and `dialogue_guidance_rate` — the fraction of conversations where a proactiveness deficit was flagged on that dimension (lower is better).

## Example Output

```json
{
  "name": "proactive_dialogue",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "explanation": {
      "dimensions": {
        "target_planning": {"evidence": "The assistant pursued the booking steps once prompted, but did not take the obvious next step after the search results returned.", "flagged": false, "rating": 3},
        "dialogue_guidance": {"evidence": "The assistant answered each question but never volunteered next steps or asked a forward-looking clarifying question.", "flagged": true, "rating": 1}
      },
      "explanation": "Goal-directed but reactive in guiding the dialogue."
    },
    "num_turns": 9
  },
  "sub_metrics": {
    "target_planning_rate": {"score": 0.0, "normalized_score": 0.0},
    "dialogue_guidance_rate": {"score": 1.0, "normalized_score": 1.0}
  }
}
```

## Related Metrics

- [conversation_progression.md](conversation_progression.md) - The complementary axis: whether the assistant avoided regressing (efficiency), vs. taking initiative (this metric)
- [conciseness.md](conciseness.md) - Evaluates response brevity (per-turn)

## Implementation Details

- **File**: `src/eva/metrics/experience/proactive_dialogue.py`
- **Class**: `ProactiveDialogueJudgeMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.proactive_dialogue`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
