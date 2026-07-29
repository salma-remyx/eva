# Information Acquisition

> **Accuracy Metric**: A polished final answer is not a valid outcome if the assistant never gathered the information it needed. This metric scores the *elicited history* independently of output quality.

## Overview

LLM-based metric that evaluates whether the assistant elicited the information required to address the user's goal. It is **diagnosis-decoupled**: it judges only what was gathered in the conversation history, deliberately ignoring how well the final answer was phrased or whether the task ultimately succeeded. This isolates information-acquisition quality from generation quality — strong generation can no longer compensate for a thin history, nor can weak generation obscure a rich one.

The judge lists the information items the assistant needed to obtain from the user and marks which were captured in the elicited history. The metric then aggregates these into a **coverage ratio** (recall of required items). Complementary to [`faithfulness`](faithfulness.md) (output groundedness) and [`task_completion`](task_completion.md) (final goal state): a conversation can complete the task and stay faithful yet still have gathered information inefficiently or incompletely.

Adapted from *MedDDC-Eval: Diagnosis-Decoupled Evaluation of Multi-Turn Medical Consultation Agents* (arXiv:2607.18999). The paper's two-step measurement is ported at full fidelity: directional semantic coverage of required items against the elicited history (LLM judge) followed by deterministic one-to-one assignment (code, at most one credit per required item). The medical D/T/E harness is replaced by EVA's judge infrastructure; the paper's benchmark suite and GRPO post-training are out of scope.

### Capabilities Measured

- **Language Model**: Did the model ask the right questions and collect the information needed to resolve the goal?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: Claude Opus 4.6
- **Granularity**: Conversation-level (single rating for the whole conversation)

### Scoring

- **`normalized_score`**: deterministic coverage ratio in `[0, 1]` = `items_covered / items_required`. Each required item is credited at most once (the paper's one-to-one assignment).
- **`rating`** (1–3, in details): a coarser human-readable summary of acquisition sufficiency (3 = adequate, 2 = partial, 1 = insufficient).

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: Full conversation (via `format_transcript`) — the elicited history under evaluation
- `user_goal`, `user_persona`: Source for the required information items
- `agent_instructions`, `agent_role`, `agent_tools`: Define what the assistant needed to collect and verify
- `current_date_time`: Simulated date/time
- `pipeline_type` / `is_audio_native`: Controls the cascade vs. audio-native user/assistant-turn disclaimers

### Decoupling Discipline

The judge is instructed to credit an item only when it was genuinely elicited from the user — not when the assistant asserted, assumed, or guessed it. Final-answer eloquence and task success are explicitly out of scope (covered by `faithfulness` and `task_completion`), so this metric cannot be inflated by good generation over a thin history.
