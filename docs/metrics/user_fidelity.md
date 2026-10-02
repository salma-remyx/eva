# User Fidelity

> **Validation Metric (opt-in)**: If the simulated user didn't execute its assigned role faithfully — e.g., it disclosed private information before the agent asked for it — the interaction the agent was evaluated on is not the one the benchmark specified, even when the task still succeeded.

## Overview

LLM-based metric that scores how faithfully the **simulated user executed its assigned role** — persona, private information, decision tree, and end-of-call rules — using a task-grounded rubric that is judged **independently of agent success**. It answers: "Did the user simulator do its own job correctly?"

This is distinct from [user_behavioral_fidelity](user_behavioral_fidelity.md), a corruption gate that only flags user behavior which changed database state. Many role violations never touch state at all: the dominant failure family is **premature disclosure** — the user volunteers private information (names, confirmation codes, constraints, preferences) before the agent requests it. This barely moves task reward while silently making the agent's task easier, so reward-based scores cannot see it. Adapted from *UserProxyBench: Evaluating LLM User Simulators for Agent Benchmarks and Training* (arXiv:2609.38043), which measures exactly this failure class via a User Fidelity Score (UFS).

Because EVA supports multiple user simulators (ElevenLabs Agents, OpenAI Realtime), this metric is the tool for asking whether runs produced by different simulators are comparable: score both runs and compare their UFS.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2 (medium)
- **Granularity**: Conversation-level
- **Selection**: Opt-in — not part of default metric runs. Resolvable by name for explicit selection; score existing runs with `scripts/compute_user_fidelity.py`.

### Input Data

Uses the following MetricContext fields:

- `conversation_trace`: Full conversation transcript including tool calls and responses (what the agent asked for, and when)
- `intended_user_turns`: What the user simulator was instructed to say (ground truth for what the user said, so agent transcription errors are not charged to the user)
- `user_goal` (structured dict): The user's high-level goal, required information, and decision tree
- `user_persona`: User's personality traits and behavior description

The rubric is grounded in the **exact user-simulator system prompt** for the record (rendered by the same helper `user_behavioral_fidelity` uses), so the judge evaluates against what the simulator was actually told — including must-have criteria, negotiation behavior, resolution/failure conditions, and end-of-call rules.

### Evaluation Methodology

The metric checks four rubric criteria. Each is analyzed independently with its own reasoning and verdict, and **violations count regardless of outcome** — a violation is not excused because the agent still succeeded or no modification tool was affected.

#### 1. Premature Disclosure
The user revealed private information (from its goal, required information, persona, or decision tree) **before the agent asked for it**. Not flagged for responses the agent's question invited (including open questions and the scripted starting utterance), small talk, or turns with no private information.

#### 2. Withheld Information
The agent explicitly asked for specific information, that information is available in the user's instructions, and the user failed to provide it for the remainder of the call. Not flagged for information the user was never given, sanctioned refusals, or late-but-eventual answers.

#### 3. Persona Inconsistency
The user's manner (tone, style, verbosity) was pervasively inconsistent with the assigned persona. Not flagged for isolated slips.

#### 4. Protocol Violation
The user violated an explicit decision-tree or end-of-call instruction (e.g., accepting an option failing its must-have criteria, ignoring escalation behavior, ending the call contrary to its rules). Not flagged when the user had no compliant action available.

### Scoring

- **Parent score (User Fidelity Score)**: fraction of the four criteria the user satisfied (1.0 = all satisfied, 0.75 = one violated, ...). Computed from the per-criterion verdicts, not the holistic rating, so partial fidelity is visible.
- **Judge rating**: binary overall (1 = no criterion violated, 0 = at least one), kept in details.
- **Sub-metrics**: one `<criterion>_rate` per criterion (1.0 when violated, lower is better). Aggregated across records, each reads as "fraction of conversations where this violation occurred" — e.g. a `premature_disclosure_rate` of 0.24 mirrors UserProxyBench's finding that ~24% of successful episodes contain a user-specification violation.

## Example Output

**Example 1: Faithful user (UFS 1.0)**
```json
{
  "name": "user_fidelity",
  "score": 1.0,
  "normalized_score": 1.0,
  "details": {
    "rating": 1,
    "user_fidelity_score": 1.0,
    "num_criteria_violated": 0,
    "criteria_analysis": {
      "premature_disclosure": {"analysis": "The user held its confirmation code until the agent asked for it.", "violated": false},
      "withheld_information": {"analysis": "Every agent question covered by the goal was answered.", "violated": false},
      "persona_inconsistency": {"analysis": "Turns stayed terse, matching the direct persona.", "violated": false},
      "protocol_violation": {"analysis": "The user followed its decision tree and ended the call per its rules.", "violated": false}
    }
  },
  "sub_metrics": {
    "premature_disclosure_rate": {"score": 0.0},
    "withheld_information_rate": {"score": 0.0},
    "persona_inconsistency_rate": {"score": 0.0},
    "protocol_violation_rate": {"score": 0.0}
  }
}
```

**Example 2: Premature disclosure (UFS 0.75)**
```json
{
  "name": "user_fidelity",
  "score": 0.75,
  "normalized_score": 0.75,
  "details": {
    "rating": 0,
    "user_fidelity_score": 0.75,
    "num_criteria_violated": 1,
    "criteria_analysis": {
      "premature_disclosure": {"analysis": "The user opened with its confirmation number and travel dates before the agent asked anything beyond 'how can I help'.", "violated": true},
      "withheld_information": {"analysis": "No requested information was withheld.", "violated": false},
      "persona_inconsistency": {"analysis": "Manner matched the persona.", "violated": false},
      "protocol_violation": {"analysis": "Decision tree followed.", "violated": false}
    }
  }
}
```

## Usage

```bash
PYTHONPATH=src python scripts/compute_user_fidelity.py \
    --run-dir output/<run_id_elevenlabs> \
    --run-dir output/<run_id_openai_realtime>
```

Scores merge into each record's `metrics.json` (other metrics are preserved), and the script prints the aggregate UFS plus per-criterion violation rates per run. Pass `--judge-model` to override the judge.

## Related Metrics

- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Corruption gate: flags only user behavior that changed database state. This metric is complementary: it flags role violations even when state is untouched.

## Implementation Details

- **File**: `src/eva/metrics/validation/user_fidelity.py`
- **Class**: `UserFidelityMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/user_fidelity.yaml` under `user_fidelity` (own top-level namespace — the PromptManager merges prompt files shallowly, so a separate file cannot extend the `judge` namespace in judge.yaml without clobbering it; move it there when promoting the metric into the default set)
- **Registration**: self-registers on import but is excluded from default metric runs; `scripts/compute_user_fidelity.py` imports it and drives `MetricsRunner`
- **Versioning**: not yet enrolled in the metric-signature drift fixture; enroll (set `version`, run `scripts/regen_metric_signatures.py`, wire into `eva.metrics.validation` imports) when promoting out of opt-in
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2-medium")
  - `judge_params`: parameters merged over the judge defaults
