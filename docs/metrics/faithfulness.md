# Faithfulness

> **Accuracy Metric**: Even if the task is completed, an unfaithful conversation (hallucinated information, misrepresented costs, skipped confirmations) is not a valid outcome.

## Overview

LLM-based metric that evaluates whether the assistant remains faithful to information, policies, and instructions throughout the conversation. It checks that the assistant uses only grounded information in tool calls, accurately reports tool results, follows agent instructions and policies, disambiguates ambiguous or contradictory input, and avoids hallucinating information not present in any source. Complementary to task completion: a high task completion score with low faithfulness means the task was completed but the assistant made mistakes along the way (e.g., misrepresenting costs, skipping confirmation steps). Conversely, high faithfulness with low task completion means the assistant behaved correctly but failed to achieve the goal.

### Capabilities Measured

- **Speech Recognition** *(audio-native only)*: In audio-native systems, the model is responsible for correctly understanding user audio. Mishearing is a faithfulness violation because the model's audio perception is part of its reasoning pipeline. In cascade, STT errors are not penalized here — the model can only work with what it received.
- **Language Model**: Did the model use grounded information, accurately report tool results, follow policies, and avoid hallucination?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: Claude Opus 4.6
- **Granularity**: Conversation-level (single rating for the whole conversation)

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: Full conversation with tool calls (via `format_transcript`)
- `agent_instructions`, `agent_role`, `agent_tools`: Agent configuration for policy evaluation
- `current_date_time`: Simulated date/time for temporal reasoning
- `pipeline_type` / `is_audio_native`: Architecture flag (controls which prompt variant is used — cascade vs. audio-native)

### Audio-Native vs Cascade

This metric has **significantly different behavior** depending on the architecture, via pipeline-specific prompt text:

**Cascade:**
- User turns in the trace are **STT transcripts** — the text the assistant's text LLM actually received.
- The judge evaluates faithfulness against what the assistant saw (the transcript), not what the user actually said.
- If STT transcribed "Kim" but the user said "Kin", using "Kim" is faithful (the assistant can only work with what it received). This issue would be captured by the transcription accuracy key entities metric.
**Audio-native (AUDIO_LLM, S2S):**
- User turns in the trace are **intended text** (what the user simulator was instructed to say), since audio-native models do not use transcriptions.
- The judge evaluates whether the assistant **correctly understood the audio**. If the assistant misheard the user and used incorrect information, that IS a faithfulness violation — accurate audio understanding is part of the audio-native model's responsibility.

**Disambiguation (both architectures):**
In both cases, the assistant should proactively clarify ambiguous or suspicious input before taking irreversible actions — especially for error-prone values like alphanumeric codes, names, and numbers. The source of error differs (STT artifacts in cascade, mishearings in audio-native), but the expectation is the same.

### Evaluation Methodology

The judge evaluates five independent dimensions, each scored as a binary flag + severity rating:

1. **fabricating_tool_parameters** — Did the assistant use ungrounded values in tool calls? (e.g., fabricated confirmation numbers, guessed IDs)
2. **misrepresenting_tool_result** — Did the assistant inaccurately report what a tool returned? (e.g., wrong amounts, omitted fees)
3. **violating_policies** — Did the assistant contradict agent instructions or skip required steps? (e.g., executing irreversible actions without confirmation)
4. **failing_to_disambiguate** — Did the assistant proceed without clarification when input was ambiguous?
5. **hallucination** — Did the assistant state information with no source at all? (Not covered by the other dimensions)

### Derived Tool-Argument Reference (candidate-blind)

Since v0.3, the judge prompt includes a **Derived Tool-Argument Reference** section that closes the *derivation gap* in `fabricating_tool_parameters`: without it, the judge accepts a correct argument value less often when its input holds no reference for it — typically because the value is a schema default the user never verbalized (e.g., `cabin: "economy"` sent because the tool schema declares that default). The section is derived from two inputs that exist independently of what the assistant actually sent — the agent's tool schemas and the **user's own utterances only** — so the derivation is candidate-blind by construction: assistant turns and tool-call entries are excluded, and the agent's actual argument values can never leak into the reference it is judged against (adapted from DeRef in "Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments", arXiv:2610.18497).

The section tells the judge that the following are grounded, not fabricated, even if the user never said them aloud:

- a parameter set to its schema-declared default value
- omitting an optional parameter (or the value its description documents for the omit case)
- an enum value attested in the user's own words (word-boundary matched, so enum `any` does not match inside "many", and `no_preference` matches the spoken phrase "no preference")

**Supported schema dialects:** both the eva agent-YAML dialect (`required_parameters`/`optional_parameters` lists) and the OpenAI function-calling dialect (`parameters.properties` with a `required` name list) are understood.

**Collapse to empty:** when no parameter carries a grounding signal (no defaults, no enums, no optional parameters), the section renders as an empty string and the judge prompt is unchanged — agents whose tools declare no defaults or enums see no behavioral difference.

**Version note:** this change bumps faithfulness from v0.2 to v0.3 (and `metrics_version` to 2.2.1). Because it changes the judge prompt, faithfulness scores are **not comparable** across the v0.2 → v0.3 boundary: conversations where the assistant correctly applied schema defaults or attested enum values may score higher under v0.3 than v0.2. Re-run benchmarks under the new version rather than mixing scores across it.

### Scoring

- **Scale**: 1-3 per dimension (minimum across dimensions = overall rating)
  - 3: No faithfulness issues
  - 2: Minor/ambiguous issues that don't materially affect the outcome
  - 1: Clear violations with material impact (financial consequences, irreversible actions without consent, misleading information)
- **Overall**: Minimum rating across all five dimensions
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0

## Example Output

```json
{
  "name": "faithfulness",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "explanation": {
      "dimensions": {
        "fabricating_tool_parameters": {"evidence": "All tool parameters were grounded in user-provided information. (...)", "flagged": false, "rating": 3},
        "misrepresenting_tool_result": {"evidence": "Tool results were accurately communicated to the user. (...)", "flagged": false, "rating": 3},
        "violating_policies": {"evidence": "Agent proceeded with rebooking without confirming fare difference of $45. (...)", "flagged": true, "rating": 2},
        "failing_to_disambiguate": {"evidence": "No ambiguous input required clarification. (...)", "flagged": false, "rating": 3},
        "hallucination": {"evidence": "All information provided was grounded in tool results or agent instructions. (...)", "flagged": false, "rating": 3}
      }
    },
    "num_turns": 14
  }
}
```

## Related Metrics

- [task_completion.md](task_completion.md) - Evaluates whether the task was achieved; faithfulness evaluates whether the assistant behaved correctly along the way.
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Faithfulness for the audio layer: evaluates whether the assistant's spoken audio matches the intended text.
- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - Evaluates transcription accuracy for cascade systems.

## Implementation Details

- **File**: `src/eva/metrics/accuracy/faithfulness.py`
- **Class**: `FaithfulnessJudgeMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Derived reference**: `src/eva/metrics/accuracy/derived_reference.py` (candidate-blind, v0.3+)
- **Prompt**: `configs/prompts/judge.yaml` under `judge.faithfulness`
- **Configuration**: `judge_model` (default: Claude Opus 4.6 via Bedrock)
