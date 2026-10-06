# STT Error Propagation

> **Diagnostic Metric**: Traces mis-transcribed user words into tool-call arguments — useful for telling accent/noise-induced STT failures that actually corrupted the agent's actions apart from errors the agent recovered from.

## Overview

Deterministic metric that measures how often words the STT got wrong in user speech show up — in their corrupted form — in the arguments of the tool calls the agent made. It connects the two failure layers that `stt_wer` and `tool_call_validity` measure independently: a high value means transcription errors are carrying through to the LLM stage and corrupting tool-call arguments, the failure mode accent-robustness benchmarks find most costly.

### Capabilities Measured

- **Speech Recognition, Language Model**: The STT error itself is a Speech Recognition failure, but the propagated outcome depends on the language model accepting the corrupted value into a tool call without confirming it.

## How It Works

### Evaluation Method

- **Type**: Deterministic (normalized token diff, no LLM judge)
- **Granularity**: Per-turn token classification, conversation-level aggregation

### Input Data

Uses the following MetricContext fields:
- `intended_user_turns`: What the user simulator intended to say (reference text)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis text)
- `tool_params`: Every tool call with its arguments

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — the STT transcript is the LLM's actual input, so errors in it can corrupt downstream tool arguments.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`), same as `stt_wer`: audio-native models receive raw audio, not STT transcripts.

### Evaluation Methodology

Both texts go through the same language-aware normalization pipeline as `stt_wer`. Content words present in the intended text but missing from the transcript are **mis-transcribed** tokens (function words and single letters are ignored). Each is classified against every tool-call argument value in the conversation:

| Outcome | Condition |
|---------|-----------|
| **propagated** | The corrupted form the STT produced instead appears in a tool argument — the error reached the tool layer |
| **recovered** | The correct form appears in a tool argument anyway (agent confirmed, repeated, or inferred it) |
| **unattributed** | Neither form appears in any tool argument (the word was not tool-relevant) |

### Scoring

- **Scale**: 0.0-1.0
  - `score` = propagated / mis-transcribed tokens (lower is better)
  - `normalized_score` = 1 − score (argument integrity, higher is better)
- **Sub-metrics**: `propagation_rate`, `mis_transcribed_token_rate` (mis-transcribed / content tokens), `num_propagated` (count)

## Example Output

```json
{
  "name": "stt_error_propagation",
  "score": 1.0,
  "normalized_score": 0.0,
  "details": {
    "mis_transcribed_tokens": 1,
    "propagated": 1,
    "recovered": 0,
    "unattributed": 0,
    "propagated_examples": [
      {
        "turn_id": 1,
        "intended_token": "edinburgh",
        "transcribed_as": "edinburrow",
        "tool_calls": ["search_flights"]
      }
    ],
    "per_turn": {
      "1": {
        "dropped": ["edinburgh"],
        "inserted": ["edinburrow"],
        "classifications": [{"token": "edinburgh", "outcome": "propagated", "corrupted_as": "edinburrow"}]
      }
    }
  }
}
```

## Related Metrics

- [stt_wer.md](stt_wer.md) - Word-level STT quality (upstream signal, WER predicts but does not equal task impact)
- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - LLM-judged entity-level STT accuracy (complementary)
- [tool_call_validity.md](tool_call_validity.md) - Whether tool calls were well-formed (downstream signal)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/stt_error_propagation.py`
- **Class**: `STTErrorPropagation`
- **Base Class**: `CodeMetric`
- **Configuration**: Language is taken from `MetricContext.language` (no per-metric config needed)
