# Response Grounding

> **Experience Metric**: Did the agent speak and act on what the user actually said — or produce content and tool arguments that trace to nothing in the conversation?

## Overview

Deterministic metric that grounds every content-bearing assistant turn and every scalar tool-call argument in the user's speech and tool responses. It targets the failure modes highlighted by [MSI-Bench](https://arxiv.org/abs/2609.24812): agents that *respond when no one has addressed them* (conversational restraint) and fail *speaker-scoped decision making* (acting on values not attributable to the addressing speaker). EVA conversations are dyadic, so "speaker-scoped" is adapted to "user-utterance-scoped", and the benchmark's LLM-judged atomic rubrics are replaced by parameter-free lexical checks in the spirit of the deterministic `turn_taking` metric. The converse failure (staying silent when addressed) is already covered by `turn_taking`'s missed-turn signal.

### Capabilities Measured

- **Language Model**: Does the agent's response content derive from the user's request or from tool output, rather than from content nobody introduced?
- **Language Model**: Are tool-call argument values traceable to user speech, tool output, or the tool schema's declared enum vocabulary?

## How It Works

### Evaluation Method

- **Type**: Deterministic (lexical grounding — no LLM judge)
- **Granularity**: Atomic rubric units (assistant turns and tool-call arguments)

### Input Data

Uses the following MetricContext fields:

- `transcribed_user_turns` / `transcribed_assistant_turns`: per-turn transcripts
- `conversation_trace`: tool_call entries (name, parameters, turn_id) and tool_response entries
- `agent_tools`: tool schema (declared enum values are fixed vocabulary)
- `current_date_time`: the run's date is legitimate vocabulary

### Evaluation Methodology

**Turn rubric (conversational restraint)** — for each assistant turn ≥ 1 (the greeting is excluded):

1. Extract content words (lowercase alphanumeric tokens, length ≥ 2, stopwords dropped).
2. A turn is *not applicable* (excluded) if it asks a question or has fewer than 3 content words — interactional turns ("Sure, anything else?") make no substantive claim.
3. Compute the grounding ratio: fraction of content words that appear (after digit↔word canonicalization, e.g. "two" ↔ "2") in the user's speech so far, tool responses so far, enum values, or the current date.
4. The turn **passes** if the ratio ≥ 0.3 *or* at least one grounded token is entity-like (contains a digit or is ≥ 5 chars) — echoing the right confirmation code counts even if the phrasing is novel.

**Tool-argument rubric (speaker-scoped binding)** — for each scalar value in a tool call's parameters (booleans/nulls skipped):

- Tokenize the value; it **passes** when every token is traceable to the user's speech so far, prior tool responses, the tool schema's enum vocabulary, or the current date.
- A value that traces to nothing (another speaker's ID in the multi-speaker setting, a hallucinated confirmation code) fails. In dyadic logs a wrong-speaker binding is not directly observable; it surfaces as a value with no source at all.

### Scoring

- **Scale**: 0.0-1.0 — fraction of atomic rubrics passed (content-bearing turns + tool arguments)
- **Normalization**: Same as raw score (already 0-1)
- **Edge case**: No applicable rubrics → returns 1.0 with a note

### Sub-metrics

| Sub-metric | Direction | Meaning |
|---|---|---|
| `turn_grounding_accuracy` | ↑ | Fraction of content-bearing turns that passed |
| `unaddressed_turn_rate` | ↓ | Fraction of turns with unaddressed content |
| `mean_turn_grounding` | ↑ | Mean per-turn grounding ratio |
| `tool_argument_accuracy` | ↑ | Fraction of tool arguments traceable |
| `fabricated_tool_argument_rate` | ↓ | Fraction of arguments with no source |
| `num_unaddressed_turns` | — | Count (null when zero) |

### Known Limitations

- The stopword list is English. For other languages, function words appear on both sides and ground each other, so the metric degrades gracefully rather than failing — but restraint failures in non-English turns are detected with less precision.
- Lexical grounding is conservative by design: turns are only flagged when *no* content overlaps any legitimate source. Paraphrased-but-relevant content usually shares enough tokens (or one entity) to pass.

## Example Output

```json
{
  "name": "response_grounding",
  "score": 0.8,
  "normalized_score": 0.8,
  "details": {
    "per_turn_grounding": {
      "1": {"ratio": 0.5, "num_content_words": 4, "num_grounded": 2, "passed": true}
    },
    "per_tool_argument": {
      "turn1:update_fare_class.confirmation_number": {"value": "XYZ789", "passed": false},
      "turn1:update_fare_class.last_name": {"value": "Black", "passed": true}
    },
    "num_turns": 2,
    "num_evaluated": 1,
    "num_not_applicable": 1,
    "total_tool_arguments": 2,
    "grounded_tool_arguments": 1,
    "total_rubrics": 3,
    "passed_rubrics": 2
  }
}
```

## Related Metrics

- [turn_taking.md](turn_taking.md) - timestamp-based turn allocation; covers the converse failure (staying silent when addressed)
- [tool_call_validity.md](tool_call_validity.md) - format validity of tool calls (this metric checks value provenance instead)

## Implementation Details

- **File**: `src/eva/metrics/experience/response_grounding.py`
- **Class**: `ResponseGroundingMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
- **Adapted from**: MSI-Bench (arXiv:2609.24812) — atomic-rubric respond-vs-silent and speaker-scoped tool correctness, re-grounded on dyadic transcripts
