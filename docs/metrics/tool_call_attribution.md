# Tool Call Attribution

> **Diagnostic Metric**: Separates serving-stack failures from model behavior in tool-call outcomes — so a conversation the serving layer never let the model answer is not read as "the model made no calls" (or as perfect fidelity because there are no calls to invalidate).

## Overview

Deterministic metric that attributes each tool-call protocol outcome to the layer responsible for it. Serving-layer failures (request rejected before inference, retry exhaustion, parser artifacts) surface in the conversation trace as the generic LLM error message rather than as tool calls, so fidelity metrics that only inspect `tool_responses` misattribute them to the model.

### Capabilities Measured

- **Serving stack / infrastructure**: Did the serving layer return completions at all, or did every attempt die before the model could act?
- **Language Model**: Among the attempts the model actually got to make, how many produced well-formed tool calls?

## How It Works

### Evaluation Method

- **Type**: Deterministic (conversation trace analysis)
- **Granularity**: Conversation-level, with both per-call and per-turn estimates

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: Ordered trace entries (`tool_call`, `tool_response`, intended assistant text) with turn ids
- `intended_assistant_turns`: Assistant text per turn; the generic LLM error message is the trace-level fingerprint of a serving-layer failure
- `conversation_ended_reason`: `"error"` marks whole-conversation serving failures

### Outcome Taxonomy

- `valid_call`: tool call executed without a format/parse error
- `parse_failure`: call emitted but not executable — error types shared with `tool_call_validity` (`CALL_ERROR_TYPES`: `tool_not_found`, `invalid_parameter`, field-validation errors, …)
- `serving_failure`: assistant turn produced by the error path — the serving layer never returned a model completion
- `model_non_call`: real assistant reply that simply made no tool call

### Scoring

- **Scale**: 0.0-1.0 — tool-selection fidelity over model-reachable attempts (`valid_call / (valid_call + parse_failure)`)
  - Serving failures are excluded from the denominator and reported separately
  - **Edge case**: no tool calls *and* serving failures present → score is `None` with `skipped: true` and `confounded: true`; do not read this record as a model non-call or as 0%/100% fidelity
  - **Edge case**: no tool calls and no serving failures → `skipped: true` (the model made no attempts)

### Sub-metrics

- `num_tool_calls`: count of paired calls (normalized_score is `None`)
- `valid_call_rate`: per-instance estimate of call validity
- `parse_failure_rate`: share of calls that were not executable
- `serving_failure_rate`: share of assistant turns on the error path
- `turn_pooled_fidelity`: share of calling turns whose calls are all valid — diverges from `valid_call_rate` when failures cluster inside turns (`aggregation_gap` in details reports the difference)

## Example Output

```json
{
  "name": "tool_call_attribution",
  "score": 0.75,
  "normalized_score": 0.75,
  "details": {
    "total_tool_calls": 4,
    "outcome_counts": {
      "valid_call": 3,
      "parse_failure": 1,
      "serving_failure": 0,
      "model_non_call": 1
    },
    "attribution": {
      "serving_layer": {"failed_turns": 0, "failed_turn_ids": [], "conversation_ended_in_error": false},
      "model": {"parse_failures": 1, "non_call_turns": 1}
    },
    "per_instance_fidelity": 0.75,
    "turn_pooled_fidelity": 0.5,
    "aggregation_gap": 0.25,
    "confounded": false
  }
}
```

## Related Metrics

- [tool_call_validity.md](tool_call_validity.md) - Fraction of calls with correctly formatted parameters; consumes `tool_responses` only, so it scores a serving-rejected record 1.0 ("no tool calls to evaluate")

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/tool_call_attribution.py`
- **Class**: `ToolCallAttributionMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)

Adapted from *"Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation"* (arXiv:2609.26693). The paper probes live serving stacks (Ollama, llama.cpp, vLLM, SGLang); this metric reads the same outcome taxonomy off eva's recorded conversation traces.
