# Judge Stability

> **Diagnostic Metric**: Audits the measurement stability of the LLM judge itself rather than the agent. Not directly used in final evaluation scores; excluded from pass@k and from default runs (opt-in).

## Overview

Every text-judge metric in EVA rests on one assumption: the same judge request, sent to the same model, returns the same rating. This metric tests that assumption directly. It builds one judge prompt for a record and submits the **exact same request** several times through the shared judge call path — same model resolution and params as every other text judge, including the `service_tier: flex` default — then reports how often the judge agrees with itself.

The score is the exact-agreement rate: the share of repeats that match the modal rating. `1.0` means the instrument behaved like a frozen snapshot on this record; anything lower means the judge wobbled on byte-identical input, and `noise_floor_normalized` in the details gives the smallest between-model gap (in normalized units) this judge can resolve on this record. Differences smaller than that are noise, no matter what the leaderboard says.

Adapted from [*Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints*](https://arxiv.org/abs/2609.04198), which measured same-window repeat agreement of black-box LLM judges far below the stability levels leaderboard gating assumes, and recommends a small repeat pilot before trusting any gate built on a shared endpoint.

### Capabilities Measured

- **None (instrument audit)**: This metric measures the judge, not the agent. A low score here means the evaluation instrument is noisy on this record — it does not mean the agent performed poorly.

## How It Works

### Evaluation Method

- **Type**: LLM judge (text), repeated N times
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: The full conversation, formatted with the same transcript helper as the other conversation-level text judges.

### Evaluation Methodology

1. Format the transcript and build the judge prompt **once**, so every repeat sends byte-identical content.
2. Submit that prompt `repeats` times (default 5, configurable) via the shared `TextJudgeMetric.call_judge` path. Per-call token usage is logged to `judge_token_usage/` like any other judge metric.
3. Collect the ratings; count parse failures and out-of-schema ratings as readout noise instead of silently dropping them.
4. Score the exact-agreement rate (repeats matching the modal rating) and report the rating histogram, distinct-rating count, parse stability, and the normalized noise floor (max − min rating over the scale span).

### Scoring

- **Scale**: 0.0–1.0 (agreement rate)
  - 1.0: Every repeat returned the same rating.
  - 0.6 with ratings `[2, 3, 2, 3, 2]`: modal rating 2 in 3 of 5 repeats; noise floor 0.5 (one scale step).
- **Normalization**: Already 0-1 scale.
- **Aggregation**: Mean across records (excluded from pass@k). Note that the *details*, not the score, carry the actionable number: the noise floor other metrics should be read against.

## Example Output

```json
{
  "name": "judge_stability",
  "score": 0.6,
  "normalized_score": 0.6,
  "details": {
    "num_repeats": 5,
    "num_valid": 5,
    "num_parse_failures": 0,
    "num_invalid_ratings": 0,
    "per_repeat_ratings": [2, 3, 2, 3, 2],
    "rating_histogram": {"2": 3, "3": 2},
    "modal_rating": 2,
    "distinct_ratings": 2,
    "agreement_rate": 0.6,
    "noise_floor_normalized": 0.5,
    "parse_stability_rate": 1.0
  }
}
```

## When To Run It

Opt-in (`exclude_from_default_metrics = True`) because it multiplies judge calls by `repeats`. Run it as a small pilot when validating a judge configuration — e.g. over a handful of records after switching `JUDGE_MODEL` or editing judge prompts — before reading small score differences between systems as meaningful. This mirrors the source paper's finding that a pilot at ~2% of a study's call volume exposes an unreliable instrument in advance.

## Related Metrics

- [metric_context.md](../metric_context.md) - Data available to all metrics.
- [`bootstrap.py` confidence intervals](../README.md) - Between-record sampling uncertainty; this metric covers the complementary source of error, within-instrument instability.

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/judge_stability.py`
- **Class**: `JudgeStabilityMetric`
- **Base Class**: `TextJudgeMetric`
- **Category**: `diagnostic`
- **`exclude_from_pass_at_k`**: `True`
- **`exclude_from_default_metrics`**: `True`
- **Configuration**: `repeats` (int, default 5, minimum 2), plus the inherited `judge_model` / `judge_params` overrides
- **Judge prompt**: `judge.judge_stability.user_prompt` in `configs/prompts/judge.yaml`
