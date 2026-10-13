# Judge Reliability Audit

> **Opt-in audit, not a metric**: bootstrap CIs quantify sampling noise and `prompt_hash` quantifies prompt drift, but neither tells you whether a judge's verdict would flip if you simply asked again or handed the transcript over in the other direction. This audit measures exactly that, per verdict.

## Overview

Every text-judge score is normally treated as a deterministic point estimate. The judge reliability audit re-runs the judge under two stress tests — identical repetitions and a reversed-order transcript — and attaches a `judge_reliability` entry to the score's `details` in `records/<record_id>/metrics.json` reporting whether the verdict survived them.

Adapted from [*All Verdicts are Not Equal: Rethinking LLM Judge Reliability*](https://arxiv.org/abs/2610.12083). The paper stress-tests pairwise-comparison judges across repetitions and position swaps and collapses the failure modes into the *trustworthy verdict rate*. EVA's judges score a single conversation against a rubric rather than compare a pair, so the paper's A/B position swap maps to reversing the order in which the transcript turns are presented (`reverse_context_order`).

Only text judges (`TextJudgeMetric`) are audited. Code metrics have no sampling noise, and audio judges don't read a transcript, so the order swap means nothing for them.

## Opting In

The audit is off by default — no extra judge calls, unchanged scores. Two run-level knobs (fields on `RunConfig`, env vars with the `EVA_` prefix):

```bash
# Extra identical re-runs per text-judge verdict (reproducibility)
EVA_JUDGE_RELIABILITY_REPS=2

# Also re-run the judge on the transcript presented back-to-front (order invariance)
EVA_JUDGE_RELIABILITY_ORDER_SWAP=true
```

Or via CLI: `--judge-reliability-reps 2 --judge-reliability-order-swap`.

Per-metric config wins over the run-level values, so a single judge can be audited (or excluded) individually:

```json
{"faithfulness": {"reliability_reps": 2, "reliability_order_swap": true}}
```

### Cost

Each audited verdict costs extra judge calls: `reliability_reps` forward re-runs, plus `reliability_reps + 1` reversed runs when the order swap is on (the reversed condition has no primary run of its own). With the defaults (`reps=0`, no swap) the audit is free; with `reps=2` and the swap on, a verdict costs `1 + 2 + 3 = 6` judge calls.

## What the `judge_reliability` Entry Reports

| Key | Meaning |
|-----|---------|
| `num_samples` | Total verdict samples taken (primary + re-runs) |
| `unparsed_verdicts` | Samples where no verdict could be parsed from the judge response |
| `verdicts` | The observed verdicts per order: `{"forward": [...], "reversed": [...]}` |
| `modal_verdict` | Most common forward verdict (ties break toward the lower verdict) |
| `reproducible` | All forward samples agree; `null` when fewer than two forward samples exist |
| `verdict_agreement_rate` | Fraction of forward verdicts matching the modal one |
| `order_invariant` | Forward and reversed modal verdicts agree; `null` with no reversed sample |
| `accurate` | Modal verdict matches a ground-truth verdict; `null` unless one is supplied |
| `trustworthy` | Conjunction of whichever components were estimable |

EVA's rubric judges produce no per-item ground truth, so `accurate` (and therefore the accuracy leg of the paper's T) is `null` in run output and does not gate `trustworthy`, which degrades to *reproducible ∧ order-invariant*.

### Example Output

```json
{
  "judge_reliability": {
    "num_samples": 4,
    "unparsed_verdicts": 0,
    "verdicts": {"forward": [3.0, 3.0], "reversed": [1.0, 1.0]},
    "modal_verdict": 3.0,
    "reproducible": true,
    "verdict_agreement_rate": 1.0,
    "order_invariant": false,
    "accurate": null,
    "trustworthy": false
  }
}
```

The verdict the run reports (`score`, aggregates, leaderboards) is never changed by the audit — only the `judge_reliability` entry is added.

## Module API (`eva.metrics.judge_reliability`)

The audit lives in `src/eva/metrics/judge_reliability.py` and is wired into `MetricsRunner._run_record` after each text-judge `compute()`. Its statistics are reusable for consumers that want run-level rates:

- `summarize_verdicts(samples, ground_truth=None)` — collapse one item's verdict samples into the per-item summary above.
- `trustworthy_verdict_rate(summaries)` — aggregate per-item summaries into population rates: the paper's T (`trustworthy_verdict_rate`), plus `reproducibility_rate`, `order_invariance_rate`, `accuracy_rate`, `flip_rate`, and `position_bias_accuracy_bound` (a verdict that flips under an arbitrary presentation order is correct in at most one of the two orders, so flipped items average at most half correct — the bound is `(1 + order_invariance_rate) / 2`).
- `audit_metric_verdict(compute, context, primary_verdict=..., reps=0, order_swap=False)` — run the audit against any metric's `compute`; returns `{}` when disabled, costing nothing.
- `reverse_context_order(context)` / `verdict_of(score)` — the order perturbation and the verdict reader.

Aggregating T across a whole run is left to consumers: per-record `judge_reliability` details land in the output JSON, and `trustworthy_verdict_rate` is a plain function over those summaries.

## Related Documentation

- [README.md](README.md) - Metrics documentation index
- [../metric_context.md](../metric_context.md) - MetricContext data structure
- [judge_validation_datasets/](judge_validation_datasets/) - Judge accuracy against human labels
