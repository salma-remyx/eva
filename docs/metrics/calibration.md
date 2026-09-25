# Judge Calibration

> **Scoring Utility**: Not a registered benchmark metric — this module scores a judge's outputs for calibration so the scores can be reported *alongside* accuracy, not instead of it.

## Overview

Accuracy and macro-F1 say whether a judge is right on average; they say nothing about whether the judge's confidence is empirically meaningful. A judge can be 84% accurate while being confidently wrong on the records it gets wrong, which is exactly the failure mode accuracy hides. Calibration scoring pairs every performance number with a check on its confidence.

EVA's judges emit no explicit confidence (no logprobs, no self-reported score), so confidence is derived from **self-consistency across repeat runs**: the share of runs that agree with the modal verdict. A judge that returns the same rating in every repeat run is treated as fully confident; one that flip-flops between ratings is treated as proportionally less confident. Correctness is measured against human labels.

## Scores

Given one confidence score and one binary correctness judgment per record:

- **Expected Calibration Error (ECE)**: bin-weighted `|empirical accuracy − mean confidence|` over equal-width confidence bins on [0, 1] (default 10 bins). 0.0 means perfectly calibrated.
- **Brier score**: mean squared error between confidence and correctness. Lower is better.
- **Confidence gap**: signed `mean confidence − empirical accuracy`. Positive means the judge is overconfident, negative underconfident.

`calibration_report` returns all three plus the per-bin reliability curve (`reliability_bins`, empty bins included with `None` stats), and `summary_fields()` flattens them into a `calibration_n` / `mean_confidence` / `judge_accuracy` / `ece` / `brier` / `confidence_gap` dict.

## Scoring the Judge-Validation Protocol

`judge_calibration_from_repeats` scores the same repeat-run protocol used for the [judge validation datasets](judge_validation_datasets/): confidence is the share of repeat runs agreeing with the modal rating, correctness is whether that modal rating matches the human `expected_rating`. The resulting ECE/Brier/confidence-gap numbers pair with the accuracy / macro-F1 results reported in that protocol. `load_validation_labels` loads the human labels from a validation dataset (one JSON object per line; records flagged `invalid` or missing `expected_rating` are skipped):

```python
from eva.utils.calibration import judge_calibration_from_repeats, load_validation_labels

labels = load_validation_labels("docs/metrics/judge_validation_datasets/faithfulness.jsonl")
report = judge_calibration_from_repeats(per_record_predictions, list(labels.values()))
print(report.summary_fields())  # ece, brier, confidence_gap, judge_accuracy, ...
```

## Scoring a Multi-Trial Benchmark Run

`metric_trial_agreement` groups a run's records by base scenario id (via `parse_trial_record_id`) and treats each trial's binary verdict (`get_score >= threshold`, default 0.5) as one repeat prediction — trials where the metric is missing or errored are excluded, mirroring how pass@k treats invalid trials. `judge_calibration_from_records` then combines those agreements with human pass/fail labels keyed by base scenario id: confidence comes from cross-trial verdict agreement, correctness from agreement with the human label. Scenarios without a human label are excluded — calibration is computed on the labeled subset only.

## Related Documentation

- [judge_validation_datasets/](judge_validation_datasets/) - Human-annotated validation datasets and judge accuracy / macro-F1 results
- [README.md](README.md) - Metrics system overview

## Implementation Details

- **File**: `src/eva/utils/calibration.py`
- **Configuration**: None (pure scoring helpers — no config knob or metric-registry entry)
- **Adapted from**: *Calibration as a First-Class Criterion in LLM Evaluation* (arXiv:2609.26489), which argues that every performance metric should be paired with a calibration score reporting whether confidence is empirically meaningful.
