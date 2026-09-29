# Evidence Grounding

> **Diagnostic Sub-metric**: A judge score justified by fabricated citations is a confident-sounding error. Evidence grounding string-verifies the transcript citations behind every flagged judge dimension, at zero extra cost.

## Overview

Conversation-level judge metrics ([`faithfulness`](faithfulness.md) and [`conversation_progression`](conversation_progression.md)) ask the judge to justify every flagged dimension with an `evidence` field citing the transcript. After each judge call, EVA string-verifies those citations: quoted spans in a flagged dimension's evidence are checked against the transcript the judge saw (case- and whitespace-insensitive), and each flagged dimension receives a verdict.

Verification is deterministic — no additional LLM call — and never changes a score. It is a reliability diagnostic, not a reweighting: it tells you whether the judge's flags rest on quotes that actually exist in the transcript.

### Verdicts

| Verdict | Meaning |
|---|---|
| `grounded` | Every cited quote appears in the transcript. |
| `fabricated` | At least one cited quote does not appear in the transcript. |
| `unquoted` | The dimension was flagged but its evidence quotes nothing, so there is nothing to verify. |

Clean (unflagged) dimensions are not adjudicated: the judge prompts ask for a *reason* there, not a conviction, and reasons are not expected to be quotes. For the same reason, fields holding free-form reasoning rather than citations are skipped — [`user_behavioral_fidelity`](user_behavioral_fidelity.md)'s `corruption_analysis` entries use `analysis` fields, so they are not verified. The hook runs after every conversation-level text judge call, so any metric whose prompt starts asking for quoted evidence is adjudicated automatically.

A citation may bridge omitted material with an ellipsis (`"first part ... last part"`); each side of the ellipsis is verified on its own, and every fragment must be found.

## How It Works

### Evaluation Method

- **Type**: Deterministic (string verification, no LLM)
- **Granularity**: Per flagged judge dimension, attached to the parent metric's score
- **Where it runs**: In `ConversationTextJudgeMetric.compute`, after the judge response is parsed and the score is built

### Output

Per record, the adjudication report lands in `details["evidence_grounding"]` of the parent metric's `MetricScore`:

- `codes`: per-dimension verdicts, each with its extracted citations and a `verified` flag per citation
- Summary counts: `codes_flagged`, `codes_grounded`, `codes_fabricated`, `codes_unquoted`, `citations_total`, `citations_verified`, and `citations_grounding_rate`

When the judge cited anything, an `evidence_grounding` sub-metric (share of cited quotes that verified) is merged into the parent score's `sub_metrics` as `<metric>.evidence_grounding`, alongside any sub-metrics the metric itself emits. The runner's generic sub-metric aggregation reports it as a run-level grounding rate per judge metric in `metrics_summary.json` — a low rate means the judge is flagging dimensions with quotes it cannot actually find in the transcript, and the affected records are worth inspecting before trusting their scores.

### Opting Out

> [!NOTE]
> Verification is on by default. Set `verify_evidence: false` in a metric's config to disable it for that metric.

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
        "violating_policies": {"evidence": "Agent \"proceeded with rebooking without confirming the fare difference of $45\".", "flagged": true, "rating": 2},
        "hallucination": {"evidence": "Agent mentioned \"a complimentary first-class upgrade\" that no tool result provides.", "flagged": true, "rating": 1}
      }
    },
    "num_turns": 14,
    "evidence_grounding": {
      "codes": {
        "violating_policies": {
          "flagged": true,
          "verdict": "grounded",
          "citations": [
            {"quote": "proceeded with rebooking without confirming the fare difference of $45", "verified": true}
          ]
        },
        "hallucination": {
          "flagged": true,
          "verdict": "fabricated",
          "citations": [
            {"quote": "a complimentary first-class upgrade", "verified": false}
          ]
        }
      },
      "codes_flagged": 2,
      "codes_grounded": 1,
      "codes_fabricated": 1,
      "codes_unquoted": 0,
      "citations_total": 2,
      "citations_verified": 1,
      "citations_grounding_rate": 0.5
    }
  },
  "sub_metrics": {
    "violating_policies_rate": {"score": 1.0, "normalized_score": 1.0},
    "hallucination_rate": {"score": 1.0, "normalized_score": 1.0},
    "evidence_grounding": {
      "name": "faithfulness.evidence_grounding",
      "score": 0.5,
      "normalized_score": 0.5,
      "details": {
        "codes_flagged": 2,
        "codes_grounded": 1,
        "codes_fabricated": 1,
        "codes_unquoted": 0,
        "citations_total": 2,
        "citations_verified": 1,
        "citations_grounding_rate": 0.5
      }
    }
  }
}
```

## Related Metrics

- [faithfulness.md](faithfulness.md) - Flagged dimensions are string-verified; the report and sub-metric are attached to its score.
- [conversation_progression.md](conversation_progression.md) - Same verification as faithfulness.
- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Runs through the same hook, but its `analysis` reasoning fields are deliberately not verified.
- [judge_validation_datasets/](judge_validation_datasets/) - Offline, human-annotated judge validation; evidence grounding is the runtime complement.

## Implementation Details

- **File**: `src/eva/metrics/evidence_grounding.py`
- **Hook**: `ConversationTextJudgeMetric.compute` in `src/eva/metrics/base.py`, behind the `verify_evidence` config flag (default: `true`)
- **Sub-metric builder**: `make_rate_sub_metric` in `src/eva/metrics/utils.py`
- **Configuration**: `verify_evidence` (default: `true`)
- **Provenance**: Adapted from the string-verified adjudication discipline of "Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases" (arXiv:2609.30012).
