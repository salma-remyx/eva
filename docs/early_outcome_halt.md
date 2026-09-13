# Early Outcome Halt

Opt-in cost saving for development iterations: halt a conversation simulation
once its predicted outcome is already evident, instead of running every
conversation to its natural end. Adapted from *"EarlyEval: Cheaper Agent
Evaluation via Early Outcome Prediction"* (arXiv:2609.02783); see
[README → Early Outcome Halt](../README.md#early-outcome-halt-optional) for
the feature overview.

> **Leave this unset for EVA-X runs** and any run where full-conversation
> metrics matter. Halted conversations intentionally fail the goodbye
> validation and are counted as `not_finished`.

---

## Configuration

The feature is disabled unless an `EVA_EARLY_OUTCOME__*` variable (or the
equivalent YAML field on `RunConfig.early_outcome`) is set:

| Variable | Default | Description |
|---|---|---|
| `EVA_EARLY_OUTCOME__CONFIDENCE_THRESHOLD` | `0.8` | Predicted-outcome confidence required before halting (valid range 0.5–1.0). |
| `EVA_EARLY_OUTCOME__MIN_USER_TURNS` | `2` | Only evaluate the predictor after this many completed user turns. |
| `EVA_EARLY_OUTCOME__HALT_ON` | `failure` | `failure` halts only predicted failures (saves cost on doomed runs); `any` also halts predicted successes, for cost/accuracy experiments. |

```bash
EVA_EARLY_OUTCOME__CONFIDENCE_THRESHOLD=0.85
EVA_EARLY_OUTCOME__MIN_USER_TURNS=2
EVA_EARLY_OUTCOME__HALT_ON=failure
```

## How the prediction works

After every completed user turn, the predictor scores the partial
conversation from behavioral signals (repeated assistant turns, generic LLM
error apologies, user frustration markers, error events) and textual overlap
with the record's goal decision tree (must-have criteria, resolution and
failure conditions). When the score crosses the confidence threshold, the run
halts with end reason `early_halt`.

Scores are heuristic evidence strengths, not calibrated probabilities — pick
the threshold empirically for your workload (e.g. by comparing early-halted
runs against full runs on a record subset).

## Artifacts

- **`<record>/early_halt.json`** — written when a conversation is halted.
  Contains the predicted `outcome` (`success` | `failure`), the `confidence`,
  `user_turns` / `assistant_turns` at the halt point, and the behavioral
  `features` behind the prediction.
- **`evaluation_summary.json`** — the `simulation` block gains two keys:
  - `early_halted_count` — number of records halted by early outcome prediction.
  - `early_halted_record_ids` — the record IDs that were halted.

Halted records appear in the standard summary counts as `not_finished`.
