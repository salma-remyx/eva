# Disclosure-Sharding Condition Suite

This suite measures how well a voice agent integrates user requirements that
are disclosed incrementally across turns, adapted from the full/concat/sharded
protocol of [SCB (SpeechConversationBench)](https://arxiv.org/abs/2609.40198):

- **full** — the record's original `high_level_user_goal`, delivered naturally;
- **concat** — the goal's clause-level shards restated as a numbered list the
  user communicates in full on its first turn (isolates sensitivity to goal
  *reformulation*);
- **sharded** — the same shards disclosed one per conversational turn, in order
  (adds the cost of *incremental spoken interaction*).

The paired `sharded − concat` delta in a metric such as `task_completion` is
the incremental-disclosure cost; `full − concat` tells you how much of any
observed gap is just reformulation.

Each condition is a normal evaluation record differing only in its id
(`::<condition>` suffix) and the goal text the user simulator reads. Personas,
decision trees, information requirements, starting utterances, and ground
truths are carried over verbatim, so every condition is graded against the
same final outcome and per-condition scores are directly comparable.

## Files

| File | Role |
|------|------|
| `disclosure_sharding.py` | Deterministic clause sharding of `high_level_user_goal` and construction of the three condition variants. |
| `make_condition_dataset.py` | Expands an existing dataset into condition-variant records via `EvaluationRecord.load_dataset`/`save_dataset`. |
| `condition_deltas.py` | Per-condition means and paired cross-condition deltas with bootstrap CIs, seeded through `eva.utils.bootstrap` like the framework's other statistics. |

## How to run

```bash
# 1. Expand a dataset into full/concat/sharded variants
uv run python analysis/disclosure_sharding/make_condition_dataset.py \
    --input data/airline_dataset.json --output output/disclosure/airline_conditions.json

# 2. Point a regular benchmark run at the expanded dataset, then analyze the trial scores
uv run python analysis/disclosure_sharding/condition_deltas.py \
    --input output/disclosure/trial_scores.csv --output-dir output_processed/disclosure \
    --metrics task_completion
```

`condition_deltas.py` reads the same long-format `trial_scores.csv` as the
perturbation pipeline (`system_alias`, `scenario_id`, `metric`, `trial`,
`value`; `record_id`/`alias` column names are also accepted) and writes
`condition_summary.csv` and `condition_deltas.csv`.

## Notes and limitations

- Sharding is parameter-free: sentences split on terminal punctuation, then on
  comma/semicolon boundaries; fragments too short to disclose alone are folded
  back; the tail is merged beyond `--max-shards` (default 4). The first shard
  is the core request; later shards are additional requirements.
- Goals that yield a single shard get only a `full` variant — there is nothing
  to deliver incrementally — and the generator lists their ids.
- Starting utterances are carried over unchanged, so localized openers keep
  working; the staged script governs when each requirement is revealed. The
  shard text itself is derived from the (English) goal statement.
- Conditions are an experimental design, not a new registered metric: compare
  conditions on the metrics EVA already scores.
