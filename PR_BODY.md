## Summary

Every judge eva ships today scores the assistant from the outside: an evaluator reads the
transcript and grades it against a rubric. Nothing in the metric suite reports what the
*user* was privately thinking while the call was going on, so a conversation can look clean
to a third-party judge while the caller was quietly annoyed. This PR adds
`intrinsic_user_feedback`, an opt-in experience judge adapted from
["Training User Simulators with Latent Intrinsic Feedback"](https://arxiv.org/abs/2610.13617):
instead of a third-party evaluator, the judge role-plays the user — grounded in the same
persona and goal the user simulator was given — and reports, after every assistant turn, the
latent evaluation states the user was having at that moment (`internal_thought`, `reaction`,
and a 1–5 satisfaction `rating`).

Evidence — per-turn latent states from one replayed airline conversation ("move my AUS→LAX
flight to March 25", judge response replayed in-sandbox; this environment has no live API keys):

```json
{
  "name": "intrinsic_user_feedback",
  "score": 4.0,
  "normalized_score": 0.75,
  "details": {
    "per_turn_ratings": {"1": 5, "2": 2, "3": 5},
    "per_turn_internal_thoughts": {
      "1": "Straight to the point, no messing around.",
      "2": "I already told them the date, why are they asking again?",
      "3": "Everything I asked for, said once."
    },
    "per_turn_reactions": {
      "1": "reassured — they picked up right away",
      "2": "annoyed — repeating myself",
      "3": "relieved — got the seat and the price"
    },
    "num_turns": 3,
    "num_evaluated": 3
  },
  "sub_metrics": {"intrinsic_user_feedback.dissatisfaction_rate": {"score": 0.333}}
}
```

Turn 2 is the interesting one: externally nothing failed — the tool call succeeded and the
reply is polite — but the user was annoyed at being asked for the date twice, and that lands
as a 2/5 plus a `dissatisfaction_rate` of 0.333 localized to that turn.

## Changes

- **New metric** `src/eva/metrics/experience/intrinsic_user_feedback.py` —
  `IntrinsicUserFeedbackMetric` (`version = "v0.1"`, `category = "experience"`,
  `exclude_from_default_metrics = True`), a `PerTurnConversationJudgeMetric` that rates every
  *assistant* turn from the user's first-person perspective. From the paper it carries over:
  - turn-level signals rather than one conversation score — dissatisfaction is localized to
    the turn that caused it;
  - no hindsight — each turn is rated as the user experienced it at that point, so a mistake
    fixed two turns later still counts as earlier dissatisfaction.
- **First-person grounding** — `_render_user_brief` renders the `user_persona` and
  `user_goal` the simulator was given (structured goal dicts with `must_have_criteria` and
  plain-string goals), so latent states are judged against what *this* user wanted rather
  than a generic quality rubric.
- **Sub-metric** `intrinsic_user_feedback.dissatisfaction_rate` — share of rated turns at or
  below `dissatisfaction_threshold` (default 2, configurable via the metric's config dict).
- **Judge prompt** `judge.intrinsic_user_feedback` in `configs/prompts/judge.yaml`, with the
  regenerated entry in `tests/fixtures/metric_signatures.json`.
- **Call site** — no new wiring: the metric registers itself on import of
  `eva.metrics.experience` and runs through the existing `MetricsRunner` when named in
  `--metrics` / `EVA_METRICS`.
- **Docs** — new page `docs/metrics/intrinsic_user_feedback.md`; `docs/metrics/README.md`
  (15 → 16 metrics, Experience 3 → 4, opt-in note); `.env.example` `EVA_METRICS` entry with
  the opt-in worked example.
- **Versioning** — `metrics_version` 2.2.0 → 2.2.1 in `src/eva/__init__.py`.

## Tests

- [x] `tests/unit/metrics/test_intrinsic_user_feedback.py` restructured into `Test*` classes;
  `test_only_assistant_turns_are_rated`, `test_judge_prompt_is_grounded_in_the_users_own_goal`,
  and `test_plain_string_goal_still_renders` parametrized over airline/itsm/medical_hr —
  10 passed (`pytest tests/unit/metrics/test_intrinsic_user_feedback.py`)
- [x] `test_latent_states_collected_per_turn`, `test_ratings_aggregate_and_normalize`,
  `test_dissatisfaction_rate_sub_metric`, `test_null_rating_excluded_from_aggregation` —
  pass; driven directly via `asyncio.run` in-sandbox because `pytest-asyncio` is not
  installed there (CI runs them under `asyncio_mode = "auto"` in `pyproject.toml`)
- [x] Rest of `tests/unit/metrics` unaffected: 252 passed vs 246 before the change; the 182
  failures before and after are identical missing-async-plugin collection errors, none new
- [x] `test_metric_signatures.py` passes with the regenerated fixture entry (1 passed)
- [x] `tests/unit/test_config_io.py`, `tests/unit/test_config_editor_integration.py` pass
  after the `.env.example` edit (22 passed)
- [ ] `ruff check` / `ruff format` — ruff is not installed in the patch sandbox; line lengths
  in the touched files verified ≤ 120 by hand, but the pre-commit pass still needs to run in CI

## Notes

- **Score comparability** — `intrinsic_user_feedback` is opt-in
  (`exclude_from_default_metrics = True`), so default-run scores are unchanged and existing
  metrics' computation is untouched; the `metrics_version` bump to 2.2.1 records the new
  metric in run provenance.
- Per-domain caller profiles (persona / goal / trace for airline, itsm, medical_hr) are added
  to `tests/unit/metrics/conftest.py` as `make_domain_metric_context`, reusable by the other
  metric suites.

## Not addressed here

- **Online latent state** — the paper trains simulators that emit latent state mid-call;
  eva's user simulators generate speech inside hosted voice models (ElevenLabs Agents,
  OpenAI Realtime) that cannot surface it, so this PR only adds post-hoc replay elicitation.
- **Judge validation** — no entry under `docs/metrics/judge_validation_datasets/` and no
  development note in `docs/metrics/metric_development/` yet; follow-up once live-judge
  ratings have been collected and reviewed.
- **Aggregation** — the metric is not wired into any EVA aggregate score or weighting.
