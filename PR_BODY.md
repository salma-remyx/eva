## Summary

Adds string-verified evidence grounding to EVA's conversation-level judge metrics. After each judge call, the quoted citations in a flagged dimension's `evidence` field are checked against the transcript the judge saw, and each flagged dimension receives a `grounded` / `fabricated` / `unquoted` verdict — a zero-cost judge-reliability diagnostic that lands in `details["evidence_grounding"]` and aggregates into a run-level grounding rate, without ever changing a score. The verification discipline is adapted from "Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases" (arXiv:2609.30012).

## Changes

- `src/eva/metrics/evidence_grounding.py` — new module: citation extraction (straight/curly/guillemet quotes, ellipsis-split fragments), case- and whitespace-insensitive verification against the transcript, per-code adjudication report, and the `evidence_grounding` sub-metric merge
- `src/eva/metrics/base.py` — `ConversationTextJudgeMetric.compute` now calls `attach_evidence_grounding` behind the `verify_evidence` config flag (default `true`)
- `src/eva/metrics/accuracy/faithfulness.py` — version bump v0.2 → v0.3
- `src/eva/metrics/experience/conversation_progression.py` — version bump v0.1 → v0.2
- `src/eva/metrics/validation/user_behavioral_fidelity.py` — version bump v0.1 → v0.2
- `src/eva/__init__.py` — `metrics_version` 2.2.0 → 2.3.0 (metric output shapes changed)
- `tests/fixtures/metric_signatures.json` — regenerated `source_hash`/`version` for the three metrics
- `tests/unit/metrics/test_evidence_grounding.py` — new test module: citation extraction, verification, adjudication report, sub-metric merge, and compute wiring for faithfulness / conversation_progression
- `docs/metrics/evidence_grounding.md` — new per-metric page: verdict definitions, report location, run-level aggregation, `verify_evidence` opt-out
- `docs/metrics/README.md` — lists the new page under "Validating the Judges"
- `docs/metrics/faithfulness.md`, `docs/metrics/conversation_progression.md`, `docs/metrics/user_behavioral_fidelity.md` — short notes that these metrics now run through the shared verification hook
- `README.md` — one-line pointer to the new docs page

## Testing

- `tests/unit/metrics/test_evidence_grounding.py`: 23 passed.
- Full suite verified: 1890 passed, 0 failed, 52 skipped, 3 xfailed (async tests driven via a per-test `asyncio.run` harness; `pytest-asyncio` was not available in the authoring environment).
- `ruff check` clean.
- mypy: no new errors — `evidence_grounding.py` reports 0 errors; the pre-existing strict-mode baseline (bare `dict` annotations, etc.) is unchanged.

## Not addressed here

- No judge prompts changed (`prompt_hash` values in the fixture are unchanged), so `configs/prompts/judge.yaml` and the judge validation datasets are untouched.
- Verification never reweights a score: a `fabricated` verdict is reported, not penalized.
- Audio judge metrics (`agent_speech_fidelity`, `user_speech_fidelity`, `tts_fidelity`) are not verified — they do not emit text `evidence` fields.
- `user_behavioral_fidelity`'s `corruption_analysis` reasoning fields are deliberately not verified (reasons are not expected to be quotes); the shared hook is in place should its prompt start citing verbatim evidence.
- No historical runs are re-scored; existing results keep their original `metrics_version` provenance.
