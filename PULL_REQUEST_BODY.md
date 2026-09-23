## Summary

`stt_wer` is our only default STT quality diagnostic and it is word-level only: it charges function-word drift ("a" vs "the", "I'd" vs "I would") at full price, so transcripts that preserved meaning rank the same as transcripts that changed what was said. Per "Rethinking Human-Aligned Evaluation: An Analysis of Semantic Metrics Beyond WER" (arXiv:2609.21663), WER also agrees least with human judgment among ASR metrics, and the paper recommends CER plus a semantic distance measure instead.

This PR adds a `transcription_semantic_accuracy` diagnostic metric that separates the two signals: character error rate (CER) for lexical drift and a SemDist-style content-token semantic distance for meaning change. Both are deterministic and offline (jiwer + token comparison, no LLM), so the metric joins the default cascade run at no extra cost. `metrics_version` is bumped 2.2.0 → 2.2.1 because the default metric set changes.

## Changes

- `src/eva/metrics/diagnostic/transcription_semantic_accuracy.py` (new): `TranscriptionSemanticAccuracyMetric` (`CodeMetric`, `v0.1`) — corpus-level CER surfaced as a `character_error_rate` rate sub-metric, plus mean per-turn SemDist (1 − cosine similarity of stopword-filtered content-token frequency vectors; negations deliberately count as content words). `supported_pipeline_types = {CASCADE}`, `exclude_from_pass_at_k = True`, `language` config (default `"en"`).
- `src/eva/metrics/diagnostic/__init__.py`: import + `__all__` registration so `@register_metric` fires and the metric is included in `registry.list_metrics()` (i.e. the default run — this is what makes the version bump necessary).
- `src/eva/__init__.py`: `metrics_version` 2.2.0 → 2.2.1 — every default cascade evaluation now emits this metric (and its `character_error_rate` sub-metric), which changes metric outputs.
- `tests/fixtures/metric_signatures.json`: new `TranscriptionSemanticAccuracyMetric` signature entry (`v0.1`) so the drift/signature test stays green. `configs/prompts/judge.yaml` is untouched (the metric is deterministic), so no judge signatures changed.
- `tests/unit/metrics/test_transcription_semantic_accuracy.py` (new): 12 tests — registry wiring (registered + default-on, created from registry with config) and compute behavior (perfect transcription, lexical drift that preserves meaning vs `stt_wer` on the same turn, content-word substitution, dropped negation, CER sub-metric accounting, multi-turn per-turn details, bracket-annotation stripping, empty-turn skipping, no-common-turns error, non-English all-tokens fallback).
- `docs/metrics/transcription_semantic_accuracy.md` (new): metric page — the two signals, how to read them together, inputs, cascade-vs-audio-native behavior, example output.
- `docs/metrics/README.md`: metric count 15 → 16, Diagnostic 7 → 8, new table row linking the metric page.

## Testing

- New module `tests/unit/metrics/test_transcription_semantic_accuracy.py`: 12/12 passed (2 sync wiring tests via pytest; the 10 async compute tests were driven with an inline `asyncio.run` harness because `pytest-asyncio` is not installed in this sandbox — see note below).
- `tests/unit/metrics/test_metric_signatures.py`: passed with the regenerated fixture — the new `v0.1` entry matches the registered metric.
- Full unit suite (`PYTHONPATH=src python -m pytest tests/unit`): 1545 passed, 52 skipped, 3 xfailed, 308 failed — **all 308 failures are the sandbox's missing `pytest-asyncio` plugin** (`async def functions are not natively supported`; every failure line is that error, zero assertion failures). CI runs `uv run pytest tests/` with the dev extra installed and `asyncio_mode = "auto"`, where these collect normally.
- `metrics_version` bump 2.2.0 → 2.2.1 recorded in `src/eva/__init__.py` and picked up by provenance (`eva.utils.provenance`).
- ruff: not runnable in this sandbox (binary absent, package installs blocked); changed Python files pass `python -m compileall` clean. CI's tests job runs ruff via pre-commit.

## Known limitations

- The paper's sentence-embedding SemDist is replaced by stopword-filtered content-token cosine similarity to keep the metric deterministic and offline. The proxy catches content-word substitutions, deletions and insertions, but not free paraphrase ("would like" vs "want").
- The stopword list is English-only; other languages conservatively compare all tokens, so function-word drift in non-English counts as semantic.
- CASCADE-only by design: audio-native (AUDIO_LLM / S2S) models receive raw audio rather than STT transcripts, so STT meaning preservation is not meaningful there and the metric is skipped.
- Diagnostic only — excluded from pass@k and final evaluation scores, like the rest of the diagnostic category.
- Judge prompts, pricing tables and postprocessors are untouched (out of scope); no `judge.yaml` signature regen was needed beyond the new code-metric signature entry.
