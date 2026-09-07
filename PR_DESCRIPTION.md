## Summary

New deterministic diagnostic metric `stt_reference_leakage` that flags STT turns recovering reference text the audio cannot support, adapted from the reference-disagreement and masked-number probe families of "Towards Quantifying Benchmark Optimization in ASR Models" (arXiv:2608.19936). Two probes run on `intended_user_turns` vs `transcribed_user_turns`: directive leakage (a non-speech directive such as `[slow]` appearing verbatim in the transcript) and truncation recovery (full intended-utterance recovery on a barge-in turn, where only a prefix was audible). The metric is CASCADE-only and diagnostic — excluded from pass@k and final evaluation scores.

## Changes

- `src/eva/metrics/diagnostic/stt_reference_leakage.py`: new `STTReferenceLeakageMetric` (v0.1) — CASCADE-only, deterministic, with directive-leakage and truncation-recovery probes, per-turn evidence (`leaked_directives`, `coverage`, `digit_runs`), and `directive_leakage_rate` / `truncation_recovery_rate` sub-metrics.
- `src/eva/metrics/diagnostic/__init__.py`: register the new module (import side effect wires it into the global metric registry).
- `src/eva/orchestrator/runner.py`: per-language metric config — `stt_reference_leakage` receives `{"language": config.language.value}` alongside `stt_wer`.
- `tests/unit/metrics/test_stt_reference_leakage.py`: new tests grouped per feature — `TestDirectiveLeakage`, `TestTruncationRecovery`, `TestRegistryWiring`.
- `tests/fixtures/metric_signatures.json`: new signature entry for `STTReferenceLeakageMetric` (regenerated).
- `docs/metrics/stt_reference_leakage.md` + `docs/metrics/README.md`: new metric page, plus the diagnostic-table row and count update (now 8 diagnostic metrics).

## Testing

Adds 7 unit tests in `tests/unit/metrics/test_stt_reference_leakage.py`. Full unit suite (1903 tests) and ruff pass; the signature-drift test passes with the regenerated fixture.

## Known limitations

- High rates do not prove benchmark optimization on their own: barge-in cut timing is approximate, and a fast speaker may genuinely finish before the cut lands. Single-turn flags are weak evidence; run-level rates are the meaningful signal.
- CASCADE pipeline only — audio-native pipelines where STT is internal to the stack are out of scope.
- Verifying the leakage channel's offline controllability (reachability experiments beyond flagging) is out of scope for this PR.
