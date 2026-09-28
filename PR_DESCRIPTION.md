## Summary

Adds `tool_call_attribution`, a new deterministic diagnostic metric that splits tool-call protocol failures into serving-layer failures versus model non-calls. Serving-layer failures (request rejected before inference, retry exhaustion, parser artifacts) surface in the conversation trace as the generic LLM error message rather than as tool calls, so a fidelity metric that only inspects `tool_responses` reads them as "the model made no calls" — or as a perfect score when no calls exist at all. This metric separates the two so a diagnostic report can say *which layer* failed, and marks a record whose serving stack rejected every request as `skipped` rather than scoring it 0% (model never called) or 100% (no calls to invalidate).

The metric registers itself via `@register_metric` and is auto-discovered through the existing `eva.metrics.diagnostic` package import, so it is reached through the standard `MetricsRunner` path (`--metrics tool_call_attribution`) — no new entry point. Its sub-metrics flow through the generic sub-metric aggregation in `MetricsRunner`, so they appear in run-level aggregates without further wiring.

Adapted from *"Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation"* (arXiv:2609.26693). The paper probes live serving stacks (Ollama, llama.cpp, vLLM, SGLang); this port reads the same outcome taxonomy off eva's recorded conversation traces instead.

## Changes

- `src/eva/metrics/diagnostic/tool_call_attribution.py` — new metric module. `ToolCallAttributionMetric` (`CodeMetric`, `category = "diagnostic"`, `version = "v0.1"`, `exclude_from_pass_at_k = True`) pairs each `tool_call` with its following `tool_response` in trace order, classifies each into a four-way outcome taxonomy (`valid_call`, `parse_failure`, `serving_failure`, `model_non_call`), and reports tool-selection fidelity at both per-call and turn-pooled granularity plus the `aggregation_gap` between them. Parse failures reuse the `CALL_ERROR_TYPES` taxonomy shared with `tool_call_validity`; serving failures are detected from the generic LLM error message in `intended_assistant_turns` and excluded from the fidelity denominator. Sub-metrics: `num_tool_calls`, `valid_call_rate`, `parse_failure_rate`, `serving_failure_rate`, `turn_pooled_fidelity`.
- `src/eva/metrics/diagnostic/__init__.py` — registration: one `from . import tool_call_attribution  # noqa` import plus the `__all__` entry. This import is the whole wiring step — the `@register_metric` decorator on the class does the rest.
- `docs/metrics/tool_call_attribution.md` — new user-facing doc: capabilities measured, input `MetricContext` fields, the outcome taxonomy, scoring rules and both no-tool-call edge cases, sub-metric list, and example output.
- `docs/metrics/README.md` — new row in the Diagnostic table (pipeline + language model, deterministic) and the Diagnostic count bumped 7 → 8.
- `tests/unit/metrics/test_tool_call_attribution.py` — new suite, 7 tests grouped under `Test*` classes by behavior: registration through the diagnostic package, the serving-failure confound (including a regression case showing `tool_call_validity` scores the same serving-rejected record a perfect 1.0), per-instance vs. turn-pooled divergence, `CALL_ERROR_TYPES` reuse, and the two no-call edge cases.
- `tests/fixtures/metric_signatures.json` — regenerated to include `ToolCallAttributionMetric` (`prompt_hash: null` — deterministic, no judge prompt — `source_hash: 33cc02d5e8db`, `version: v0.1`).
- `src/eva/__init__.py` — `metrics_version` 2.2.0 → 2.2.1; adding a metric changes metric outputs, so the version bump accompanies the regenerated signature fixture.

## Testing

- `pytest tests/unit/metrics/test_tool_call_attribution.py` — **7 passed**.
- `pytest tests/unit/metrics/test_metric_signatures.py` — **1 passed** (signature/drift test confirms the fixture is in sync with the new metric module).
- `pytest tests/unit/metrics` — **427 passed**, confirming the existing metric suites are unaffected.
- `python scripts/check_version_bump.py` — exit 0 (the `metrics_version` bump is staged alongside the `src/eva/metrics/` change).

## Not addressed here

- **Live serving-stack probe harness.** The paper probes Ollama, llama.cpp, vLLM and SGLang directly; this PR only reads the recorded traces eva already produces. Porting the probe harness is out of scope.
- **Constrained-decoding ablation.** The paper's ablation over constrained/grammar-based decoding is not reproduced.
- **Mitigation experiments.** No attempt to reduce serving-layer failure rates (retry budgets, backoff tuning, parser hardening).
- **Two-stack comparison.** No head-to-head benchmark run comparing serving stacks; no leaderboard results are published here.

## Notes

- Diagnostic-only: the metric is excluded from pass@k (`exclude_from_pass_at_k = True`) and is not a component of any EVA composite, consistent with the other `eva.metrics.diagnostic` metrics. Existing metric and composite behavior is unchanged.
- No judge prompt is added: the metric is deterministic, so there is no `configs/prompts/judge.yaml` entry and `prompt_hash` is `null` in the signature fixture.
- No operational configuration is introduced, so `.env.example` needs no update.
