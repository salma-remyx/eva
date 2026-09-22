## Summary

Adds **prosody expressiveness** — an opt-in audio-judge diagnostic metric (`prosody_expressiveness`) implementing the decoupled multi-dimensional prosody judgment of D-LPJ from [Multi-Dimensional Prosody Judgment For Live Streaming Speech Synthesis](https://arxiv.org/abs/2609.20124v1). The metric subclasses `SpeechFidelityBaseMetric` (same `MetricContext` contract as `tts_fidelity`) and rates each agent turn 0–3 on three independently judged dimensions — emotion, intonation, energy — over the existing Gemini audio-judge path, with deliberately **no overall-verdict target** in the rubric so per-dimension judgments cannot collapse onto a single preference bit. Dimensions the judge is unsure about are returned as null and excluded from that dimension's aggregation instead of coerced onto the scale; results surface as independent per-dimension sub-metrics plus a `dimension_collapse_rate` diagnostic for verdict coupling, written to each record's `metrics.json` and the run's `metrics_summary.json` via `scripts/run_prosody_expressiveness.py` driving the standard `MetricsRunner`. Bumped `metrics_version` 2.2.0 → 2.3.0 and regenerated `tests/fixtures/metric_signatures.json`.

## Changes

- **`configs/prompts/prosody.yaml`** — new prosody judge prompt in its own top-level `prosody:` namespace (PromptManager merges yaml files at the top level, so a second file redefining `judge:` would clobber judge.yaml's entire judge section): per-dimension 0–3 rubric with no overall verdict, and an explicit instruction to return null for any dimension the judge is unsure about (inference-time uncertainty masking).
- **`src/eva/metrics/diagnostic/prosody_expressiveness.py`** — new `ProsodyExpressivenessMetric(SpeechFidelityBaseMetric)`: per-turn 0–3 ratings on emotion/intonation/energy with no overall-verdict target in the parsed rubric; null-masking of uncertain dimensions (masked ratings excluded from that dimension's aggregation, `num_masked_ratings` surfaced in details); sub-metrics `prosody_expressiveness.emotion|intonation|energy` with independent aggregation, so a strong energy score can no longer hide a flat intonation score; `dimension_collapse_rate` sub-metric — the fraction of multi-rated turns where every dimension received the same rating (1.0 = fully coupled judge). Carries `version = "v0.1"` and `prompt_namespace = "prosody"` so the signature drift test tracks it like every other judge metric.
- **`src/eva/metrics/diagnostic/__init__.py`** — import the module like every other diagnostic metric so it registers in the global metric registry; `exclude_from_default_metrics = True` keeps it out of default runs (same shape as `tts_fidelity`).
- **`src/eva/metrics/base.py` + `src/eva/metrics/signatures.py`** — new `BaseMetric.prompt_namespace` class attribute (default `"judge"`): `get_judge_prompt()` and the signature drift test now both resolve judge prompts as `{prompt_namespace}.{name}.{prompt_key}`, so a judge metric whose prompt lives outside judge.yaml stays drift-tracked instead of duplicating `get_judge_prompt`. Behavior unchanged for every existing metric.
- **`scripts/run_prosody_expressiveness.py`** — entry point that drives the standard `MetricsRunner` on an existing benchmark run-dir, so results land in each record's `metrics.json` and the run's `metrics_summary.json` with the usual aggregation, sub-metrics, and bootstrap confidence intervals.
- **`tests/unit/metrics/test_prosody_expressiveness.py`** — 10 tests covering the decoupling properties: registry wiring and opt-out, prompt decoupling, no-overall-verdict parsing, null masking, invalid-rating masking, per-dimension sub-metrics, and collapse-rate computation.
- **`src/eva/__init__.py` + `tests/fixtures/metric_signatures.json`** — bumped `metrics_version` 2.2.0 → 2.3.0 (metric/judge-prompt change) and regenerated the signature fixture via `scripts/regen_metric_signatures.py` (18 → 19 entries; `ProsodyExpressivenessMetric` v0.1 with its `prosody.`-namespace prompt hash) so the drift/signature test stays green.
- **`docs/metrics/prosody_expressiveness.md`** — metric definition page: rubric, scoring, example output, and implementation/wiring details.
- **`README.md`** — short opt-in usage note pointing at the script and the docs page.

## Testing

- `pytest tests/unit/metrics/test_prosody_expressiveness.py tests/unit/metrics/test_metric_signatures.py`: **11 passed, 0 failed** (the 10 new metric tests plus the signature drift test against the regenerated fixture).
- Full suite in this sandbox: **1567 passed, 52 skipped, 3 xfailed, 310 failed** — every one of the 310 failures is `Failed: async def functions are not natively supported`, i.e. the sandbox lacks `pytest-asyncio` (a dev dependency CI installs via `uv sync --extra dev` before `pytest tests/`); **0 assertion/logic failures** anywhere in the suite.
- `scripts/check_version_bump.py` exits 0 (metrics_version bump staged alongside the `src/eva/metrics/` changes), and `scripts/regen_metric_signatures.py` output is identical to the checked-in fixture. ruff/mypy are not installed in this sandbox — CI's pre-commit (`ruff-check`, `ruff-format`) remains the lint gate.

## Known limitations

- **Paper components intentionally out of scope** (training infra an eval framework doesn't host): the distilled student model Live-ProsodyJudge (Qwen3-Omni distilled from Gemini — would require a training harness plus curated pairwise preference data); the span-local GRPO RL loop and its rationale-span advantage normalization (needs an RL trainer and reward infra the repo cannot host); and the pairwise A/B comparison protocol with balanced-order 10-sample voting and Best-of-8 tournament selection for TTS preference optimization (an RL-feedback use case outside an eval framework's needs). The decoupling result is instead reproduced through rubric and response design on the paper's own teacher model (Gemini) via the pointwise per-turn rubric on eva's `MetricContext`.

## Notes

- The initial draft's self-review flagged this as **orphan-shaped** (module not imported by `src/eva/metrics/diagnostic/__init__.py`, no metric `version`, no signature-fixture entry — default-run scores and saved signatures stayed unchanged). Addressed in this revision: the module is now imported like every diagnostic metric (registered in the global registry, still excluded from default runs via `exclude_from_default_metrics`), carries `version = "v0.1"`, and is tracked in `tests/fixtures/metric_signatures.json`; the only user-facing invocation path remains the opt-in script.
- Per-record judge cost/latency is one Gemini audio call, the same as `tts_fidelity`; `dimension_collapse_rate` is lower-is-better via the `_rate` suffix convention.
- The full-suite async-plugin caveat above applies to any local run without `uv sync --extra dev`.

_Drafted by [Outrider](https://github.com/remyxai/outrider) — paper: [arXiv:2609.20124](https://arxiv.org/abs/2609.20124v1)._

<details>
<summary>Discovery context</summary>

Drafted by an autonomous discovery loop — Remyx ranks recent arXiv papers against this team's research interest and shipping history; Claude Code selects the candidate most directly implementable against this repo from the lookback window and drafts it.

**Research interest**: eva · **Implementation by**: Claude Code as autonomous agent

**Why this paper for this team**: This paper is highly relevant to the eva team's effort to 'Improve faithfulness and progression metrics for S2S models' by offering a novel approach to evaluating nuanced conversational qualities. While focused on TTS, the core contribution of `Decoupled-Live-ProsodyJudge (D-LPJ)` provides a robust method for assessing 'fine-grained, highly expressive prosody such as emotion, intonation, and energy,' which is critical for human-like conversational quality in S2S systems. The team's ambition to capture the 'nuances of spoken interaction' extends beyond simple accuracy. The paper's solution to 'verdict coupling' in LLM judges by masking uncertain pair-dimensions offers a sophisticated technique for developing more granular, reliable metrics for prosody, aligning with eva's emphasis on statistical rigor and moving beyond traditional MOS scores.

**Why this candidate (selected from the lookback pool)**: D-LPJ's contribution maps onto eva's audio-judge metric contract (MetricContext agent audio + per-turn timestamps → per-dimension per-turn ratings → aggregated MetricScore).

**License / code availability**: no code repository or model URL surfaced in the paper, recommendation envelope, or arXiv abstract page; no license detected (class `no-code-link`, compat 0.30). Worth confirming the paper has an open release before investing in adoption.

</details>

Co-Authored-By: remyx-ai[bot] <289541483+remyx-ai[bot]@users.noreply.github.com>
