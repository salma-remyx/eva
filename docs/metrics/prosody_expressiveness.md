# Prosody Expressiveness

> **Diagnostic Metric**: An agent can say every word correctly and still sound robotic. This metric rates *how* the agent speaks — the expressiveness of its prosody — dimension by dimension.

## Overview

Audio-judge metric that rates each assistant turn's prosody on three **independently judged** dimensions: **emotion**, **intonation**, and **energy**. It complements [tts_fidelity.md](tts_fidelity.md), which measures *what* was said; this metric measures *how* it was delivered, regardless of whether speech comes from a cascade TTS engine or direct audio generation.

The design is adapted from *Multi-Dimensional Prosody Judgment For Live Streaming Speech Synthesis* (D-LPJ, arXiv:2609.20124). The paper's central finding is **verdict coupling**: asked for a multi-dimensional rubric, an LLM judge lazily aligns every dimension score with its overall impression, collapsing the rubric into a single preference bit. The metric carries the paper's decoupling remedies into eva's pointwise per-turn contract:

- **No overall verdict** — the rubric never asks for an overall rating, so there is nothing for dimension scores to blindly follow.
- **Uncertainty masking** — the judge returns `null` for any dimension it is not confident about; masked ratings are excluded from that dimension's aggregation instead of being forced onto the scale.
- **Per-dimension rationale** — each rating carries its own evidence sentence, so one dimension's justification cannot stand in for the others.

The paper's pairwise (A/B) comparator, its distilled Qwen3-Omni judge, and its span-local GRPO training loop are intentionally out of scope: the decoupling is achieved through rubric and response design on the stock Gemini judge (the paper's own teacher model).

> [!NOTE]
> By default, this diagnostic metric is excluded. Run it on an existing benchmark run with `PYTHONPATH=src python scripts/run_prosody_expressiveness.py --run-dir output/<run_id>` (optionally `--records 1.1.1,2.1.3`).

### Capabilities Measured

- **Prosody / delivery quality**: emotional coloring, pitch contour naturalness, and vocal energy of the spoken output — the qualities MOS predictors miss, per the paper.

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3 Flash
- **Granularity**: Per-turn, per-dimension

### Input Data

Uses the following MetricContext fields:
- `audio_assistant_path`: Path to assistant-only audio file
- `intended_assistant_turns`: Intended text, used as *context for the intended delivery* — audio-direction tags such as `[warm]` or `[firm]` tell the judge what emotion was supposed to come through. Word-level fidelity is explicitly out of scope for this judge.

### Scoring

- **Scale**: 0-3 per dimension, per turn (0 = robotic/wrong, 3 = expressive and appropriate), with `null` = masked (uncertain)
- **Parent score**: mean over per-turn dimension means, normalized to 0-1
- **Sub-metrics**:
  - `emotion`, `intonation`, `energy` — each dimension's own normalized mean, so a strong energy score cannot hide a flat intonation score
  - `dimension_collapse_rate` (lower is better) — fraction of turns where every rated dimension received the *same* rating, the pointwise signature of verdict coupling. A fully-coupled judge yields 1.0.
- **Masked ratings**: excluded from their dimension's aggregation; counted in `details.num_masked_ratings`

## Example Output

```json
{
  "name": "prosody_expressiveness",
  "score": 2.333,
  "normalized_score": 0.778,
  "sub_metrics": {
    "emotion": {"score": 2.5, "normalized_score": 0.833},
    "intonation": {"score": 2.0, "normalized_score": 0.667},
    "energy": {"score": 2.5, "normalized_score": 0.833},
    "dimension_collapse_rate": {"score": 0.5, "details": {"collapsed_turns": [1], "num_rated": 2}}
  },
  "details": {
    "dimensions": ["emotion", "intonation", "energy"],
    "num_turns": 2,
    "num_evaluated": 2,
    "num_masked_ratings": 0,
    "per_turn_ratings": {"0": {"emotion": 3, "intonation": 2, "energy": 3}}
  }
}
```

## Related Metrics

- [tts_fidelity.md](tts_fidelity.md) - Measures *what* was said (word/entity fidelity); this metric measures *how* it was said
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Entity articulation clarity from the conversation trace

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/prosody_expressiveness.py`
- **Class**: `ProsodyExpressivenessMetric`
- **Base Class**: `SpeechFidelityBaseMetric` → `AudioJudgeMetric` (reuses role-audio loading, silence trimming, and the Gemini call/retry/file-upload fallback path)
- **Prompt**: `configs/prompts/prosody.yaml` under `prosody.prosody_expressiveness` (kept in its own top-level namespace because the prompt manager merges yaml files at the top level; resolved via the class's `prompt_namespace = "prosody"`)
- **Versioning**: `version = "v0.1"`; tracked in `tests/fixtures/metric_signatures.json` (the drift test hashes the `prosody.`-namespace prompt like any other judge metric)
- **Configuration**: `audio_judge_model` (default: Gemini 3 Flash), `aggregation` (default: "mean")
- **Wiring**: opt-in via `scripts/run_prosody_expressiveness.py` (`exclude_from_default_metrics = True`; the module is imported by `src/eva/metrics/diagnostic/__init__.py` like every metric, so it is registered but not run by default). To include it in every run, drop `exclude_from_default_metrics`
