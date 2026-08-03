# Clean Transcription WER

> **Diagnostic Metric**: A disfluency-aware companion to `stt_wer` — measures STT accuracy against the speaker's *final intent* rather than verbatim output, so fillers, repetitions, and false starts do not inflate the error rate. Not scored directly.

## Overview

Deterministic metric that measures Speech-to-Text (STT) accuracy as Word Error Rate (WER), computed on **disfluency-cleaned** transcripts. Verbatim transcripts retain hesitation fillers (`um`, `uh`), stuttering repetitions (`I I want`), and abandoned false starts that inflate WER without reflecting a real recognition error. This metric strips those disfluencies from both the reference and the hypothesis before running the existing WER normalization pipeline, so the score reflects recognition accuracy against what the speaker ultimately meant to say.

Inspired by the AgenticSR "audio-to-clean-text" formulation (*AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach*, arXiv:2607.28175). The paper cleans transcripts with a learned text-to-text "Refiner" model; this metric substitutes that learned component with a parameter-free, deterministic cleaner so the result is reproducible and side-effect-free for a diagnostic metric.

### Capabilities Measured

- **Speech Recognition**: Measures the quality of the speech-to-text pipeline against intent-preserving clean text, complementing the verbatim `stt_wer` baseline.

## How It Works

### Evaluation Method

- **Type**: Deterministic (uses the `jiwer` library)
- **Granularity**: Per-turn with conversation-level aggregation

### Disfluency Cleaning

`clean_disfluencies` is applied symmetrically to the reference and hypothesis (so it cannot bias WER) and targets the disfluency classes a deterministic rule can resolve correctly:

- **Hesitation fillers** (`um`, `uh`, `er`, `mm`, `hmm`, …) — English only; dropped as standalone tokens. Discourse words (`like`, `so`, `well`, …) are intentionally kept because they are context-dependent.
- **Immediate repetitions** from stuttering (`I I want`, `the the`) — collapsed to a single token.
- **Trailing cutoffs** from interrupted speech (`I want to--`) — dangling dashes/ellipses tidied.

Self-correction resolution (the paper's third disfluency class) is intentionally out of scope: a naive deterministic rule for it distorts meaning (an abandoned fragment is not always a whole-utterance restart), so it is left to a learned Refiner.

### Input Data

Uses the following `MetricContext` fields:

- `intended_user_turns`: What the user simulator intended to say (reference text)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis text)

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — measures the quality of the assistant's STT pipeline (`supported_pipeline_types = {CASCADE}`).
- **Audio-native (AUDIO_LLM / S2S):** Skipped, for the same reason as `stt_wer`.

## Scoring

- **Scale**: 0.0–∞ (clean WER, unbounded but typically 0.0–1.0). Lower is better.
- **Normalization**: `1 - clean_WER` (clamped to 0–1).
- **Comparison field**: `details.verbatim_wer` reports the WER *without* cleaning, and `details.disfluency_inflation` is `max(0, verbatim_wer - clean_wer)` — the gap quantifies how much disfluency was inflating the verbatim error rate.

## Running

Runs by default on cascade pipelines as a diagnostic metric (like `stt_wer`). Select it explicitly to recompute only this metric on an existing run:

```bash
python main.py --run-id <existing_run_id> --metrics clean_transcription_wer
```
