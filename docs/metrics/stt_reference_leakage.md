# STT Reference Leakage

> **Diagnostic Metric**: Flags STT transcripts that reproduce reference text the audio cannot support — a signal that transcription accuracy may be inflated by context leakage or benchmark-conditioned behavior rather than acoustic fidelity.

## Overview

Deterministic metric that quantifies how often the assistant's STT recovers content from the reference text (`tts_text_user`) that was never present in the audio. Adapted from the probe families of ["Towards Quantifying Benchmark Optimization in ASR Models"](https://arxiv.org/abs/2608.19936) (reference disagreement, masked-number recovery), which show that high-scoring ASR models can output verbatim reference spans even when the audio is contradictory, masked, or ambiguous.

EVA is well positioned to run these probes: it knows the exact spoken ground truth and which turns were truncated by a barge-in, so "the audio cannot support this span" is checkable deterministically.

### Capabilities Measured

- **Speech Recognition**: Trustworthiness rather than raw accuracy — whether the STT's agreement with the reference is explainable by what was actually audible.

## How It Works

### Evaluation Method

- **Type**: Deterministic (no LLM)
- **Granularity**: Per-turn evidence with conversation-level rate aggregation

### Input Data

Uses the following MetricContext fields:
- `intended_user_turns`: What the user simulator intended to say, including non-speech bracket directives (e.g. `[slow]`, `[laughs]`)
- `transcribed_user_turns`: What the assistant's STT transcribed
- `assistant_interrupted_turns`: Turns where the assistant barged in on the user, cutting the user's audio mid-utterance

### Probe Families

**Directive leakage** (paper family: reference disagreement)
Bracketed directives in the intended text are stage directions for TTS delivery — they are never vocalized, and the framework's own WER metric strips them from both sides before scoring. Framework bookkeeping labels (`[assistant interrupts]`, `[pause]`, …) are excluded, since the log processor may insert those into the transcript itself. Any remaining directive whose words appear verbatim (alphanumeric-normalized substring match) in the transcript is a span the audio cannot contain — for example, reference text reaching the STT through conversation-context carryover rather than through listening.

**Truncation recovery** (paper family: masked-number recovery)
On turns the assistant interrupted, the user's audio was cut mid-utterance, so the STT can only have heard a prefix. The probe measures multiset coverage of intended content words in the transcript (after the standard WER normalization pipeline, so digit-words vs digits converge) and flags turns where coverage reaches the full-recovery threshold (default: every intended word recovered). Long digit runs (default: ≥ 4 digits — confirmation codes, phone numbers) are recorded in the evidence since they are the paper's strongest signal.

### Scoring

- **Scale**: 0.0-1.0, **lower is better** (`higher_is_better = false`)
  - `score = flagged turns / evaluated turns`
- **Sub-metrics**:
  - `stt_reference_leakage.directive_leakage_rate` — turns with directive leakage / evaluated turns
  - `stt_reference_leakage.truncation_recovery_rate` — fully-recovered turns / barge-in turns (only when barge-in turns exist)
- **Skipped** when no probe surface exists (no directives in any intended turn and no barge-in turns), rather than reporting a misleading clean 0.0.

### Interpretation

A high rate does not prove benchmark optimization — barge-in cut timing is not exact, and a fast speaker may genuinely finish before the cut lands. Treat flags as review pointers: inspect the per-turn evidence (`leaked_directives`, `coverage`, `digit_runs`) and the STT configuration (e.g. whether context carryover is enabled) before drawing conclusions. Rates across a full run are the meaningful signal; single-turn flags are weak evidence.

## Example Output

```json
{
  "name": "stt_reference_leakage",
  "score": 0.25,
  "normalized_score": 0.25,
  "details": {
    "language": "en",
    "full_recovery_threshold": 1.0,
    "num_turns": 8,
    "num_flagged": 2,
    "flagged_turn_ids": [3, 6],
    "num_directive_probe_turns": 4,
    "num_truncation_eligible": 3,
    "per_turn_evidence": {
      "3": {
        "leaked_directives": [],
        "interrupted": true,
        "coverage": 1.0,
        "digit_runs": ["4821"]
      }
    }
  },
  "sub_metrics": {
    "directive_leakage_rate": {"name": "stt_reference_leakage.directive_leakage_rate", "score": 0.125},
    "truncation_recovery_rate": {"name": "stt_reference_leakage.truncation_recovery_rate", "score": 0.333}
  }
}
```

## Related Metrics

- [stt_wer.md](stt_wer.md) — overall word-level STT accuracy this metric audits for inflation
- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) — entity-level accuracy whose trustworthiness these probes contextualize

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/stt_reference_leakage.py`
- **Class**: `STTReferenceLeakageMetric`
- **Base Class**: `CodeMetric`
- **Configuration**:
  - `language` (default: "en") — WER normalization locale for coverage matching
  - `full_recovery_threshold` (default: 1.0) — coverage counting as full recovery on barge-in turns
  - `min_digit_run` (default: 4) — minimum digit-run length recorded as evidence
  - `min_turn_words` (default: 3) — turns shorter than this are not probed for truncation recovery
