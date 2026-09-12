# TTS Robustness

> **Diagnostic Metric**: Complex text (numbers, dates, named entities, code-switching) is where speech synthesis fails first — this metric separates those failures from the average so they stop hiding inside aggregate scores.

## Overview

Deterministic metric that measures how faithfully the agent's spoken output survives the TTS → STT round trip on **assistant turns**, with a focus on complex text. It is the assistant-side counterpart to [`stt_wer`](stt_wer.md) (which covers user turns only), and fills the gap between that user-side WER and the binary audio-judge [`tts_fidelity`](tts_fidelity.md).

Adapted from *"Complex-Text Robustness Evaluation and Failure Diagnosis for Low-Resource Multilingual Text-to-Speech"* (arXiv:2609.11545). The paper's diagnostic axes are kept — content consistency, language consistency, and a pre-synthesis Text Risk Score — while its trained language-identification model is substituted with a parameter-free script + function-word heuristic so the metric stays fully deterministic.

### Capabilities Measured

- **Speech Synthesis**: measures whether spoken assistant output preserves the intended text through the TTS → STT round trip, split by text complexity.

## How It Works

### Evaluation Method

- **Type**: Deterministic (uses jiwer character error rate; no LLM, no model training)
- **Granularity**: Per-turn with conversation-level aggregation

### Input Data

Uses the following MetricContext fields:

- `intended_assistant_turns`: What the agent intended to speak (reference text)
- `transcribed_assistant_turns`: What the assistant's STT heard in the synthesized audio (hypothesis text)

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — compares intended text against the round-trip transcription of the TTS output.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`). In audio-native systems there is no intended-text-to-speech step to compare against.

### Evaluation Methodology

Three axes, mirroring the paper:

1. **Content consistency (CER)** — Both sides go through the same WER normalization pipeline as `stt_wer` (language-aware, configured per run), then character error rate is computed per turn and reported separately for **complex** and **simple** turns. A persistent gap between `complex_text_accuracy` and `simple_text_accuracy` is the signature of a synthesizer that mishandles hard inputs.
2. **Language consistency** — Each transcript turn is checked against the expected language profile: a `wrong_script` flag when most letters are in another script, and a `no_target_language_markers` flag when a long turn contains none of the language's function words or signature diacritics (catches same-script drift such as English output during a French run).
3. **Text Risk Score (TRS)** — A pre-synthesis risk estimate computed from the *intended text alone*, before any audio exists: digit groups, numeric dates/times, acronyms and proper nouns, code-switching (multiple script families), very long turns, and symbol clutter. Binary triggers with interpretable weights; the capped sum drives the complex/simple split (threshold 0.5 by default, configurable via `risk_threshold`).

### Scoring

| Score | Direction | Description |
|-------|-----------|-------------|
| `score` | Lower is better | Conversation-level character error rate over assistant turns |
| `normalized_score` | Higher is better | `1 - CER` (character accuracy) |
| `wrong_language_rate` | Lower is better | Fraction of evaluated turns that left the target language |
| `complex_text_accuracy` | Higher is better | Character accuracy on high-risk (TRS ≥ threshold) turns |
| `simple_text_accuracy` | Higher is better | Character accuracy on low-risk turns |

Per-turn diagnostics in `details` include CER, TRS, triggered risk features, language flags, and top character-level substitutions — useful for spotting systematic failures like dropped dates or mangled confirmation codes.

## Related Metrics

- [`stt_wer`](stt_wer.md) — user-turn counterpart (simulator TTS → agent STT).
- [`tts_fidelity`](tts_fidelity.md) — binary audio-judge view of the same round trip; requires an audio model, this metric does not.
- [`agent_speech_fidelity`](agent_speech_fidelity.md) — key-entity audio fidelity.

## Implementation Details

Source: [`src/eva/metrics/diagnostic/tts_robustness.py`](../../src/eva/metrics/diagnostic/tts_robustness.py)
