# Speech Naturalness

> **Diagnostic Metric**: A single "naturalness" score hides *where* speech breaks down. This metric probes the agent's spoken audio across 10 linguistically grounded dimensions so the distribution of failures is visible.

## Overview

Audio-based metric that evaluates how **natural** the assistant's spoken audio sounds, using an audio LLM for multimodal analysis. Rather than collapsing quality into one opaque score, it deconstructs naturalness into 10 perceptual dimensions (segmental, prosodic, fluency, voice-quality, affect, register) and asks the judge to flag, per turn, which dimensions are defective. Each dimension is surfaced as its own sub-metric so teams can see whether a model's speech is, for example, intelligible but inexpressive, or fluent but unnaturally paced.

Adapted from ["Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions"](https://arxiv.org/abs/2608.09930), which deconstructs TTS naturalness into linguistically grounded dimensions and probes automated evaluators per-dimension. This metric ports that probing *mechanism* (per-dimension binary rating by an audio judge with per-dimension rates) onto EVA's audio-judge contract. It is **complementary** to [tts_fidelity](tts_fidelity.md) (word-level fidelity to the intended text), not duplicative: `tts_fidelity` asks "did it say the right words?"; this metric asks "did it say them naturally?".

> [!NOTE]
> By default, this diagnostic metric is excluded. Enable it explicitly with `--metrics speech_naturalness` (or include it in a comma-separated `--metrics` list).

### Capabilities Measured

- **Perceptual naturalness**: How human the agent's speech sounds across linguistic dimensions, independent of whether the words match the script.

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3 Flash
- **Granularity**: Per-turn (each assistant turn evaluated independently)

### Input Data

Uses the following MetricContext fields:
- `audio_assistant_path`: Path to assistant-only audio file
- `intended_assistant_turns`: What the assistant intended to say (provided as context only; the metric judges the *speech*, not word fidelity)

### Linguistic Dimensions (10)

Each dimension is a binary "is this kind of unnaturalness present?" probe:

| Key | Dimension |
|---|---|
| `mispronunciation` | A word or phoneme mispronounced, distorted, or replaced by the wrong sound (segmental error) |
| `unnatural_intonation` | Pitch contour / melody is flat, robotic, or does not match the utterance type |
| `inappropriate_stress` | Lexical or sentential stress on the wrong syllable or word |
| `unnatural_pacing` | Speech rate or rhythm is off (too fast/slow, mechanical, irregular timing) |
| `pause_placement` | Pauses missing, misplaced, or of unnatural length relative to syntactic structure |
| `disfluency` | Unscripted hesitations, repetitions, prolongations, false starts, stutter-like breaks |
| `unnatural_voice_quality` | Timbre is harsh, breathy, creaky, metallic, cracked, or non-human |
| `inexpressive_affect` | Emotion/affect is flat or mismatched to the conversational context |
| `phrasing_boundary_error` | Intonational phrasing / boundary tones break clauses unnaturally |
| `register_tone_mismatch` | Tone, formality, or register does not fit the dialogue |

### Scoring

- **Parent scale**: 0-1 (binary per turn)
  - 1: Natural — none of the ten dimensions are defective
  - 0: Unnatural — at least one dimension is defective
- **Sub-metrics**: one per dimension, `speech_naturalness.<dimension>_rate` = `flagged turns / rated turns`
- **Normalization**: Already 0-1 scale
- **Aggregation**: Mean across all assistant turns (parent score)

## Example Output

```json
{
  "name": "speech_naturalness",
  "score": 0.5,
  "normalized_score": 0.5,
  "details": {
    "aggregation": "mean",
    "num_turns": 2,
    "num_evaluated": 2,
    "per_turn_ratings": {"0": 0, "1": 1},
    "per_turn_failure_modes": {"0": ["unnatural_intonation", "unnatural_pacing"], "1": []}
  },
  "sub_metrics": {
    "unnatural_intonation_rate": {"name": "speech_naturalness.unnatural_intonation_rate", "score": 0.5},
    "unnatural_pacing_rate": {"name": "speech_naturalness.unnatural_pacing_rate", "score": 0.5},
    "mispronunciation_rate": {"name": "speech_naturalness.mispronunciation_rate", "score": 0.0}
  }
}
```

## Related Metrics

- [tts_fidelity.md](tts_fidelity.md) - Word-level fidelity to the intended text (orthogonal: "right words?" vs "natural speech?")
- [speakability.md](speakability.md) - Checks if the *text* is voice-friendly before it is spoken (upstream concern)
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Speech clarity and articulation of entities

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/speech_naturalness.py`
- **Class**: `SpeechNaturalnessMetric`
- **Base Class**: `SpeechFidelityBaseMetric` → `AudioJudgeMetric`
- **Prompt**: `configs/prompts/judge.yaml` under `judge.speech_naturalness`
- **Configuration**: `audio_judge_model` (default: Gemini 3 Flash), `aggregation` (default: "mean")
