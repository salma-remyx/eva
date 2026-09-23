# Transcription Semantic Accuracy

> **Diagnostic Metric**: Provides a meaning-level view of STT quality to complement the word-level (`stt_wer`) and entity-level (`transcription_accuracy_key_entities`) views — useful for telling lexical drift that preserved meaning apart from errors that changed what was said.

## Overview

Deterministic metric that measures how well the *meaning* of the user's utterances survived Speech-to-Text transcription. It compares what the user simulator intended to say against what was transcribed and reports two signals:

- **CER** (character error rate): computed for **every language**, not only whitespace-less writing systems. Cheap, interpretable, and closer to human judgment of transcript quality than word error rate.
- **SemDist** (semantic distance): `1 - cosine similarity` between stopword-filtered content-token frequency vectors. Function-word drift ("a" vs "the", "I'd" vs "I would") does not move this score; substituted, deleted, or inserted content words (including negations) do.

Reading the two together answers the question `stt_wer` cannot: high CER with low SemDist means the STT made lexical errors that preserved meaning, while both high means the transcript actually changed what was said.

Adapted from [Rethinking Human-Aligned Evaluation: An Analysis of Semantic Metrics Beyond WER](https://arxiv.org/abs/2609.21663), which finds that WER agrees least with human judgment among tested ASR metrics, recommends shifting to CER (for English as well as morphosyllabic writing systems), and recommends SemDist as a complementary semantic measure. The paper's sentence-embedding SemDist is replaced here by content-token cosine similarity so the metric stays deterministic and offline; this proxy catches content-word changes but not free paraphrase ("would like" vs "want").

### Capabilities Measured

- **Speech Recognition**: Measures whether the transcription preserved the meaning of the spoken input, complementing word-level and entity-level views of the same pipeline.

## How It Works

### Evaluation Method

- **Type**: Deterministic (jiwer + token comparison, no LLM)
- **Granularity**: Per-turn with conversation-level aggregation (mean of per-turn SemDist; corpus-level CER)

### Input Data

Uses the following MetricContext fields:
- `intended_user_turns`: What the user simulator intended to say (reference text)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis text)

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — same inputs and same pipeline restriction as `stt_wer`.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`). Audio-native models receive raw audio, not STT transcripts, so measuring STT meaning preservation is not meaningful.

### Evaluation Methodology

Both texts go through the same normalization pipeline as `stt_wer` (Unicode conversion, digits to words, apostrophe normalization, …) before comparison. Bracket annotations such as `[slow]` are stripped. The semantic comparison then removes English function words (a curated stopword list that deliberately keeps negations, since dropping a "not" reverses meaning) and compares the remaining content tokens as frequency vectors. Languages other than English currently compare all tokens (conservative: function-word drift counts as semantic).

## Example Output

```json
{
  "name": "transcription_semantic_accuracy",
  "score": 0.0,
  "normalized_score": 1.0,
  "details": {
    "cer": 0.079,
    "semdist": 0.0,
    "semantic_accuracy": 1.0,
    "language": "en",
    "num_turns": 1,
    "per_turn_cer": {"1": 0.079},
    "per_turn_semdist": {"1": 0.0},
    "total_substitutions": 3,
    "total_deletions": 0,
    "total_insertions": 0,
    "reference_characters": 38
  },
  "sub_metrics": {
    "character_error_rate": {"name": "transcription_semantic_accuracy.character_error_rate", "score": 0.079}
  }
}
```

(The example is "I would like to book a flight to Paris" transcribed as "I'd like to book the flight to Paris": lexical drift visible in CER, meaning preserved.)

## Related Metrics

- [stt_wer.md](stt_wer.md) - Word-level error rate over the same turns
- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - Entity-level accuracy (names, dates, numbers)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/transcription_semantic_accuracy.py`
- **Class**: `TranscriptionSemanticAccuracyMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: `language` (default: "en")
