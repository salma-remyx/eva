# STT Semantic Similarity

> **Diagnostic Metric**: Provides a semantic view of STT quality to complement the word-level `stt_wer` — useful for telling lexically different but semantically faithful transcriptions apart from errors that change the meaning.

## Overview

Measures the semantic fidelity of Speech-to-Text (STT) transcription by comparing what the user simulator intended to say against what was transcribed using sentence-embedding cosine similarity (SemDist), with per-turn WER reported alongside for contrast.

Adapted from [Generative vs. Encoder Large Language Models for ASR Evaluation: A Comparative Study](https://arxiv.org/abs/2608.25574), which shows encoder-based semantic similarity correlates with human semantic judgments better than WER. The paper's SemDist configuration is used; its BERT-family encoders are swapped for a LiteLLM-routed embedding model.

### Capabilities Measured

- **Speech Recognition**: Semantic view of STT quality — robust to lexical drift that preserves meaning (including entity drift into adjacent turns) and sensitive to meaning-changing errors.

## How It Works

### Evaluation Method

- **Type**: Deterministic rule-based scoring over encoder embeddings (one embedding API call per conversation)
- **Granularity**: Per-turn similarity, conversation-level mean

### Input Data

Uses the following MetricContext fields (same I/O as `stt_wer`):
- `intended_user_turns`: What the user simulator intended to say (reference text)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis text)

Both sides go through the same bracket-annotation stripping and text normalization pipeline as `stt_wer` before embedding.

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — measures the quality of the assistant's STT pipeline.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`), for the same reason as `stt_wer`: audio-native models receive raw audio, not STT transcripts.

### Evaluation Methodology

For each turn with both reference and hypothesis text:

1. Embed the normalized reference and hypothesis with the configured embedding model (one batched call per conversation).
2. Compute cosine similarity between the two sentence embeddings and rescale from [-1, 1] to [0, 1]: `semdist = (cosine + 1) / 2`.
3. Compute the turn's WER (CER for ja/zh/ko) alongside, so the lexical and semantic views can be contrasted: a turn with high WER but high `semdist` drifted lexically while preserving meaning, while low `semdist` flags meaning-changing errors.

The conversation score is the mean of per-turn `semdist` values.

### Scoring

- **Scale**: 0.0-1.0 (higher is better)
  - 1.0: Transcription semantically identical to the intended utterance
  - ~0.5: Semantically unrelated (orthogonal embeddings)
- **Normalization**: Identity — `semdist` is already on a 0-1 scale.

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `language` | `"en"` | Language used for text normalization (same as `stt_wer`) |
| `embedding_model` | `text-embedding-3-small` (or `EMBEDDING_MODEL` env var) | LiteLLM embedding model name; credentials come from standard provider environment variables |

**Opt-in**: the metric costs one embedding API call per conversation, so it is excluded from default runs. Enable it explicitly:

```bash
python main.py \
    --run-id <existing_run_id> \
    --metrics stt_wer,stt_semantic_similarity
```

## Example Output

```json
{
  "name": "stt_semantic_similarity",
  "score": 0.971,
  "normalized_score": 0.971,
  "details": {
    "semdist": 0.971,
    "mean_wer": 0.088,
    "embedding_model": "text-embedding-3-small",
    "language": "en",
    "use_cer": false,
    "num_turns": 8,
    "per_turn_semdist": {"1": 1.0, "2": 0.982, "3": 1.0, "4": 0.901},
    "per_turn_wer": {"1": 0.0, "2": 0.25, "3": 0.0, "4": 0.1}
  }
}
```

Turn 2 above is the divergence case this metric exists to surface: WER of 0.25 with near-perfect semantic similarity.

## Related Metrics

- [stt_wer.md](stt_wer.md) - Lexical counterpart (word error rate); same I/O
- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - Entity-level accuracy (LLM judge)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/stt_semantic_similarity.py`
- **Class**: `STTSemanticSimilarityMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: `language` (default: "en"), `embedding_model` (default: `text-embedding-3-small` or `EMBEDDING_MODEL`)
