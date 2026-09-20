# TTS Streaming Speed

> **Diagnostic Metric**: Characterizes the TTS stage as a streaming component — how fast it starts speaking and whether it sustains real-time synthesis. Useful for diagnosing latency bottlenecks, but not directly scorable since acceptable TTFA/RTF targets depend on the deployment. Turn taking and response speed provide the end-to-end timing assessment.

## Overview

Deterministic metric that measures the streaming behavior of a cascade's TTS stage, reporting the two numbers the streaming-TTS literature uses for exactly this question:

- **TTFA (time-to-first-audio)**: seconds from the TTS request to the first audible sample, i.e. time-to-first-byte plus any leading silence the service pads onto the response. Reported as a distribution (mean/p50/p95 in ms) across all TTS calls in the conversation.
- **RTF (real-time factor)**: total TTS synthesis wall-time divided by the total duration of the recorded assistant audio. Below 1 means the synthesizer keeps up with real-time playback; above 1 means audio queues up behind the synthesizer even after first audio.

## Capabilities Measured

- **Speech Synthesis**: Isolates the TTS stage's streaming speed (time to first audio, sustained synthesis rate), separate from end-to-end latency.

## How It Works

### Evaluation Method

- **Type**: Deterministic (per-record analysis of the Pipecat metrics stream)
- **Granularity**: Per-TTS-call samples with conversation-level aggregation

### Input Data

Reads artifacts a cascade run already produces for each record:

- `pipecat_metrics.jsonl` (from `context.output_dir`):
  - `TTFAMetricsData` entries for TTS processors — serialized field-wise as `{ttfa, ttfb, leading_silence}` (seconds). Preferred source: TTFA includes the leading-silence correction.
  - `TTFBMetricsData` entries (or `LatencyMetric` with `stage="tts"`) — fallback for older pipecat streams that don't emit TTFA; lacks the leading-silence correction.
  - `ProcessingMetricsData` entries for TTS processors — per-request synthesis wall-time.
- `audio_assistant.wav` (from `context.audio_assistant_path`) — the recorded assistant audio, whose duration is the RTF denominator.

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — TTFA and RTF describe the discrete TTS stage.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`). Audio-native models generate speech as part of a single model call with no per-request TTS stage, so the metrics-stream signals this metric reads are absent.

### Sanity Checks

- TTS entries are matched the same way the orchestrator's latency stats match them: the processor name contains `TTSService`, or the entry is a `LatencyMetric` with `stage="tts"`. Non-TTS processors (STT, LLM) are ignored.
- Latency values outside `(0, 10)` seconds and processing values outside `(0, 300)` seconds are discarded — a single `run_tts` call covers one utterance, so anything beyond these windows is a stuck request, not synthesis time.
- An unreadable or missing metrics file (or no usable TTS entries) yields a *skipped* result, not an error.

### Scoring

- **Scale**: Parent score is the conversation-level RTF (ratio; lower is better, < 1 keeps up with playback). The TTFA sub-metric is in milliseconds.
- **Normalization**: None — these are physical quantities, not quality judgments.
- **Aggregation**: TTFA mean/p50/p95 over per-call values (nearest-rank percentiles, matching the orchestrator worker's latency stats); RTF = sum of per-call processing time / assistant audio duration.
- **Attribution**: `details.source` records whether TTFA figures came from `ttfa` (silence-corrected) or the `ttfb` fallback, so readers can tell which signal a run used.

## Example Output

```json
{
  "name": "tts_streaming_speed",
  "score": 0.24,
  "normalized_score": null,
  "details": {
    "rtf": 0.24,
    "ttfa_mean_ms": 300.0
  },
  "sub_metrics": {
    "ttfa": {
      "name": "tts_streaming_speed.ttfa",
      "score": 300.0,
      "details": {
        "source": "ttfa",
        "num_calls": 2,
        "mean_ms": 300.0,
        "p50_ms": 200.0,
        "p95_ms": 400.0,
        "leading_silence_mean_ms": 75.0
      }
    },
    "rtf": {
      "name": "tts_streaming_speed.rtf",
      "score": 0.24,
      "details": {
        "num_calls": 2,
        "total_processing_seconds": 3.0,
        "audio_duration_seconds": 12.5
      }
    }
  }
}
```

## Related Metrics

- [response_speed.md](response_speed.md) - End-to-end user-utterance-end → assistant-response-start latency (all pipeline stages)
- [tts_fidelity.md](tts_fidelity.md) - Correctness of the synthesized speech (what was said, not how fast)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/tts_streaming_speed.py`
- **Class**: `TTSStreamingSpeedMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
