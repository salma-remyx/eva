"""TTS streaming speed metric: time-to-first-audio and real-time factor.

Diagnostic metric characterizing how a cascade's TTS stage behaves as a
streaming component, using the two numbers the streaming-TTS literature
reports for exactly this question (e.g. TontaubeV1, arXiv:2609.08703:
~200 ms time-to-first-audio, RTF 0.08 single-input / 0.02 aggregate):

- TTFA (time-to-first-audio): seconds from the TTS request to the first
  audible sample, i.e. time-to-first-byte plus any leading silence the
  service pads onto the response.
- RTF (real-time factor): synthesis wall-time divided by synthesized audio
  duration. Below 1 means the TTS keeps up with real-time playback; above 1
  means audio queues up behind the synthesizer even after first audio.

Both are computed from the per-record pipecat metrics stream
(pipecat_metrics.jsonl): ``TTFAMetricsData`` entries when the pipecat
version emits them (falling back to ``TTFBMetricsData`` / ``LatencyMetric``
stage="tts" for older streams), and ``ProcessingMetricsData`` entries for
TTSService processors as per-request synthesis wall-time. RTF divides the
sum of the latter by the duration of the recorded assistant audio
(audio_assistant.wav).

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import json
import math
import wave
from pathlib import Path
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore

# Same sanity window the orchestrator's latency stats use for TTS entries.
_TTS_LATENCY_MAX_SECONDS = 10.0
# A single run_tts call covers one assistant utterance; anything beyond this
# is a stuck request, not synthesis time.
_TTS_PROCESSING_MAX_SECONDS = 300.0


def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate the p-th percentile using the nearest-rank method.

    Mirrors the orchestrator worker's percentile helper so latency figures
    match across result.json and this metric.

    Args:
        sorted_data: Pre-sorted list of values (ascending).
        p: Percentile in (0, 100].

    Returns:
        The percentile value.
    """
    rank = math.ceil(p / 100.0 * len(sorted_data))
    return sorted_data[rank - 1]


def _as_float(value: object) -> float | None:
    """Return ``value`` as float if it is a real number, else None.

    Args:
        value: Parsed JSON value of arbitrary type.

    Returns:
        The value as a float, or None for non-numeric input (bool included).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_tts_stream_metrics(metrics_path: Path) -> dict[str, list[float]]:
    """Collect per-request TTS streaming signals from pipecat_metrics.jsonl.

    Args:
        metrics_path: Path to the pipecat_metrics.jsonl file for one record.

    Returns:
        Dict with per-request lists (seconds): ``"ttfa"`` and
        ``"leading_silence"`` from TTFAMetricsData entries, ``"ttfb"`` from
        TTFBMetricsData / LatencyMetric stage="tts" entries (TTFA fallback),
        and ``"processing"`` from ProcessingMetricsData entries. Missing
        signals are absent; an unreadable file yields an empty dict.
    """
    signals: dict[str, list[float]] = {"ttfa": [], "leading_silence": [], "ttfb": [], "processing": []}
    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    metric = json.loads(line)
                except ValueError:
                    continue

                # Same TTS-processor match the worker's latency stats use:
                # every TTS service class name contains "TTSService".
                processor = metric.get("processor") or ""
                metric_type = metric.get("type")
                is_tts_latency = metric_type == "LatencyMetric" and metric.get("stage") == "tts"
                if "TTSService" not in processor and not is_tts_latency:
                    continue

                value = metric.get("value")
                if metric_type == "TTFAMetricsData":
                    # MetricsFileObserver serializes field-wise metrics as a
                    # dict of their non-base fields: {ttfa, ttfb, leading_silence}.
                    if isinstance(value, dict):
                        ttfa = _as_float(value.get("ttfa"))
                        if ttfa is not None and 0 < ttfa < _TTS_LATENCY_MAX_SECONDS:
                            signals["ttfa"].append(ttfa)
                        silence = _as_float(value.get("leading_silence"))
                        if silence is not None and 0 <= silence < _TTS_LATENCY_MAX_SECONDS:
                            signals["leading_silence"].append(silence)
                elif metric_type in ("TTFBMetricsData", "LatencyMetric"):
                    ttfb = _as_float(value)
                    if ttfb is not None and 0 < ttfb < _TTS_LATENCY_MAX_SECONDS:
                        signals["ttfb"].append(ttfb)
                elif metric_type == "ProcessingMetricsData" and "TTSService" in processor:
                    processing = _as_float(value)
                    if processing is not None and 0 < processing < _TTS_PROCESSING_MAX_SECONDS:
                        signals["processing"].append(processing)
    except Exception:
        return {"ttfa": [], "leading_silence": [], "ttfb": [], "processing": []}

    return signals


def _audio_duration_seconds(audio_path: str | None) -> float | None:
    """Return the duration of a PCM wav file in seconds, or None if unreadable.

    Args:
        audio_path: Path to a wav file (typically the record's assistant audio).

    Returns:
        Duration in seconds, or None when the path is missing/unreadable.
    """
    if not audio_path:
        return None
    try:
        with wave.open(str(audio_path), "rb") as wav:
            frame_rate = wav.getframerate()
            if frame_rate <= 0:
                return None
            return wav.getnframes() / frame_rate
    except (OSError, wave.Error):
        return None


@register_metric
class TTSStreamingSpeedMetric(CodeMetric):
    """TTS streaming speed metric.

    Measures how quickly a cascade's TTS stage starts speaking (TTFA) and
    whether it sustains real-time synthesis (RTF) across a conversation.

    Parent score is the conversation-level RTF: total TTS processing
    wall-time divided by total synthesized (assistant) audio duration.
    Sub-metrics carry the TTFA distribution (mean/p50/p95 in ms, plus the
    leading-silence share when the metrics stream reports TTFA).

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "tts_streaming_speed"
    version = "v0.1"
    description = "Diagnostic metric: TTS streaming speed — time-to-first-audio and real-time factor"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    higher_is_better = False  # TTFA (ms) and RTF (ratio) — lower is better.
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            metrics_path = Path(context.output_dir) / "pipecat_metrics.jsonl"
            signals = _load_tts_stream_metrics(metrics_path) if metrics_path.exists() else {}

            ttfa_values = signals.get("ttfa") or signals.get("ttfb") or []
            # TTFB lacks the leading-silence correction; record which signal
            # the TTFA figures came from so readers can tell.
            ttfa_source = "ttfa" if signals.get("ttfa") else "ttfb"

            if not ttfa_values and not signals.get("processing"):
                return MetricScore(name=self.name, score=None, normalized_score=None, skipped=True)

            sub_metrics: dict[str, MetricScore] = {}

            if ttfa_values:
                sorted_ttfa = sorted(ttfa_values)
                num_calls = len(sorted_ttfa)
                ttfa_details: dict[str, Any] = {
                    "source": ttfa_source,
                    "num_calls": num_calls,
                    "mean_ms": round(sum(sorted_ttfa) / num_calls * 1000, 1),
                    "p50_ms": round(_percentile(sorted_ttfa, 50) * 1000, 1),
                    "p95_ms": round(_percentile(sorted_ttfa, 95) * 1000, 1),
                }
                leading_silence = signals.get("leading_silence") or []
                if leading_silence:
                    mean_silence_ms = sum(leading_silence) / len(leading_silence) * 1000
                    ttfa_details["leading_silence_mean_ms"] = round(mean_silence_ms, 1)
                sub_metrics["ttfa"] = MetricScore(
                    name=f"{self.name}.ttfa",
                    score=ttfa_details["mean_ms"],
                    normalized_score=None,
                    details=ttfa_details,
                )

            # RTF = synthesis wall-time / synthesized audio duration. Reported
            # only when both signals exist for the record.
            rtf: float | None = None
            processing = signals.get("processing") or []
            audio_seconds = _audio_duration_seconds(context.audio_assistant_path)
            if processing and audio_seconds:
                total_processing = sum(processing)
                rtf = total_processing / audio_seconds
                sub_metrics["rtf"] = MetricScore(
                    name=f"{self.name}.rtf",
                    score=round(rtf, 4),
                    normalized_score=None,
                    details={
                        "num_calls": len(processing),
                        "total_processing_seconds": round(total_processing, 3),
                        "audio_duration_seconds": round(audio_seconds, 3),
                    },
                )

            if not sub_metrics:
                return MetricScore(name=self.name, score=None, normalized_score=None, skipped=True)

            return MetricScore(
                name=self.name,
                score=round(rtf, 4) if rtf is not None else None,
                normalized_score=None,
                details={
                    "rtf": round(rtf, 4) if rtf is not None else None,
                    "ttfa_mean_ms": sub_metrics["ttfa"].score if "ttfa" in sub_metrics else None,
                },
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)
