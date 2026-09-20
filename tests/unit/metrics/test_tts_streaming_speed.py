"""Tests for the TTSStreamingSpeedMetric."""

import json
import wave

import pytest

import eva.metrics.diagnostic  # noqa: F401  (registers all diagnostic metrics)
from eva.metrics.diagnostic.tts_streaming_speed import TTSStreamingSpeedMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context

_TTS_PROCESSOR = "OpenAITTSService#0(Kokoro)"


def _write_metrics_jsonl(path, entries):
    """Write a synthetic pipecat_metrics.jsonl from plain dict entries."""
    with open(path, "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)


def _write_wav(path, seconds, sample_rate=24000):
    """Write a silent mono 16-bit PCM wav of the given duration."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate))


class TestTTSStreamingSpeedRegistration:
    def test_registered_with_diagnostic_package(self):
        """Importing eva.metrics.diagnostic registers the metric (wiring in diagnostic/__init__.py)."""
        assert get_global_registry().get("tts_streaming_speed") is TTSStreamingSpeedMetric

    def test_included_in_default_metrics(self):
        """The metric is not excluded, so benchmark runs pick it up by default."""
        assert "tts_streaming_speed" in get_global_registry().list_metrics()


class TestTTSStreamingSpeedMetric:
    @pytest.mark.asyncio
    async def test_skipped_without_metrics_file(self, tmp_path):
        """No pipecat_metrics.jsonl means no data: skipped, not errored."""
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path))

        result = await metric.compute(ctx)

        assert result.name == "tts_streaming_speed"
        assert result.score is None
        assert result.error is None
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_ttfa_from_ttfa_metrics(self, tmp_path):
        """TTFAMetricsData entries give TTFA stats plus the leading-silence share."""
        _write_metrics_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "type": "TTFAMetricsData",
                    "processor": _TTS_PROCESSOR,
                    "value": {"ttfa": 0.2, "ttfb": 0.15, "leading_silence": 0.05},
                },
                {
                    "type": "TTFAMetricsData",
                    "processor": _TTS_PROCESSOR,
                    "value": {"ttfa": 0.4, "ttfb": 0.3, "leading_silence": 0.1},
                },
            ],
        )
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path))

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.skipped is False
        ttfa = result.sub_metrics["ttfa"]
        assert ttfa.details["source"] == "ttfa"
        assert ttfa.details["num_calls"] == 2
        assert ttfa.score == pytest.approx(300.0)  # mean of 200 ms and 400 ms
        assert ttfa.details["p50_ms"] == pytest.approx(200.0)
        assert ttfa.details["p95_ms"] == pytest.approx(400.0)
        assert ttfa.details["leading_silence_mean_ms"] == pytest.approx(75.0)

    @pytest.mark.asyncio
    async def test_ttfa_falls_back_to_ttfb(self, tmp_path):
        """Older streams without TTFAMetricsData fall back to TTFB values."""
        _write_metrics_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": 0.25},
            ],
        )
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path))

        result = await metric.compute(ctx)

        ttfa = result.sub_metrics["ttfa"]
        assert ttfa.details["source"] == "ttfb"
        assert ttfa.score == pytest.approx(250.0)
        assert "leading_silence_mean_ms" not in ttfa.details

    @pytest.mark.asyncio
    async def test_rtf_from_processing_and_audio(self, tmp_path):
        """RTF = total TTS processing time / assistant audio duration."""
        _write_metrics_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": 0.2},
                {"type": "ProcessingMetricsData", "processor": _TTS_PROCESSOR, "value": 2.0},
                {"type": "ProcessingMetricsData", "processor": _TTS_PROCESSOR, "value": 1.0},
            ],
        )
        audio_path = tmp_path / "audio_assistant.wav"
        _write_wav(audio_path, seconds=12.5)
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path), audio_assistant_path=str(audio_path))

        result = await metric.compute(ctx)

        # 3.0 s of synthesis for 12.5 s of audio → 0.24 (paper-style RTF).
        assert result.score == pytest.approx(0.24)
        rtf = result.sub_metrics["rtf"]
        assert rtf.score == pytest.approx(0.24)
        assert rtf.details["num_calls"] == 2
        assert rtf.details["total_processing_seconds"] == pytest.approx(3.0)
        assert rtf.details["audio_duration_seconds"] == pytest.approx(12.5)

    @pytest.mark.asyncio
    async def test_ttfa_only_when_no_audio(self, tmp_path):
        """Without recorded assistant audio, RTF is absent and TTFA still reports."""
        _write_metrics_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": 0.3},
                {"type": "ProcessingMetricsData", "processor": _TTS_PROCESSOR, "value": 1.0},
            ],
        )
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path))

        result = await metric.compute(ctx)

        assert result.score is None
        assert result.skipped is False
        assert "rtf" not in result.sub_metrics
        assert "ttfa" in result.sub_metrics

    @pytest.mark.asyncio
    async def test_ignores_non_tts_and_invalid_entries(self, tmp_path):
        """STT entries and out-of-range/non-numeric values are filtered out."""
        _write_metrics_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                # STT processor: contains "STTService", not "TTSService".
                {"type": "TTFBMetricsData", "processor": "DeepgramSTTService#0", "value": 0.5},
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": -1.0},
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": 42.0},
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": "slow"},
                {"type": "TTFBMetricsData", "processor": _TTS_PROCESSOR, "value": 0.2},
                {"type": "ProcessingMetricsData", "processor": "LLMService#0", "value": 5.0},
            ],
        )
        metric = TTSStreamingSpeedMetric()
        ctx = make_metric_context(output_dir=str(tmp_path))

        result = await metric.compute(ctx)

        ttfa = result.sub_metrics["ttfa"]
        assert ttfa.details["num_calls"] == 1  # only the valid 0.2 s entry
        assert ttfa.score == pytest.approx(200.0)
        assert "rtf" not in result.sub_metrics  # non-TTS processing ignored
