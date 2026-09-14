"""Unit tests for the serving-route capability preflight."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eva.models.config import ModelConfig, RunConfig
from eva.orchestrator.route_preflight import RouteContractError, check_route_contract
from eva.orchestrator.runner import BenchmarkRunner
from tests.unit.conftest import make_evaluation_record

_MODEL_LIST = [{"model_name": "test", "litellm_params": {"model": "test"}}]
_BASE_ENV = {"EVA_MODEL_LIST": json.dumps(_MODEL_LIST)}


def _cascade_config(tmp_path: Path, **updates) -> RunConfig:
    """Cascade route (deepgram STT + cartesia TTS + litellm LLM) on default metrics."""
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        return RunConfig(
            model=ModelConfig(
                llm="test-model",
                stt="deepgram",
                tts="cartesia",
                stt_params={"api_key": "k", "model": "nova-2"},
                tts_params={"api_key": "k", "model": "sonic"},
            ),
            output_dir=tmp_path / "output",
            run_id="preflight-run",
            **updates,
        )


def _s2s_config(tmp_path: Path, **updates) -> RunConfig:
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        return RunConfig(
            model=ModelConfig(s2s="gpt-realtime-mini", s2s_params={"api_key": "k", "model": "rt"}),
            output_dir=tmp_path / "output",
            run_id="preflight-run",
            **updates,
        )


class TestCheckRouteContract:
    def test_cascade_with_default_metrics_binds(self, tmp_path):
        result = check_route_contract(_cascade_config(tmp_path))

        assert result.passed
        assert result.failures == []
        assert "stt_wer" in result.bound_metrics  # cascade-only metric binds on cascade
        assert result.unsupported_metrics == []

    def test_route_with_no_bindable_metrics_fails(self, tmp_path):
        """A route that cannot execute any requested metric fails the gate."""
        config = _s2s_config(tmp_path, metrics=["stt_wer"])

        result = check_route_contract(config)

        assert not result.passed
        assert "stt_wer" in result.unsupported_metrics
        assert any("no requested metric" in f for f in result.failures)

    def test_unsupported_metrics_do_not_fail_when_others_bind(self, tmp_path):
        """Cascade-only metrics on an s2s route are excluded, not failures."""
        result = check_route_contract(_s2s_config(tmp_path))

        assert result.passed
        assert result.unsupported_metrics == [
            "speakability",
            "stt_wer",
            "transcription_accuracy_key_entities",
        ]
        assert result.bound_metrics

    def test_unknown_metric_names_are_reported(self, tmp_path):
        result = check_route_contract(_cascade_config(tmp_path, metrics=["task_completion", "no_such_metric"]))

        assert result.passed
        assert result.unknown_metrics == ["no_such_metric"]
        assert result.bound_metrics == ["task_completion"]

    def test_metric_subset_reports_partial_composites(self, tmp_path):
        result = check_route_contract(_cascade_config(tmp_path, metrics=["response_speed"]))

        assert result.passed
        assert "EVA-A_pass" in result.partial_composites
        assert set(result.partial_composites["EVA-A_pass"]) == {
            "task_completion",
            "faithfulness",
            "agent_speech_fidelity",
        }

    def test_latency_lever_on_non_cascade_is_a_warning(self, tmp_path):
        with patch.dict(os.environ, _BASE_ENV, clear=True):
            config = RunConfig(
                model=ModelConfig(
                    s2s="gpt-realtime-mini",
                    s2s_params={"api_key": "k", "model": "rt"},
                    llm_streaming=True,
                ),
                output_dir=tmp_path / "output",
                run_id="preflight-run",
            )

        result = check_route_contract(config)

        assert result.passed
        assert result.lever_warnings == [
            "latency levers ['llm_streaming'] do not apply to pipeline 's2s' and are ignored"
        ]
        assert result.route["latency_levers"] == ["llm_streaming"]

    def test_no_metrics_requested_still_passes(self, tmp_path):
        result = check_route_contract(_cascade_config(tmp_path, metrics=[]))

        assert result.passed
        assert result.bound_metrics == []


class TestRunnerWiring:
    async def test_run_aborts_before_conversations_when_unbindable(self, tmp_path):
        """BenchmarkRunner.run() raises RouteContractError before any conversation runs."""
        config = _s2s_config(tmp_path, metrics=["stt_wer"])
        with patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()):
            runner = BenchmarkRunner(config)

        with pytest.raises(RouteContractError, match="no requested metric"):
            await runner.run([make_evaluation_record("rec-1")])

        # The preflight verdict is persisted for post-hoc inspection...
        preflight = json.loads((runner.output_dir / "route_preflight.json").read_text())
        assert preflight["passed"] is False
        assert preflight["pipeline_type"] == "s2s"
        # ...and no conversation was burned.
        assert not (runner.output_dir / "records" / "rec-1").exists()

    async def test_run_writes_preflight_on_pass(self, tmp_path):
        """A bindable route records its preflight verdict and continues."""
        config = _cascade_config(tmp_path, debug=True)

        async def _no_conversations(record, output_id):
            return MagicMock(), None

        with (
            patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()),
            patch.object(BenchmarkRunner, "_run_conversation", side_effect=_no_conversations),
        ):
            runner = BenchmarkRunner(config)
            await runner.run([make_evaluation_record("rec-1")])

        preflight = json.loads((runner.output_dir / "route_preflight.json").read_text())
        assert preflight["passed"] is True
        assert "task_completion" in preflight["bound_metrics"]
