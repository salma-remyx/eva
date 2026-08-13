"""Tests for SpokenFunctionCallingMetric.

The first test proves the registry wiring (the call-site edit in
``eva.metrics.accuracy.__init__``): importing ``eva.metrics.accuracy`` runs the
package ``__init__`` which registers the metric. The rest exercise the metric's
behavior following the ``test_faithfulness`` pattern.
"""

import json

import pytest

from eva.metrics.accuracy.spoken_function_calling import SpokenFunctionCallingMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestSpokenFunctionCalling:
    def setup_method(self):
        self.metric = make_judge_metric(SpokenFunctionCallingMetric, mock_llm=True)

    def test_registered_via_accuracy_package(self):
        """Importing eva.metrics.accuracy auto-registers the metric (call-site wiring)."""
        # The module-level import above triggers eva.metrics.accuracy.__init__, which
        # imports spoken_function_calling and registers it via @register_metric.
        registry = get_global_registry()
        assert registry.get("spoken_function_calling") is SpokenFunctionCallingMetric
        assert "spoken_function_calling" in registry.list_metrics()

    def test_metric_attributes(self):
        assert self.metric.name == "spoken_function_calling"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)

    def test_get_prompt_variables_cascade(self):
        ctx = make_metric_context(
            agent_instructions="Be helpful",
            agent_role="Airline assistant",
            agent_tools=[{"name": "search_flights"}],
            current_date_time="2026-01-01",
            pipeline_type="cascade",
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["agent_role"] == "Airline assistant"
        assert "search_flights" in variables["available_tools"]
        assert "conversation_trace" in variables
        assert "transcript" in variables["perception_note"]  # cascade mode
        assert "STT" in variables["user_turns_disclaimer"]

    def test_get_prompt_variables_audio_native(self):
        ctx = make_metric_context(pipeline_type="s2s")
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert "audio-native" in variables["perception_note"]
        assert "raw audio" in variables["perception_note"]

    def test_build_metric_score_surfaces_accuracy_sub_metrics(self):
        ctx = make_metric_context(
            conversation_trace=[{"role": "user"}, {"role": "assistant"}],
            tool_responses=[{"tool_name": "get_reservation"}],
        )
        response = {
            "rating": 2,
            "explanation": "right functions, one wrong value",
            "calls": [
                {
                    "turn_id": 3,
                    "tool_name": "get_reservation",
                    "correct_function": True,
                    "correct_arguments": True,
                },
                {
                    "turn_id": 5,
                    "tool_name": "cancel_reservation",
                    "correct_function": True,
                    "correct_arguments": False,
                },
            ],
        }

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,
            response=response,
            prompt="p",
            context=ctx,
            raw_response="{...}",
        )

        assert score.name == "spoken_function_calling"
        assert score.score == 2.0
        assert score.normalized_score == 0.5
        assert score.details["rating"] == 2
        assert score.details["num_tool_calls"] == 1
        assert score.sub_metrics is not None
        # 2 calls, 2 correct functions, 1 correct argument value.
        fn_acc = score.sub_metrics["function_name_accuracy"]
        assert fn_acc.name == "spoken_function_calling.function_name_accuracy"
        assert fn_acc.score == 1.0  # 2/2
        assert fn_acc.details == {"correct": 2, "total": 2}
        arg_acc = score.sub_metrics["argument_value_accuracy"]
        assert arg_acc.score == 0.5  # 1/2

    def test_build_metric_score_no_calls(self):
        ctx = make_metric_context()
        score = self.metric.build_metric_score(
            rating=3,
            normalized=1.0,
            response={"rating": 3, "calls": []},
            prompt="p",
            context=ctx,
        )
        assert score.sub_metrics is None

    @pytest.mark.asyncio
    async def test_compute_success(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 3,
                    "explanation": "all correct",
                    "calls": [
                        {
                            "turn_id": 3,
                            "tool_name": "get_reservation",
                            "correct_function": True,
                            "correct_arguments": True,
                        },
                    ],
                }
            ),
            None,
        )
        ctx = make_metric_context(
            tool_responses=[{"tool_name": "get_reservation", "tool_response": {}}],
            conversation_trace=[
                {"role": "user", "content": "find my reservation"},
                {"role": "assistant", "content": "let me look that up"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.sub_metrics is not None
        assert score.sub_metrics["function_name_accuracy"].score == 1.0

    @pytest.mark.asyncio
    async def test_compute_no_tool_calls_is_skipped(self):
        ctx = make_metric_context(tool_responses=[])
        score = await self.metric.compute(ctx)
        assert score.skipped is True
        assert score.score is None
        assert score.normalized_score is None

    @pytest.mark.asyncio
    async def test_compute_unparseable_response(self):
        self.metric.llm_client.generate_text.return_value = ("not json ~~~", None)
        ctx = make_metric_context(
            tool_responses=[{"tool_name": "get_reservation", "tool_response": {}}],
            conversation_trace=[{"role": "user", "content": "hi"}],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None
