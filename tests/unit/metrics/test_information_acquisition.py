"""Tests for InformationAcquisitionMetric."""

import json

import pytest

from eva.metrics.accuracy.information_acquisition import (
    InformationAcquisitionMetric,
    coverage_ratio,
)
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestInformationAcquisition:
    def setup_method(self):
        self.metric = make_judge_metric(InformationAcquisitionMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "information_acquisition"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)

    def test_registered_in_global_registry(self):
        # Importing the eva.metrics.accuracy package fires the @register_metric
        # hook via the package __init__ (the call-site wiring). The metric must
        # resolve by name so the runner can select it via --metrics.
        import eva.metrics.accuracy  # noqa: F401

        assert get_global_registry().get("information_acquisition") is InformationAcquisitionMetric

    def test_coverage_ratio_deterministic_one_to_one(self):
        # MedDDC-Eval's deterministic assignment: at most one credit per required item.
        items = [
            {"item": "name", "covered": True},
            {"item": "date", "covered": False},
            {"item": "code", "covered": True},
        ]
        assert coverage_ratio(items) == (2 / 3, 2, 3)
        assert coverage_ratio([]) == (0.0, 0, 0)
        assert coverage_ratio(None) == (0.0, 0, 0)
        # Non-dict entries are ignored, not counted.
        assert coverage_ratio([{"item": "x", "covered": True}, "junk"]) == (1.0, 1, 1)

    def test_get_prompt_variables(self):
        ctx = make_metric_context(
            user_goal="Book a flight",
            user_persona="A traveler",
            agent_instructions="Collect trip details",
            agent_role="Travel agent",
            agent_tools=[{"name": "search_flights"}],
            current_date_time="2026-01-01",
            pipeline_type="cascade",
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")

        assert variables["user_goal"] == "Book a flight"
        assert variables["user_persona"] == "A traveler"
        assert "search_flights" in variables["available_tools"]
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        assert "STT" in variables["user_turns_disclaimer"]  # cascade mode

    def test_build_metric_score_coverage_ratio(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 2,
            "explanation": "Got name and code, missed the date.",
            "required_information": [
                {"item": "name", "covered": True, "evidence": "Turn 1", "elicited_in_turn": 1},
                {"item": "date", "covered": False, "evidence": "never asked", "elicited_in_turn": None},
                {"item": "code", "covered": True, "evidence": "Turn 3", "elicited_in_turn": 3},
            ],
        }

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,  # ignored — the deterministic coverage ratio is the headline
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response='{"rating": 2}',
        )

        assert score.name == "information_acquisition"
        assert score.score == 2.0  # raw judge rating preserved
        assert score.normalized_score == 2 / 3  # deterministic coverage ratio is the headline
        assert score.details["coverage_ratio"] == 2 / 3
        assert score.details["items_covered"] == 2
        assert score.details["items_required"] == 3
        assert score.details["rating"] == 2
        assert score.details["num_turns"] == 2
        assert len(score.details["required_information"]) == 3

    def test_build_metric_score_no_required_items(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}])
        response = {"rating": 3, "explanation": "Nothing to collect.", "required_information": []}

        score = self.metric.build_metric_score(
            rating=3, normalized=1.0, response=response, prompt="p", context=ctx, raw_response="{}"
        )

        assert score.normalized_score == 0.0
        assert score.details["items_required"] == 0

    @pytest.mark.asyncio
    async def test_compute_success(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 3,
                    "explanation": "All collected.",
                    "required_information": [
                        {"item": "name", "covered": True, "evidence": "Turn 1", "elicited_in_turn": 1},
                        {"item": "destination", "covered": True, "evidence": "Turn 2", "elicited_in_turn": 2},
                    ],
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "I'm Jane"},
                {"role": "assistant", "content": "Where to?"},
                {"role": "user", "content": "Paris"},
            ],
        )

        score = await self.metric.compute(ctx)

        assert score.normalized_score == 1.0
        assert score.error is None

    @pytest.mark.asyncio
    async def test_compute_empty_transcript(self):
        ctx = make_metric_context(conversation_trace=[])
        score = await self.metric.compute(ctx)

        assert score.score == 0.0
        assert "No transcript" in score.error
