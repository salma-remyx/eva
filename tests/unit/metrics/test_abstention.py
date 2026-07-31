"""Tests for AbstentionJudgeMetric.

Covers the integration wiring (the metric is registered via the accuracy package
import — the call-site edit) plus the judge I/O contract: prompt rendering,
rating normalization, and per-dimension sub-metrics.
"""

import json
from unittest.mock import AsyncMock

import pytest

# `get_global_registry` is imported from a NON-NEW module to prove the
# registry-add wiring lands; the metric class itself is the new code.
from eva.metrics.accuracy.abstention import AbstentionJudgeMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Can I get a refund?"},
    {"turn_id": 1, "role": "assistant", "content": "Yes, your refund of $40 is approved and will post in 3 days."},
]


def test_metric_is_registered():
    """Importing eva.metrics.accuracy wires the metric into the global registry."""
    registry = get_global_registry()
    assert registry.get("abstention") is AbstentionJudgeMetric
    # Opt-in: resolvable by name but excluded from the default metric set.
    assert "abstention" not in registry.list_metrics()
    assert registry.create("abstention") is not None


def test_prompt_renders_from_yaml(metric):
    """The judge.abstention.user_prompt template loads and substitutes all variables."""
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    variables = metric.get_prompt_variables(context, transcript_text="dummy transcript")
    # Every variable the template references is provided.
    expected_keys = {
        "agent_instructions",
        "agent_role",
        "available_tools",
        "conversation_trace",
        "current_date_time",
        "user_goal",
        "user_persona",
        "abstention_context",
    }
    assert expected_keys <= set(variables.keys())
    # Renders without a missing-variable / stray-brace error.
    prompt = metric.get_judge_prompt(**variables)
    assert "CALIBRATION" in prompt
    assert "dummy transcript" in prompt


@pytest.mark.asyncio
async def test_overcommitment_response_scores_low(metric):
    """A confident answer with no basis yields rating 1 and flags the right dimensions."""
    mock_response = json.dumps(
        {
            "dimensions": {
                "unrecognized_missing_information": {
                    "evidence": "Did not ask which order or why.",
                    "flagged": True,
                    "rating": 1,
                },
                "overcommitment": {
                    "evidence": "Asserted a specific $40 refund as certain with no basis.",
                    "flagged": True,
                    "rating": 1,
                },
                "inappropriate_abstention": {"evidence": "None", "flagged": False, "rating": 3},
            },
            "rating": 1,
        }
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == 1.0
    assert result.normalized_score == 0.0  # rating 1 on a (1, 3) scale
    # Per-dimension issue-frequency sub-metrics are emitted.
    assert set(result.sub_metrics.keys()) == {
        "unrecognized_missing_information_rate",
        "overcommitment_rate",
        "inappropriate_abstention_rate",
    }
    assert result.sub_metrics["overcommitment_rate"].score == 1.0
    assert result.sub_metrics["unrecognized_missing_information_rate"].score == 1.0
    assert result.sub_metrics["inappropriate_abstention_rate"].score == 0.0


@pytest.mark.asyncio
async def test_well_calibrated_response_scores_high(metric):
    """A response that appropriately qualifies/clarifies scores 3 with no flags."""
    mock_response = json.dumps(
        {
            "dimensions": {
                "unrecognized_missing_information": {"evidence": "None", "flagged": False, "rating": 3},
                "overcommitment": {"evidence": "None", "flagged": False, "rating": 3},
                "inappropriate_abstention": {"evidence": "None", "flagged": False, "rating": 3},
            },
            "rating": 3,
        }
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == 3.0
    assert result.normalized_score == 1.0
    assert all(sm.score == 0.0 for sm in result.sub_metrics.values())


@pytest.mark.asyncio
async def test_unparseable_judge_response_returns_error(metric):
    """A response that cannot be parsed surfaces the standard judge error."""
    metric.llm_client.generate_text = AsyncMock(return_value=("not json at all", None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error == "Failed to parse judge response"
    assert result.score == 0.0


@pytest.fixture
def metric():
    return make_judge_metric(AbstentionJudgeMetric)
