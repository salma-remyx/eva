"""Tests for ResponseGroundingMetric (response_grounding)."""

import pytest

import eva.metrics  # noqa: F401  (package import is what registers the metric modules)
from eva.metrics.base import BaseMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


@pytest.fixture
def metric() -> BaseMetric:
    """Resolve the metric through the global registry, not a direct module import."""
    instance = get_global_registry().create("response_grounding")
    assert instance is not None, "response_grounding must be registered via the eva.metrics package"
    return instance


def test_registered_via_metrics_package() -> None:
    """The metric is discoverable through the package import chain and runs by default."""
    registry = get_global_registry()
    assert registry.get("response_grounding") is not None
    assert "response_grounding" in registry.list_metrics()


@pytest.mark.asyncio
async def test_grounded_conversation_scores_perfect(metric: BaseMetric) -> None:
    """Turns and tool arguments that trace to user speech all pass."""
    context = make_metric_context(
        transcribed_user_turns={
            1: "I'd like to rebook my flight AA123 to Boston on Friday.",
            2: "Yes, my confirmation code is ABC123. The name is White.",
        },
        transcribed_assistant_turns={
            0: "Hello, this is the airline. How can I help you?",
            1: "Your flight AA123 has been rebooked to Boston.",
            2: "Is there anything else I can help with?",
        },
        conversation_trace=[
            {
                "type": "tool_call",
                "tool_name": "get_reservation",
                "turn_id": 2,
                "parameters": {"confirmation_number": "ABC123", "last_name": "White"},
            },
        ],
    )
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.error is None
    # Turn 2 is a question turn -> not applicable; turn 0 is the greeting and excluded.
    assert result.details["num_not_applicable"] == 1
    assert result.details["num_evaluated"] == 1
    assert result.details["grounded_tool_arguments"] == 2
    assert result.sub_metrics is not None
    assert result.sub_metrics["turn_grounding_accuracy"].score == 1.0
    assert result.sub_metrics["tool_argument_accuracy"].score == 1.0
    assert result.sub_metrics["num_unaddressed_turns"].score is None


@pytest.mark.asyncio
async def test_unaddressed_turn_flagged(metric: BaseMetric) -> None:
    """Assistant content overlapping nothing the user said is flagged (restraint failure)."""
    context = make_metric_context(
        transcribed_user_turns={1: "I need help with my internet connection."},
        transcribed_assistant_turns={
            0: "Hello, how can I help you?",
            1: "Your hotel reservation in Paris has been confirmed for Thursday.",
        },
    )
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.details["per_turn_grounding"][1]["passed"] is False
    assert result.sub_metrics is not None
    assert result.sub_metrics["unaddressed_turn_rate"].score == 1.0
    assert result.sub_metrics["turn_grounding_accuracy"].score == 0.0
    assert result.sub_metrics["num_unaddressed_turns"].score == 1.0


@pytest.mark.asyncio
async def test_fabricated_tool_argument_flagged(metric: BaseMetric) -> None:
    """Tool arguments traceable to user speech or schema enums pass; others fail."""
    context = make_metric_context(
        transcribed_user_turns={1: "My name is Robert Black, confirmation ABC123, for two passengers."},
        transcribed_assistant_turns={
            0: "Hello, how can I help you?",
            1: "Looking up reservation ABC123 for Robert.",
        },
        conversation_trace=[
            {
                "type": "tool_call",
                "tool_name": "update_fare_class",
                "turn_id": 1,
                "parameters": {
                    "confirmation_number": "XYZ789",  # never uttered anywhere
                    "last_name": "Black",  # user speech
                    "fare_class": "premium_economy",  # declared enum
                    "passenger_count": 2,  # "two" spoken by the user
                },
            },
        ],
        agent_tools=[
            {
                "id": "update_fare_class",
                "required_parameters": [
                    {"name": "fare_class", "type": "string", "enum": ["basic_economy", "premium_economy", "business"]}
                ],
            }
        ],
    )
    result = await metric.compute(context)

    per_arg = result.details["per_tool_argument"]
    assert per_arg["turn1:update_fare_class.confirmation_number"]["passed"] is False
    assert per_arg["turn1:update_fare_class.last_name"]["passed"] is True
    assert per_arg["turn1:update_fare_class.fare_class"]["passed"] is True
    assert per_arg["turn1:update_fare_class.passenger_count"]["passed"] is True
    assert result.sub_metrics is not None
    assert result.sub_metrics["tool_argument_accuracy"].score == 0.75
    assert result.sub_metrics["fabricated_tool_argument_rate"].score == 0.25
    # 1 grounded turn + 3 of 4 grounded arguments = 4 of 5 rubrics.
    assert result.score == 0.8


@pytest.mark.asyncio
async def test_interactional_turns_not_applicable(metric: BaseMetric) -> None:
    """Question turns and interjections are excluded rather than scored."""
    context = make_metric_context(
        transcribed_user_turns={1: "Thanks."},
        transcribed_assistant_turns={
            0: "Hello, how can I help you?",
            1: "Could you share your confirmation code please?",
            2: "Okay, you're all set.",
        },
    )
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.details["num_not_applicable"] == 2
    assert result.details["total_rubrics"] == 0
    assert "note" in result.details


@pytest.mark.asyncio
async def test_empty_transcripts_no_error(metric: BaseMetric) -> None:
    """Missing transcripts produce the vacuous perfect score, not an error."""
    context = make_metric_context()
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == 1.0
