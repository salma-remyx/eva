"""Tests for IntrinsicUserFeedbackMetric."""

import json
from unittest.mock import AsyncMock

import pytest

# Importing the package module is what wires the metric into the global registry.
from eva.metrics.experience import intrinsic_user_feedback  # noqa: F401
from eva.metrics.experience.intrinsic_user_feedback import IntrinsicUserFeedbackMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need to move my flight to March 25th."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "It's ABC123."},
    {"turn_id": 2, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
    {"turn_id": 2, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
    {"turn_id": 2, "role": "assistant", "content": "Found it. Which date did you want again?"},
    {"turn_id": 3, "role": "user", "content": "March 25th, like I said."},
    {"turn_id": 3, "role": "assistant", "content": "Done — rebooked for March 25th, window seat 12A, $90 total."},
]

STRUCTURED_GOAL = {
    "high_level_user_goal": (
        "Move your AUS to LAX flight to March 25, arriving by 4 PM, under $120, keep a window seat."
    ),
    "decision_tree": {
        "must_have_criteria": [
            "New travel date is 2026-03-25 for AUS to LAX.",
            "Total rebooking cost is $120 or less.",
        ],
        "resolution_condition": "You have a confirmed window seat and the total cost.",
    },
}


def _judge_response():
    """Judge output carrying the three latent states per assistant turn."""
    return json.dumps(
        [
            {
                "turn_id": 1,
                "internal_thought": "Straight to the point, no messing around.",
                "reaction": "reassured — they picked up right away",
                "rating": 5,
            },
            {
                "turn_id": 2,
                "internal_thought": "I already told them the date, why are they asking again?",
                "reaction": "annoyed — repeating myself",
                "rating": 2,
            },
            {
                "turn_id": 3,
                "internal_thought": "Everything I asked for, said once.",
                "reaction": "relieved — got the seat and the price",
                "rating": 5,
            },
        ]
    )


@pytest.fixture
def metric():
    return make_judge_metric(IntrinsicUserFeedbackMetric)


def _context(**overrides):
    defaults = {
        "conversation_trace": SAMPLE_TURNS,
        "user_goal": STRUCTURED_GOAL,
        "user_persona": "Robert White. You're direct and to the point.",
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


def test_registered_but_opt_in():
    """Importing eva.metrics.experience registers the metric; it stays opt-in for default runs."""
    registry = get_global_registry()
    assert registry.get("intrinsic_user_feedback") is IntrinsicUserFeedbackMetric
    # Opt-in: excluded from the default metric list so existing runs gain no extra judge call.
    assert "intrinsic_user_feedback" not in registry.list_metrics()


def test_only_assistant_turns_are_rated():
    """The user reacts to assistant turns, so user-only/tool-only turns are not rated."""
    metric = IntrinsicUserFeedbackMetric()
    assert metric.get_expected_turn_ids(_context()) == [1, 2, 3]
    user_only = [
        {"turn_id": 1, "role": "user", "content": "Hello?"},
        {"turn_id": 1, "type": "tool_call", "tool_name": "x", "parameters": {}},
    ]
    assert metric.get_expected_turn_ids(_context(conversation_trace=user_only)) == []


@pytest.mark.asyncio
async def test_latent_states_collected_per_turn(metric):
    """Internal thoughts and reactions land in details alongside the ratings."""
    metric.llm_client.generate_text = AsyncMock(return_value=(_judge_response(), None))
    result = await metric.compute(_context())

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 5, 2: 2, 3: 5}
    assert result.details["per_turn_internal_thoughts"][2] == "I already told them the date, why are they asking again?"
    assert "annoyed" in result.details["per_turn_reactions"][2]


@pytest.mark.asyncio
async def test_ratings_aggregate_and_normalize(metric):
    """Mean rating and its 1-5 normalization become the metric score."""
    metric.llm_client.generate_text = AsyncMock(return_value=(_judge_response(), None))
    result = await metric.compute(_context())

    assert result.error is None
    # mean of [5, 2, 5]
    assert result.score == pytest.approx(4.0, abs=0.001)
    # normalized: [1.0, 0.25, 1.0] -> mean
    assert result.normalized_score == pytest.approx(0.75, abs=0.001)
    assert result.details["num_turns"] == 3
    assert result.details["num_evaluated"] == 3


@pytest.mark.asyncio
async def test_dissatisfaction_rate_sub_metric(metric):
    """Turns at or below the threshold count as latent dissatisfaction."""
    metric.llm_client.generate_text = AsyncMock(return_value=(_judge_response(), None))
    result = await metric.compute(_context())

    assert result.sub_metrics is not None
    dissatisfaction = result.sub_metrics["dissatisfaction_rate"]
    assert dissatisfaction.name == "intrinsic_user_feedback.dissatisfaction_rate"
    assert dissatisfaction.score == pytest.approx(1 / 3, abs=0.001)
    assert dissatisfaction.details["turn_ids"] == [2]

    # The threshold is configurable, e.g. count merely-tolerable turns too.
    mixed = json.dumps(
        [
            {"turn_id": 1, "internal_thought": "ok", "reaction": "fine", "rating": 5},
            {"turn_id": 2, "internal_thought": "again?", "reaction": "annoyed", "rating": 2},
            {"turn_id": 3, "internal_thought": "worked out", "reaction": "mild", "rating": 3},
        ]
    )
    strict = IntrinsicUserFeedbackMetric(config={"dissatisfaction_threshold": 3})
    strict.llm_client.generate_text = AsyncMock(return_value=(mixed, None))
    strict_result = await strict.compute(_context())
    assert strict_result.sub_metrics is not None
    assert strict_result.sub_metrics["dissatisfaction_rate"].score == pytest.approx(2 / 3, abs=0.001)


def test_judge_prompt_is_grounded_in_the_users_own_goal(metric):
    """The rendered prompt replays from the user's persona and goal, not a generic rubric."""
    context = _context()
    prompt = metric.get_judge_prompt(**metric.get_prompt_variables(context, metric.format_transcript(context)))

    assert "Robert White" in prompt
    assert "Move your AUS to LAX flight" in prompt
    assert "Total rebooking cost is $120 or less." in prompt
    # First-person framing: the judge is the user, not an evaluator scoring a transcript.
    assert "You are the user from this conversation" in prompt


def test_plain_string_goal_still_renders(metric):
    """Records with an unstructured goal fall back to rendering it as-is."""
    context = _context(user_goal="Rebook my flight to March 25th.")
    prompt = metric.get_judge_prompt(**metric.get_prompt_variables(context, metric.format_transcript(context)))

    assert "Rebook my flight to March 25th." in prompt


@pytest.mark.asyncio
async def test_null_rating_excluded_from_aggregation(metric):
    """Turns without assistant content are stored but excluded from the score."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "internal_thought": "ok", "reaction": "fine", "rating": None},
            {"turn_id": 2, "internal_thought": "hmm", "reaction": "mild", "rating": 4},
            {"turn_id": 3, "internal_thought": "good", "reaction": "happy", "rating": 5},
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    result = await metric.compute(_context())

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: None, 2: 4, 3: 5}
    assert result.details["num_evaluated"] == 2
    assert result.details["per_turn_internal_thoughts"][1] == ""
    assert result.sub_metrics is not None
    assert result.sub_metrics["dissatisfaction_rate"].score == 0.0
