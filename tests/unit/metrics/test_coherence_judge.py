"""Tests for CoherenceJudgeMetric — turn-level dialogue coherence (ECoh-inspired).

The first two tests exercise the integration wiring through existing (non-new)
modules: the metric registry (populated by importing the ``eva.metrics.experience``
package, whose ``__init__`` wires in the coherence module) and the PromptManager
(which loads the coherence judge prompt from ``configs/prompts/judge.yaml``).
The remaining tests drive the metric's ``compute()`` pipeline end-to-end with a
mocked judge LLM.
"""

import json
from unittest.mock import AsyncMock

import pytest

from eva.metrics.experience.coherence import CoherenceJudgeMetric
from eva.metrics.registry import get_global_registry
from eva.utils.prompt_manager import get_prompt_manager

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
    {"turn_id": 2, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
    {"turn_id": 2, "role": "assistant", "content": "Found it. When would you like to fly?"},
    {"turn_id": 3, "role": "user", "content": "March 25th please."},
    # Incoherent assistant turn: ignores the date the user just gave and changes topic.
    {"turn_id": 3, "role": "assistant", "content": "The weather in Paris is lovely this time of year."},
]


def test_coherence_registered_via_experience_package():
    """Importing eva.metrics.experience registers the coherence metric by name."""
    import eva.metrics.experience  # noqa: F401 — triggers @register_metric via the package wiring

    cls = get_global_registry().get("coherence")
    assert cls is CoherenceJudgeMetric
    assert cls.category == "experience"
    assert cls.rating_scale == (1, 3)
    assert cls.version == "v0.1"


def test_coherence_prompt_template_renders():
    """The coherence judge prompt exists in judge.yaml and substitutes its variables.

    ``interruption_tags_reference`` is auto-injected from the ``_shared`` section;
    the conversation transcript and language are passed explicitly.
    """
    rendered = get_prompt_manager().get_prompt(
        "judge.coherence.user_prompt",
        conversation_turns="USER(1): hi\nASSISTANT(1): hello, how can I help?",
        language_display_name="English",
    )
    assert "turn-level coherence" in rendered.lower()
    # the language variable was substituted
    assert "conversation may be in english" in rendered.lower()
    # literal JSON braces in the example survived str.format()
    assert '"turn_id"' in rendered


@pytest.fixture
def metric():
    return make_judge_metric(CoherenceJudgeMetric)


@pytest.mark.asyncio
async def test_compute_rates_turns_end_to_end(metric):
    """compute() produces per-turn coherence ratings and aggregates them."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "Coherent greeting exchange.", "failure_modes": []},
            {"turn_id": 2, "rating": 3, "explanation": "Coherent follow-up question.", "failure_modes": []},
            {
                "turn_id": 3,
                "rating": 1,
                "explanation": "Ignores the date; off-topic tangent.",
                "failure_modes": ["ignores_user_input", "topic_drift"],
            },
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3, 2: 3, 3: 1}
    assert result.details["num_evaluated"] == 3
    # mean of [3, 3, 1]
    assert result.score == pytest.approx(2.333, abs=0.001)
    # normalized: [1.0, 1.0, 0.0] -> mean
    assert result.normalized_score == pytest.approx(0.667, abs=0.001)


@pytest.mark.asyncio
async def test_compute_surfaces_coherence_failure_mode_sub_metrics(metric):
    """Sub-metrics surface per-failure-mode coherence rates across rated turns."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "clean", "failure_modes": []},
            {"turn_id": 2, "rating": 2, "explanation": "mild tangent", "failure_modes": ["topic_drift"]},
            {
                "turn_id": 3,
                "rating": 1,
                "explanation": "ignores the user",
                "failure_modes": ["ignores_user_input", "topic_drift"],
            },
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.sub_metrics is not None
    assert set(result.sub_metrics.keys()) == {
        "non_sequitur_rate",
        "contradicts_context_rate",
        "topic_drift_rate",
        "ignores_user_input_rate",
    }
    # topic_drift flagged on 2 of 3 rated turns
    drift = result.sub_metrics["topic_drift_rate"]
    assert drift.name == "coherence.topic_drift_rate"
    assert drift.score == pytest.approx(2 / 3, abs=0.001)
    assert set(drift.details["turn_ids"]) == {2, 3}
    # a failure mode with zero occurrences is still emitted at rate 0
    assert result.sub_metrics["non_sequitur_rate"].score == 0.0


@pytest.mark.asyncio
async def test_compute_multilingual_language_injected(metric):
    """The judge prompt is built with the conversation's language (multilingual eval)."""
    captured: dict = {}

    async def fake_generate(messages):
        captured["prompt"] = messages[0]["content"]
        return (json.dumps([{"turn_id": 1, "rating": 3, "explanation": "cohérent", "failure_modes": []}]), None)

    metric.llm_client.generate_text = AsyncMock(side_effect=fake_generate)
    context = make_metric_context(conversation_trace=SAMPLE_TURNS, language="fr")
    result = await metric.compute(context)

    assert result.error is None
    prompt = captured["prompt"]
    assert "turn-level coherence" in prompt.lower()
    # the French display name was injected into the prompt
    assert context.language_display_name  # non-empty
    assert context.language_display_name.lower() in prompt.lower()
