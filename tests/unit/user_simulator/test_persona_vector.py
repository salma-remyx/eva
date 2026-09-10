from __future__ import annotations

from pathlib import Path

import pytest

from eva.models.config import OpenAIRealtimeSimulatorConfig, PerturbationConfig
from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator
from eva.user_simulator.persona_vector import (
    NAMED_PROFILES,
    build_persona_vector,
    persona_fragment_from_config,
    render_persona_fragment,
)
from eva.utils.culture import resolve_user_config

_GOAL = {
    "high_level_user_goal": "Rebook my connecting flight after a delay.",
    "decision_tree": {
        "must_have_criteria": ["Rebook onto the next available flight."],
        "escalation_behavior": "Escalate if the agent cannot rebook today.",
        "nice_to_have_criteria": ["A seat with extra legroom."],
        "negotiation_behavior": "Accept a valid rebooking.",
        "resolution_condition": "The new itinerary is confirmed.",
        "failure_condition": "No flight is available today.",
        "edge_cases": [],
    },
    "information_required": ["confirmation code"],
    "starting_utterance": "My connection was missed.",
}


def _simulator(
    tmp_path: Path,
    persona_config: dict,
    *,
    perturbation_config: PerturbationConfig | None = None,
) -> OpenAIRealtimeUserSimulator:
    return OpenAIRealtimeUserSimulator(
        current_date_time="2026-06-05T12:00:00",
        persona_config=persona_config,
        goal=_GOAL,
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_airline",
        perturbation_config=perturbation_config,
        simulator_config=OpenAIRealtimeSimulatorConfig(),
    )


def test_persona_vector_replaces_default_fragment_in_instructions(tmp_path):
    simulator = _simulator(
        tmp_path,
        {
            "user_persona_id": 1,
            "persona_vector": {"profile": "impatient_executive", "seed": 7},
        },
    )

    instructions = simulator._build_session_config()["instructions"]

    assert "Simulated caller profile: impatient_executive" in instructions
    assert "Query phrasing (complexity level: direct)" in instructions
    assert "direct and to the point" not in instructions  # flat default no longer used


def test_without_persona_vector_prompt_is_unchanged(tmp_path):
    simulator = _simulator(tmp_path, {"user_persona_id": 1})

    instructions = simulator._build_session_config()["instructions"]

    assert "direct and to the point" in instructions
    assert "Simulated caller profile" not in instructions


def test_perturbation_behavior_takes_precedence_over_persona_vector(tmp_path):
    simulator = _simulator(
        tmp_path,
        {"user_persona_id": 1, "persona_vector": "impatient_executive"},
        perturbation_config=PerturbationConfig(behavior="aggressive_impatient"),
    )

    prompt = simulator._build_prompt()

    assert "You are impatient and easily frustrated" in prompt
    assert "Simulated caller profile" not in prompt


def test_persona_vector_survives_resolve_user_config():
    user_config = {
        "name": "{first_name}",
        "user_persona_id": 1,
        "persona_vector": {"profile": "elderly_cautious", "seed": 3, "traits": {"patience": 0.9}},
    }

    resolved = resolve_user_config(user_config, {"en": {"first_name": "Ada", "last_name": "Lovelace"}}, "en")
    fragment = persona_fragment_from_config(resolved, _GOAL)

    assert fragment is not None
    assert "Simulated caller profile: elderly_cautious" in fragment
    assert fragment == persona_fragment_from_config(user_config, _GOAL)


def test_seed_reproduces_the_same_caller():
    spec = {"profile": "skeptical_negotiator", "seed": 11}

    first = build_persona_vector(spec, "cancel my booking")
    second = build_persona_vector(spec, "cancel my booking", seed=11)
    other = build_persona_vector({**spec, "seed": 12}, "cancel my booking")

    assert first["traits"] == second["traits"]  # spec-carried and keyword seeds agree
    assert render_persona_fragment(first) == render_persona_fragment(second)
    assert first["traits"] != other["traits"]
    assert render_persona_fragment(first) != render_persona_fragment(other)


def test_extreme_profiles_render_opposite_trait_bands():
    impatient = render_persona_fragment(build_persona_vector({"profile": "impatient_executive", "seed": 0}, ""))
    patient = render_persona_fragment(build_persona_vector({"profile": "elderly_cautious", "seed": 0}, ""))

    assert "little patience" in impatient
    assert "You are patient and calmly work through" in patient
    assert "struggle with apps, codes, and online forms" in patient


def test_scenario_pressure_shifts_emotional_states():
    calm = build_persona_vector({"profile": "distressed_urgent", "seed": 5}, "update my mailing address")
    heated = build_persona_vector(
        {"profile": "distressed_urgent", "seed": 5}, "urgent: refund a wrong charge, error, escalate now"
    )

    assert heated["scenario_pressure"] > calm["scenario_pressure"]
    assert heated["emotional_states"]["frustration"] > calm["emotional_states"]["frustration"]
    assert heated["emotional_states"]["trust"] < calm["emotional_states"]["trust"]


def test_query_complexity_overlay_directs_phrasing():
    spec = {"profile": "balanced_default", "seed": 1, "query_complexity": "deliberately_vague"}
    vector = build_persona_vector(spec, "")

    fragment = render_persona_fragment(vector)

    assert "Deliberately phrase requests vaguely" in fragment
    assert "complexity level: deliberately_vague" in fragment


def test_correlation_rules_are_audited_and_explicit_overrides_win():
    vector = build_persona_vector(
        {"profile": "impatient_executive", "seed": 4, "traits": {"politeness": 0.9}},
        "",
    )

    adjustments = vector["correlation_adjustments"]
    assert adjustments
    assert all({"source", "target", "delta"} <= set(entry) for entry in adjustments)
    assert vector["traits"]["politeness"] == 0.9  # explicit override skips the rules
    assert not any(entry["target"] == "politeness" for entry in adjustments)


@pytest.mark.parametrize("seed", range(25))
def test_sampled_dimensions_stay_in_the_unit_interval(seed):
    vector = build_persona_vector({"profile": "anxious_first_timer"}, "my flight was delayed", seed=seed)

    assert all(0.0 <= value <= 1.0 for value in vector["traits"].values())
    assert all(0.0 <= value <= 1.0 for value in vector["emotional_states"].values())


@pytest.mark.parametrize("profile_name", sorted(NAMED_PROFILES))
def test_every_named_profile_renders(profile_name):
    fragment = persona_fragment_from_config(
        {"name": "Ada Lovelace", "persona_vector": profile_name},
        _GOAL,
    )

    assert f"Simulated caller profile: {profile_name}" in fragment
    assert "Trait vector:" in fragment
    assert "Query phrasing" in fragment


def test_unknown_profile_and_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown persona profile"):
        persona_fragment_from_config({"persona_vector": {"profile": "angry_customer"}}, _GOAL)

    with pytest.raises(ValueError, match="Unknown persona_vector.traits keys"):
        persona_fragment_from_config(
            {"persona_vector": {"profile": "balanced_default", "traits": {"grumpiness": 0.9}}}, _GOAL
        )

    with pytest.raises(ValueError, match="Unknown query complexity"):
        persona_fragment_from_config({"persona_vector": {"query_complexity": "cryptic"}}, _GOAL)
