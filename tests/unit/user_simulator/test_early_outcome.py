"""Tests for early outcome prediction and its user-simulator wiring."""

import copy
import json
from pathlib import Path

from eva.models.config import EarlyOutcomeConfig, ElevenLabsSimulatorConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.early_outcome import (
    EARLY_HALT_ARTIFACT_FILENAME,
    EarlyOutcomeMonitor,
    is_early_halted,
)
from eva.user_simulator.elevenlabs import ElevenLabsUserSimulator
from eva.user_simulator.factory import create_user_simulator

GENERIC_ERROR = "I'm sorry, I encountered an error processing your request."

_GOAL = {
    "high_level_user_goal": "Move the AUS to LAX flight to March 25 under $120.",
    "decision_tree": {
        "must_have_criteria": [
            "New travel date is 2026-03-25 for AUS to LAX.",
            "Total rebooking cost is $120 or less.",
        ],
        "resolution_condition": "The flight is rebooked and a confirmation number is provided.",
        "failure_condition": "The agent cannot rebook the flight.",
    },
    "starting_utterance": "I need to change my flight.",
}

# A conversation that satisfies every must-have criterion and the resolution
# condition (dates, route, cost, confirmation number all stated).
_SUCCESS_EXCHANGES = [
    ("Thanks for calling, how can I help?", "I need to change my flight."),
    ("Which date would you like?", "2026-03-25, from AUS to LAX."),
    ("The total rebooking cost is $120 or less.", "Great, please book it."),
    ("Your flight is rebooked, confirmation number 42A.", "Perfect, thanks."),
]


class _StubSimulator(AbstractUserSimulator):
    """Minimal concrete simulator used to exercise the base-class wiring."""

    async def run_conversation(self) -> str:
        return "goodbye"


def _make_simulator(
    tmp_path: Path,
    early_outcome_config: EarlyOutcomeConfig | None = EarlyOutcomeConfig(),
) -> _StubSimulator:
    return _StubSimulator(
        current_date_time="2026-09-13T12:00:00",
        persona_config={"user_persona_id": 1},
        goal=copy.deepcopy(_GOAL),
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_airline",
        early_outcome_config=early_outcome_config,
        provider="stub",
    )


def _exchange(simulator: AbstractUserSimulator, assistant_text: str, user_text: str) -> None:
    simulator.event_logger.log_event("assistant_speech", {"text": assistant_text, "source": "assistant"})
    simulator.event_logger.log_event("user_speech", {"text": user_text, "source": "simulated_user"})


def _halt_events(simulator: AbstractUserSimulator) -> list[dict]:
    return [event for event in simulator.event_logger.get_events() if event["type"] == "early_halt"]


class TestEarlyOutcomeHalt:
    def test_halts_on_repeated_errors_and_frustration(self, tmp_path):
        simulator = _make_simulator(tmp_path)
        for _ in range(3):
            _exchange(simulator, GENERIC_ERROR, "This is unacceptable, transfer me to a supervisor.")

        assert simulator._end_reason == "early_halt"
        assert simulator._conversation_done.is_set()

        halt_events = _halt_events(simulator)
        assert len(halt_events) == 1
        assert halt_events[0]["data"]["outcome"] == "failure"
        assert halt_events[0]["data"]["user_turns"] == 3
        assert halt_events[0]["data"]["confidence"] >= 0.8

        artifact = json.loads((tmp_path / EARLY_HALT_ARTIFACT_FILENAME).read_text())
        assert artifact["outcome"] == "failure"
        assert artifact["features"]["generic_error_count"] == 3
        assert is_early_halted(tmp_path)

    def test_no_halt_before_min_user_turns(self, tmp_path):
        simulator = _make_simulator(tmp_path, EarlyOutcomeConfig(min_user_turns=5))
        for _ in range(4):
            _exchange(simulator, GENERIC_ERROR, "This is unacceptable, get me a supervisor.")

        assert not simulator._conversation_done.is_set()
        assert not is_early_halted(tmp_path)

    def test_healthy_conversation_runs_to_completion(self, tmp_path):
        simulator = _make_simulator(tmp_path)
        exchanges = [
            ("Thanks for calling, how can I help?", "I need to change my flight."),
            ("Which date would you like to move to?", "March 25 please, from AUS to LAX."),
            ("Let me look up the options for you.", "Take your time."),
            ("The total cost for the rebooking is $120.", "That works, please book it."),
            ("All set, have a great trip.", "Thank you very much."),
        ]
        for assistant_text, user_text in exchanges:
            _exchange(simulator, assistant_text, user_text)

        assert not simulator._conversation_done.is_set()
        assert not is_early_halted(tmp_path)

    def test_success_halt_requires_opt_in(self, tmp_path):
        default_simulator = _make_simulator(tmp_path)
        for assistant_text, user_text in _SUCCESS_EXCHANGES:
            _exchange(default_simulator, assistant_text, user_text)
        # Default policy halts on predicted failure only: a predicted success
        # is logged nowhere and the conversation keeps running.
        assert not default_simulator._conversation_done.is_set()
        assert _halt_events(default_simulator) == []

        any_simulator = _make_simulator(tmp_path, EarlyOutcomeConfig(halt_on="any", confidence_threshold=0.7))
        for assistant_text, user_text in _SUCCESS_EXCHANGES:
            _exchange(any_simulator, assistant_text, user_text)

        assert any_simulator._end_reason == "early_halt"
        halt_events = _halt_events(any_simulator)
        assert len(halt_events) == 1
        assert halt_events[0]["data"]["outcome"] == "success"

    def test_monitor_absent_without_config(self, tmp_path):
        simulator = _make_simulator(tmp_path, early_outcome_config=None)
        assert simulator._early_outcome_monitor is None
        assert simulator.event_logger.on_event is None

        for _ in range(3):
            _exchange(simulator, GENERIC_ERROR, "This is unacceptable, supervisor please.")

        assert not simulator._conversation_done.is_set()
        assert not is_early_halted(tmp_path)


class TestEarlyOutcomeMonitor:
    def test_features_track_behavioral_signals(self, tmp_path):
        monitor = EarlyOutcomeMonitor(EarlyOutcomeConfig(), goal=copy.deepcopy(_GOAL), output_dir=tmp_path)
        monitor.on_event({"type": "assistant_speech", "data": {"text": GENERIC_ERROR}})
        monitor.on_event({"type": "assistant_speech", "data": {"text": GENERIC_ERROR}})
        monitor.on_event({"type": "error", "data": {}})
        monitor.on_event({"type": "user_speech", "data": {"text": "This is unacceptable."}})

        features = monitor.features()
        assert features["user_turns"] == 1
        assert features["assistant_turns"] == 2
        assert features["generic_error_count"] == 2
        assert features["error_event_count"] == 1
        assert features["max_repeated_assistant_turn"] == 2
        assert features["frustration_kinds"] == ["dissatisfaction"]

        success_confidence, failure_confidence = monitor.predict()
        assert 0.0 < failure_confidence < 0.8
        assert success_confidence == 0.0

    def test_ignores_events_after_halt(self, tmp_path):
        monitor = EarlyOutcomeMonitor(
            EarlyOutcomeConfig(confidence_threshold=0.5), goal=copy.deepcopy(_GOAL), output_dir=tmp_path
        )
        for _ in range(2):
            monitor.on_event({"type": "assistant_speech", "data": {"text": GENERIC_ERROR}})
            monitor.on_event({"type": "user_speech", "data": {"text": "This is unacceptable."}})
        assert monitor.halted

        assert monitor.on_event({"type": "user_speech", "data": {"text": "more"}}) is None
        assert monitor.on_event({"type": "assistant_speech", "data": {"text": "more"}}) is None


class TestFactoryWiring:
    def test_factory_passes_early_outcome_config(self, tmp_path):
        simulator = create_user_simulator(
            ElevenLabsSimulatorConfig(),
            current_date_time="2026-09-13T12:00:00",
            persona_config={"user_persona_id": 1},
            goal=copy.deepcopy(_GOAL),
            server_url="ws://localhost:9999/ws",
            output_dir=tmp_path,
            agent_id="agent_airline",
            early_outcome_config=EarlyOutcomeConfig(),
        )

        assert isinstance(simulator, ElevenLabsUserSimulator)
        assert simulator._early_outcome_monitor is not None
        assert simulator.event_logger.on_event is not None
