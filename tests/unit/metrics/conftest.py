"""Shared fixtures for metric tests."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eva.metrics.base import MetricContext
from eva.metrics.processor import MetricsContextProcessor
from eva.models.results import MetricScore, RecordMetrics


def make_metric_context(**overrides) -> MetricContext:
    """Create a MetricContext with sensible defaults, overridable via kwargs.

    If audio_timestamps_*_turns are provided but latency_assistant_turns is not,
    it is automatically derived from the timestamps.

    Usage::

        ctx = make_metric_context(conversation_trace=[...])
        ctx = make_metric_context(tool_responses=[...], tool_params=[...])
    """
    defaults = {
        "record_id": "test_record",
        "user_goal": "Test goal",
        "user_persona": "Test persona",
        "expected_scenario_db": {},
        "initial_scenario_db": {},
        "final_scenario_db": {},
        "initial_scenario_db_hash": "",
        "final_scenario_db_hash": "",
        "agent_role": "Test role",
        "agent_instructions": "Test instructions",
        "agent_tools": [],
        "agent_id": "agent_test",
        "current_date_time": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)

    # Auto-derive latency_assistant_turns from audio timestamps if not provided
    if "latency_assistant_turns" not in overrides:
        user_ts = defaults.get("audio_timestamps_user_turns")
        asst_ts = defaults.get("audio_timestamps_assistant_turns")
        if user_ts and asst_ts:
            tmp = SimpleNamespace(
                audio_timestamps_user_turns=user_ts,
                audio_timestamps_assistant_turns=asst_ts,
                latency_assistant_turns={},
            )
            MetricsContextProcessor._compute_per_turn_latency(tmp)
            defaults["latency_assistant_turns"] = tmp.latency_assistant_turns

    return MetricContext(**defaults)


# Real-shaped caller profiles for the three benchmark domains, for tests that must hold
# across airline/itsm/medical_hr. Goals use the dataset shape (high_level_user_goal plus a
# decision_tree of must-have criteria) so metrics that read context.user_goal see what a real
# record carries.
DOMAIN_USER_SAMPLES = {
    "airline": {
        "user_persona": "Robert White. You're direct and to the point.",
        "user_goal": {
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
        },
        "conversation_trace": [
            {"turn_id": 1, "role": "user", "content": "Hi, I need to move my flight to March 25th."},
            {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
            {"turn_id": 2, "role": "user", "content": "It's ABC123."},
            {"turn_id": 2, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
            {"turn_id": 2, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
            {"turn_id": 2, "role": "assistant", "content": "Found it. Which date did you want again?"},
            {"turn_id": 3, "role": "user", "content": "March 25th, like I said."},
            {"turn_id": 3, "role": "assistant", "content": "Done — rebooked for March 25th, window seat 12A, $90."},
        ],
        "assistant_turn_ids": [1, 2, 3],
    },
    "itsm": {
        "user_persona": "Dana Kowalski. You're polite, but you want this fixed before you hang up.",
        "user_goal": {
            "high_level_user_goal": (
                "You were locked out of Active Directory this morning after too many failed login attempts, "
                "and you need the service desk to unlock your account so you can log in again."
            ),
            "decision_tree": {
                "must_have_criteria": ["Your Active Directory account is unlocked and you can log in again."],
                "resolution_condition": "The agent has confirmed the account is unlocked. End the call.",
            },
        },
        "conversation_trace": [
            {"turn_id": 1, "role": "user", "content": "I got locked out of my account this morning."},
            {"turn_id": 1, "role": "assistant", "content": "Sorry about that. Can I have your employee ID?"},
            {"turn_id": 2, "role": "user", "content": "It's E-4471."},
            {"turn_id": 2, "type": "tool_call", "tool_name": "unlock_account", "parameters": {"employee_id": "E-4471"}},
            {"turn_id": 2, "type": "tool_response", "tool_name": "unlock_account", "tool_response": {"status": "ok"}},
            {"turn_id": 2, "role": "assistant", "content": "Unlocked — try logging in now and it should work."},
        ],
        "assistant_turn_ids": [1, 2],
    },
    "medical_hr": {
        "user_persona": "Priya Nair. You're organized and have your NPI written down.",
        "user_goal": {
            "high_level_user_goal": (
                "You are a provider whose Florida medical license is close to expiring while your renewal is "
                "still pending, and you need to request a 60-day supervised extension so you can keep seeing patients."
            ),
            "decision_tree": {
                "must_have_criteria": [
                    "Your 60-day supervised license extension request is submitted and you receive a case ID."
                ],
                "resolution_condition": "You have a case ID for the extension request. End the call.",
            },
        },
        "conversation_trace": [
            {"turn_id": 1, "role": "user", "content": "I need a temporary extension on my medical license."},
            {"turn_id": 1, "role": "assistant", "content": "I can help with that. What is your NPI?"},
            {"turn_id": 2, "role": "user", "content": "1487654320, and my facility code is FL-229."},
            {"turn_id": 2, "type": "tool_call", "tool_name": "submit_extension", "parameters": {"npi": "1487654320"}},
            {"turn_id": 2, "type": "tool_response", "tool_name": "submit_extension", "tool_response": {"id": "EXT-88"}},
            {"turn_id": 2, "role": "assistant", "content": "Filed — your case ID is EXT-88 and it's good for 60 days."},
        ],
        "assistant_turn_ids": [1, 2],
    },
}


def make_domain_metric_context(domain: str, **overrides) -> MetricContext:
    """Create a MetricContext carrying one domain's caller profile (airline/itsm/medical_hr).

    Usage::

        ctx = make_domain_metric_context("itsm")
        ctx = make_domain_metric_context("medical_hr", conversation_trace=[...])
    """
    try:
        sample = DOMAIN_USER_SAMPLES[domain]
    except KeyError:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(DOMAIN_USER_SAMPLES)}") from None
    defaults = {
        "user_persona": sample["user_persona"],
        "user_goal": sample["user_goal"],
        "conversation_trace": sample["conversation_trace"],
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


def make_judge_metric(metric_cls, *, mock_llm: bool = False, logger_name: str | None = None):
    """Instantiate a judge metric, loading real prompts from configs/prompts/.

    Args:
        metric_cls: The metric class to instantiate.
        mock_llm: If True, also replace ``llm_client`` with a MagicMock.
        logger_name: If provided, replace the metric's logger.

    Returns:
        An instance of *metric_cls* ready for testing.
    """
    m = metric_cls()
    if logger_name:
        m.logger = logging.getLogger(logger_name)
    if mock_llm:
        m.llm_client = MagicMock()
        m.llm_client.generate_text = AsyncMock()
        m.llm_client.params = {}
    if hasattr(m, "_trim_silence"):
        m._trim_silence = lambda audio, _ctx: audio
    return m


def make_metric_score(
    name: str,
    score: float = 1.0,
    normalized_score: float | None = None,
    error: str | None = None,
    details: dict | None = None,
) -> MetricScore:
    """Shorthand to build a MetricScore."""
    return MetricScore(
        name=name,
        score=score,
        normalized_score=normalized_score if normalized_score is not None else score,
        error=error,
        details=details or {},
    )


def make_record_metrics(scores: dict[str, float], record_id: str = "1.1.1") -> RecordMetrics:
    """Create RecordMetrics from a {name: normalized_score} dict."""
    metrics = {}
    for name, value in scores.items():
        metrics[name] = MetricScore(name=name, score=value, normalized_score=value)
    return RecordMetrics(record_id=record_id, metrics=metrics)
