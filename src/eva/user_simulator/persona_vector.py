"""Three-tier persona vectors for the simulated caller.

Flat role descriptions ("you are an angry customer") produce near-identical
conversations regardless of the scenario. This module replaces them with a
controllable persona vector, adapted from "A Three-Tier Persona Vector for
Controllable User Simulation in Agentic Evaluation" (arXiv:2609.08592):

* Tier 1 — six categorical demographics (jurisdiction, age, channel, device,
  language proficiency, time availability).
* Tier 2 — twelve continuous behavioral traits sampled with Gaussian noise
  around curated named-profile base vectors, then adjusted by seven
  rule-described trait correlations so co-occurrence patterns stay auditable
  without a learned covariance matrix.
* Tier 3 — five continuous emotional states that shift in response to
  scenario context, approximated here by a keyword-pressure proxy derived
  from the record's user goal.
* Orthogonally, a four-level query-complexity overlay controls utterance
  phrasing from direct to deliberately vague.

A record opts in by carrying a ``persona_vector`` entry in its ``user_config``.
The rendered fragment flows through the existing persona contract
(``resolve_user_config`` -> ``persona_config`` -> ``AbstractUserSimulator._build_prompt``)
and drives the caller's instructions for both simulator providers.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

DEMOGRAPHIC_KEYS: tuple[str, ...] = (
    "jurisdiction",
    "age",
    "channel",
    "device",
    "language_proficiency",
    "time_availability",
)
TRAIT_KEYS: tuple[str, ...] = (
    "patience",
    "assertiveness",
    "digital_literacy",
    "verbosity",
    "cooperativeness",
    "attention_to_detail",
    "decisiveness",
    "skepticism",
    "politeness",
    "formality",
    "humor",
    "resilience",
)
EMOTION_KEYS: tuple[str, ...] = ("frustration", "anxiety", "trust", "confidence", "stress")
QUERY_COMPLEXITY_LEVELS: tuple[str, ...] = ("direct", "contextual", "indirect", "deliberately_vague")

_TRAIT_SIGMA = 0.08
_EMOTION_SIGMA = 0.06
_LOW_BAND = 0.34
_HIGH_BAND = 0.67

# Curated tier-2 base vectors for the eight named profiles. Traits and
# emotional states are 0-1 floats; missing demographics simply render as
# unspecified.
NAMED_PROFILES: dict[str, dict[str, Any]] = {
    "balanced_default": {
        "demographics": {"jurisdiction": "US", "age": "35-44", "channel": "phone", "device": "smartphone"},
        "traits": {key: 0.5 for key in TRAIT_KEYS},
        "emotional_states": {"frustration": 0.2, "anxiety": 0.3, "trust": 0.6, "confidence": 0.6, "stress": 0.2},
    },
    "impatient_executive": {
        "demographics": {"age": "45-54", "channel": "phone", "device": "smartphone", "time_availability": "rushed"},
        "traits": {
            "patience": 0.12,
            "assertiveness": 0.9,
            "digital_literacy": 0.8,
            "verbosity": 0.35,
            "cooperativeness": 0.3,
            "attention_to_detail": 0.6,
            "decisiveness": 0.9,
            "skepticism": 0.5,
            "politeness": 0.3,
            "formality": 0.7,
            "humor": 0.2,
            "resilience": 0.3,
        },
        "emotional_states": {
            "frustration": 0.75,
            "anxiety": 0.3,
            "trust": 0.35,
            "confidence": 0.85,
            "stress": 0.6,
        },
    },
    "elderly_cautious": {
        "demographics": {"age": "65+", "channel": "phone", "device": "landline", "time_availability": "ample"},
        "traits": {
            "patience": 0.85,
            "assertiveness": 0.25,
            "digital_literacy": 0.1,
            "verbosity": 0.7,
            "cooperativeness": 0.8,
            "attention_to_detail": 0.7,
            "decisiveness": 0.2,
            "skepticism": 0.4,
            "politeness": 0.9,
            "formality": 0.8,
            "humor": 0.4,
            "resilience": 0.4,
        },
        "emotional_states": {"frustration": 0.25, "anxiety": 0.6, "trust": 0.65, "confidence": 0.3, "stress": 0.35},
    },
    "anxious_first_timer": {
        "demographics": {
            "age": "18-24",
            "channel": "mobile_app",
            "device": "smartphone",
            "language_proficiency": "fluent",
        },
        "traits": {
            "patience": 0.6,
            "assertiveness": 0.2,
            "digital_literacy": 0.4,
            "verbosity": 0.5,
            "cooperativeness": 0.9,
            "attention_to_detail": 0.5,
            "decisiveness": 0.25,
            "skepticism": 0.3,
            "politeness": 0.8,
            "formality": 0.5,
            "humor": 0.2,
            "resilience": 0.25,
        },
        "emotional_states": {"frustration": 0.35, "anxiety": 0.85, "trust": 0.55, "confidence": 0.25, "stress": 0.6},
    },
    "friendly_chatterbox": {
        "demographics": {"age": "25-34", "channel": "web_chat", "device": "laptop", "time_availability": "ample"},
        "traits": {
            "patience": 0.7,
            "assertiveness": 0.45,
            "digital_literacy": 0.6,
            "verbosity": 0.95,
            "cooperativeness": 0.9,
            "attention_to_detail": 0.4,
            "decisiveness": 0.5,
            "skepticism": 0.2,
            "politeness": 0.95,
            "formality": 0.35,
            "humor": 0.85,
            "resilience": 0.7,
        },
        "emotional_states": {"frustration": 0.15, "anxiety": 0.2, "trust": 0.8, "confidence": 0.6, "stress": 0.15},
    },
    "skeptical_negotiator": {
        "demographics": {"jurisdiction": "EU", "age": "35-44", "channel": "web_chat", "device": "laptop"},
        "traits": {
            "patience": 0.45,
            "assertiveness": 0.8,
            "digital_literacy": 0.7,
            "verbosity": 0.55,
            "cooperativeness": 0.25,
            "attention_to_detail": 0.8,
            "decisiveness": 0.7,
            "skepticism": 0.95,
            "politeness": 0.4,
            "formality": 0.6,
            "humor": 0.3,
            "resilience": 0.6,
        },
        "emotional_states": {"frustration": 0.45, "anxiety": 0.25, "trust": 0.1, "confidence": 0.75, "stress": 0.3},
    },
    "detail_auditor": {
        "demographics": {"age": "45-54", "channel": "web_chat", "device": "laptop", "language_proficiency": "native"},
        "traits": {
            "patience": 0.75,
            "assertiveness": 0.5,
            "digital_literacy": 0.85,
            "verbosity": 0.6,
            "cooperativeness": 0.6,
            "attention_to_detail": 0.95,
            "decisiveness": 0.6,
            "skepticism": 0.6,
            "politeness": 0.6,
            "formality": 0.75,
            "humor": 0.15,
            "resilience": 0.65,
        },
        "emotional_states": {"frustration": 0.3, "anxiety": 0.35, "trust": 0.4, "confidence": 0.7, "stress": 0.3},
    },
    "distressed_urgent": {
        "demographics": {"channel": "phone", "device": "smartphone", "time_availability": "rushed"},
        "traits": {
            "patience": 0.15,
            "assertiveness": 0.6,
            "digital_literacy": 0.45,
            "verbosity": 0.6,
            "cooperativeness": 0.4,
            "attention_to_detail": 0.3,
            "decisiveness": 0.4,
            "skepticism": 0.5,
            "politeness": 0.4,
            "formality": 0.4,
            "humor": 0.05,
            "resilience": 0.1,
        },
        "emotional_states": {"frustration": 0.85, "anxiety": 0.7, "trust": 0.3, "confidence": 0.4, "stress": 0.9},
    },
}

# Seven rule-described trait correlations, applied after sampling as
# deterministic nudges (source trait, direction, target trait, strength):
# the target moves by direction * strength * (source - 0.5) * 2, so the
# co-occurrence pattern is auditable without a learned covariance matrix.
_CORRELATION_RULES: tuple[tuple[str, float, str, float], ...] = (
    ("resilience", 1.0, "patience", 0.12),
    ("decisiveness", 1.0, "assertiveness", 0.1),
    ("assertiveness", -1.0, "politeness", 0.1),
    ("cooperativeness", 1.0, "politeness", 0.08),
    ("digital_literacy", 1.0, "attention_to_detail", 0.08),
    ("skepticism", -1.0, "cooperativeness", 0.1),
    ("patience", -1.0, "verbosity", 0.08),
)

# Tier-3 scenario-pressure proxy: urgency and adversity keywords found in the
# record's goal raise the emotional pressure of the call.
_URGENCY_TOKENS = ("urgent", "immediately", "asap", "right now", "deadline", "last minute", "within the hour", "today")
_ADVERSITY_TOKENS = (
    "error",
    "failed",
    "failure",
    "cancel",
    "refund",
    "complaint",
    "broken",
    "lost",
    "charged twice",
    "escalate",
    "wrong",
    "delay",
    "denied",
    "overbooked",
    "outage",
)

_TRAIT_DIRECTIVES: dict[str, dict[str, str]] = {
    "patience": {
        "low": "You have little patience: push the agent to hurry and show irritation at delays or repeated steps.",
        "high": "You are patient and calmly work through slow, step-by-step processes.",
    },
    "assertiveness": {
        "low": "You accept the agent's proposals and rarely push back.",
        "high": "You firmly state what you want, push back, and ask for escalation when needed.",
    },
    "digital_literacy": {
        "low": "You struggle with apps, codes, and online forms — ask the agent to walk you through anything digital.",
        "high": "You are comfortable with technology and handle confirmation codes and portals without help.",
    },
    "verbosity": {
        "low": "You keep your answers short.",
        "high": "You give long answers with plenty of background detail.",
    },
    "cooperativeness": {
        "low": "You volunteer little information and make the agent work for every detail.",
        "high": "You volunteer context and answer questions helpfully.",
    },
    "attention_to_detail": {
        "low": "You gloss over details and may misread numbers or codes.",
        "high": "You double-check codes, amounts, and spellings.",
    },
    "decisiveness": {
        "low": "You hesitate and ask the agent to decide for you.",
        "high": "You make decisions quickly and commit to them.",
    },
    "skepticism": {
        "low": "You take the agent's statements at face value.",
        "high": "You question the agent's claims and ask them to confirm specifics.",
    },
    "politeness": {
        "low": "You are blunt, occasionally rude.",
        "high": "You are courteous and quick to apologize.",
    },
    "formality": {
        "low": "You speak casually, with slang.",
        "high": "You speak formally and precisely.",
    },
    "humor": {
        "low": "You stay serious throughout the call.",
        "high": "You joke and make small talk.",
    },
    "resilience": {
        "low": "Setbacks visibly throw you off track.",
        "high": "You recover quickly from setbacks.",
    },
}

_EMOTION_DIRECTIVES: dict[str, dict[str, str]] = {
    "frustration": {
        "high": "You are visibly frustrated: sigh, repeat yourself, and voice dissatisfaction when progress stalls.",
        "low": "You stay unbothered even when things go wrong.",
    },
    "anxiety": {
        "high": "You sound anxious: double-check everything and seek reassurance.",
        "low": "You are untroubled by the situation.",
    },
    "trust": {
        "high": "You trust the agent and readily accept their help.",
        "low": "You distrust the agent's motives and hesitate to share information.",
    },
    "confidence": {
        "high": "You speak with certainty about what you need.",
        "low": "You sound unsure and second-guess yourself.",
    },
    "stress": {
        "high": "You are under real pressure and mention it.",
        "low": "You are relaxed about the outcome.",
    },
}

_QUERY_COMPLEXITY_DIRECTIVES: dict[str, str] = {
    "direct": "Phrase your requests plainly: state exactly what you need, with all details, up front.",
    "contextual": "Give brief context before each request, but keep the actual ask clear.",
    "indirect": "Approach requests indirectly: hint at what you need and let the agent draw it out of you.",
    "deliberately_vague": (
        "Deliberately phrase requests vaguely: omit key details and use imprecise words; "
        "only clarify when the agent asks the right question."
    ),
}


def _band(value: float) -> str:
    """Map a 0-1 continuous dimension onto a qualitative band."""
    if value < _LOW_BAND:
        return "low"
    if value > _HIGH_BAND:
        return "high"
    return "moderate"


def _clip(value: float) -> float:
    """Clamp a continuous dimension to the unit interval."""
    return min(1.0, max(0.0, value))


def _scenario_text(goal: Mapping[str, Any]) -> str:
    """Flatten the record's user goal into the scenario context for tier 3."""
    parts = [str(goal.get("high_level_user_goal", ""))]
    decision_tree = goal.get("decision_tree")
    if isinstance(decision_tree, Mapping):
        for key in ("must_have_criteria", "edge_cases", "escalation_behavior", "failure_condition"):
            value = decision_tree.get(key, "")
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
    information_required = goal.get("information_required", [])
    if isinstance(information_required, list):
        parts.extend(str(item) for item in information_required)
    return " ".join(part for part in parts if part)


def _scenario_pressure(scenario_text: str) -> float:
    """Score scenario pressure in [0, 1] from urgency and adversity keywords."""
    text = scenario_text.lower()
    urgency = sum(1 for token in _URGENCY_TOKENS if token in text)
    adversity = sum(1 for token in _ADVERSITY_TOKENS if token in text)
    return min(1.0, 0.14 * urgency + 0.11 * adversity)


def _apply_scenario_shifts(vector: dict[str, Any], pressure: float) -> None:
    """Shift tier-3 emotional states in response to scenario pressure.

    Less resilient callers react more, and low patience amplifies the
    frustration response — the same profile therefore behaves differently
    across scenarios of different difficulty.
    """
    if pressure <= 0.0:
        vector["scenario_pressure"] = 0.0
        return
    traits = vector["traits"]
    reactivity = 0.4 + 0.6 * (1.0 - float(traits["resilience"]))
    frustration_reactivity = reactivity * (0.3 + 0.7 * (1.0 - float(traits["patience"])))
    deltas = {
        "frustration": 0.35 * pressure * frustration_reactivity,
        "anxiety": 0.25 * pressure * reactivity,
        "stress": 0.30 * pressure * reactivity,
        "trust": -0.20 * pressure * reactivity,
        "confidence": -0.15 * pressure * reactivity,
    }
    emotions = vector["emotional_states"]
    for key, delta in deltas.items():
        emotions[key] = _clip(float(emotions[key]) + delta)
    vector["scenario_pressure"] = pressure


def _stable_seed(persona_config: Mapping[str, Any], profile: str) -> int:
    """Derive a seed from the caller identity so a record reproduces its caller."""
    identity = f"{persona_config.get('name', '')}|{profile}"
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:4], "big")


def build_persona_vector(
    spec: Mapping[str, Any],
    scenario_text: str = "",
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Instantiate a three-tier persona vector from a spec.

    Args:
        spec: Either a bare profile name or a mapping with a ``profile`` key
            plus optional ``demographics``, ``traits``, ``emotional_states``
            and ``query_complexity`` overrides. Explicit overrides are exact
            and are exempt from sampling noise and correlation rules.
        scenario_text: Scenario context driving the tier-3 emotional shifts.
        seed: Seed for the Gaussian trait sampling; the same seed always
            reproduces the same simulated caller.

    Returns:
        The sampled vector with ``profile``, ``demographics``, ``traits``,
        ``emotional_states``, ``query_complexity``, ``scenario_pressure`` and
        the audited ``correlation_adjustments``.
    """
    if isinstance(spec, str):
        spec = {"profile": spec}
    profile_name = str(spec.get("profile", "balanced_default"))
    if profile_name not in NAMED_PROFILES:
        raise ValueError(
            f"Unknown persona profile {profile_name!r}. Available: {', '.join(sorted(NAMED_PROFILES))}."
        )
    if seed is None:
        seed = spec.get("seed")
    rng = random.Random(int(seed) if seed is not None else 0)

    base = NAMED_PROFILES[profile_name]
    vector: dict[str, Any] = {
        "profile": profile_name,
        "demographics": {**base["demographics"]},
        "traits": dict(base["traits"]),
        "emotional_states": dict(base["emotional_states"]),
        "query_complexity": str(spec.get("query_complexity", "direct")),
    }
    if vector["query_complexity"] not in QUERY_COMPLEXITY_LEVELS:
        raise ValueError(
            f"Unknown query complexity {vector['query_complexity']!r}. "
            f"Available: {', '.join(QUERY_COMPLEXITY_LEVELS)}."
        )

    explicit: set[str] = set()
    for section, keys, sigma in (
        ("demographics", DEMOGRAPHIC_KEYS, 0.0),
        ("traits", TRAIT_KEYS, _TRAIT_SIGMA),
        ("emotional_states", EMOTION_KEYS, _EMOTION_SIGMA),
    ):
        overrides = spec.get(section, {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"persona_vector.{section} must be a mapping, got {type(overrides).__name__}.")
        unknown = set(overrides) - set(keys)
        if unknown:
            raise ValueError(f"Unknown persona_vector.{section} keys: {', '.join(sorted(unknown))}.")
        vector[section].update(overrides)
        explicit.update(f"{section}.{key}" for key in overrides)
        if sigma:
            for key in keys:
                if f"{section}.{key}" not in explicit:
                    vector[section][key] = _clip(float(vector[section][key]) + rng.gauss(0.0, sigma))

    adjustments = []
    for source, direction, target, strength in _CORRELATION_RULES:
        if f"traits.{target}" in explicit:
            continue
        delta = direction * strength * (float(vector["traits"][source]) - 0.5) * 2.0
        if abs(delta) < 0.02:
            continue
        vector["traits"][target] = _clip(float(vector["traits"][target]) + delta)
        adjustments.append({"source": source, "target": target, "delta": round(delta, 4)})
    vector["correlation_adjustments"] = adjustments

    _apply_scenario_shifts(vector, _scenario_pressure(scenario_text))
    return vector


def render_persona_fragment(vector: Mapping[str, Any]) -> str:
    """Render a persona vector as the user-persona prompt fragment."""
    lines = [f"Simulated caller profile: {vector['profile']} (sampled persona vector — stay in character)."]

    demographics = vector.get("demographics", {})
    if demographics:
        rendered = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in demographics.items())
        lines.append(f"Caller context — {rendered}.")

    traits = vector["traits"]
    directives = [
        _TRAIT_DIRECTIVES[key][_band(float(traits[key]))]
        for key in TRAIT_KEYS
        if _band(float(traits[key])) in _TRAIT_DIRECTIVES[key]
    ]
    if directives:
        lines.append("Stable behavioral traits: " + " ".join(directives))
    lines.append("Trait vector: " + ", ".join(f"{key} {float(traits[key]):.2f}" for key in TRAIT_KEYS) + ".")

    emotions = vector["emotional_states"]
    lines.append(
        "Entering the call you feel: "
        + ", ".join(f"{key} {float(emotions[key]):.2f} ({_band(float(emotions[key]))})" for key in EMOTION_KEYS)
        + "."
    )
    emotion_directives = [
        _EMOTION_DIRECTIVES[key][_band(float(emotions[key]))]
        for key in EMOTION_KEYS
        if _band(float(emotions[key])) in _EMOTION_DIRECTIVES[key]
    ]
    if emotion_directives:
        lines.append(" ".join(emotion_directives))

    complexity = vector["query_complexity"]
    lines.append(f"Query phrasing (complexity level: {complexity}): {_QUERY_COMPLEXITY_DIRECTIVES[complexity]}")
    return "\n".join(lines)


def persona_fragment_from_config(
    persona_config: Mapping[str, Any] | None,
    goal: Mapping[str, Any] | None = None,
) -> str | None:
    """Build the user-persona fragment carried by a record's user_config.

    Returns None when the record did not opt into a persona vector, so the
    caller falls back to the shared behavior prompts. Seeds default to a
    stable per-record draw (caller name + profile) for reproducibility.
    """
    if persona_config is None:
        return None
    spec = persona_config.get("persona_vector")
    if spec is None:
        return None
    if not isinstance(spec, str | Mapping):
        raise ValueError("persona_vector must be a profile name or a mapping.")
    seed = spec.get("seed") if isinstance(spec, Mapping) else None
    if seed is None:
        profile = spec if isinstance(spec, str) else str(spec.get("profile", "balanced_default"))
        seed = _stable_seed(persona_config, profile)
    vector = build_persona_vector(spec, _scenario_text(goal or {}), seed=int(seed))
    return render_persona_fragment(vector)
