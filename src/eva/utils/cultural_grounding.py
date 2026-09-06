"""Cultural grounding layer for multilingual evaluation records.

On the multilingual path, culture is currently expressed only superficially:
localized names, translated locations, and a "speak this language" directive.
This module adds the missing layer — per-language cultural grounding profiles
describing the *implicit* expectations a simulated user carries (formality
register, date/time/number formats, greeting norms). The user never states
these expectations; a culturally competent assistant is expected to infer them
from the user's own speech and accommodate them over the conversation.

Two renderings of the same profile:
- ``build_user_cultural_directive`` — appended to the user-simulator persona
  (wired in :mod:`eva.utils.culture`). The user behaves per the profile but
  never announces it, so the assistant must infer rather than follow rules.
- ``build_agent_cultural_brief`` — the rubric the ``cultural_appropriateness``
  judge scores the agent against.

Methodology adapted (clean-room, no code or data copied) from CultureConverse
(arXiv:2608.28405): evaluate cultural competency through multi-turn practical
assistance in which the user carries implicit cultural constraints the
assistant must infer from partial information, rather than through explicit
single-turn instructions.

Profiles live in ``configs/cultural_grounding.yaml``. English (and any
language without a profile) gets no grounding — behavior is unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_GROUNDING_PATH = Path(__file__).resolve().parents[3] / "configs" / "cultural_grounding.yaml"

_USER_DIRECTIVE_TEMPLATE = (
    "## Cultural grounding (implicit)\n"
    "You are {community}. The following behaviors are second nature to you.\n"
    "You never announce them as rules and never explain them unless the agent asks\n"
    "directly — you simply expect the agent to notice and accommodate them:\n"
    "{aspects}\n"
    "If the agent violates one of these expectations, continue naturally: at most a\n"
    "mild hint or a polite restatement in your own wording. Do not explicitly correct\n"
    'the agent (e.g. do not say "you should use the day-month format").'
)

_AGENT_BRIEF_TEMPLATE = (
    "The simulated user is {community}. They carry the following implicit cultural\n"
    "expectations, which they never state aloud. A culturally competent assistant\n"
    "infers them from the user's speech and accommodates them:\n"
    "{aspects}"
)

_USER_ASPECT_LINE = "- {aspect}: {user_behavior}"
_AGENT_ASPECT_LINE = "- {aspect}: {agent_expectation}"


@lru_cache(maxsize=1)
def _load_profiles() -> dict[str, dict[str, Any]]:
    """Load the per-language grounding profiles; ``{}`` when the file is missing."""
    if not _GROUNDING_PATH.exists():
        logger.warning(f"Cultural grounding profiles not found at {_GROUNDING_PATH}; skipping grounding.")
        return {}
    return yaml.safe_load(_GROUNDING_PATH.read_text(encoding="utf-8")) or {}


def _normalize_language(language: str) -> str:
    """Reduce a language code to the lowercase two-letter form the profiles are keyed by."""
    return language.strip().lower()[:2]


def get_cultural_grounding(language: str) -> dict[str, Any] | None:
    """Return the grounding profile for ``language``, or None for English/unknown languages."""
    code = _normalize_language(language)
    if not code or code == "en":
        return None
    profile = _load_profiles().get(code)
    if profile is None:
        logger.debug(f"No cultural grounding profile for language {language!r}; skipping grounding.")
    return profile


def build_user_cultural_directive(language: str) -> str | None:
    """Render the simulator-side grounding block for ``language``, or None when ungrounded.

    The block is appended to the user persona (see
    :func:`eva.utils.culture.add_user_language_directive`), so both the simulator
    and any judge re-rendering the persona see the exact same grounding.
    """
    profile = get_cultural_grounding(language)
    if profile is None:
        return None
    aspects = profile.get("aspects") or {}
    lines = [
        _USER_ASPECT_LINE.format(aspect=slug, user_behavior=a.get("user_behavior", ""))
        for slug, a in aspects.items()
        if isinstance(a, dict)
    ]
    if not lines:
        return None
    return _USER_DIRECTIVE_TEMPLATE.format(
        community=profile.get("community", f"a speaker of {language}"), aspects="\n".join(lines)
    )


def build_agent_cultural_brief(language: str) -> str | None:
    """Render the judge-side rubric for ``language``, or None when ungrounded."""
    profile = get_cultural_grounding(language)
    if profile is None:
        return None
    aspects = profile.get("aspects") or {}
    lines = [
        _AGENT_ASPECT_LINE.format(aspect=slug, agent_expectation=a.get("agent_expectation", ""))
        for slug, a in aspects.items()
        if isinstance(a, dict)
    ]
    if not lines:
        return None
    return _AGENT_BRIEF_TEMPLATE.format(
        community=profile.get("community", f"a speaker of {language}"), aspects="\n".join(lines)
    )


def cultural_aspect_keys(language: str) -> tuple[str, ...]:
    """Return the stable aspect slugs for ``language`` (empty when ungrounded)."""
    profile = get_cultural_grounding(language)
    if profile is None:
        return ()
    return tuple(slug for slug, a in (profile.get("aspects") or {}).items() if isinstance(a, dict))
