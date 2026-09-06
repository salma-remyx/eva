"""Tests for the cultural grounding layer and its wiring into culture.py.

The grounding profiles are the adapted core of CultureConverse (arXiv:2608.28405):
implicit, per-language cultural constraints the simulated user carries without
announcing, which the agent is expected to infer (see configs/cultural_grounding.yaml).
"""

from eva.utils.cultural_grounding import (
    build_agent_cultural_brief,
    build_user_cultural_directive,
    cultural_aspect_keys,
    get_cultural_grounding,
)
from eva.utils.culture import add_user_language_directive


class TestGetCulturalGrounding:
    def test_profile_loaded_for_supported_languages(self):
        for language in ("fr", "de", "es"):
            profile = get_cultural_grounding(language)
            assert profile is not None
            assert profile["community"]
            assert cultural_aspect_keys(language)

    def test_english_and_unknown_languages_are_ungrounded(self):
        assert get_cultural_grounding("en") is None
        assert get_cultural_grounding("english") is None
        assert get_cultural_grounding("it") is None
        assert cultural_aspect_keys("it") == ()

    def test_language_codes_are_normalized(self):
        assert get_cultural_grounding("FR") is not None
        assert get_cultural_grounding(" fr ") is not None


class TestUserDirective:
    def test_directive_states_constraints_are_implicit(self):
        directive = build_user_cultural_directive("fr")
        assert directive is not None
        # The paper's core move: constraints are implicit — the user must not
        # announce them, so the agent has to infer them from partial information.
        assert "never announce" in directive.lower()
        assert "formality_register" in directive

    def test_directive_none_for_ungrounded_languages(self):
        assert build_user_cultural_directive("en") is None
        assert build_user_cultural_directive("it") is None


class TestAgentBrief:
    def test_brief_describes_agent_expectations(self):
        brief = build_agent_cultural_brief("de")
        assert brief is not None
        assert "Sie" in brief
        for aspect in cultural_aspect_keys("de"):
            assert aspect in brief

    def test_brief_none_for_ungrounded_languages(self):
        assert build_agent_cultural_brief("en") is None
        assert build_agent_cultural_brief("it") is None


class TestCultureWiring:
    """The grounding rides culture.py's existing just-in-time persona resolution."""

    PERSONA = "You're direct and to the point."

    def test_french_persona_gains_grounding_after_language_directive(self):
        resolved = add_user_language_directive("fr", "French", self.PERSONA)
        assert resolved.startswith(self.PERSONA)
        assert "Speak ONLY in French." in resolved
        assert "Cultural grounding (implicit)" in resolved
        assert "formality_register" in resolved

    def test_english_persona_is_unchanged(self):
        assert add_user_language_directive("en", "English", self.PERSONA) == self.PERSONA

    def test_ungrounded_language_keeps_language_directive_only(self):
        resolved = add_user_language_directive("it", "Italian", self.PERSONA)
        assert "Speak ONLY in Italian." in resolved
        assert "Cultural grounding" not in resolved

    def test_judge_side_brief_matches_simulator_side_aspects(self):
        # The judge rubric and the simulator directive must come from the same
        # profile so the agent is scored against exactly what the user exhibits.
        for language in ("fr", "de", "es"):
            directive = build_user_cultural_directive(language)
            brief = build_agent_cultural_brief(language)
            for aspect in cultural_aspect_keys(language):
                assert aspect in (directive or "")
                assert aspect in (brief or "")
