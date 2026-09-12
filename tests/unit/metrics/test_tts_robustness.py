"""Tests for the TTS complex-text robustness diagnostic metric."""

import asyncio

import pytest

from eva.metrics.diagnostic.tts_robustness import TTSRobustnessMetric, detect_wrong_language, text_risk_score
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


def test_metric_registered_via_diagnostic_package():
    """The diagnostic package wiring registers the metric in the global registry."""
    import eva.metrics  # noqa: F401 — importing the package pulls in eva.metrics.diagnostic

    registry = get_global_registry()
    assert registry.get("tts_robustness") is TTSRobustnessMetric
    assert "tts_robustness" in registry.list_metrics()


class TestTextRiskScore:
    def test_simple_text_is_low_risk(self):
        score, features = text_risk_score("Have a nice day.")
        assert score == 0.0
        assert not any(features.values())

    def test_complex_text_triggers_multiple_features(self):
        text = "Flight AF 402 departs on 12/09 at gate B12."
        score, features = text_risk_score(text)
        assert features["numbers"] is True
        assert features["dates"] is True
        assert features["named_entities"] is True
        assert score == pytest.approx(0.65)

    def test_code_switching_detected(self):
        text = "Votre réservation для вас est prête."
        score, features = text_risk_score(text)
        assert features["code_switching"] is True
        assert score >= 0.25

    def test_long_text_detected(self):
        score, features = text_risk_score("word " * 100)
        assert features["long_text"] is True


class TestDetectWrongLanguage:
    def test_wrong_script_flagged(self):
        assert detect_wrong_language("Ваш рейс подтверждён", "fr") == "wrong_script"

    def test_same_script_without_markers_flagged(self):
        text = "I am sorry but I did not catch what you just said there."
        assert detect_wrong_language(text, "fr") == "no_target_language_markers"

    def test_target_language_passes(self):
        assert detect_wrong_language("Votre réservation pour Lyon est confirmée.", "fr") is None

    def test_short_turns_skip_lexical_check(self):
        # Too few words to judge — only the script check applies.
        assert detect_wrong_language("Okay, sounds good.", "fr") is None

    def test_unknown_language_skipped(self):
        assert detect_wrong_language("un deux trois", "sw") is None


class TestTTSRobustnessCompute:
    def test_perfect_round_trip(self):
        metric = TTSRobustnessMetric()
        ctx = make_metric_context(
            intended_assistant_turns={1: "Have a nice day."},
            transcribed_assistant_turns={1: "Have a nice day."},
        )
        result = asyncio.run(metric.compute(ctx))
        assert result.details["cer"] == 0.0
        assert result.details["accuracy"] == 1.0
        assert result.details["num_turns"] == 1
        assert result.details["num_simple_turns"] == 1

    def test_complex_text_degrades_more_than_simple_text(self):
        """CER is split by text risk: errors on a complex turn must not hide in the average."""
        metric = TTSRobustnessMetric()
        ctx = make_metric_context(
            intended_assistant_turns={
                1: "Have a nice day.",
                2: "Flight AF 402 departs on 12/09 at gate B12.",
            },
            transcribed_assistant_turns={
                1: "Have a nice day.",
                2: "Flight AF 402 at gate B12.",  # TTS dropped the date clause
            },
        )
        result = asyncio.run(metric.compute(ctx))

        assert result.score > 0.0
        assert result.details["num_complex_turns"] == 1
        assert result.details["num_simple_turns"] == 1
        assert result.details["per_turn_cer"][1] == 0.0
        assert result.details["per_turn_cer"][2] > 0.0
        assert result.details["per_turn_text_risk"][2] == pytest.approx(0.65)

        assert result.sub_metrics is not None
        assert result.sub_metrics["simple_text_accuracy"].score == 1.0
        assert result.sub_metrics["complex_text_accuracy"].score < 1.0
        assert result.sub_metrics["wrong_language_rate"].score == 0.0

    def test_wrong_language_rate_for_foreign_script_output(self):
        """TTS falling back to another language is flagged per turn."""
        metric = TTSRobustnessMetric(config={"language": "fr"})
        ctx = make_metric_context(
            intended_assistant_turns={1: "Votre vol est confirmé."},
            transcribed_assistant_turns={1: "Ваш рейс подтверждён."},
        )
        result = asyncio.run(metric.compute(ctx))

        assert result.details["per_turn_language_flag"] == {1: "wrong_script"}
        assert result.sub_metrics is not None
        wrong = result.sub_metrics["wrong_language_rate"]
        assert wrong.score == 1.0
        assert wrong.details["turn_ids"] == [1]

    def test_no_common_turns(self):
        metric = TTSRobustnessMetric()
        ctx = make_metric_context(
            intended_assistant_turns={1: "hello"},
            transcribed_assistant_turns={2: "hello"},
        )
        result = asyncio.run(metric.compute(ctx))
        assert result.score == 0.0
        assert result.error is not None

    def test_risk_features_reported_per_turn(self):
        metric = TTSRobustnessMetric()
        ctx = make_metric_context(
            intended_assistant_turns={1: "Call me at 555-0123."},
            transcribed_assistant_turns={1: "Call me at 555-0123."},
        )
        result = asyncio.run(metric.compute(ctx))
        assert result.details["per_turn_risk_features"][1]["numbers"] is True
        assert result.details["risk_feature_rate"]["numbers"] == 1.0
