"""Tests for eva.metrics.diagnostic.transcription_semantic_accuracy."""

import pytest

from eva.metrics.diagnostic.stt_wer import STTWERMetric
from eva.metrics.diagnostic.transcription_semantic_accuracy import TranscriptionSemanticAccuracyMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context


class TestTranscriptionSemanticAccuracyWiring:
    def test_registered_and_default_on(self):
        """Importing eva.metrics.diagnostic registers the metric for the default run."""
        import eva.metrics.diagnostic  # noqa: F401

        registry = get_global_registry()
        assert registry.get("transcription_semantic_accuracy") is TranscriptionSemanticAccuracyMetric
        assert "transcription_semantic_accuracy" in registry.list_metrics()

    def test_created_from_registry_with_config(self):
        metric = get_global_registry().create("transcription_semantic_accuracy", {"language": "fr"})
        assert isinstance(metric, TranscriptionSemanticAccuracyMetric)
        assert metric.language == "fr"


class TestTranscriptionSemanticAccuracyCompute:
    @pytest.mark.asyncio
    async def test_perfect_transcription(self):
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.normalized_score == 1.0
        assert result.details["cer"] == 0.0
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_lexical_drift_that_preserves_meaning(self):
        """Function-word drift moves CER (and WER) but not the semantic score."""
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "I would like to book a flight to Paris"},
            transcribed_user_turns={1: "I'd like to book the flight to Paris"},
        )
        result = await metric.compute(ctx)
        assert result.details["cer"] > 0
        assert result.normalized_score == 1.0
        assert result.details["per_turn_semdist"][1] == 0.0

        # The word-level diagnostic still charges the same turn — the two
        # metrics are meant to be read together.
        wer_result = await STTWERMetric().compute(ctx)
        assert wer_result.details["wer"] > 0

    @pytest.mark.asyncio
    async def test_content_word_substitution_changes_meaning(self):
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "book a flight to Paris"},
            transcribed_user_turns={1: "book a flight to Tokyo"},
        )
        result = await metric.compute(ctx)
        assert result.details["cer"] > 0
        assert result.normalized_score < 1.0
        assert result.details["semdist"] == pytest.approx(result.score)

    @pytest.mark.asyncio
    async def test_dropped_negation_is_a_semantic_error(self):
        """Negations stay content words: losing a "not" must lower the score."""
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "I do not want a window seat"},
            transcribed_user_turns={1: "I want a window seat"},
        )
        result = await metric.compute(ctx)
        assert result.normalized_score < 1.0
        assert result.details["per_turn_semdist"][1] > 0

    @pytest.mark.asyncio
    async def test_cer_sub_metric(self):
        """CER rides along as a rate sub-metric over the reference character count."""
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the quick brown fox"},
            transcribed_user_turns={1: "the quick green fox"},
        )
        result = await metric.compute(ctx)

        assert result.sub_metrics is not None
        sub = result.sub_metrics["character_error_rate"]
        assert sub.name == "transcription_semantic_accuracy.character_error_rate"
        assert sub.details["reference_characters"] == result.details["reference_characters"]
        assert sub.details["count"] == (
            result.details["total_substitutions"]
            + result.details["total_deletions"]
            + result.details["total_insertions"]
        )
        # make_rate_sub_metric rounds to 3 decimals
        assert sub.score == pytest.approx(sub.details["count"] / sub.details["reference_characters"], abs=0.001)

    @pytest.mark.asyncio
    async def test_multiple_turns_per_turn_details(self):
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello there", 2: "book a flight to Paris", 3: "thank you"},
            transcribed_user_turns={1: "hello there", 2: "book a flight to Tokyo", 3: "thank you"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 3
        assert set(result.details["per_turn_cer"]) == {1, 2, 3}
        assert set(result.details["per_turn_semdist"]) == {1, 2, 3}
        # Only the turn whose content changed carries semantic distance.
        assert result.details["per_turn_semdist"][1] == 0.0
        assert result.details["per_turn_semdist"][2] > 0

    @pytest.mark.asyncio
    async def test_bracket_annotations_stripped(self):
        """Bracket annotations like [slow] are removed before comparison."""
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[slow] hello world [likely cut off]"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.details["cer"] == 0.0
        assert result.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_empty_turns_skipped(self):
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[likely interruption]", 2: "hello world"},
            transcribed_user_turns={1: "", 2: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_no_common_turns(self):
        metric = TranscriptionSemanticAccuracyMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello"},
            transcribed_user_turns={2: "hello"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_non_english_falls_back_to_all_tokens(self):
        """Languages without a stopword list compare every token (conservative)."""
        metric = TranscriptionSemanticAccuracyMetric(config={"language": "fr"})
        ctx = make_metric_context(
            intended_user_turns={1: "je voudrais annuler le vol"},
            transcribed_user_turns={1: "je voudrais annuler mon vol"},
        )
        result = await metric.compute(ctx)
        assert result.normalized_score < 1.0
