"""Tests for eva.metrics.diagnostic.stt_semantic_similarity compute and registration."""

import pytest

import eva.metrics.diagnostic  # noqa: F401  (imports register every diagnostic metric)
from eva.metrics.diagnostic.stt_semantic_similarity import EmbeddingResult, STTSemanticSimilarityMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context

# Unit vectors keyed on the *normalized* turn text the metric embeds.
_REF_FOX = [1.0, 0.0]
# 30 degrees off _REF_FOX: cosine 0.96 → semdist (0.96 + 1) / 2 = 0.98
_HYP_PARAPHRASE = [0.96, 0.28]
_ORTHOGONAL = [0.0, 1.0]

_STUB_VECTORS = {
    "hello world": _REF_FOX,
    "the quick brown fox": _REF_FOX,
    "a fast brown animal": _HYP_PARAPHRASE,
    "goodbye": _REF_FOX,
    "bye now": _ORTHOGONAL,
}


class StubEmbeddingClient:
    """Deterministic stand-in for the LiteLLM embedding client."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.seen_batches: list[list[str]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.seen_batches.append(list(texts))
        return EmbeddingResult(vectors=[self.vectors[text] for text in texts])


def make_metric(
    vectors: dict[str, list[float]] | None = None,
) -> tuple[STTSemanticSimilarityMetric, StubEmbeddingClient]:
    """Build the metric with a stub embedding client wired in."""
    metric = STTSemanticSimilarityMetric(config={"embedding_model": "stub-embedding"})
    stub = StubEmbeddingClient(vectors if vectors is not None else dict(_STUB_VECTORS))
    metric.embedding_client = stub
    return metric, stub


class TestSTTSemanticSimilarityCompute:
    @pytest.mark.asyncio
    async def test_identical_transcription(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.error is None
        assert result.score == 1.0
        assert result.normalized_score == 1.0
        assert result.details["semdist"] == 1.0
        assert result.details["per_turn_semdist"] == {1: 1.0}
        assert result.details["per_turn_wer"] == {1: 0.0}
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_lexical_drift_that_preserves_meaning(self):
        """High WER with high semantic similarity is the divergence this metric exists to surface."""
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "the quick brown fox"},
            transcribed_user_turns={1: "a fast brown animal"},
        )
        result = await metric.compute(ctx)
        assert result.details["per_turn_wer"][1] == pytest.approx(0.75)
        assert result.details["per_turn_semdist"][1] == pytest.approx(0.98)
        assert result.score == pytest.approx(0.98)
        assert result.details["mean_wer"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_semantically_different_turn_scores_lower(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "goodbye"},
            transcribed_user_turns={1: "bye now"},
        )
        result = await metric.compute(ctx)
        # Orthogonal embeddings → cosine 0 → rescaled similarity 0.5
        assert result.details["per_turn_semdist"][1] == pytest.approx(0.5)
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_multiple_turns_averaged(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world", 2: "goodbye"},
            transcribed_user_turns={1: "hello world", 2: "bye now"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx((1.0 + 0.5) / 2)
        assert set(result.details["per_turn_semdist"]) == {1, 2}

    @pytest.mark.asyncio
    async def test_embeddings_batched_into_single_call(self):
        metric, stub = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world", 2: "goodbye"},
            transcribed_user_turns={1: "hello world", 2: "bye now"},
        )
        await metric.compute(ctx)
        assert len(stub.seen_batches) == 1
        assert len(stub.seen_batches[0]) == 4  # all references and hypotheses in one batch

    @pytest.mark.asyncio
    async def test_bracket_annotations_stripped(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "[slow] hello world [likely cut off]"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        # Both sides normalize to "hello world" after bracket stripping
        assert result.details["semdist"] == 1.0
        assert result.details["per_turn_wer"][1] == 0.0

    @pytest.mark.asyncio
    async def test_empty_turns_skipped(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "[likely interruption]", 2: "hello world"},
            transcribed_user_turns={1: "", 2: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_no_common_turns(self):
        metric, _ = make_metric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello"},
            transcribed_user_turns={2: "hello"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_embedding_failure_surfaces_error(self):
        class FailingClient:
            async def embed(self, texts: list[str]) -> EmbeddingResult:
                raise RuntimeError("embedding backend unavailable")

        metric = STTSemanticSimilarityMetric()
        metric.embedding_client = FailingClient()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.error is not None
        assert "embedding backend unavailable" in result.error

    def test_config_language_and_model(self):
        metric = STTSemanticSimilarityMetric(config={"language": "fr", "embedding_model": "multilingual-embed"})
        assert metric.language == "fr"
        assert metric.embedding_model == "multilingual-embed"

    def test_embedding_model_resolves_from_settings_variable(self, monkeypatch):
        """Without metric config, the EVA_EMBEDDING_MODEL settings variable is the fallback tier."""
        monkeypatch.setenv("EVA_EMBEDDING_MODEL", "text-embedding-3-large")
        assert STTSemanticSimilarityMetric().embedding_model == "text-embedding-3-large"

    def test_metric_config_overrides_settings_variable(self, monkeypatch):
        """Explicit metric config wins over the settings variable."""
        monkeypatch.setenv("EVA_EMBEDDING_MODEL", "text-embedding-3-large")
        metric = STTSemanticSimilarityMetric(config={"embedding_model": "custom-embed"})
        assert metric.embedding_model == "custom-embed"


class TestSTTSemanticSimilarityRegistration:
    def test_registered_via_diagnostic_package(self):
        """The diagnostic package import registers the metric in the global registry."""
        assert get_global_registry().get("stt_semantic_similarity") is STTSemanticSimilarityMetric

    def test_created_from_registry_with_config(self):
        metric = get_global_registry().create("stt_semantic_similarity", config={"language": "en"})
        assert isinstance(metric, STTSemanticSimilarityMetric)
        assert metric.embedding_model == "text-embedding-3-small"

    def test_opt_in_excluded_from_default_metrics(self):
        """Costs embedding API calls, so it must not run unless named explicitly."""
        assert "stt_semantic_similarity" not in get_global_registry().list_metrics()
