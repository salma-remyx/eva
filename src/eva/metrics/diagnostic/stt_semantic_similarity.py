"""STT (Speech-to-Text) semantic similarity diagnostic metric.

Semantic counterpart to `stt_wer`: measures how well transcriptions preserve
the *meaning* of what the user simulator said, using sentence-embedding
cosine similarity (SemDist), and reports per-turn WER alongside it so turns
where the lexical and semantic views disagree are easy to spot.

Adapted from "Generative vs. Encoder Large Language Models for ASR
Evaluation: A Comparative Study" (arXiv:2608.25574), which shows that
encoder-based semantic similarity correlates with human semantic judgments
better than WER. The paper's SemDist configuration is kept; its BERT-family
encoders are swapped for a LiteLLM-routed embedding model, and the paper's
layer/pooling study is out of scope.
"""

import os
from typing import Any, NamedTuple

import jiwer
import litellm
import numpy as np

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.diagnostic.stt_wer import _BRACKET_PATTERN, _CER_LANGUAGES
from eva.metrics.registry import register_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingResult(NamedTuple):
    """Batch embedding result with token usage for cost tracking."""

    vectors: list[list[float]]
    usage: dict[str, Any] | None = None


def _embedding_vector(item: Any) -> list[float]:
    """Extract the embedding vector from one LiteLLM response entry.

    Entries are pydantic objects on current litellm versions but plain dicts
    on older ones, so both shapes are accepted.
    """
    embedding = item.get("embedding") if isinstance(item, dict) else item.embedding
    return [float(component) for component in embedding]


class EmbeddingClient:
    """Sentence-embedding client backed by LiteLLM's embedding endpoint.

    eva.utils.router only exposes completion deployments, so embeddings go
    through litellm.aembedding directly and rely on standard provider
    environment variables for credentials.
    """

    def __init__(self, model: str, timeout: int = 60) -> None:
        """Initialize the client with a LiteLLM embedding model name."""
        self.model = model
        self.timeout = timeout

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts, returning one vector per text."""
        response = await litellm.aembedding(model=self.model, input=list(texts), timeout=self.timeout, num_retries=3)
        vectors = [_embedding_vector(item) for item in response.data]
        usage = None
        response_usage = getattr(response, "usage", None)
        if response_usage:
            usage = {"prompt_tokens": response_usage.prompt_tokens, "model_name": self.model}
        return EmbeddingResult(vectors=vectors, usage=usage)


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either is all zeros."""
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    norm_product = float(np.linalg.norm(u_arr) * np.linalg.norm(v_arr))
    if norm_product == 0.0:
        return 0.0
    return float(np.dot(u_arr, v_arr) / norm_product)


def rescale_cosine(cosine: float) -> float:
    """Rescale a cosine similarity from [-1, 1] to a [0, 1] similarity score."""
    return min(1.0, max(0.0, (cosine + 1.0) / 2.0))


@register_metric
class STTSemanticSimilarityMetric(CodeMetric):
    """Speech-to-Text semantic similarity metric (SemDist).

    Measures the semantic fidelity of STT transcription by comparing what
    the user simulator said (tts_text_user) to what was transcribed
    (transcript_user) via sentence-embedding cosine similarity, rescaled to
    0-1 (higher is better).

    Per-turn WER is reported next to the semantic score so the two metric
    families can be contrasted: a turn with high WER but high semantic
    similarity drifted lexically while preserving meaning, while low
    semantic similarity flags meaning-changing errors WER alone may miss.

    This is a diagnostic metric used for diagnosing model performance
    issues. It is not directly used in final evaluation scores.

    Opt-in: it costs one embedding API call per conversation, so it is
    excluded from default runs (enable via `--metrics stt_semantic_similarity`).
    """

    name = "stt_semantic_similarity"
    version = "v0.1"
    description = "Diagnostic metric: STT semantic similarity via sentence embeddings (SemDist)"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    def __init__(self, config: dict | None = None):
        """Initialize the metric with language and embedding model configuration.

        The embedding model resolves from config (`embedding_model`, injected
        from RunConfig.embedding_model by the orchestrator), then the
        `EVA_EMBEDDING_MODEL` settings variable, then a small default.
        """
        super().__init__(config)
        self.language = self.config.get("language", "en")
        self.embedding_model = (
            self.config.get("embedding_model") or os.environ.get("EVA_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_client = EmbeddingClient(self.embedding_model)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute per-turn semantic similarity between intended and transcribed user turns."""
        try:
            # Collect reference/hypothesis pairs for turns present in both dicts
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

            evaluated_turn_ids: list[int] = []
            references_clean: list[str] = []
            hypotheses_clean: list[str] = []

            for turn_id in common_turn_ids:
                ref = _BRACKET_PATTERN.sub("", context.intended_user_turns[turn_id]).strip()
                hyp = _BRACKET_PATTERN.sub("", context.transcribed_user_turns[turn_id]).strip()
                if not (ref and hyp):
                    continue
                ref_clean = normalize_text(ref, self.language)
                hyp_clean = normalize_text(hyp, self.language)
                # Embedding APIs reject empty strings, so drop pairs that
                # normalize away entirely (degenerate either way).
                if ref_clean and hyp_clean:
                    evaluated_turn_ids.append(turn_id)
                    references_clean.append(ref_clean)
                    hypotheses_clean.append(hyp_clean)

            if not references_clean:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            result = await self.embedding_client.embed(references_clean + hypotheses_clean)
            self._log_token_usage(
                context,
                self.embedding_model,
                {},
                "\n".join(references_clean + hypotheses_clean),
                result.usage,
                response_text=None,
            )

            reference_embeddings = result.vectors[: len(references_clean)]
            hypothesis_embeddings = result.vectors[len(references_clean) :]

            use_cer = self.language in _CER_LANGUAGES

            per_turn_semdist: dict[int, float] = {}
            per_turn_wer: dict[int, float] = {}
            raw_semdist: list[float] = []

            for turn_id, ref_clean, hyp_clean, ref_emb, hyp_emb in zip(
                evaluated_turn_ids, references_clean, hypotheses_clean, reference_embeddings, hypothesis_embeddings
            ):
                cosine = cosine_similarity(ref_emb, hyp_emb)
                semdist = rescale_cosine(cosine)
                raw_semdist.append(semdist)
                per_turn_semdist[turn_id] = round(semdist, 3)
                turn_rate = jiwer.cer(ref_clean, hyp_clean) if use_cer else jiwer.wer(ref_clean, hyp_clean)
                per_turn_wer[turn_id] = round(turn_rate, 3)

            mean_semdist = sum(raw_semdist) / len(raw_semdist)
            mean_wer = sum(per_turn_wer.values()) / len(per_turn_wer)

            return MetricScore(
                name=self.name,
                score=round(mean_semdist, 3),
                normalized_score=round(mean_semdist, 3),
                details={
                    "semdist": round(mean_semdist, 3),
                    "mean_wer": round(mean_wer, 3),
                    "embedding_model": self.embedding_model,
                    "language": self.language,
                    "use_cer": use_cer,
                    "num_turns": len(references_clean),
                    "per_turn_semdist": per_turn_semdist,
                    "per_turn_wer": per_turn_wer,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
