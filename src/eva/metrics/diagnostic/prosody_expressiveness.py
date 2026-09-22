"""Prosody expressiveness metric: decoupled multi-dimensional prosody judgment.

Judges the agent's spoken prosody per turn across independent dimensions
(emotion, intonation, energy) using the existing Gemini audio-judge pipeline.
The design follows the decoupling principles of Decoupled-Live-ProsodyJudge
(D-LPJ, "Multi-Dimensional Prosody Judgment For Live Streaming Speech
Synthesis", arXiv:2609.20124), adapted from the paper's pairwise TTS
comparator to eva's pointwise per-turn metric contract:

- **No overall verdict.** The judge rubric never asks for an overall rating,
  so per-dimension judgments cannot collapse onto a single preference bit
  (the "verdict coupling" failure mode the paper identifies).
- **Uncertainty masking.** The judge returns ``null`` for any dimension it is
  not confident about; masked ratings are excluded from that dimension's
  aggregation instead of being forced onto the scale (the inference-time
  analogue of the paper's SFT pair-dimension masking).
- **Per-dimension rationale.** Each dimension rating carries its own evidence
  sentence, so one dimension's justification cannot stand in for the others.

The paper's distilled Qwen3-Omni student model and its span-local GRPO
training loop are out of scope: eva is an evaluation framework, so the
decoupling is achieved through rubric and response design on top of the stock
Gemini judge (the paper's own teacher model).

A ``dimension_collapse_rate`` sub-metric reports the fraction of turns where
every rated dimension received the same rating — the pointwise signature of
verdict coupling. A fully-coupled judge yields 1.0; healthy decoupled
judgments land well below it.

This metric is opt-in (``exclude_from_default_metrics``): run it on an
existing benchmark run via ``scripts/run_prosody_expressiveness.py``, which
drives the standard ``MetricsRunner``. Like every metric module it is imported
by ``eva/metrics/diagnostic/__init__.py`` (registering it in the global metric
registry) and tracked by the signature drift test — its prompt lives outside
``judge.yaml`` under the ``prosody`` namespace (``prompt_namespace``), so
``tests/fixtures/metric_signatures.json`` hashes it from there. To fold it
into every run, drop ``exclude_from_default_metrics``.
"""

from typing import Any

from eva.metrics.base import MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import aggregate_per_turn_scores, normalize_rating, resolve_turn_id
from eva.models.results import MetricScore


@register_metric
class ProsodyExpressivenessMetric(SpeechFidelityBaseMetric):
    """Audio-judge metric rating agent prosody on decoupled per-turn dimensions.

    Each assistant turn is rated 0-3 on three independent dimensions (emotion,
    intonation, energy). There is no overall rating: the parent score averages
    per-turn dimension means, and each dimension is surfaced as its own
    sub-metric so a strong energy score cannot hide a flat intonation score.
    """

    name = "prosody_expressiveness"
    version = "v0.1"
    description = "Diagnostic metric: decoupled per-dimension prosody judgment (emotion, intonation, energy)"
    category = "diagnostic"
    role = "assistant"
    rating_scale = (0, 3)
    # The paper's prosody dimensions, kept fixed so the prompt's per-dimension
    # rating anchors always match the parsed response.
    dimensions: tuple[str, ...] = ("emotion", "intonation", "energy")
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    # The judge prompt lives in ``prosody.yaml`` under a top-level ``prosody:`` key
    # rather than in ``judge.yaml`` because PromptManager merges yaml files at
    # the top level — a second file redefining ``judge:`` would clobber
    # judge.yaml's entire judge section. Inherited ``get_judge_prompt`` and the
    # signature drift test both resolve prompts through this namespace.
    prompt_namespace = "prosody"

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute decoupled prosody scores for every assistant turn.

        Args:
            context: MetricContext with the agent audio and intended turns.

        Returns:
            MetricScore whose normalized score averages per-turn dimension
            means, with one sub-metric per dimension plus
            ``dimension_collapse_rate``.
        """
        try:
            audio_segment = self.load_role_audio(context, self.role)
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"No {self.role} audio file available",
                )

            if self.trim_silence:
                audio_segment = self._trim_silence(audio_segment, context)

            intended_turns = self._get_intended_turns(context)
            tts_turn_ids = sorted(intended_turns.keys())

            prompt = self.get_judge_prompt(
                intended_turns_formatted=self._format_intended_turns(intended_turns),
                expected_language=context.language_display_name,
            )
            messages = self.create_audio_message(self.encode_audio_segment(audio_segment), prompt)

            response_text, turns = await self._call_and_parse(messages, context, audio_segment, prompt)

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            self.logger.debug(f"Raw judge response: {response_text[:200]}")

            if len(turns) != len(tts_turn_ids):
                self.logger.warning(
                    f"[{context.record_id}] Expected {len(tts_turn_ids)} prosody ratings "
                    f"for {self.role}, got {len(turns)}"
                )

            return self.build_prosody_score(turns, tts_turn_ids, prompt, response_text)

        except Exception as e:
            return self._handle_error(e, context)

    def build_prosody_score(
        self,
        turns: list[dict[str, Any]],
        expected_turn_ids: list[int],
        prompt: str,
        response_text: str,
    ) -> MetricScore:
        """Aggregate parsed per-turn dimension ratings into a MetricScore.

        Args:
            turns: Per-turn items from the judge response, each expected to
                carry ``dimensions: {<name>: {rating, explanation}}``.
            expected_turn_ids: Turn IDs the judge was asked to rate.
            prompt: The rendered judge prompt (stored in details for auditing).
            response_text: The raw judge response (stored in details for auditing).

        Returns:
            MetricScore with per-dimension sub-metrics and the
            ``dimension_collapse_rate`` coupling diagnostic.
        """
        min_rating, max_rating = self.rating_scale

        per_turn_ratings: dict[int, dict[str, int | None]] = {}
        per_turn_explanations: dict[int, dict[str, str]] = {}

        for response_item in turns:
            turn_id = resolve_turn_id(response_item, expected_turn_ids, self.name)
            if turn_id is None:
                self.logger.warning(f"Could not resolve turn ID for {response_item} turn_ids {expected_turn_ids}")
                continue
            dimensions_item = response_item.get("dimensions")
            if not isinstance(dimensions_item, dict):
                dimensions_item = {}
            turn_ratings: dict[str, int | None] = {}
            turn_explanations: dict[str, str] = {}
            for dimension in self.dimensions:
                rating, explanation = self._parse_dimension_rating(
                    dimensions_item.get(dimension), min_rating, max_rating, dimension, turn_id
                )
                turn_ratings[dimension] = rating
                turn_explanations[dimension] = explanation
            per_turn_ratings[turn_id] = turn_ratings
            per_turn_explanations[turn_id] = turn_explanations

        sub_metrics = self._build_dimension_sub_metrics(per_turn_ratings, min_rating, max_rating)

        details: dict[str, Any] = {
            "aggregation": self.aggregation,
            "dimensions": list(self.dimensions),
            "num_turns": len(expected_turn_ids),
            "num_evaluated": sum(1 for t in per_turn_ratings.values() if any(r is not None for r in t.values())),
            "num_masked_ratings": sum(1 for t in per_turn_ratings.values() for r in t.values() if r is None),
            "per_turn_ratings": per_turn_ratings,
            "per_turn_explanations": per_turn_explanations,
            "judge_prompt": prompt,
            "judge_raw_response": response_text,
        }

        if sub_metrics is None:
            return MetricScore(
                name=self.name,
                score=0.0,
                normalized_score=0.0,
                error="No prosody dimension ratings parsed from judge response",
                details=details,
            )

        # Per-turn value = mean of the turn's rated (non-masked) dimensions.
        # Turns with no rated dimensions contribute None and drop out of the
        # parent aggregation, matching how other per-turn metrics treat N/A turns.
        per_turn_normalized: dict[int, float | None] = {}
        for turn_id, turn_ratings in per_turn_ratings.items():
            rated = [normalize_rating(r, min_rating, max_rating) for r in turn_ratings.values() if r is not None]
            per_turn_normalized[turn_id] = sum(rated) / len(rated) if rated else None

        aggregated_score = aggregate_per_turn_scores(list(per_turn_normalized.values()), self.aggregation)
        all_rated = [r for t in per_turn_ratings.values() for r in t.values() if r is not None]
        avg_rating = sum(all_rated) / len(all_rated) if all_rated else 0.0

        return MetricScore(
            name=self.name,
            score=round(avg_rating, 3),
            normalized_score=round(aggregated_score, 3) if aggregated_score is not None else 0.0,
            details=details,
            error="Aggregation failed" if aggregated_score is None else None,
            sub_metrics=sub_metrics,
        )

    def _parse_dimension_rating(
        self,
        dimension_item: Any,
        min_rating: int,
        max_rating: int,
        dimension: str,
        turn_id: int,
    ) -> tuple[int | None, str]:
        """Extract and validate one dimension rating from a judge turn item.

        Explicit nulls (the judge flagging uncertainty) and invalid values are
        both returned as None — a masked rating that the aggregation excludes
        rather than a coerced guess.

        Args:
            dimension_item: The dimension's ``{rating, explanation}`` dict, or None.
            min_rating: Minimum valid rating.
            max_rating: Maximum valid rating.
            dimension: Dimension name (for logging).
            turn_id: Turn ID (for logging).

        Returns:
            Tuple of (validated rating or None, explanation).
        """
        if isinstance(dimension_item, dict):
            rating: int | str | None = dimension_item.get("rating")
            raw_explanation = dimension_item.get("explanation", "")
        else:
            rating = None
            raw_explanation = ""
        if isinstance(rating, str):
            try:
                rating = int(rating)
            except ValueError:
                rating = None
        if rating is not None and not (min_rating <= rating <= max_rating):
            self.logger.warning(f"Invalid {dimension} rating {rating} for turn {turn_id}")
            rating = None
        explanation = raw_explanation if isinstance(raw_explanation, str) else ""
        return rating, explanation

    def _build_dimension_sub_metrics(
        self,
        per_turn_ratings: dict[int, dict[str, int | None]],
        min_rating: int,
        max_rating: int,
    ) -> dict[str, MetricScore] | None:
        """Build one sub-metric per dimension plus the coupling diagnostic.

        Args:
            per_turn_ratings: {turn_id: {dimension: rating or None}}.
            min_rating: Minimum valid rating.
            max_rating: Maximum valid rating.

        Returns:
            Sub-metrics keyed by dimension name and ``dimension_collapse_rate``,
            or None when every dimension rating was masked.
        """
        sub_metrics: dict[str, MetricScore] = {}

        for dimension in self.dimensions:
            raw: list[int] = []
            normalized: list[float] = []
            masked = 0
            for turn_ratings in per_turn_ratings.values():
                rating = turn_ratings.get(dimension)
                if rating is None:
                    masked += 1
                    continue
                raw.append(rating)
                normalized.append(normalize_rating(rating, min_rating, max_rating))
            if not raw:
                # Fully masked dimension — report nothing rather than a guessed 0.
                continue
            dim_normalized = aggregate_per_turn_scores(normalized, self.aggregation)
            sub_metrics[dimension] = MetricScore(
                name=f"{self.name}.{dimension}",
                score=round(sum(raw) / len(raw), 3),
                normalized_score=round(dim_normalized, 3) if dim_normalized is not None else None,
                details={"num_rated": len(raw), "num_masked": masked},
            )

        # Verdict-coupling diagnostic: a turn "collapses" when every rated
        # dimension received the same rating — the signature of a judge that
        # pattern-matched one impression onto the whole rubric (D-LPJ's verdict
        # coupling). The ``_rate`` suffix makes the runner treat this as
        # lower-is-better.
        collapsed_turn_ids: list[int] = []
        multi_dimension_turn_ids: list[int] = []
        for turn_id, turn_ratings in per_turn_ratings.items():
            rated = [r for r in turn_ratings.values() if r is not None]
            if len(rated) < 2:
                continue
            multi_dimension_turn_ids.append(turn_id)
            if len(set(rated)) == 1:
                collapsed_turn_ids.append(turn_id)
        if multi_dimension_turn_ids:
            collapse_rate = len(collapsed_turn_ids) / len(multi_dimension_turn_ids)
            sub_metrics["dimension_collapse_rate"] = MetricScore(
                name=f"{self.name}.dimension_collapse_rate",
                score=round(collapse_rate, 3),
                normalized_score=round(collapse_rate, 3),
                details={"collapsed_turns": collapsed_turn_ids, "num_rated": len(multi_dimension_turn_ids)},
            )

        return sub_metrics or None
