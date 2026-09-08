"""Judge stability diagnostic metric.

Audits the measurement stability of eva's LLM judge instrument itself: the
same byte-identical judge request is submitted to the same model several
times within the same window, and the agreement of the returned ratings is
reported as the judge's own noise floor.

Adapted from "Clean Engineering, Unstable Measurement: A Preregistered
Reliability Failure of Black-Box LLM Observers on Shared Endpoints"
(arXiv:2609.04198), which found that same-window repeat agreement of
black-box LLM judges falls far below the stability levels that
leaderboard-style gating assumes, and recommends measuring the instrument
with a small repeat pilot before trusting any gate built on it.

Debug metric for diagnosing the evaluation instrument, not model
performance; not directly used in final evaluation scores.
"""

from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import format_transcript_with_tools
from eva.models.results import MetricScore


@register_metric
class JudgeStabilityMetric(TextJudgeMetric):
    """Agreement of the judge across byte-identical repeat calls.

    Builds one judge prompt for the record and submits the exact same
    request ``repeats`` times through the shared ``call_judge`` path — the
    same model resolution and params as every other eva text judge
    (including the ``service_tier: flex`` default). Any variance in the
    returned ratings is the instrument's, not the request's.

    The score is the exact-agreement rate: the share of repeats that match
    the modal rating. 1.0 means a frozen instrument; below that,
    ``noise_floor_normalized`` in the details gives the smallest
    between-model gap (in normalized units) this judge can resolve.
    """

    name = "judge_stability"
    version = "v0.1"
    description = "Diagnostic metric: agreement of the LLM judge across byte-identical repeat calls"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    # Opt-in: multiplies judge calls by `repeats`, so it is a pilot to run
    # when validating the judge, not part of every benchmark run.
    exclude_from_default_metrics = True
    default_repeats = 5

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        try:
            repeats = int(self.config.get("repeats", self.default_repeats))
        except (TypeError, ValueError):
            self.logger.warning(f"Invalid 'repeats' {self.config.get('repeats')!r}, using {self.default_repeats}")
            repeats = self.default_repeats
        if repeats < 2:
            self.logger.warning(f"'repeats' must be >= 2 to measure agreement, got {repeats}; using 2")
            repeats = 2
        self.repeats = repeats

    def format_transcript(self, context: MetricContext) -> str:
        """Format conversation content for the judge prompt."""
        return format_transcript_with_tools(context.conversation_trace)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Submit the identical judge request ``repeats`` times and score agreement.

        Args:
            context: MetricContext containing all data for the conversation

        Returns:
            MetricScore whose score is the exact-agreement rate across repeats
        """
        try:
            transcript_text = self.format_transcript(context)
            if not transcript_text:
                return MetricScore(name=self.name, score=0.0, normalized_score=0.0, error="No transcript available")

            # One prompt, built once: every repeat sends byte-identical content,
            # so rating variance measures the judge, not the request.
            prompt = self.get_judge_prompt(conversation_turns=transcript_text)

            valid_range = set(range(self.rating_scale[0], self.rating_scale[1] + 1))
            ratings: list[int] = []
            invalid_ratings = 0
            parse_failures = 0

            for _ in range(self.repeats):
                response, _raw_response = await self.call_judge(prompt, context)
                if response is None:
                    parse_failures += 1
                    continue
                rating = response.get("rating")
                if isinstance(rating, str):
                    try:
                        rating = int(rating)
                    except ValueError:
                        pass
                if isinstance(rating, int) and rating in valid_range:
                    ratings.append(rating)
                else:
                    # An out-of-schema rating is readout noise too — the paper's
                    # label-to-meaning failure mode — so count it, don't drop it.
                    invalid_ratings += 1

            details: dict[str, Any] = {
                "judge_prompt": prompt,
                "judge_model": self.llm_client.model,
                "num_repeats": self.repeats,
                "num_valid": len(ratings),
                "num_parse_failures": parse_failures,
                "num_invalid_ratings": invalid_ratings,
                "per_repeat_ratings": ratings,
            }

            if not ratings:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No judge call returned a usable rating",
                    details=details,
                )

            histogram = {rating: ratings.count(rating) for rating in sorted(set(ratings))}
            modal_rating, modal_count = max(histogram.items(), key=lambda item: item[1])
            agreement_rate = modal_count / len(ratings)
            min_r, max_r = self.rating_scale
            noise_floor = (max(ratings) - min(ratings)) / (max_r - min_r)

            details.update(
                {
                    "rating_histogram": histogram,
                    "modal_rating": modal_rating,
                    "distinct_ratings": len(histogram),
                    "agreement_rate": round(agreement_rate, 3),
                    # Smallest gap (normalized units) this judge resolves on this
                    # record: between-model differences below it are noise.
                    "noise_floor_normalized": round(noise_floor, 3),
                    "parse_stability_rate": round(len(ratings) / self.repeats, 3),
                }
            )

            return MetricScore(
                name=self.name,
                score=round(agreement_rate, 3),
                normalized_score=round(agreement_rate, 3),
                details=details,
            )

        except Exception as e:
            return self._handle_error(e, context)
