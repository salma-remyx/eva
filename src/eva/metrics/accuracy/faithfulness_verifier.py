"""Faithfulness metric scored as a continuous LLM-as-a-Verifier score.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). Where :class:`~eva.metrics.accuracy.faithfulness.FaithfulnessJudgeMetric`
elicits a discrete integer rating, this metric asks the judge to emit a rating
token and scores faithfulness as the expectation over that token's logprob
distribution (see :func:`eva.metrics.verifier_scoring.expectation_score_from_logprobs`).
The result is a continuous, finer-grained faithfulness score for S2S / cascade
voice agents — the paper's "score granularity" scaling axis. Setting
``n_samples > 1`` enables the paper's "repeated evaluation" axis (the mean of
independent draws).

This metric is deliberately unversioned and excluded from the default metric set:
it is an opt-in experimental metric for comparing verifier-score granularity
against the discrete faithfulness judge. Promote it to a versioned, drift-tracked
metric — regenerating ``tests/fixtures/metric_signatures.json`` and providing a
``judge.faithfulness_verifier.user_prompt`` template — once the prompt stabilizes.
"""

import json
from statistics import mean
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext, MetricType
from eva.metrics.registry import register_metric
from eva.metrics.utils import parse_judge_response, validate_rating
from eva.metrics.verifier_scoring import (
    VerifierDistribution,
    call_judge_with_logprobs,
    expectation_score_from_logprobs,
)
from eva.models.results import MetricScore

# The model emits the rating as the very first token, then explains itself in
# compact JSON. Reading that leading token's logprobs yields the verifier
# distribution over the rating scale. Braces in the JSON example are escaped so
# ``str.format`` renders the surrounding template variables verbatim.
_VERIFIER_PROMPT = """\
You are evaluating a voice agent's faithfulness in a single conversation.

Faithfulness means the assistant stays grounded in the information, tool results,
and policies available to it: no hallucination, no fabricated tool parameters, no
misrepresentation of results, and proper disambiguation of ambiguous or
perception-prone user input (names, codes, numbers, write / irreversible actions).

Agent role: {agent_role}
Agent instructions: {agent_instructions}
Available tools: {available_tools}

Conversation:
{conversation_trace}

Rate the assistant's overall faithfulness on this scale:
  1 = serious violations (hallucination, fabrication, policy breach)
  2 = partial / mixed faithfulness
  3 = faithful

Respond with the rating as the very first character — a single digit (1, 2, or 3)
and nothing else before it — followed by a newline, then a compact JSON object:
{{"rating": <1|2|3>, "evidence": "<one short sentence>"}}
"""


@register_metric
class FaithfulnessVerifierMetric(ConversationTextJudgeMetric):
    """Continuous, logprob-based faithfulness judge (LLM-as-a-Verifier).

    Unlike the discrete :class:`FaithfulnessJudgeMetric`, the score is the
    expected rating over the judge's scoring-token logprob distribution, so two
    conversations that both round to "3" can still be distinguished by how
    confidently the judge placed them there.
    """

    name = "faithfulness_verifier"
    description = "LLM-as-a-Verifier continuous faithfulness score (logprob expectation)"
    category = "accuracy"
    metric_type = MetricType.TEXT_JUDGE
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params: dict[str, Any] = {"max_tokens": 1024}
    rating_scale = (1, 3)
    higher_is_better = True
    exclude_from_default_metrics = True
    # Deliberately no `version`: see module docstring (excluded from the drift fixture).

    # Paper scaling axes: ``top_logprobs`` = score granularity (distribution
    # resolution); ``n_samples`` = repeated evaluation (mean of independent draws).
    top_logprobs: int = 20
    n_samples: int = 1

    async def compute(self, context: MetricContext) -> MetricScore:
        """Score faithfulness as the expected rating over the judge's logprobs."""
        try:
            transcript_text = self.format_transcript(context)
            if not transcript_text:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No transcript available",
                )
            prompt = _VERIFIER_PROMPT.format(**self.get_prompt_variables(context, transcript_text))
            return await self._score_with_verifier(prompt, context)
        except Exception as e:
            return self._handle_error(e, context)

    async def _score_with_verifier(self, prompt: str, context: MetricContext) -> MetricScore:
        """Run the verifier judge ``n_samples`` times and aggregate the scores."""
        scale = list(range(self.rating_scale[0], self.rating_scale[1] + 1))
        expectations: list[float] = []
        probability_totals: dict[int, float] = dict.fromkeys(scale, 0.0)
        last_text = ""

        for _ in range(max(1, self.n_samples)):
            response = await call_judge_with_logprobs(
                model=self.llm_client.model,
                prompt=prompt,
                params=self.llm_client.params,
                timeout=self.llm_client.timeout,
                top_logprobs=self.top_logprobs,
            )
            self._log_token_usage(
                context,
                self.llm_client.model,
                self.llm_client.params,
                prompt,
                response.usage,
                response.text,
            )
            last_text = response.text

            distribution = expectation_score_from_logprobs(response.logprobs, self.rating_scale)
            if distribution is None:
                distribution = self._discrete_fallback(response.text, context)
            expectations.append(distribution.expectation)
            for value, probability in distribution.probabilities.items():
                probability_totals[value] = probability_totals.get(value, 0.0) + probability

        samples = len(expectations)
        expectation = mean(expectations)
        low, high = self.rating_scale
        # Mirrors eva.metrics.utils.normalize_rating, inlined because the verifier
        # score is a continuous float rather than the int that helper is typed for.
        normalized = (expectation - low) / (high - low) if high != low else 1.0
        averaged_distribution = {str(value): round(probability_totals.get(value, 0.0) / samples, 6) for value in scale}

        return MetricScore(
            name=self.name,
            score=round(expectation, 4),
            normalized_score=round(normalized, 4),
            details={
                "scoring_method": "llm_as_a_verifier_logprob_expectation",
                "expectation": round(expectation, 4),
                "n_samples": samples,
                "rating_scale": scale,
                "probability_distribution": averaged_distribution,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": last_text,
            },
        )

    def _discrete_fallback(self, text: str, context: MetricContext) -> VerifierDistribution:
        """Fall back to a discrete rating when no scoring-token logprobs are usable."""
        parsed = parse_judge_response(text, context.record_id, self.logger)
        rating = validate_rating(
            parsed.get("rating") if parsed else None,
            list(range(self.rating_scale[0], self.rating_scale[1] + 1)),
            default=self.rating_scale[0],
            record_id=context.record_id,
            metric_logger=self.logger,
        )
        self.logger.info(
            "No usable scoring-token logprobs for %s; falling back to discrete rating %d.",
            context.record_id,
            rating,
        )
        return VerifierDistribution(
            probabilities={rating: 1.0},
            expectation=float(rating),
            scoring_token=str(rating),
            from_top_logprobs=False,
        )

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return the variables used to render the verifier prompt."""
        return {
            "agent_role": context.agent_role,
            "agent_instructions": context.agent_instructions,
            "available_tools": json.dumps(context.agent_tools, indent=4),
            "conversation_trace": transcript_text,
        }
