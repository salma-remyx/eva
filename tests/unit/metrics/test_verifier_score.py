"""Tests for continuous verifier scoring (LLM-as-a-Verifier, arXiv:2607.05391).

Covers both the pure scoring functions and their integration into the shared
``TextJudgeMetric`` scoring path, exercised through the real
``FaithfulnessJudgeMetric`` metric.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.verifier_score import expected_rating, normalize_expected_rating
from tests.unit.metrics.conftest import make_metric_context


def _logprob_tokens(distribution: dict[int, float], prefix: str = '{"rating": '):
    """Build fake per-token logprob content.

    A prefix token that contains the ``rating`` anchor, then a scoring token
    carrying the given distribution.
    """
    sampled = max(distribution, key=distribution.get)
    top = [{"token": str(r), "logprob": math.log(p)} for r, p in distribution.items()]
    return [
        {"token": prefix, "top_logprobs": []},
        {"token": str(sampled), "top_logprobs": top},
    ]


class TestExpectedRating:
    def test_expectation_over_distribution(self):
        # E = 1*0.2 + 2*0.5 + 3*0.3 = 2.1
        tokens = _logprob_tokens({1: 0.2, 2: 0.5, 3: 0.3})
        assert expected_rating(tokens, 1, 3) == pytest.approx(2.1)

    def test_renormalizes_over_valid_tokens_only(self):
        # Non-rating mass (a quote token) is dropped and the rest renormalized.
        tokens = [
            {"token": '{"rating": ', "top_logprobs": []},
            {
                "token": "3",
                "top_logprobs": [
                    {"token": '"', "logprob": math.log(0.5)},  # non-rating: ignored
                    {"token": "1", "logprob": math.log(0.1)},
                    {"token": "3", "logprob": math.log(0.4)},
                ],
            },
        ]
        # Renormalized over {1: 0.1, 3: 0.4} -> {1: 0.2, 3: 0.8}; E = 1*0.2 + 3*0.8 = 2.6
        assert expected_rating(tokens, 1, 3) == pytest.approx(2.6)

    def test_point_mass_when_no_alternatives(self):
        tokens = [{"token": '{"rating": ', "top_logprobs": []}, {"token": "2", "top_logprobs": []}]
        assert expected_rating(tokens, 1, 3) == 2.0

    def test_anchor_skips_earlier_digits(self):
        # A digit appearing before the "rating" key must not be treated as the score.
        tokens = [
            {"token": '{"turn": ', "top_logprobs": []},
            {"token": "7", "top_logprobs": [{"token": "7", "logprob": 0.0}]},
            {"token": ', "rating": ', "top_logprobs": []},
            {
                "token": "1",
                "top_logprobs": [{"token": "1", "logprob": math.log(0.9)}, {"token": "2", "logprob": math.log(0.1)}],
            },
        ]
        assert expected_rating(tokens, 1, 3) == pytest.approx(1.1)

    def test_returns_none_without_logprobs(self):
        assert expected_rating(None, 1, 3) is None
        assert expected_rating([], 1, 3) is None

    def test_returns_none_when_no_rating_token(self):
        tokens = [{"token": "no digits here", "top_logprobs": []}]
        assert expected_rating(tokens, 1, 3) is None


class TestNormalizeExpectedRating:
    def test_matches_endpoints(self):
        assert normalize_expected_rating(1.0, 1, 3) == 0.0
        assert normalize_expected_rating(3.0, 1, 3) == 1.0

    def test_continuous_midpoint(self):
        assert normalize_expected_rating(2.1, 1, 3) == pytest.approx(0.55)

    def test_clamps_out_of_range(self):
        assert normalize_expected_rating(3.2, 1, 3) == 1.0
        assert normalize_expected_rating(0.5, 1, 3) == 0.0

    def test_degenerate_scale(self):
        assert normalize_expected_rating(1.0, 1, 1) == 1.0


class TestTextJudgeMetricIntegration:
    """Exercise the wiring edit in eva.metrics.base.TextJudgeMetric."""

    def test_validate_and_normalize_uses_expected_rating(self):
        metric = FaithfulnessJudgeMetric({"continuous_scoring": True})
        ctx = SimpleNamespace(record_id="rec-1")

        rating, normalized = metric.validate_and_normalize_rating({"rating": 2, "_expected_rating": 2.1}, ctx)

        # Discrete rating preserved for the raw score; normalized is continuous.
        assert rating == 2
        assert normalized == pytest.approx(0.55)

    def test_validate_and_normalize_falls_back_to_discrete(self):
        metric = FaithfulnessJudgeMetric({"continuous_scoring": True})
        ctx = SimpleNamespace(record_id="rec-1")

        rating, normalized = metric.validate_and_normalize_rating({"rating": 2}, ctx)

        assert rating == 2
        assert normalized == pytest.approx(0.5)  # (2-1)/(3-1)

    @pytest.mark.asyncio
    async def test_compute_produces_continuous_score(self):
        metric = FaithfulnessJudgeMetric({"continuous_scoring": True})
        metric.llm_client = MagicMock()
        metric.llm_client.params = {}
        metric.llm_client.generate_text = AsyncMock(
            return_value=(
                json.dumps({"rating": 2, "dimensions": {"hallucination": {"flagged": False}}}),
                {"logprobs": _logprob_tokens({1: 0.2, 2: 0.5, 3: 0.3})},
            )
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )

        score = await metric.compute(ctx)

        # Raw score stays the discrete rating; normalized is the continuous expectation.
        assert score.score == 2.0
        assert score.normalized_score == pytest.approx(0.55)
        # continuous_scoring on -> logprobs were requested from the judge call
        _, kwargs = metric.llm_client.generate_text.call_args
        assert kwargs["logprobs"] == metric.score_top_logprobs

    @pytest.mark.asyncio
    async def test_compute_discrete_when_scoring_disabled(self):
        metric = FaithfulnessJudgeMetric()  # continuous scoring off by default
        metric.llm_client = MagicMock()
        metric.llm_client.params = {}
        metric.llm_client.generate_text = AsyncMock(return_value=(json.dumps({"rating": 2, "dimensions": {}}), None))
        ctx = make_metric_context(
            conversation_trace=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        )

        score = await metric.compute(ctx)

        assert score.normalized_score == pytest.approx(0.5)
        _, kwargs = metric.llm_client.generate_text.call_args
        assert kwargs["logprobs"] is None
