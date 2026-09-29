"""Tests for string-verified adjudication of judge evidence citations.

Unit tests cover citation extraction, verbatim verification and the
adjudication report. Integration tests run the real conversation-level judge
metrics (with a mocked judge LLM) to confirm the wiring in
ConversationTextJudgeMetric.compute attaches the report and the
``evidence_grounding`` sub-metric to every metric score.

Async compute() is driven with asyncio.run() rather than pytest-asyncio so
the tests exercise the exact production path without an event-loop plugin.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.evidence_grounding import (
    VERDICT_FABRICATED,
    VERDICT_GROUNDED,
    VERDICT_UNQUOTED,
    adjudicate_evidence,
    attach_evidence_grounding,
    extract_citations,
    verify_citation,
)
from eva.metrics.experience.conversation_progression import ConversationProgressionJudgeMetric
from eva.models.results import MetricScore
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

_TRANSCRIPT = "Turn 0:\n  user: I want to book flight ABC123 to Rome.\n  assistant: Done, booked."


def _faithfulness_response(evidence_by_dimension: dict[str, dict[str, object]], flagged: bool = True) -> str:
    """Build a faithfulness-shaped judge response with the given evidence texts."""
    dimensions = {
        "fabricating_tool_parameters": {"flagged": flagged, "rating": 2 if flagged else 3, "evidence": "clean"},
        "hallucination": {"flagged": False, "rating": 3, "evidence": "no unsupported claims"},
    }
    dimensions.update(evidence_by_dimension)
    return json.dumps({"rating": 2, "dimensions": dimensions})


class TestExtractCitations:
    def test_extracts_straight_double_quotes(self):
        assert extract_citations('The user said "book it now" please.') == ["book it now"]

    def test_extracts_curly_quotes_and_guillemets(self):
        text = "Judge cited “flight ABC123” and «Rome»."
        assert extract_citations(text) == ["flight ABC123", "Rome"]

    def test_ignores_apostrophes_inside_words(self):
        assert extract_citations("The user's don't") == []

    def test_extracts_single_quoted_span(self):
        assert extract_citations("He said 'book it now' twice.") == ["book it now"]

    def test_ignores_very_short_spans(self):
        assert extract_citations('He said "ok" and left.') == []


class TestVerifyCitation:
    def test_verbatim_citation_verifies(self):
        assert verify_citation("book flight ABC123 to Rome", _TRANSCRIPT) is True

    def test_case_and_whitespace_insensitive(self):
        assert verify_citation("Book   FLIGHT abc123 to rome.", _TRANSCRIPT) is True

    def test_missing_fragment_fails(self):
        assert verify_citation("book flight ABC123 to Paris", _TRANSCRIPT) is False

    def test_ellipsis_fragments_all_required(self):
        assert verify_citation("I want to book ... to Rome.", _TRANSCRIPT) is True
        assert verify_citation("I want to book ... to Paris.", _TRANSCRIPT) is False

    def test_quote_style_differences_are_tolerated(self):
        transcript = "user: I want to book “flight ABC123” to Rome."
        assert verify_citation("flight ABC123", transcript) is True


class TestAdjudicateEvidence:
    def _response(self):
        return {
            "dimensions": {
                "fabricating_tool_parameters": {
                    "flagged": True,
                    "evidence": 'The assistant booked "flight ABC123 to Rome" unprompted.',
                },
                "hallucination": {
                    "flagged": True,
                    "evidence": "The assistant invented a loyalty program without citing anything.",
                },
                "violating_policies": {"flagged": False, "evidence": "clean"},
                "failing_to_disambiguate": {"flagged": True, "evidence": 'Offered "a free upgrade to first class".'},
            }
        }

    def test_per_code_verdicts(self):
        report = adjudicate_evidence(self._response(), _TRANSCRIPT)
        codes = report["codes"]
        assert codes["fabricating_tool_parameters"]["verdict"] == VERDICT_GROUNDED
        assert codes["hallucination"]["verdict"] == VERDICT_UNQUOTED
        assert codes["failing_to_disambiguate"]["verdict"] == VERDICT_FABRICATED
        assert "violating_policies" not in codes  # clean codes are not adjudicated

    def test_summary_counts(self):
        report = adjudicate_evidence(self._response(), _TRANSCRIPT)
        assert report["codes_flagged"] == 3
        assert report["codes_grounded"] == 1
        assert report["codes_unquoted"] == 1
        assert report["codes_fabricated"] == 1
        assert report["citations_total"] == 2
        assert report["citations_verified"] == 1
        assert report["citations_grounding_rate"] == 0.5

    def test_ignores_reasoning_fields(self):
        # user_behavioral_fidelity's "analysis" fields hold reasoning, not citations.
        response = {"corruption_analysis": {"extra_modifications": {"detected": True, "analysis": "did it"}}}
        assert adjudicate_evidence(response, _TRANSCRIPT)["codes_flagged"] == 0

    def test_detected_flag_field_adjudicates(self):
        response = {"dimensions": {"some_code": {"detected": True, "evidence": 'cites "book flight ABC123"'}}}
        report = adjudicate_evidence(response, _TRANSCRIPT)
        assert report["codes"]["some_code"]["verdict"] == VERDICT_GROUNDED

    def test_no_flagged_codes_yields_empty_report(self):
        response = {"dimensions": {"hallucination": {"flagged": False, "evidence": "clean"}}}
        report = adjudicate_evidence(response, _TRANSCRIPT)
        assert report["codes_flagged"] == 0
        assert report["citations_grounding_rate"] is None


class TestAttachEvidenceGrounding:
    def test_noop_when_nothing_flagged(self):
        score = MetricScore(name="faithfulness", score=3.0, normalized_score=1.0)
        attach_evidence_grounding(score, {"dimensions": {}}, _TRANSCRIPT)
        assert "evidence_grounding" not in score.details
        assert score.sub_metrics is None

    def test_merges_sub_metric_without_clobbering_existing(self):
        score = MetricScore(name="faithfulness", score=2.0, normalized_score=0.5)
        score.sub_metrics = {"hallucination_rate": MetricScore(name="faithfulness.hallucination_rate", score=1.0)}
        response = {"dimensions": {"hallucination": {"flagged": True, "evidence": 'said "to Rome"'}}}
        attach_evidence_grounding(score, response, _TRANSCRIPT)

        assert score.details["evidence_grounding"]["codes"]["hallucination"]["verdict"] == VERDICT_GROUNDED
        assert set(score.sub_metrics) == {"hallucination_rate", "evidence_grounding"}
        grounding = score.sub_metrics["evidence_grounding"]
        assert grounding.name == "faithfulness.evidence_grounding"
        assert grounding.score == 1.0
        assert grounding.details["citations_total"] == 1


class TestFaithfulnessComputeWiring:
    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)

    def _compute(self, judge_response: str, trace: list[dict[str, object]] | None = None) -> MetricScore:
        self.metric.llm_client.generate_text.return_value = (judge_response, None)
        ctx = make_metric_context(
            conversation_trace=trace
            or [
                {"role": "user", "content": "I want to book flight ABC123 to Rome."},
                {"role": "assistant", "content": "Done, booked."},
            ],
        )
        return asyncio.run(self.metric.compute(ctx))

    def test_grounded_citation_adds_sub_metric(self):
        response = _faithfulness_response(
            {
                "fabricating_tool_parameters": {
                    "flagged": True,
                    "rating": 2,
                    "evidence": 'The assistant booked "flight ABC123 to Rome" without confirming.',
                }
            }
        )
        score = self._compute(response)

        report = score.details["evidence_grounding"]
        assert report["codes"]["fabricating_tool_parameters"]["verdict"] == VERDICT_GROUNDED
        assert report["codes_flagged"] == 1
        # Existing dimension sub-metrics are preserved and the grounding rate is merged in.
        assert score.sub_metrics is not None
        assert "fabricating_tool_parameters_rate" in score.sub_metrics
        assert score.sub_metrics["evidence_grounding"].score == 1.0

    def test_fabricated_citation_lowers_grounding_rate(self):
        response = _faithfulness_response(
            {
                "fabricating_tool_parameters": {
                    "flagged": True,
                    "rating": 2,
                    "evidence": 'The assistant booked "flight ABC123 to Rome" and mentioned "a free upgrade".',
                }
            }
        )
        score = self._compute(response)

        report = score.details["evidence_grounding"]
        assert report["codes"]["fabricating_tool_parameters"]["verdict"] == VERDICT_FABRICATED
        assert score.sub_metrics["evidence_grounding"].score == 0.5

    def test_unquoted_flag_reports_without_sub_metric(self):
        response = _faithfulness_response(
            {"fabricating_tool_parameters": {"flagged": True, "rating": 2, "evidence": "Repeated the same question."}}
        )
        score = self._compute(response)

        report = score.details["evidence_grounding"]
        assert report["codes"]["fabricating_tool_parameters"]["verdict"] == VERDICT_UNQUOTED
        assert "evidence_grounding" not in (score.sub_metrics or {})

    def test_all_clean_dimensions_leaves_details_untouched(self):
        score = self._compute(_faithfulness_response({}, flagged=False))
        assert "evidence_grounding" not in score.details

    def test_verification_can_be_disabled_by_config(self):
        metric = FaithfulnessJudgeMetric(config={"verify_evidence": False})
        metric.llm_client = MagicMock()
        metric.llm_client.generate_text = AsyncMock()
        metric.llm_client.params = {}
        metric.llm_client.generate_text.return_value = (
            _faithfulness_response(
                {"fabricating_tool_parameters": {"flagged": True, "rating": 2, "evidence": 'quoted "nothing real"'}}
            ),
            None,
        )
        ctx = make_metric_context(conversation_trace=[{"role": "user", "content": "hello"}])
        score = asyncio.run(metric.compute(ctx))

        assert score.score == 2.0
        assert "evidence_grounding" not in score.details


class TestConversationProgressionComputeWiring:
    def test_shared_base_class_wiring(self):
        metric = make_judge_metric(ConversationProgressionJudgeMetric, mock_llm=True)
        metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 2,
                    "dimensions": {
                        "redundant_statements": {
                            "flagged": True,
                            "rating": 2,
                            "evidence": 'The assistant asked "I want to book flight ABC123" twice.',
                        }
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[{"role": "user", "content": "I want to book flight ABC123 to Rome."}],
        )
        score = asyncio.run(metric.compute(ctx))

        report = score.details["evidence_grounding"]
        assert report["codes"]["redundant_statements"]["verdict"] == VERDICT_GROUNDED
        assert score.sub_metrics["evidence_grounding"].name == "conversation_progression.evidence_grounding"
