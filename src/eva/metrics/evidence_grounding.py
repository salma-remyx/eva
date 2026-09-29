"""String-verified adjudication of LLM-judge evidence citations.

Adapted from the assay discipline of "Low-Cost Assays for Measuring Model
Behavior Across Vendors and Releases" (arXiv:2609.30012): a codebook label
applied by an LLM judge is only trusted when it cites verbatim transcript
substrings, the citation is machine-verified against the transcript
("string-verified adjudication"), and the verification rate is reported per
codebook code.

EVA's conversation-level judge metrics (faithfulness, conversation_progression)
already ask the judge to justify every flagged dimension with an ``evidence``
field citing the transcript. This module closes that loop: it extracts the
quoted citations from each flagged dimension's evidence, checks that every
citation is a (case- and whitespace-insensitive) substring of the transcript
the judge saw, and records a per-dimension verdict:

- ``grounded`` — every cited quote appears in the transcript.
- ``fabricated`` — at least one cited quote does not appear in the transcript.
- ``unquoted`` — the dimension was flagged but its evidence quotes nothing,
  so there is nothing to verify.

Clean (unflagged) dimensions are not adjudicated: the judge prompts ask for a
*reason* there, not a conviction, and reasons are not expected to be quotes.
Fields named ``analysis`` (e.g. user_behavioral_fidelity's corruption analysis)
hold reasoning rather than citations and are likewise skipped.

The per-record report lands in ``MetricScore.details["evidence_grounding"]``,
and an ``evidence_grounding`` sub-metric (share of cited quotes that verified)
lets the runner's generic aggregation report a run-level grounding rate per
judge metric — the role the paper's per-code judge/human agreement statistics
play for its assays, with the transcript itself as the zero-cost second
labeler.
"""

import re
from collections.abc import Iterator
from typing import Any

from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore

# Judge-response field that carries citation-style evidence. Fields holding
# free-form reasoning ("analysis") are deliberately not verified.
_EVIDENCE_FIELD = "evidence"
# Boolean fields that mark a codebook code as triggered (a "conviction").
_FLAG_FIELDS = ("flagged", "detected")

# Verdicts for a flagged code, once its evidence has been string-verified.
VERDICT_GROUNDED = "grounded"
VERDICT_FABRICATED = "fabricated"
VERDICT_UNQUOTED = "unquoted"

# Fragments shorter than this (after normalization) are ignored: sub-word
# fragments match almost anywhere and carry no verification signal.
_MIN_FRAGMENT_CHARS = 3

# Quote characters that don't care whether the judge or the transcript used
# the curly or straight form, plus dash unification for the same reason.
_MATCH_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "\N{SOFT HYPHEN}": "",
    }
)

# Punctuation trimmed from the edges of a citation fragment before matching:
# judges routinely fold the surrounding sentence punctuation into a quote.
_FRAGMENT_TRIM_CHARS = " \t\n\r\"'.,;:!?…"

# Quoted spans: straight/curly double quotes, guillemets, and straight single
# quotes that look like quotations rather than apostrophes (word characters on
# either side rule out "don't" / "user's").
_CITATION_RE = re.compile(r'"([^"]{3,})"' r"|“([^”]{3,})”" r"|«([^»]{3,})»" r"|(?<!\w)'([^']{3,})'(?!\w)")


def _normalize_for_match(text: str) -> str:
    """Return the case- and whitespace-insensitive form used for verbatim matching."""
    collapsed = re.sub(r"\s+", " ", text.translate(_MATCH_TRANSLATION).casefold())
    return collapsed.strip()


def _citation_fragments(citation: str) -> list[str]:
    """Split a citation on ellipsis omissions and drop unusable fragments.

    A judge may cite "first part ... last part" to bridge omitted material;
    each side of the ellipsis is verified on its own.
    """
    fragments = []
    for raw in re.split(r"\.\.\.|…", citation):
        fragment = _normalize_for_match(raw).strip(_FRAGMENT_TRIM_CHARS)
        if len(fragment) >= _MIN_FRAGMENT_CHARS:
            fragments.append(fragment)
    return fragments


def extract_citations(text: str) -> list[str]:
    """Extract quoted citation spans from judge evidence text, in order of appearance.

    Apostrophes inside words ("don't", "user's") are not treated as quotes.

    Args:
        text: Evidence text produced by the judge.

    Returns:
        The quoted spans, without their delimiting quote characters.
    """
    citations: list[str] = []
    for match in _CITATION_RE.finditer(text):
        span = next((group for group in match.groups() if group is not None), None)
        if span:
            citations.append(span)
    return citations


def verify_citation(citation: str, transcript: str) -> bool:
    """Check that a cited span appears verbatim (modulo case/whitespace) in the transcript.

    Every fragment of the citation must be found; a single missing fragment
    means the judge quoted text that is not in the transcript.

    Args:
        citation: Quoted span from judge evidence (without delimiters).
        transcript: The transcript text the judge was shown.

    Returns:
        True when the citation has at least one usable fragment and every
        fragment is contained in the transcript.
    """
    fragments = _citation_fragments(citation)
    if not fragments:
        return False
    normalized_transcript = _normalize_for_match(transcript)
    return all(fragment in normalized_transcript for fragment in fragments)


def _iter_flagged_evidence_codes(node: Any, code: str | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(code, entry)`` for every flagged entry carrying an evidence field.

    The code is the key of the mapping holding the entry — the codebook
    dimension the judge labeled (e.g. ``dimensions.fabricating_tool_parameters``
    yields ``"fabricating_tool_parameters"``). Entries that are not flagged are
    skipped: their evidence is a reason, not a conviction.
    """
    if isinstance(node, dict):
        if code is not None and isinstance(node.get(_EVIDENCE_FIELD), str):
            if any(node.get(flag_field) is True for flag_field in _FLAG_FIELDS):
                yield code, node
        for key, value in node.items():
            child_code = key if isinstance(value, dict | list) else None
            yield from _iter_flagged_evidence_codes(value, child_code)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_flagged_evidence_codes(item, code)


def adjudicate_evidence(response: dict[str, Any], transcript: str) -> dict[str, Any]:
    """String-verify the evidence cited for every flagged code in a judge response.

    Args:
        response: Parsed judge response, with evidence-bearing entries nested
            at any depth (e.g. ``dimensions.<code>.evidence``).
        transcript: The transcript text the judge was shown.

    Returns:
        Adjudication report: per-code verdicts and citations under ``"codes"``,
        plus summary counts (codes flagged/grounded/fabricated/unquoted and
        citations total/verified).
    """
    codes: dict[str, Any] = {}
    citations_total = 0
    citations_verified = 0

    for code, entry in _iter_flagged_evidence_codes(response):
        citations = []
        for citation in extract_citations(entry[_EVIDENCE_FIELD]):
            verified = verify_citation(citation, transcript)
            citations.append({"quote": citation, "verified": verified})
            citations_total += 1
            citations_verified += int(verified)
        if not citations:
            verdict = VERDICT_UNQUOTED
        elif all(item["verified"] for item in citations):
            verdict = VERDICT_GROUNDED
        else:
            verdict = VERDICT_FABRICATED
        codes[code] = {"flagged": True, "verdict": verdict, "citations": citations}

    return {
        "codes": codes,
        "codes_flagged": len(codes),
        "codes_grounded": sum(1 for c in codes.values() if c["verdict"] == VERDICT_GROUNDED),
        "codes_fabricated": sum(1 for c in codes.values() if c["verdict"] == VERDICT_FABRICATED),
        "codes_unquoted": sum(1 for c in codes.values() if c["verdict"] == VERDICT_UNQUOTED),
        "citations_total": citations_total,
        "citations_verified": citations_verified,
        "citations_grounding_rate": round(citations_verified / citations_total, 3) if citations_total else None,
    }


def attach_evidence_grounding(score: MetricScore, response: dict[str, Any], transcript: str) -> None:
    """Adjudicate a judge metric score's cited evidence, in place.

    Writes the per-code adjudication report into ``score.details`` and, when
    the judge cited anything, merges an ``evidence_grounding`` sub-metric
    (share of citations verified) into ``score.sub_metrics`` so the runner's
    generic aggregation reports a run-level grounding rate. Everything else
    about the score is left untouched — this is a reliability diagnostic, not
    a reweighting.

    Args:
        score: MetricScore produced by ``build_metric_score``.
        response: The parsed judge response it was built from.
        transcript: The transcript text the judge was shown.
    """
    report = adjudicate_evidence(response, transcript)
    if not report["codes_flagged"]:
        return

    score.details["evidence_grounding"] = report
    if not report["citations_total"]:
        return

    sub_metric = make_rate_sub_metric(
        parent_name=score.name,
        key="evidence_grounding",
        numerator=int(report["citations_verified"]),
        denominator=int(report["citations_total"]),
        details={key: report[key] for key in report if key != "codes"},
    )
    if score.sub_metrics is None:
        score.sub_metrics = {}
    score.sub_metrics["evidence_grounding"] = sub_metric
