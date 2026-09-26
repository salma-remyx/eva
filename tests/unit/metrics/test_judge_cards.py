"""Tests for the judge cards generated off the metric signature walk.

The cards must be derived from the same machinery the drift test uses —
`compute_all_metric_signatures` and the judge prompt templates — so a card can
never report a version or prompt hash that disagrees with the released state.
"""

from pathlib import Path

import pytest

from eva.metrics.judge_cards import (
    CONTEXT_CARD,
    EVALUATION_CARD,
    PIPELINE_CARD,
    PROTOTYPING_CARD,
    JudgeCard,
    audit_judge_card,
    audit_judge_cards,
    build_judge_cards,
    render_markdown,
)
from eva.metrics.signatures import compute_all_metric_signatures
from eva.metrics.versioning import hash_prompt_template
from eva.utils.prompt_manager import get_prompt_manager

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cards() -> dict[str, JudgeCard]:
    return build_judge_cards(REPO_ROOT)


@pytest.fixture(scope="module")
def signatures() -> dict[str, dict[str, str | None]]:
    return compute_all_metric_signatures()


def test_every_judge_metric_in_the_signature_walk_gets_a_card(
    cards: dict[str, JudgeCard],
    signatures: dict[str, dict[str, str | None]],
) -> None:
    """The card set is exactly the metrics the walk identifies as LLM judges."""
    judged_by_walk = {entry["name"] for entry in signatures.values() if entry["prompt_hash"] is not None}
    assert judged_by_walk
    assert set(cards) == judged_by_walk


def test_card_hashes_come_from_the_signature_walk(
    cards: dict[str, JudgeCard],
    signatures: dict[str, dict[str, str | None]],
) -> None:
    """Version, prompt and source hashes match the walk and the live prompt template."""
    for card in cards.values():
        recorded = signatures[card.class_name]
        assert card.version == recorded["version"]
        assert card.source_hash == recorded["source_hash"]
        assert card.prompt_hash == recorded["prompt_hash"]
        template = get_prompt_manager().get_template(card.prompt_key)
        assert card.prompt_template == template
        assert card.prompt_hash == hash_prompt_template(template)


def test_audio_judges_pin_decoding_and_text_judges_do_not(cards: dict[str, JudgeCard]) -> None:
    """The reproducibility gap is reported for judges that pin no temperature."""
    findings = audit_judge_cards(cards)
    unpinned = {name for name, issues in findings.items() if any(f.code == "decoding_unpinned" for f in issues)}
    assert unpinned == {name for name, card in cards.items() if card.judge_kind == "text_judge"}
    # Every audio judge ships temperature=0.0, so none of them are flagged.
    for card in cards.values():
        if card.judge_kind == "audio_judge":
            assert card.params["temperature"] == 0.0


def test_checked_in_validity_evidence_is_picked_up(cards: dict[str, JudgeCard]) -> None:
    """Judges with a human-labelled dataset and a development doc are not flagged for them."""
    findings = audit_judge_cards(cards)

    faithfulness = findings["faithfulness"]
    assert not any(f.code == "no_validity_evidence" for f in faithfulness)
    assert not any(f.code == "no_tuning_record" for f in faithfulness)
    assert cards["faithfulness"].validation_records

    # The transcription dataset is stored under a different stem than the metric.
    assert cards["transcription_accuracy_key_entities"].validation_dataset is not None
    assert not any(f.code == "no_validity_evidence" for f in findings["transcription_accuracy_key_entities"])


def test_audit_reports_every_missing_reporting_item() -> None:
    """A card with no evidence at all is flagged once per LLJ card item."""
    bare = JudgeCard(
        metric_name="some_judge",
        class_name="SomeJudgeMetric",
        category="experience",
        description="A judge with nothing behind it",
        judge_kind="text_judge",
        model="gpt-5.2",
        model_override_path='config["judge_model"] -> env JUDGE_MODEL -> class default',
        params={"max_tokens": 100},
        prompt_key="judge.some_judge.user_prompt",
        prompt_template="Rate the turn: {turn}",
        prompt_hash="0" * 12,
        rating_scale=None,
        version="v0.1",
        source_hash="0" * 12,
        validation_dataset=None,
        validation_records=None,
        development_doc=None,
        metric_doc=None,
    )
    findings = audit_judge_card(bare)
    assert [(f.card, f.code) for f in findings] == [
        (EVALUATION_CARD, "no_validity_evidence"),
        (PROTOTYPING_CARD, "no_tuning_record"),
        (PIPELINE_CARD, "decoding_unpinned"),
        (PIPELINE_CARD, "scale_undeclared"),
        (CONTEXT_CARD, "metric_undocumented"),
    ]


def test_markdown_discloses_the_pipeline_and_the_gaps(cards: dict[str, JudgeCard]) -> None:
    """The rendered card carries the judge spec, the prompt, and any gap by card."""
    document = render_markdown(cards, audit_judge_cards(cards))

    assert "## faithfulness" in document
    assert "`us.anthropic.claude-opus-4-6-v1`" in document
    assert "judge.faithfulness.user_prompt" in document
    # The unrendered prompt is published in full, not just hashed.
    assert get_prompt_manager().get_template("judge.faithfulness.user_prompt").strip() in document
    assert "**LLJ Pipeline · Model**" in document
    assert "**LLJ Evaluation · Validity & reliability**" in document
    assert "`decoding_unpinned` (LLJ Pipeline)" in document
