"""Judge cards: structured reporting for EVA's LLM judges.

Adapted from "LLJ Cards: Best practices for the Use of LLMs as Judges"
(Chehbouni et al., arXiv:2609.24516), which organizes LLM-as-judge reporting
into five cards — Context, Evaluation Criteria, LLJ Pipeline, LLJ Evaluation
and LLJ Prototyping — so an automated evaluation can be checked for validity,
reliability and reproducibility instead of taken on trust.

EVA already records most of what those cards ask for, but scattered across
metric class attributes, `configs/prompts/judge.yaml`, the per-metric docs and
the metric signature walk. This module gathers it into one card per judge
metric and audits each card against the paper's reporting items, so a judge
with thin evidence is named by `scripts/generate_judge_cards.py` rather than
discovered at review time.

Items the paper fills with a human study (inter-rater agreement with experts,
API budget, tuning stopping criterion) are not invented here: they are read
from the checked-in evidence when it exists, and reported as a gap when it
does not.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eva.metrics.base import AudioJudgeMetric, TextJudgeMetric
from eva.metrics.signatures import _all_concrete_versioned_metric_classes, compute_all_metric_signatures
from eva.utils.prompt_manager import get_prompt_manager

# The five LLJ cards. Evaluation Criteria is filled by each metric's rubric
# doc rather than generated here, so it has no audit item of its own.
CONTEXT_CARD = "Context"
PIPELINE_CARD = "LLJ Pipeline"
EVALUATION_CARD = "LLJ Evaluation"
PROTOTYPING_CARD = "LLJ Prototyping"

# Validation datasets are named after the metric they validate, except where
# history picked a different stem.
_DATASET_STEM_ALIASES = {"transcription_accuracy_key_entities": "key_entity_transcription"}

_JudgeMetric = type[TextJudgeMetric] | type[AudioJudgeMetric]


@dataclass(frozen=True)
class JudgeCard:
    """The machine-checkable subset of the LLJ cards, for one judge metric."""

    metric_name: str
    class_name: str
    category: str
    description: str
    # LLJ Pipeline / Model
    judge_kind: str
    model: str
    model_override_path: str
    params: dict[str, Any]
    # LLJ Pipeline / Prompt
    prompt_key: str
    prompt_template: str
    prompt_hash: str | None
    # LLJ Pipeline / Score
    rating_scale: tuple[int, int] | None
    # Reproducibility
    version: str | None
    source_hash: str | None
    # LLJ Evaluation / Validity & reliability, plus Context
    validation_dataset: Path | None
    validation_records: int | None
    development_doc: Path | None
    metric_doc: Path | None


@dataclass(frozen=True)
class CardFinding:
    """A reporting item the card cannot back with evidence."""

    card: str
    code: str
    message: str


def _default_repo_root() -> Path:
    """Repo root, derived the same way PromptManager derives its prompts dir."""
    return Path(__file__).resolve().parents[3]


def _judge_metric_classes() -> dict[str, _JudgeMetric]:
    """Filter the signature walk down to LLM judges, keyed on class qualname.

    Reuses the walk the drift test uses so the card set can never disagree
    with tests/fixtures/metric_signatures.json.
    """
    judges: dict[str, _JudgeMetric] = {}
    for qualname, cls in _all_concrete_versioned_metric_classes().items():
        if issubclass(cls, TextJudgeMetric) or issubclass(cls, AudioJudgeMetric):
            judges[qualname] = cls
    return judges


def _model_override_path(cls: _JudgeMetric) -> str:
    """How the checked-in default judge model can be replaced at run time."""
    if issubclass(cls, AudioJudgeMetric):
        return 'config["audio_judge_model"] -> class default'
    return 'config["judge_model"] -> env JUDGE_MODEL -> class default'


def _dataset_path(repo_root: Path, metric_name: str) -> Path:
    stem = _DATASET_STEM_ALIASES.get(metric_name, metric_name)
    return repo_root / "docs" / "metrics" / "judge_validation_datasets" / f"{stem}.jsonl"


def _count_records(dataset: Path) -> int:
    """Count labelled records in a judge validation dataset."""
    with open(dataset, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def build_judge_cards(repo_root: Path | None = None) -> dict[str, JudgeCard]:
    """Build one card per judge metric, keyed on metric name.

    Version, prompt and source hashes come from `compute_all_metric_signatures`
    rather than being recomputed, so a card always matches the drift fixture.
    """
    root = repo_root or _default_repo_root()
    signatures = compute_all_metric_signatures()
    metrics_docs = root / "docs" / "metrics"
    prompt_manager = get_prompt_manager()

    cards: dict[str, JudgeCard] = {}
    for qualname, cls in _judge_metric_classes().items():
        signature = signatures[qualname]
        prompt_key = f"judge.{cls.name}.user_prompt"
        dataset = _dataset_path(root, cls.name)
        has_dataset = dataset.exists()
        cards[cls.name] = JudgeCard(
            metric_name=cls.name,
            class_name=qualname,
            category=cls.category,
            description=cls.description,
            judge_kind=cls.metric_type.value,
            model=cls.default_model,
            model_override_path=_model_override_path(cls),
            params=dict(cls.default_params),
            prompt_key=prompt_key,
            prompt_template=prompt_manager.get_template(prompt_key),
            prompt_hash=signature["prompt_hash"],
            rating_scale=getattr(cls, "rating_scale", None),
            version=signature["version"],
            source_hash=signature["source_hash"],
            validation_dataset=dataset if has_dataset else None,
            validation_records=_count_records(dataset) if has_dataset else None,
            development_doc=_existing(
                [
                    metrics_docs / "metric_development" / f"{cls.name}_development.md",
                    metrics_docs / "metric_development" / f"{cls.name}.md",
                ]
            ),
            metric_doc=_existing([metrics_docs / f"{cls.name}.md"]),
        )
    return cards


def audit_judge_card(card: JudgeCard) -> list[CardFinding]:
    """Report the LLJ card items this judge cannot back with evidence."""
    findings: list[CardFinding] = []

    if card.validation_dataset is None:
        findings.append(
            CardFinding(
                EVALUATION_CARD,
                "no_validity_evidence",
                "no human-labelled validation dataset in docs/metrics/judge_validation_datasets/ — "
                "agreement with human labels is unmeasured",
            )
        )
    if card.development_doc is None:
        findings.append(
            CardFinding(
                PROTOTYPING_CARD,
                "no_tuning_record",
                "no metric_development doc — the prompt-tuning phase and its stopping criterion are unpublished",
            )
        )
    if "temperature" not in card.params:
        findings.append(
            CardFinding(
                PIPELINE_CARD,
                "decoding_unpinned",
                "default_params pins no temperature, so judge sampling is not reproducible "
                "across reruns of the same record",
            )
        )
    if card.rating_scale is None:
        findings.append(
            CardFinding(
                PIPELINE_CARD,
                "scale_undeclared",
                "no rating_scale on the class — how a raw judge rating becomes a score is unstated",
            )
        )
    if card.metric_doc is None:
        findings.append(
            CardFinding(
                CONTEXT_CARD,
                "metric_undocumented",
                f"no docs/metrics/{card.metric_name}.md — task and construct are undocumented",
            )
        )
    return findings


def audit_judge_cards(cards: dict[str, JudgeCard]) -> dict[str, list[CardFinding]]:
    """Audit every card, keyed on metric name."""
    return {name: audit_judge_card(card) for name, card in cards.items()}


def _evidence_line(label: str, path: Path | None) -> str:
    """Render one evidence entry, marking it missing when absent."""
    if path is None:
        return f"- {label}: _missing_"
    return f"- {label}: `{path.name}`"


def render_markdown(cards: dict[str, JudgeCard], findings: dict[str, list[CardFinding]]) -> str:
    """Render cards and their audit findings as a Markdown document."""
    lines = [
        "# Judge Cards",
        "",
        "One card per LLM judge in EVA, following the LLJ Cards reporting items",
        "(Context, LLJ Pipeline, LLJ Evaluation, LLJ Prototyping). Regenerate with",
        "`python scripts/generate_judge_cards.py`.",
        "",
        "| Metric | Judge | Model | Version | Validation records | Findings |",
        "|---|---|---|---|---|---|",
    ]
    for name, card in sorted(cards.items()):
        count = "—" if card.validation_records is None else str(card.validation_records)
        issues = "none" if not findings[name] else ", ".join(f.code for f in findings[name])
        lines.append(
            f"| [`{name}`](#{name.replace('_', '-')}) | {card.judge_kind} | `{card.model}` | {card.version} | {count} | {issues} |"
        )

    for name, card in sorted(cards.items()):
        scale = "undeclared" if card.rating_scale is None else f"{card.rating_scale[0]}–{card.rating_scale[1]}"
        params = ", ".join(f"{key}={value!r}" for key, value in sorted(card.params.items())) or "none"
        lines += [
            "",
            f"## {name}",
            "",
            f"_{card.description}_ (`{card.class_name}`, category `{card.category}`)",
            "",
            f"**{PIPELINE_CARD} · Model** — `{card.model}`, resolved as {card.model_override_path}.",
            f"Params: {params}.",
            "",
            f"**{PIPELINE_CARD} · Prompt** — `{card.prompt_key}` (`prompt_hash` `{card.prompt_hash}`).",
            "",
            "```text",
            card.prompt_template.rstrip(),
            "```",
            "",
            f"**{PIPELINE_CARD} · Score** — rating scale {scale}.",
            "",
            f"**Reproducibility** — `version` {card.version}, `source_hash` `{card.source_hash}`, "
            f"`prompt_hash` `{card.prompt_hash}`.",
            "",
            f"**{EVALUATION_CARD} · Validity & reliability**",
            _evidence_line("human-labelled dataset", card.validation_dataset),
            _evidence_line("development record", card.development_doc),
            "",
            f"**{CONTEXT_CARD}**",
            _evidence_line("metric doc", card.metric_doc),
        ]
        if findings[name]:
            lines += ["", "**Gaps**"]
            lines += [f"- `{finding.code}` ({finding.card}): {finding.message}" for finding in findings[name]]

    return "\n".join(lines) + "\n"
