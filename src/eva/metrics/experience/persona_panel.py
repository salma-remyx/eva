"""Persona-panel LLM-as-judge metric (conversation-level).

Adapted (Mode 2) from "Beyond a Single Judge: Simulating Social Persona Panels
for Generative UI Evaluation" (ESPP, arxiv:2607.28439v1).

ESPP replaces a single, rater-variant LLM judge with a psychologically diverse
panel that (1) independently rates, (2) exchanges opinions under a
trait-derived, semantically-gated *bounded-confidence* dynamic, and (3)
aggregates via Delphi-inspired social weighting. Against a single judge this
lifts Pearson r with human judgment from 0.716 to 0.922; a plain
prompt-ensemble recovers only ~1/3 of that gap, isolating the persona +
exchange + Delphi core as the dominant source of improvement.

Mode 2 substitutions (the GenUI-specific pillars do not map to a voice-AI
transcript benchmark, so they are dropped and auxiliaries are target-nativized):

* The multimodal *screenshot* rating surface -> EVA's existing text transcript
  judge (text-only, same transcript-in / rating-out contract as ``conciseness``).
* The paper's *learned* semantic-confidence gate -> a parameter-free
  token-overlap (Jaccard) proxy over panelist reasoning -- the canonical
  substitution called out for adapted ports.
* The text-level re-deliberation rounds -> deterministic trait-driven numeric
  opinion dynamics (the bounded-confidence update itself), keeping cost at one
  LLM call per panelist rather than one per panelist per round.

The portable core is preserved at full fidelity: trait-diverse personas,
bounded-confidence opinion exchange, and Delphi social weighting. Calibration
against human ratings belongs in a downstream PR.
"""

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import format_transcript_with_tools
from eva.models.results import MetricScore

# Words that carry little signal for the bounded-confidence semantic gate.
_STOPWORDS = frozenset(
    """a an the and or but if then else of to in on for with without at by from as is are was were
    be been being this that these those it its their his her our your my we you they he she i
    do does did not no yes so very can could should would will may might must about into over
    than there here when where which who whom whose what why how""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PersonaProfile:
    """A psychologically diverse panelist.

    ``openness`` drives the bounded-confidence step size (how receptive the
    panelist is to moving toward neighbors' opinions); ``expertise`` is the
    base Delphi social weight (how much the final opinion counts).
    """

    name: str
    framing: str
    openness: float  # receptiveness to opinion exchange, 0..1
    expertise: float  # base Delphi social weight, > 0


# Default panel: three trait-divergent viewpoints on the same conversation.
# ``critical`` is deliberately stubborn (low openness) so its dissenting view
# survives the exchange -- the "structural disagreement" ESPP shows a single
# homogeneous judge would systematically erase.
DEFAULT_PERSONAS: tuple[PersonaProfile, ...] = (
    PersonaProfile(
        name="helpful",
        framing=(
            "You evaluate as an agreeable, user-centric reviewer. Judge whether the agent "
            "genuinely understood and satisfied the caller's need, with empathy and completeness."
        ),
        openness=0.6,
        expertise=1.0,
    ),
    PersonaProfile(
        name="critical",
        framing=(
            "You evaluate as a rigorous, skeptical reviewer. Hunt for inaccuracies, unhandled "
            "edge cases, hallucinations, and anything misleading or incorrect."
        ),
        openness=0.3,
        expertise=1.2,
    ),
    PersonaProfile(
        name="efficient",
        framing=(
            "You evaluate as an efficiency-focused reviewer. Penalize verbosity, redundancy, and "
            "wasted turns; reward concise, direct, low-friction resolution."
        ),
        openness=0.5,
        expertise=0.9,
    ),
)


@dataclass
class _Panelist:
    """A single panelist's mutable state across the opinion exchange."""

    profile: PersonaProfile
    rating: float  # current opinion on the raw rating scale
    explanation: str
    evidence: str
    initial_rating: float
    raw_response: str | None = None
    tokens: set[str] = field(default_factory=set)


@register_metric
class PersonaPanelJudgeMetric(TextJudgeMetric):
    """LLM-as-judge over a diverse persona panel (conversation-level).

    Three ESPP stages, sharing the transcript-in / rating-out contract of the
    single judges it sits alongside:

    1. **Independent rating** -- each persona rates the full transcript under
       its own trait-driven framing and must cite transcript evidence.
    2. **Bounded-confidence exchange** -- panelists iteratively revise their
       ratings toward the weighted mean of panelists whose reasoning is
       semantically close (token-overlap gate), stepping at a trait-derived
       (openness) rate.
    3. **Delphi aggregation** -- final opinions combine under
       expertise-weighted social weighting into one composite judgment.

    The mean of the *initial* independent ratings is also surfaced as a
    single-judge / prompt-ensemble baseline so the panel's added value is
    directly comparable, mirroring ESPP's control.
    """

    name = "persona_panel"
    # Intentionally unversioned: this metric builds its prompts inline rather than
    # from a shared ``judge.persona_panel.user_prompt`` template, so it is not
    # enrolled in the versioned signature-drift fixture (eva.metrics.signatures
    # skips classes with ``version is None``). Set a ``version`` and add the
    # template + fixture entry when promoting it to a tracked metric.
    version = None
    description = "LLM-as-judge over a psychologically diverse persona panel with bounded-confidence Delphi aggregation"
    category = "experience"
    rating_scale = (1, 5)
    # Opt-in: a panel run makes one judge call per persona (3 by default), so it
    # is heavier than a single judge. Select explicitly with `--metrics persona_panel`.
    exclude_from_default_metrics = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.delphi_rounds = int(self.config.get("delphi_rounds", 3))
        # Semantic gate for the bounded-confidence neighborhood.
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.15))
        self.personas = self._resolve_personas()

    def _resolve_personas(self) -> tuple[PersonaProfile, ...]:
        """Build the panel from config overrides, falling back to defaults."""
        raw = self.config.get("personas")
        if not raw:
            return DEFAULT_PERSONAS
        personas = [
            PersonaProfile(
                name=str(entry["name"]),
                framing=str(entry["framing"]),
                openness=float(entry.get("openness", 0.5)),
                expertise=float(entry.get("expertise", 1.0)),
            )
            for entry in raw
        ]
        return tuple(personas) if personas else DEFAULT_PERSONAS

    def format_transcript(self, context: MetricContext) -> str:
        """Format conversation content for the panel prompts."""
        return format_transcript_with_tools(context.conversation_trace)

    def _build_persona_prompt(self, persona: PersonaProfile, transcript: str) -> str:
        """Build the evidence-grounded rating prompt for one panelist."""
        min_r, max_r = self.rating_scale
        return (
            f"{persona.framing}\n\n"
            "You are one panelist in a diverse panel scoring the SAME conversation. "
            "Rate the agent's overall response quality from your viewpoint.\n\n"
            f"## Conversation\n{transcript}\n\n"
            "## Rating scale\n"
            f"{min_r} = very poor, {(min_r + max_r) // 2} = adequate, {max_r} = excellent.\n\n"
            "## Instructions\n"
            "- Ground your judgment in specific evidence from the transcript.\n"
            "- Respond with ONLY a JSON object:\n"
            '  {"rating": <integer>, "explanation": "<your reasoning>", '
            '"evidence": "<quote or paraphrase from the transcript>"}\n'
        )

    async def _rate_independently(self, transcript: str, context: MetricContext) -> list[_Panelist]:
        """Stage 1: each persona rates the transcript concurrently."""
        prompts = [self._build_persona_prompt(p, transcript) for p in self.personas]
        responses = await asyncio.gather(*[self.call_judge(prompt, context) for prompt in prompts])

        panelists: list[_Panelist] = []
        for persona, (parsed, raw) in zip(self.personas, responses):
            if parsed is None or "rating" not in parsed:
                self.logger.warning(f"[{context.record_id}] Panelist '{persona.name}' produced no rating; dropping")
                continue
            rating, _ = self.validate_and_normalize_rating(parsed, context)
            explanation = str(parsed.get("explanation", ""))
            evidence = str(parsed.get("evidence", ""))
            panelists.append(
                _Panelist(
                    profile=persona,
                    rating=float(rating),
                    explanation=explanation,
                    evidence=evidence,
                    initial_rating=float(rating),
                    raw_response=raw,
                    tokens=_tokenize(explanation) | _tokenize(evidence),
                )
            )
        return panelists

    def _exchange(self, panelists: list[_Panelist]) -> None:
        """Stage 2: trait-driven, semantically-gated bounded-confidence update (in place).

        Each panelist revises toward the weighted mean of its neighborhood -- itself
        plus panelists whose reasoning is semantically close (token-overlap gate) --
        stepping at a trait-derived (openness) rate. Including self keeps the update
        monotone (no oscillation) and lets gated-out panelists hold their ground.
        """
        if len(panelists) < 2:
            return
        for _ in range(self.delphi_rounds):
            updated: list[float] = []
            for me in panelists:
                total_w = 1.0  # self always counts
                weighted_sum = me.rating
                for other in panelists:
                    if other is me:
                        continue
                    sim = _jaccard(me.tokens, other.tokens)
                    if sim >= self.confidence_threshold:
                        weight = sim * other.profile.expertise
                        total_w += weight
                        weighted_sum += other.rating * weight
                neighbor_mean = weighted_sum / total_w
                updated.append(me.rating + me.profile.openness * (neighbor_mean - me.rating))
            for panelist, new_rating in zip(panelists, updated):
                panelist.rating = new_rating

    def _aggregate(self, panelists: list[_Panelist]) -> tuple[float, dict[str, float]]:
        """Stage 3: Delphi social weighting -> (composite rating, per-panelist weights)."""
        ordered = sorted(p.rating for p in panelists)
        median = ordered[len(ordered) // 2]
        # Centrality bonus: opinions nearer the median count slightly more (classic Delphi).
        raw_weights = {p.profile.name: p.profile.expertise / (1.0 + abs(p.rating - median)) for p in panelists}
        total = sum(raw_weights.values())
        weights = {name: w / total for name, w in raw_weights.items()}
        composite = sum(p.rating * weights[p.profile.name] for p in panelists)
        return composite, weights

    async def compute(self, context: MetricContext) -> MetricScore:
        """Run the three-stage persona-panel evaluation."""
        try:
            transcript = self.format_transcript(context)
            if not transcript:
                return MetricScore(name=self.name, score=0.0, normalized_score=0.0, error="No transcript available")

            panelists = await self._rate_independently(transcript, context)
            if not panelists:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No panelist produced a valid rating",
                )

            min_r, max_r = self.rating_scale

            # ESPP control: single-judge / prompt-ensemble equivalent.
            initial_mean = sum(p.initial_rating for p in panelists) / len(panelists)

            # Stages 2 & 3.
            self._exchange(panelists)
            composite_rating, weights = self._aggregate(panelists)

            normalized_composite = _unit(composite_rating, min_r, max_r)
            normalized_initial = _unit(initial_mean, min_r, max_r)
            dispersion = _std([p.rating for p in panelists])

            # Per-persona sub-metrics so the leaderboard surfaces subgroup views.
            sub_metrics = {
                f"{self.name}.{p.profile.name}_score": MetricScore(
                    name=f"{self.name}.{p.profile.name}_score",
                    score=round(p.rating, 3),
                    normalized_score=round(_unit(p.rating, min_r, max_r), 3),
                    details={"initial_rating": p.initial_rating, "weight": round(weights[p.profile.name], 4)},
                )
                for p in panelists
            }

            details: dict[str, Any] = {
                "num_panelists": len(panelists),
                "delphi_rounds": self.delphi_rounds,
                "confidence_threshold": self.confidence_threshold,
                "single_judge_baseline_normalized": round(normalized_initial, 4),
                "panel_composite_normalized": round(normalized_composite, 4),
                "panel_dispersion": round(dispersion, 4),
                "panelists": [
                    {
                        "name": p.profile.name,
                        "initial_rating": p.initial_rating,
                        "final_rating": round(p.rating, 3),
                        "openness": p.profile.openness,
                        "weight": round(weights[p.profile.name], 4),
                        "explanation": p.explanation,
                        "evidence": p.evidence,
                    }
                    for p in panelists
                ],
            }

            return MetricScore(
                name=self.name,
                score=round(composite_rating, 3),
                normalized_score=round(normalized_composite, 3),
                details=details,
                sub_metrics=sub_metrics or None,
            )
        except Exception as e:
            return self._handle_error(e, context)


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS and len(tok) > 2}


def _unit(value: float, lo: int, hi: int) -> float:
    """Normalize a rating on [lo, hi] to the unit interval [0, 1]."""
    return 1.0 if hi == lo else (value - lo) / (hi - lo)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Parameter-free semantic-similarity proxy (vocabulary overlap)."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
