"""Diagnostic metric: audit LLM-as-judge reliability for a single conversation.

Holds the candidate response (the agent transcript) fixed and re-scores it under
two configured judges, quantifying how much the score moves purely because the
evaluator changed (evaluator-replacement ambiguity). It also runs a pairwise
position-bias probe and emits a per-judge parser/fallback audit trail so any
observed shift can be attributed rather than guessed.

Adapted from "When the Judge Changes, So Does the Measurement: Auditing
LLM-as-Judge Reliability" (arXiv:2607.08535). The paper's full benchmark sweep
(Qwen3 1.7B->32B scaling, MiniMax M2->M2.7 adjacent releases, repeated-sample
juries, structured debate, and cross-dataset slices) is intentionally out of
scope here -- this ports the paper's actionable artifact for one conversation:
"with responses fixed, a judge swap can move the score, so report bias probes
and a protocol audit trail." It lands on EVA's existing TextJudgeMetric plumbing
rather than introducing a separate audit framework.
"""

import os
from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.diagnostic.measurement_stability import assess_stability
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric, parse_judge_response
from eva.metrics.versioning import hash_prompt_template
from eva.models.results import MetricScore
from eva.utils.llm_client import LLMClient


def _split_halves(items: list[str]) -> tuple[str, str]:
    """Split ``items`` into two contiguous, labelled halves (first gets the extra item on odd counts)."""
    mid = (len(items) + 1) // 2
    first = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(items[:mid]))
    second = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(items[mid:]))
    return first, second


@register_metric
class JudgeSwapAuditMetric(TextJudgeMetric):
    """Audit LLM-as-judge reliability: does a fixed response's score move when the judge changes?

    Diagnostic metric (opt-in; excluded from default scores and pass@k). With the
    agent transcript held fixed it:

    1. Scores overall conversation quality (1-3) under judge A and judge B and
       reports the evaluator-replacement shift ``|normA - normB|`` as the parent
       score (lower means the two judges agree).
    2. Runs a pairwise position-bias probe: the same two transcript halves are
       shown in both orders; a judge whose preference flips with order is flagged.
       Reported as the ``position_bias_rate`` sub-metric (lower is better).
    3. Records a per-judge audit trail (model, raw response, parse ok / fallback)
       in ``details`` so any shift can be attributed.

    When ``stability_samples > 1`` it additionally re-samples each judge and attaches
    a ``measurement_stability`` block (per-judge variance + bootstrap CI on the shift)
    so a reported shift can be told apart from each judge's own sampling noise.

    Judges are configured as:

      - Judge A: ``judge_model`` (inherited from TextJudgeMetric).
      - Judge B: ``judge_model_b`` config key (or ``JUDGE_MODEL_B`` env var). When
        unset it falls back to Judge A and ``details.judges_identical`` is set, so
        the shift then measures a single judge's run-to-run noise instead of a swap.
    """

    name = "judge_swap_audit"
    version = "v0.2"
    description = "Diagnostic metric: audit LLM-as-judge reliability across two judges (replacement ambiguity + position bias + audit trail)"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    rating_scale = (1, 3)
    # Headline number is a score *shift*: lower means the two judges agree.
    higher_is_better = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Repeated per-judge sampling (default 1 = single-shot, unchanged behavior).
        # With >1 samples the metric attaches a measurement-stability block reporting
        # per-judge variance and a bootstrap CI so the shift can be told apart from noise.
        self.stability_samples = max(1, int(self.config.get("stability_samples", 1)))
        self.bootstrap_iterations = max(1, int(self.config.get("bootstrap_iterations", 1000)))
        config_b = self.config.get("judge_model_b")
        model_b: str | None = config_b if isinstance(config_b, str) else os.environ.get("JUDGE_MODEL_B")
        self.judges_identical = model_b is None
        if model_b is None:
            # No second judge configured: reuse judge A so the metric still runs,
            # measuring sampling noise rather than an evaluator replacement.
            self.llm_client_b = self.llm_client
        else:
            params = {**self.default_params, **self.config.get("judge_params", {})}
            self.llm_client_b = LLMClient(model=model_b, params=params)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Score the fixed transcript under both judges and assemble the audit report."""
        try:
            transcript = self._format_transcript(context)
            if not transcript:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant transcript available to judge",
                )

            rating_prompt = self.get_judge_prompt(user_goal=context.user_goal, transcript=transcript)
            a_parsed, a_raw = await self._call_judge_on(self.llm_client, rating_prompt, context)
            b_parsed, b_raw = await self._call_judge_on(self.llm_client_b, rating_prompt, context)

            audit: dict[str, Any] = {
                "judge_a_model": self.llm_client.model,
                "judge_b_model": self.llm_client_b.model,
                "judges_identical": self.judges_identical,
                "rating_prompt": rating_prompt,
                "judge_a": self._audit_entry(a_parsed, a_raw, context),
                "judge_b": self._audit_entry(b_parsed, b_raw, context),
            }
            shift = self._replacement_shift(audit["judge_a"]["normalized"], audit["judge_b"]["normalized"])
            stability = None
            if self.stability_samples > 1:
                samples_a = await self._collect_samples(self.llm_client, rating_prompt, context, a_parsed, a_raw)
                samples_b = await self._collect_samples(self.llm_client_b, rating_prompt, context, b_parsed, b_raw)
                stability = assess_stability(samples_a, samples_b, iterations=self.bootstrap_iterations)
            probe = await self._run_position_bias_probe(context)

            sub_metrics: dict[str, MetricScore] = {}
            if probe is not None and probe["judge_count"] > 0:
                sub_metrics["position_bias_rate"] = make_rate_sub_metric(
                    parent_name=self.name,
                    key="position_bias_rate",
                    numerator=probe["flipped_count"],
                    denominator=probe["judge_count"],
                    details=probe,
                )

            prompt_hash = hash_prompt_template(self.prompt_manager.get_template(f"judge.{self.name}.user_prompt"))
            return MetricScore(
                name=self.name,
                score=shift,
                normalized_score=shift,
                version=self.version,
                prompt_hash=prompt_hash,
                details={
                    "evaluator_replacement_shift": shift,
                    "audit_trail": audit,
                    "position_bias_probe": probe,
                    "measurement_stability": stability,
                },
                sub_metrics=sub_metrics or None,
                skipped=shift is None,
            )
        except Exception as e:
            return self._handle_error(e, context)

    async def _call_judge_on(
        self,
        client: LLMClient,
        prompt: str,
        context: MetricContext,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Call ``client`` and parse the JSON answer; mirrors ``TextJudgeMetric.call_judge``.

        Parameterized by client so the same call_judge code path runs against two
        different judge models without mutating ``self.llm_client``.
        """
        messages = [{"role": "user", "content": prompt}]
        response_text, usage = await client.generate_text(messages)
        self._log_token_usage(context, client.model, client.params, prompt, usage, response_text)
        return parse_judge_response(response_text, context.record_id, self.logger), response_text

    async def _collect_samples(
        self,
        client: LLMClient,
        prompt: str,
        context: MetricContext,
        first_parsed: dict[str, Any] | None,
        first_raw: str | None,
    ) -> list[float]:
        """Gather ``stability_samples`` normalized ratings for one judge (reusing the first call).

        The already-issued rating call is passed in as ``first_parsed`` so the
        stability estimate costs only ``stability_samples - 1`` extra calls per judge.
        Unparseable samples are dropped rather than defaulted, so variance reflects
        real judge output, not fallback ratings.
        """
        samples: list[float] = []
        for parsed, raw in [(first_parsed, first_raw)] + [
            await self._call_judge_on(client, prompt, context) for _ in range(self.stability_samples - 1)
        ]:
            normalized = self._audit_entry(parsed, raw, context)["normalized"]
            if normalized is not None:
                samples.append(normalized)
        return samples

    def _audit_entry(
        self,
        parsed: dict[str, Any] | None,
        raw: str | None,
        context: MetricContext,
    ) -> dict[str, Any]:
        """Normalize one judge's parsed response into an audit-trail record."""
        entry: dict[str, Any] = {
            "raw_response": raw,
            "parsed_ok": parsed is not None,
            "rating": None,
            "normalized": None,
        }
        if parsed is None:
            return entry
        rating, normalized = self.validate_and_normalize_rating(parsed, context)
        entry["rating"] = rating
        entry["normalized"] = normalized
        return entry

    @staticmethod
    def _replacement_shift(norm_a: float | None, norm_b: float | None) -> float | None:
        """Absolute distance between the two judges' normalized ratings; None if either failed to parse."""
        if norm_a is None or norm_b is None:
            return None
        return round(abs(norm_a - norm_b), 3)

    async def _run_position_bias_probe(self, context: MetricContext) -> dict[str, Any] | None:
        """Show the same transcript halves in both orders; flag judges whose preference flips.

        Returns None when there are fewer than two assistant turns to split into a pair.
        """
        turns = context.transcribed_assistant_turns or context.intended_assistant_turns or {}
        texts = [text for _tid, text in sorted(turns.items()) if text]
        if len(texts) < 2:
            return None

        first_half, second_half = _split_halves(texts)
        # order 0 -> (A=first_half, B=second_half); order 1 -> (A=second_half, B=first_half)
        orderings: list[tuple[str, str]] = [(first_half, second_half), (second_half, first_half)]

        judges: list[tuple[str, LLMClient]] = [("a", self.llm_client)]
        if not self.judges_identical:
            judges.append(("b", self.llm_client_b))

        flipped_count = 0
        judge_count = 0
        per_judge: dict[str, Any] = {}
        for label, client in judges:
            winners = [
                self._fixed_winner(
                    await self._probe_choice(client, context, cand_a, cand_b),
                    order_idx,
                )
                for order_idx, (cand_a, cand_b) in enumerate(orderings)
            ]
            if any(winner is None for winner in winners):
                per_judge[label] = {"winners": winners, "flipped": None}
                continue
            flipped = winners[0] != winners[1]
            judge_count += 1
            if flipped:
                flipped_count += 1
            per_judge[label] = {"winners": winners, "flipped": flipped}

        result: dict[str, Any] = {
            "flipped_count": flipped_count,
            "judge_count": judge_count,
            "per_judge": per_judge,
        }
        return result

    async def _probe_choice(
        self,
        client: LLMClient,
        context: MetricContext,
        candidate_a: str,
        candidate_b: str,
    ) -> str | None:
        """Ask ``client`` which candidate is better; return ``"A"``, ``"B"``, or None on parse failure."""
        prompt = self.get_judge_prompt(
            prompt_key="position_bias_probe",
            user_goal=context.user_goal,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
        )
        parsed, _raw = await self._call_judge_on(client, prompt, context)
        if not isinstance(parsed, dict):
            return None
        choice = str(parsed.get("choice", "")).strip().upper()
        return choice if choice in ("A", "B") else None

    @staticmethod
    def _fixed_winner(choice: str | None, order_idx: int) -> str | None:
        """Map a raw ``A``/``B`` choice to the fixed candidate (``first_half``/``second_half``) it selected.

        Unpacking presentation order is what makes a flip attributable to position
        rather than to the candidates themselves.
        """
        if choice is None:
            return None
        if order_idx == 0:  # A=first_half, B=second_half
            return "first_half" if choice == "A" else "second_half"
        return "second_half" if choice == "A" else "first_half"  # order 1: A=second_half, B=first_half

    @staticmethod
    def _format_transcript(context: MetricContext) -> str:
        """Render the agent's spoken turns as the fixed candidate response to judge."""
        turns = context.transcribed_assistant_turns or context.intended_assistant_turns or {}
        if not turns:
            return ""
        return "\n".join(f"Assistant turn {tid}: {text}" for tid, text in sorted(turns.items()))
