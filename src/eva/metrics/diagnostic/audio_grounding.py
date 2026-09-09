"""Audio grounding metric — does the assistant answer to how something was said, not just what was said.

Adapted from "When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue"
(arXiv:2608.27176), which formalizes *cross-modal disagreement*: the transcript suggests a
plausible but incorrect surface interpretation while the acoustic cues (prosody, emotion,
speaking style) support a different one. Audio-native models still take the transcript-biased
reading in 30-40% of such conflict cases — the "transcript trap".

Instead of the paper's curated conflict-QA benchmark, this metric ports the measurement onto
EVA's own conversations: an audio judge listens to the user's audio, finds user turns whose
paralinguistic channel conflicts with the lexical content of the turn, and classifies whether
the assistant's next response followed the audio-grounded meaning or the misleading surface
reading. The parent score is the conflict-turn grounding accuracy; sub-metrics expose the trap
rate, the disagreement rate, and the consistent-turn accuracy, so the paper's
consistent-vs-conflict gap is computable from the results.
"""

from typing import Any

from eva.metrics.base import AudioJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric, resolve_turn_id
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json

# The five discourse dimensions from the paper in which a disagreement can occur.
_DIMENSION_KEYS = (
    "interaction_behavior",
    "emotion_state",
    "dialogue_act",
    "social_stance",
    "conversational_intent",
)

_RESPONSE_GROUNDED = "audio_grounded"
_RESPONSE_TRAP = "transcript_trap"
_RESPONSE_UNCLEAR = "unclear"
_VALID_RESPONSES = (_RESPONSE_GROUNDED, _RESPONSE_TRAP, _RESPONSE_UNCLEAR)


@register_metric
class AudioGroundingMetric(AudioJudgeMetric):
    """Audio judge metric for cross-modal grounding on user turns with incongruent affect.

    Sends the user-side audio plus the text conversation trace to the judge. For each user
    turn the judge reports whether the acoustic cues conflict with the surface reading of the
    words, and whether the assistant's next response was audio-grounded, took the
    transcript-biased trap reading, or was unclear.

    Parent score: fraction of conflict turns where the assistant was audio-grounded
    (1.0 = always grounded, 0.0 = always trapped or unclear). Skipped when the conversation
    contains no cross-modal disagreement turns.
    """

    name = "audio_grounding"
    version = "v0.1"
    description = (
        "Audio judge evaluation of whether the assistant grounds its response in the user's "
        "acoustic cues when they conflict with the transcript's surface reading"
    )
    category = "diagnostic"
    # Only audio-native pipelines receive the user's audio; a cascade assistant only ever sees
    # the transcript, so it takes the surface reading by construction and the metric is degenerate.
    supported_pipeline_types = frozenset({PipelineType.S2S, PipelineType.AUDIO_LLM})
    exclude_from_default_metrics = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute cross-modal grounding scores from the user audio and conversation trace."""
        try:
            audio_segment = self.load_role_audio(context, "user")
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user audio file available",
                )

            trace_formatted, user_turn_ids = self._format_conversation_trace(context)
            if not user_turn_ids:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns found in conversation trace",
                )

            prompt = self.get_judge_prompt(conversation_trace_formatted=trace_formatted)
            messages = self.create_audio_message(self.encode_audio_segment(audio_segment), prompt)

            response_text, usage = await self.llm_client.generate_text(messages)
            self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage, response_text)
            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            self.logger.debug(f"Raw judge response: {response_text[:200]}")

            parsed = extract_and_load_json(response_text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("turns"), list):
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No turns in judge response",
                )
            turns = parsed["turns"]
            if len(turns) != len(user_turn_ids):
                self.logger.warning(
                    f"[{context.record_id}] Expected {len(user_turn_ids)} audio grounding ratings, got {len(turns)}"
                )

            per_turn = self._parse_turns(turns, user_turn_ids, context)
            return self._build_score(per_turn, len(user_turn_ids), prompt, response_text)

        except Exception as e:
            return self._handle_error(e, context)

    def _parse_turns(
        self, turns: list[Any], user_turn_ids: list[int], context: MetricContext
    ) -> dict[int, dict[str, Any]]:
        """Normalize the judge's per-turn entries, dropping entries we cannot use."""
        per_turn: dict[int, dict[str, Any]] = {}
        for response_item in turns:
            if not isinstance(response_item, dict):
                continue
            turn_id = resolve_turn_id(response_item, user_turn_ids, self.name)
            if turn_id is None:
                continue

            response_kind = response_item.get("assistant_response")
            if response_kind not in _VALID_RESPONSES:
                self.logger.warning(
                    f"[{context.record_id}] Invalid assistant_response {response_kind!r} for turn {turn_id}, "
                    f"excluding this turn"
                )
                continue

            # The dimension is kept verbatim for visibility in details; only the paper's five
            # are tallied in conflicts_by_dimension, mirroring the per-category tagging convention.
            per_turn[turn_id] = {
                "has_disagreement": bool(response_item.get("has_disagreement")),
                "dimension": response_item.get("dimension"),
                "surface_interpretation": response_item.get("surface_interpretation"),
                "audio_interpretation": response_item.get("audio_interpretation"),
                "assistant_response": response_kind,
                "explanation": response_item.get("explanation", ""),
            }
        return per_turn

    def _build_score(
        self,
        per_turn: dict[int, dict[str, Any]],
        num_turns: int,
        prompt: str,
        response_text: str,
    ) -> MetricScore:
        """Aggregate per-turn classifications into the parent score and sub-metrics."""
        rated_ids = sorted(per_turn)
        conflict_ids = [tid for tid in rated_ids if per_turn[tid]["has_disagreement"]]
        consistent_ids = [tid for tid in rated_ids if not per_turn[tid]["has_disagreement"]]

        def responded(turn_ids: list[int], kind: str) -> list[int]:
            return [tid for tid in turn_ids if per_turn[tid]["assistant_response"] == kind]

        grounded_conflict = responded(conflict_ids, _RESPONSE_GROUNDED)
        trapped = responded(conflict_ids, _RESPONSE_TRAP)
        grounded_consistent = responded(consistent_ids, _RESPONSE_GROUNDED)

        # Conflict-case accuracy, the paper's headline number: anything that is not an
        # audio-grounded response on a disagreement turn (trap or unclear) counts against it.
        conflict_accuracy = len(grounded_conflict) / len(conflict_ids) if conflict_ids else None
        consistent_accuracy = len(grounded_consistent) / len(consistent_ids) if consistent_ids else None
        skipped = not conflict_ids

        conflicts_by_dimension: dict[str, list[int]] = {}
        for tid in conflict_ids:
            dimension = per_turn[tid]["dimension"]
            if dimension in _DIMENSION_KEYS:
                conflicts_by_dimension.setdefault(dimension, []).append(tid)

        sub_metrics: dict[str, MetricScore] = {}
        if rated_ids:
            sub_metrics["disagreement_rate"] = make_rate_sub_metric(
                parent_name=self.name,
                key="disagreement_rate",
                numerator=len(conflict_ids),
                denominator=len(rated_ids),
                details={"count": len(conflict_ids), "num_rated": len(rated_ids), "turn_ids": conflict_ids},
            )
        if conflict_ids:
            sub_metrics["trap_rate"] = make_rate_sub_metric(
                parent_name=self.name,
                key="trap_rate",
                numerator=len(trapped),
                denominator=len(conflict_ids),
                details={"count": len(trapped), "num_conflict": len(conflict_ids), "turn_ids": trapped},
            )
        if consistent_ids:
            sub_metrics["consistent_accuracy"] = make_rate_sub_metric(
                parent_name=self.name,
                key="consistent_accuracy",
                numerator=len(grounded_consistent),
                denominator=len(consistent_ids),
                details={
                    "count": len(grounded_consistent),
                    "num_consistent": len(consistent_ids),
                    "turn_ids": grounded_consistent,
                },
            )

        details: dict[str, Any] = {
            "num_turns": num_turns,
            "num_rated": len(rated_ids),
            "num_conflict": len(conflict_ids),
            "num_trapped": len(trapped),
            "conflicts_by_dimension": conflicts_by_dimension,
            "skipped_reason": "No cross-modal disagreement turns found" if skipped else None,
            "per_turn": per_turn,
            "judge_prompt": prompt,
            "judge_raw_response": response_text,
        }

        return MetricScore(
            name=self.name,
            score=round(conflict_accuracy, 3) if conflict_accuracy is not None else None,
            normalized_score=round(conflict_accuracy, 3) if conflict_accuracy is not None else None,
            details=details,
            skipped=skipped,
            sub_metrics=sub_metrics or None,
        )

    @staticmethod
    def _format_conversation_trace(context: MetricContext) -> tuple[str, list[int]]:
        """Format the conversation trace as text for the judge.

        Keeps one line per (role, turn_id) — a turn can carry multiple entries of the same
        role (e.g. intended and transcribed variants) and duplicating them would only confuse
        the judge. Returns the formatted trace and the sorted user turn IDs.
        """
        lines: list[str] = []
        user_turn_ids: set[int] = set()
        seen: set[tuple[str, int]] = set()
        for entry in context.conversation_trace or []:
            role = entry.get("role")
            turn_id = entry.get("turn_id")
            if role not in ("user", "assistant") or (role, turn_id) in seen:
                continue
            seen.add((role, turn_id))
            if role == "user" and turn_id is not None:
                user_turn_ids.add(turn_id)
            # Flatten whitespace so a turn never spills across unlabeled lines.
            content = " ".join(str(entry.get("content", "")).split())
            label = "User" if role == "user" else "Assistant"
            lines.append(f"Turn {turn_id} - {label}: {content}")
        return "\n".join(lines), sorted(user_turn_ids)
