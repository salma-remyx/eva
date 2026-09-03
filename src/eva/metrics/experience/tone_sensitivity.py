"""Tone-sensitivity metric: does the assistant act on vocal delivery, not just words?

Adapted from *Real-Time Voice AI Hears but Does Not Listen* (arXiv:2606.26083),
which shows that production voice systems often perceive a caller's emotion, fear,
or sarcasm yet make decisions as if the speech had been reduced to a transcript —
the "emotional intelligence gap". Every existing EVA-X (experience) judge is
transcript-only and therefore structurally blind to this signal.

This metric is the first experience judge to consume the *user* audio channel
(via the existing ``AudioJudgeMetric`` machinery). It asks an audio LLM judge
whether the assistant's replies acknowledged and appropriately reacted to the
caller's vocal delivery, rather than only the literal words. Records whose
delivery carries no meaningful tonal signal beyond the words are skipped, so the
score reflects only scenarios where words and delivery genuinely diverge — the
exact condition the paper isolates.
"""

from eva.metrics.base import AudioJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import (
    build_binary_flag_sub_metrics,
    format_transcript_with_tools,
    normalize_rating,
    parse_judge_response,
    validate_rating,
)
from eva.models.results import MetricScore

# The judge prompt is kept inline (rather than in configs/prompts/judge.yaml) so the
# metric is self-contained. Literal JSON braces are doubled for str.format().
_JUDGE_PROMPT_TEMPLATE = """You are evaluating a spoken customer-service conversation. You are given the
USER's audio (their voice only) and the full text transcript of the conversation,
including the assistant's replies.

Speech carries meaning through BOTH the words and the vocal delivery (tone, prosody,
crying, trembling, hesitation, sarcasm, urgency, fear, frustration). A competent
voice assistant should respond to how something is said, not only to the literal words.

Listen to the user's audio and read the transcript, then judge whether the assistant
attended to the user's vocal delivery.

Expected user language: {expected_language}

Conversation transcript:
{conversation}

Do the following:
1. Describe the user's vocal delivery and any emotion it conveys (distress, fear,
   sarcasm, urgency, etc.).
2. Decide whether that delivery conveys meaningful information BEYOND the literal words
   (e.g. words say "I'm fine" but the voice is clearly crying; agreement said sarcastically).
3. Judge whether the assistant's responses acknowledged or appropriately reacted to the
   delivery, rather than acting purely on the transcript.
4. Flag the "emotional intelligence gap": true when the delivery clearly signalled an
   emotional state that the assistant then ignored while making decisions or replying.

Return ONLY a JSON object:
{{
  "delivery_conveys_signal": <true|false>,
  "rating": <1|2|3>,
  "perceived_delivery": "<what the voice conveyed>",
  "explanation": "<how the assistant did or did not respond to the delivery>",
  "dimensions": {{
    "emotional_intelligence_gap": {{"flagged": <true|false>, "evidence": "<short quote or note>"}}
  }}
}}

Rating scale (only meaningful when delivery_conveys_signal is true):
  3 = assistant clearly acknowledged and appropriately acted on the vocal delivery
  2 = assistant partially attended to the delivery (noticed but under-reacted)
  1 = assistant ignored the delivery and acted only on the words

If delivery_conveys_signal is false, still return a rating of 3 (nothing to attend to);
it will be excluded from scoring."""

_TONE_DIMENSION_KEYS = ("emotional_intelligence_gap",)


@register_metric
class ToneSensitivityJudgeMetric(AudioJudgeMetric):
    """Audio LLM judge for whether the assistant reacts to vocal delivery, not just words.

    Consumes the user audio channel and the conversation transcript to measure the
    "emotional intelligence gap" described in arXiv:2606.26083. Conversation-level,
    single rating.

    Rating scale: 3 (attended to delivery), 2 (partial), 1 (ignored delivery).
    Normalized: 3->1.0, 2->0.5, 1->0.0. Records whose delivery adds no signal beyond
    the words are skipped (score is None, ``skipped=True``).
    """

    name = "tone_sensitivity"
    description = "Audio judge for whether the assistant responds to the user's vocal delivery, not just the words"
    category = "experience"
    rating_scale = (1, 3)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Score how well the assistant responded to the user's vocal delivery."""
        try:
            audio_segment = self.load_role_audio(context, "user")
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={"reason": "No user audio available for tone-sensitivity analysis"},
                )

            transcript_text = format_transcript_with_tools(context.conversation_trace)
            if not transcript_text:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={"reason": "No transcript available"},
                )

            audio_b64 = self.encode_audio_segment(audio_segment)
            prompt = _JUDGE_PROMPT_TEMPLATE.format(
                conversation=transcript_text,
                expected_language=context.language_display_name,
            )
            messages = self.create_audio_message(audio_b64, prompt)

            response_text, usage = await self.llm_client.generate_text(messages)
            self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage, response_text)

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                    details={"judge_prompt": prompt},
                )

            response = parse_judge_response(response_text, context.record_id, self.logger)
            if response is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Failed to parse judge response",
                    details={"judge_prompt": prompt, "judge_raw_response": response_text},
                )

            # The paper's premise: only scenarios where delivery diverges from the words
            # exercise tone-sensitivity. Records without such signal are skipped, not scored.
            if response.get("delivery_conveys_signal") is False:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={
                        "reason": "User vocal delivery conveyed no signal beyond the words",
                        "perceived_delivery": response.get("perceived_delivery", ""),
                        "judge_prompt": prompt,
                        "judge_raw_response": response_text,
                    },
                )

            min_rating, max_rating = self.rating_scale
            rating = validate_rating(
                response.get("rating"),
                list(range(min_rating, max_rating + 1)),
                default=min_rating,
                record_id=context.record_id,
                metric_logger=self.logger,
            )
            normalized = normalize_rating(rating, min_rating, max_rating)

            dimensions = response.get("dimensions", {}) or {}
            sub_metrics = build_binary_flag_sub_metrics(
                parent_name=self.name,
                entries=dimensions,
                entry_keys=_TONE_DIMENSION_KEYS,
                flag_field="flagged",
                detail_fields=("evidence",),
            )

            return MetricScore(
                name=self.name,
                score=float(rating),
                normalized_score=normalized,
                details={
                    "rating": rating,
                    "perceived_delivery": response.get("perceived_delivery", ""),
                    "explanation": response.get("explanation", ""),
                    "emotional_intelligence_gap": bool(
                        dimensions.get("emotional_intelligence_gap", {}).get("flagged", False)
                    ),
                    "judge_prompt": prompt,
                    "judge_raw_response": response_text,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
