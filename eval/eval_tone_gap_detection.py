"""Benchmark: does tone_sensitivity produce correct verdicts across the paper's scenario classes?

Fixture: 15 conversations — 6 tone-gap (calm words, distressed delivery, assistant
ignores: rating 1 + emotional_intelligence_gap flag), 3 no-signal (delivery adds
nothing beyond the words: skipped), 3 attended (rating 3 -> normalized 1.0), 3
partial (rating 2 -> normalized 0.5). Judge responses are scripted canned JSON;
audio loading and the LLM client are mocked exactly as in
tests/unit/metrics/experience/test_tone_sensitivity.py (load_role_audio /
encode_audio_segment patched, llm_client.generate_text stubbed). Deterministic,
CPU-only, no network, no wall-clock measurements.

Threshold derivations (see validation.yaml):
  tone_sensitivity_verdict_accuracy >= 1.0: every canned verdict is decidable
    deterministically by the real compute() on the PR head -> 15/15 = 1.0; on
    `main` the metric class is absent from the registry -> 0/15 = 0.0.
  experience_metric_registry_count  >= 3.0: baseline `main` registers exactly the
    3 pre-existing experience metrics (conciseness, conversation_progression,
    turn_taking), which the PR must not drop.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 6 tone-gap + 3 no-signal + 3 attended + 3 partial = 15 conversations.
# (count, user words, assistant reply, rating, gap_flagged, delivery_conveys_signal)
SCENARIOS = (
    (6, "No really, I'm fine, everything's fine.", "Great, I'll close your case then.", 1, True, True),
    (3, "I'd like to move my flight to Friday, please.", "Done - you're on the 9am Friday flight.", 3, False, False),
    (3, "I'm fine... it's fine, really.", "I hear how hard this is - let's keep your case open and sort it out.", 3, False, True),
    (3, "Fine. Whatever you think is best.", "Okay... is everything alright? Is there anything else?", 2, False, True),
)


def _cases():
    cases = []
    for idx, (count, user_line, assistant_line, rating, flagged, conveys) in enumerate(SCENARIOS):
        for k in range(count):
            tag = f"scenario{idx}-{k}"
            trace = [
                {"role": "user", "content": f"{tag}: {user_line}", "type": "intended", "turn_id": 0},
                {"role": "assistant", "content": f"{tag}: {assistant_line}", "type": "transcribed", "turn_id": 1},
            ]
            response = {
                "delivery_conveys_signal": conveys,
                "rating": rating,
                "perceived_delivery": "caller is audibly distressed" if conveys else "neutral, matches the words",
                "explanation": "scripted verdict driving the real compute() path",
                "dimensions": {
                    "emotional_intelligence_gap": {
                        "flagged": flagged,
                        "evidence": f"{tag}: assistant acted only on the literal words" if flagged else "",
                    }
                },
            }
            expected = (
                {"skip": True}
                if not conveys
                else {"skip": False, "normalized": (rating - 1) / 2.0, "gap": flagged}
            )
            cases.append({"trace": trace, "response": response, "expected": expected})
    return cases


def _verdict_correct(result, expected) -> bool:
    """Compare the real compute() output against the scripted ground-truth verdict."""
    if expected["skip"]:
        return bool(result.skipped) and result.score is None and result.error is None
    if result.skipped or result.score is None:
        return False
    if abs(float(result.normalized_score) - expected["normalized"]) > 1e-9:
        return False
    return bool((result.details or {}).get("emotional_intelligence_gap")) == expected["gap"]


def _count_experience_metrics(registry) -> int:
    count = 0
    for name in list(registry.get_all()):
        cls = registry.get(name)
        if cls is not None and getattr(cls, "category", None) == "experience":
            count += 1
    return count


async def _measure() -> dict:
    cases = _cases()
    # Degraded defaults: emitted unconditionally so the baseline arm (where the
    # changed module does not exist) still produces a comparable JSON metrics line.
    metrics = {
        "tone_sensitivity_verdict_accuracy": 0.0,
        "experience_metric_registry_count": 0.0,
        "verdicts_correct": 0.0,
        "conversations_total": float(len(cases)),
    }
    try:
        import eva.metrics.experience  # noqa: F401  triggers registration on both arms
        from eva.metrics.registry import get_global_registry
        from eva.models.config import PipelineType
        from tests.unit.metrics.conftest import make_judge_metric, make_metric_context
    except Exception:
        return metrics  # repo/test helpers unavailable -> zeroed metrics, never crash

    registry = get_global_registry()
    metrics["experience_metric_registry_count"] = float(_count_experience_metrics(registry))
    metric_cls = registry.get("tone_sensitivity")
    if metric_cls is None:
        return metrics  # baseline arm: metric absent -> 0/15 verdicts, guardrail count still emitted

    correct = 0
    for case in cases:
        metric = make_judge_metric(metric_cls, mock_llm=True, logger_name="eval_tone_gap_detection")
        metric.llm_client.generate_text.return_value = (json.dumps(case["response"]), None)
        context = make_metric_context(
            audio_user_path="/fake/audio_user.wav",
            pipeline_type=PipelineType.S2S,
            conversation_trace=case["trace"],
        )
        with patch.object(metric, "load_role_audio", return_value=MagicMock()), patch.object(
            metric, "encode_audio_segment", return_value="base64audio"
        ):
            result = await metric.compute(context)
        if _verdict_correct(result, case["expected"]):
            correct += 1
    metrics["verdicts_correct"] = float(correct)
    metrics["tone_sensitivity_verdict_accuracy"] = correct / float(len(cases))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", type=int, default=None)
    _args = parser.parse_args()  # accepted and ignored
    print(json.dumps(asyncio.run(_measure())))
    return 0


if __name__ == "__main__":
    sys.exit(main())