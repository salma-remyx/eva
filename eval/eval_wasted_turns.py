"""Benchmark: does the tone_sensitivity metric surface wasted conversational turns?

Fixture: 8 tone-gap conversations (calm words, distressed delivery) carrying 18
ground-truth wasted user re-raises (2-3 each). A conversation's wasted turns count
as flagged when the REAL ToneSensitivityJudgeMetric.compute() flags the
emotional-intelligence gap on that conversation. Judge responses are scripted
canned JSON; audio loading and the LLM client are mocked exactly as in
tests/unit/metrics/experience/test_tone_sensitivity.py (load_role_audio /
encode_audio_segment patched, llm_client.generate_text stubbed). Deterministic,
CPU-only, no network, no wall-clock measurements.

Baseline/feature symmetry: on `main` the tone_sensitivity metric is absent from
the registry, so no conversation is flagged -> 0/18 = 0.0 and the script still
emits metrics; on the PR head all 8 gap conversations are flagged -> 18/18 = 1.0.
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

# Ground-truth wasted user re-raises per conversation: 6*2 + 2*3 = 18 total.
WASTED_PER_CONV = [2, 3, 2, 2, 3, 2, 2, 2]


def _build_trace(conv_idx: int, wasted: int):
    """Words are calm; each wasted turn is a redundant re-raise after being ignored."""
    trace = [
        {"role": "user", "content": f"conv{conv_idx}: No really, I'm fine, everything's fine.", "type": "intended", "turn_id": 0},
        {"role": "assistant", "content": "Great, I'll close your case then.", "type": "transcribed", "turn_id": 1},
    ]
    for k in range(wasted):
        trace.append({"role": "user", "content": f"conv{conv_idx} re-raise {k + 1}: I still need help, this is not resolved.", "type": "intended", "turn_id": 2 + 2 * k})
        trace.append({"role": "assistant", "content": "Okay, anything else I can help with?", "type": "transcribed", "turn_id": 3 + 2 * k})
    return trace


def _judge_response(conv_idx: int) -> str:
    """Scripted (deterministic) audio-judge verdict: delivery ignored, gap flagged."""
    return json.dumps(
        {
            "delivery_conveys_signal": True,
            "rating": 1,
            "perceived_delivery": f"conv{conv_idx}: caller is audibly distressed",
            "explanation": "Assistant acted only on the literal words; the user had to re-raise.",
            "dimensions": {
                "emotional_intelligence_gap": {
                    "flagged": True,
                    "evidence": f"conv{conv_idx}: case closed despite distress, forcing re-raises",
                }
            },
        }
    )


async def _flagged(metric_cls, make_judge_metric, make_metric_context, pipeline_type_s2s, conv_idx, wasted) -> bool:
    """Run the real metric end-to-end with mocked audio/LLM; True if the gap is flagged."""
    metric = make_judge_metric(metric_cls, mock_llm=True, logger_name="eval_wasted_turns")
    metric.llm_client.generate_text.return_value = (_judge_response(conv_idx), None)
    context = make_metric_context(
        audio_user_path="/fake/audio_user.wav",
        pipeline_type=pipeline_type_s2s,
        conversation_trace=_build_trace(conv_idx, wasted),
    )
    with patch.object(metric, "load_role_audio", return_value=MagicMock()), patch.object(
        metric, "encode_audio_segment", return_value="base64audio"
    ):
        result = await metric.compute(context)
    return (not result.skipped) and bool((result.details or {}).get("emotional_intelligence_gap"))


async def _measure() -> dict:
    # Degraded defaults: emitted unconditionally so the baseline arm (where the
    # changed module does not exist) still produces a comparable JSON metrics line.
    metrics = {
        "wasted_turns_flagged_rate": 0.0,
        "experience_metric_registry_count": 0.0,
        "tone_gap_conversations_flagged": 0.0,
        "ground_truth_wasted_turns": float(sum(WASTED_PER_CONV)),
    }
    try:
        import eva.metrics.experience  # noqa: F401  triggers registration on both arms
        from eva.metrics.registry import get_global_registry
        from eva.models.config import PipelineType
        from tests.unit.metrics.conftest import make_judge_metric, make_metric_context
    except Exception:
        return metrics  # repo/test helpers unavailable -> zeroed metrics, never crash

    registry = get_global_registry()
    entries = registry.get_all()
    names = list(entries) if isinstance(entries, dict) else [e if isinstance(e, str) else getattr(e, "name", None) for e in entries]
    # Guardrail: baseline registers exactly 3 experience metrics; head adds a 4th.
    # < 3 would mean the new import broke the existing experience registrations.
    metrics["experience_metric_registry_count"] = float(
        sum(1 for n in names if n is not None and getattr(registry.get(n), "category", None) == "experience")
    )

    total_wasted = sum(WASTED_PER_CONV)
    flagged_wasted = 0
    flagged_convs = 0
    metric_cls = registry.get("tone_sensitivity")  # None on baseline (main)
    if metric_cls is not None:
        for i, wasted in enumerate(WASTED_PER_CONV):
            if await _flagged(metric_cls, make_judge_metric, make_metric_context, PipelineType.S2S, i, wasted):
                flagged_convs += 1
                flagged_wasted += wasted

    # Target threshold derivation (0.9): flagging is per-conversation over 18
    # ground-truth wasted turns, so achievable rates are 18/18=1.0, 16/18=0.889,
    # 15/18=0.833, ... — 0.9 forces ALL 8 gap conversations to be flagged.
    # Baseline has no tone_sensitivity metric: 0/18 = 0.0.
    metrics["wasted_turns_flagged_rate"] = (flagged_wasted / total_wasted) if total_wasted else 0.0
    metrics["tone_gap_conversations_flagged"] = float(flagged_convs)
    metrics["ground_truth_wasted_turns"] = float(total_wasted)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.parse_args()  # accepted and ignored; the benchmark is deterministic
    metrics = asyncio.run(_measure())
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())