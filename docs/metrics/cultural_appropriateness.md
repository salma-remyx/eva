# Cultural Appropriateness

> **Diagnostic Metric**: A multilingual user carries cultural expectations they never state aloud — formality register, date/time/number formats, greeting norms. An agent that makes the user re-interpret "06/14" or tutoies a formal caller is failing even when the task completes.

## Overview

Text-judge metric that scores whether the agent **inferred and respected the implicit cultural constraints** of the simulated user on the multilingual path. The user simulator persona is grounded per language (see [`eva.utils.cultural_grounding`](../../src/eva/utils/cultural_grounding.py) and `configs/cultural_grounding.yaml`): the user *behaves* per the profile but never announces it, so the agent must pick the constraints up from the user's own speech — the inference-based cultural evaluation of [CultureConverse](https://arxiv.org/abs/2608.28405), adapted to eva's languages.

> [!NOTE]
> By default, this diagnostic metric is excluded. Enable it explicitly with `--metrics cultural_appropriateness` (or include it in a comma-separated `--metrics` list).

### Capabilities Measured

- **Language Model + Multilingual**: Whether the agent accommodates the user's implicit cultural expectations (register, formats, politeness norms) rather than merely speaking the right language.

## How It Works

### Evaluation Method

- **Type**: Text Judge (LLM-as-judge)
- **Model**: GPT-5.2 (configurable via `judge_model` / `JUDGE_MODEL`)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `conversation_trace`: The conversation (including tool calls) to judge
- `language`: Selects the cultural grounding profile that becomes the rubric

### Evaluation Methodology

The judge receives the conversation plus a **cultural grounding brief** rendered from the same profile that grounded the user simulator (so the agent is scored against exactly what the user exhibited). For each aspect in the profile it decides whether the agent's own spoken behavior accommodated the expectation:

- Only the agent's spoken words are judged; tool calls are context
- An aspect the conversation never touched (e.g. no dates discussed) is **not** a violation
- The agent is not penalized for following the user's lead

### Scoring

- **Scale**: 1-3
  - 3: Culturally competent — all applicable aspects respected
  - 2: Minor slips — mostly accommodated, isolated corrected slips
  - 1: Violation — one or more aspects clearly or repeatedly disregarded
- **Normalization**: 3→1.0, 2→0.5, 1→0.0
- **Sub-metrics**: One `<aspect>_rate` per profile aspect (e.g. `formality_register_rate`) — the fraction-of-records convention for issue flags, lower is better
- **Records whose language has no grounding profile** (e.g. English) get an error score, not a pass

## Adding or Changing Grounding

Profiles live in `configs/cultural_grounding.yaml`, one entry per language with a `community` and a set of `aspects` (`user_behavior` for the simulator side, `agent_expectation` for the judge side). Adding a language there immediately grounds both the user simulator and this judge — no code changes needed. Aspect slugs become sub-metric names, so treat them as stable once shipped.

## Example Output

```json
{
  "name": "cultural_appropriateness",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "language": "fr",
    "aspect_analysis": {
      "formality_register": {"analysis": "Agent kept 'vous' throughout.", "violated": false},
      "date_time_format": {"analysis": "Agent proposed '06/14' twice before correcting to '14/06'.", "violated": true}
    }
  },
  "sub_metrics": {
    "formality_register_rate": {"score": 0.0},
    "date_time_format_rate": {"score": 1.0}
  }
}
```

## Related Metrics

- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Validates the simulator side; also re-renders the grounded persona so the judge sees the user's actual instructions
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Catches wrong-language output; this metric catches right-language-wrong-culture output

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/cultural_appropriateness.py`
- **Class**: `CulturalAppropriatenessMetric`
- **Base Class**: `ConversationTextJudgeMetric` → `TextJudgeMetric`
- **Prompt**: `configs/prompts/judge.yaml` under `judge.cultural_appropriateness`
- **Grounding profiles**: `configs/cultural_grounding.yaml` (loaded by `eva.utils.cultural_grounding`)
