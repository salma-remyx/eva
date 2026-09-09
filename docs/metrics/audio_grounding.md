# Audio Grounding

> **Diagnostic Metric**: If the agent answers to the words but not the voice, it is processing a transcript — not a conversation.

## Overview

Audio-based metric that evaluates whether the assistant **grounds its responses in the user's acoustic cues** when those cues conflict with the surface reading of the words — sarcasm, masked frustration, hesitant agreement, suppressed anger behind polite wording. This failure mode is known as *cross-modal disagreement* and responding to the word-level reading anyway is the *transcript trap*: "When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue" (arXiv:2608.27176) reports that direct AudioLLMs still take the trap reading in roughly 30-40% of conflict cases.

The judge listens to the user-side audio alongside the text conversation trace. For each user turn it reports whether the paralinguistic channel (prosody, emotion, speaking style) conflicts with the lexical content, and whether the assistant's next response was `audio_grounded`, took the `transcript_trap` reading, or was `unclear`. Only audio-native pipelines are evaluated — a cascade assistant never receives audio, so it takes the surface reading by construction.

> [!NOTE]
> By default, this diagnostic metric is excluded. Enable it explicitly with `--metrics audio_grounding` (or include it in a comma-separated `--metrics` list).

### Capabilities Measured

- **Speech Recognition (audio-native)**: Measures whether the model's responses reflect what the user's voice actually conveys, not just the transcribable words.

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3 Flash
- **Granularity**: Per user turn, aggregated to a per-record rate

### Input Data

Uses the following MetricContext fields:

- `audio_user_path`: Path to user-only audio file
- `conversation_trace`: Text of user and assistant turns (the judge compares the audio against the user turn's lexical content, then inspects the following assistant turn)

### Evaluation Methodology

For each user turn, the judge classifies:

- `has_disagreement`: whether the acoustic cues clearly conflict with the surface reading of the words
- `dimension`: the discourse dimension of the conflict — `interaction_behavior`, `emotion_state`, `dialogue_act`, `social_stance`, or `conversational_intent` (the five dimensions from the paper)
- `assistant_response`: `audio_grounded` (responds to what the voice conveys), `transcript_trap` (responds to the surface reading despite conflicting audio), or `unclear` (does not engage the user's conveyed state)

Records with no cross-modal disagreement turns are **skipped** (score `null`), not failed.

### Scoring

- **Scale**: 0-1
  - Parent score: conflict-turn grounding accuracy — fraction of disagreement turns where the assistant was audio-grounded (trap and unclear both count against it)
  - `audio_grounding.trap_rate`: fraction of conflict turns where the assistant took the transcript-biased reading (lower is better)
  - `audio_grounding.disagreement_rate`: fraction of rated user turns with cross-modal disagreement (describes the conversation, not the agent)
  - `audio_grounding.consistent_accuracy`: fraction of consistent turns where the assistant responded appropriately (higher is better)
- **Aggregation**: Mean across records; the consistent-vs-conflict gap (`consistent_accuracy` − parent score) mirrors the paper's headline comparison

### Producing conflict turns

The metric measures whatever conversations the dataset contains. To evaluate it, include records whose user simulator persona induces incongruent affect (e.g. a persona like *"You are increasingly frustrated but keep your wording polite and positive"* spoken by the TTS with the matching delivery) — the dataset's `user_persona` field is the intended lever, no code change required.

## Example Output

```json
{
  "name": "audio_grounding",
  "score": 0.0,
  "normalized_score": 0.0,
  "skipped": false,
  "details": {
    "num_turns": 2,
    "num_rated": 2,
    "num_conflict": 1,
    "num_trapped": 1,
    "conflicts_by_dimension": {"emotion_state": [0]},
    "per_turn": {
      "0": {
        "has_disagreement": true,
        "dimension": "emotion_state",
        "surface_interpretation": "pleased",
        "audio_interpretation": "bitter sarcasm",
        "assistant_response": "transcript_trap",
        "explanation": "Assistant responded cheerfully to a sarcastic turn."
      }
    }
  },
  "sub_metrics": {
    "trap_rate": {"score": 1.0},
    "disagreement_rate": {"score": 0.5},
    "consistent_accuracy": {"score": 1.0}
  }
}
```

## Related Metrics

- [faithfulness.md](faithfulness.md) - Text-layer grounding: whether responses are faithful to instructions, policies, and tool results (its S2S prompt already warns about audio perception errors)
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Output-side audio check: whether entities are correctly articulated in the agent's speech
- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Whether the user simulator itself behaved as scripted

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/audio_grounding.py`
- **Class**: `AudioGroundingMetric` → `AudioJudgeMetric`
- **Prompt**: `configs/prompts/judge.yaml` under `judge.audio_grounding`
- **Configuration**: `audio_judge_model` (default: Gemini 3 Flash), `judge_params`
- **Pipeline support**: S2S and audio-LLM only
