# Floor Control

> **Experience Metric (opt-in)**: a full-duplex agent shares the audio channel with the user — grabbing the floor during the user's pause, going silent after a backchannel, or ignoring an interruption all break the conversation even when the content is correct.

## Overview

Code-based metric (no LLM) that scores three floor-control behaviours from the per-turn audio timeline already stored on `MetricContext`:

1. **Pause handling** — during a user *mid-utterance pause* (a silence gap between two speech segments of the same turn), the agent should hold the floor. Speaking inside the pause is a **takeover**.
2. **Interruption response** — after the user barges in with a *substantive* interruption, the agent should speak again in response once the user finishes.
3. **Backchannel resume** — after a *short* user backchannel ("mm-hmm") during the agent's speech, the agent should keep speaking and finish its response.

These are the temporal behaviours that separate full-duplex speech-to-speech systems from half-duplex cascades; they are the signals full-duplex benchmarks report on (pause-handling takeover rate, takeover/response after user interruptions, backchannel resume rate). Adapted from *NemotronLabs VoiceChat: An Open Full-duplex Speech-to-Speech Model with Tool Calling Capabilities* (arXiv:2609.21967) — the behaviours are measured on EVA's own conversation recordings rather than the paper's bespoke benchmark harness, and the paper's judged post-interruption response-quality panel is out of scope (EVA's `turn_taking` already captures post-interrupt latency).

**Opt-in** — excluded from the default run; enable via `--metrics floor_control`.

## Scope

- **Greeting (turn 0) is excluded.**
- All signals come from `audio_timestamps_user_turns` / `audio_timestamps_assistant_turns` and `user_interrupted_turns`. No transcripts, no judge model.
- A conversation with **no floor-control events at all** (strictly alternating single-segment turns, no barge-ins) returns `score = None` with `skipped = True`, so it doesn't inflate cross-record averages.

## Event Detection

### Mid-utterance pause

Two consecutive user segments of the same turn with a silence gap in `[PAUSE_MIN_MS, PAUSE_MAX_MS]` (default `[500, 3000]` ms). The later segment is what makes the pause *mid-utterance* — the user resumed, so the floor was expected to stay with them.

**Takeover** = an assistant segment *in the same turn* intersecting the pause window. Assistant speech that begins only after the user resumed is overtalk (`turn_taking`'s overlap signal), not pause takeover; assistant audio from the previous turn spilling across the boundary is likewise not attributed to this turn.

### Backchannel vs. substantive interruption

Every turn in `user_interrupted_turns` (the user started speaking while assistant audio was active) is classified by the user's **total speech duration in the turn**:

| User speech in the interrupt turn | Classified as | Expected agent behaviour |
|---|---|---|
| ≤ `BACKCHANNEL_MAX_MS` (1500 ms) | backchannel | keep speaking / resume → **resumed** |
| > 1500 ms | substantive interruption | stop, then speak in response → **responded** |

Both use the same audible-outcome test: an assistant segment in the turn **ends after the user's last speech** (the agent is speaking once the user has finished). The interruption event additionally records `response_delay_ms` — the gap from the user's last speech to the first assistant segment starting at or after it; `0.0` when an assistant segment spans the user's end (the agent never stopped). Turns in `user_interrupted_turns` without user audio are dropped.

## Score

Each component counts only when its events exist, and each lives on `[0, 1]`:

| Component | Definition | Direction |
|---|---|---|
| `pause_handling` | `1 − takeover_rate` over mid-utterance pauses | higher = better |
| `interruption_response` | fraction of substantive interruptions with a spoken response | higher = better |
| `backchannel_resume` | fraction of backchannels after which the agent resumed | higher = better |

`floor_control.score = floor_control.normalized_score = mean(available components)`; `skipped` when none are available.

## Sub-metrics (flat)

Emitted as `sub_metrics` on the main `MetricScore`; the runner aggregates each into its own column.

| Key | Normalized? | When present | Meaning |
|---|---|---|---|
| `pause_handling.takeover_rate` | yes — **lower is better** (`_rate` suffix) | pauses exist | Fraction of mid-utterance pauses where the agent took the floor. |
| `pause_handling.mean_pause_ms` | no | pauses exist | Mean detected pause duration. |
| `pause_handling.num_pauses` | no (raw count) | always; `score=None` when 0 | Number of detected mid-utterance pauses. |
| `interruption.response_score` | yes | interruptions exist | Fraction of substantive interruptions the agent responded to. |
| `interruption.mean_response_delay_ms` | no | ≥ 1 response delay available | Mean gap from the user's last speech to the agent's response. |
| `interruption.num_events` | no (raw count) | always; `score=None` when 0 | Number of substantive interruptions. |
| `backchannel.resume_score` | yes | backchannels exist | Fraction of backchannels after which the agent resumed speaking. |
| `backchannel.num_events` | no (raw count) | always; `score=None` when 0 | Number of backchannels. |

Count sub-metrics carry `score=None` on clean runs so cross-record aggregates exclude them rather than averaging in zeros.

## Details Fields

`details` on the main `MetricScore` contains:

| Field | Description |
|---|---|
| `pause_events` | List of `{turn_id, pause_start_s, pause_end_s, pause_ms, took_over}`. |
| `interruption_events` | List of `{turn_id, user_speech_ms, agent_spoke_after, responded, response_delay_ms}`. |
| `backchannel_events` | List of `{turn_id, user_speech_ms, agent_spoke_after, resumed}`. |
| `num_pause_events` / `num_interruption_events` / `num_backchannel_events` | Event counts. |
| `components` | The per-behaviour component values that fed the score (omitted when skipped). |

## Tunable Constants

All thresholds live as class-level attributes on `FloorControlMetric`. Override by subclassing or editing in place.

| Constant | Default | Purpose |
|---|---|---|
| `PAUSE_MIN_MS` | 500 | Shortest silence gap that counts as a deliberate mid-utterance pause (shorter gaps are ordinary VAD splitting). |
| `PAUSE_MAX_MS` | 3000 | Longest gap that still counts — beyond it the user plausibly finished the utterance. |
| `BACKCHANNEL_MAX_MS` | 1500 | User speech duration in an interrupt turn at or below which the barge-in is a backchannel rather than a substantive interruption. |

## Example Output

```json
{
  "name": "floor_control",
  "score": 0.8333,
  "normalized_score": 0.8333,
  "details": {
    "pause_events": [
      {"turn_id": 1, "pause_start_s": 2.0, "pause_end_s": 2.8, "pause_ms": 800.0, "took_over": false},
      {"turn_id": 2, "pause_start_s": 8.0, "pause_end_s": 9.0, "pause_ms": 1000.0, "took_over": true}
    ],
    "interruption_events": [
      {"turn_id": 4, "user_speech_ms": 2500.0, "agent_spoke_after": true, "responded": true, "response_delay_ms": 500.0}
    ],
    "backchannel_events": [
      {"turn_id": 3, "user_speech_ms": 600.0, "agent_spoke_after": true, "resumed": true}
    ],
    "num_pause_events": 2,
    "num_interruption_events": 1,
    "num_backchannel_events": 1,
    "components": {"pause_handling": 0.5, "backchannel_resume": 1.0, "interruption_response": 1.0}
  },
  "sub_metrics": {
    "pause_handling.takeover_rate":         {"score": 0.5,   "normalized_score": 0.5},
    "pause_handling.mean_pause_ms":         {"score": 900.0, "normalized_score": null},
    "pause_handling.num_pauses":            {"score": 2.0,   "normalized_score": null},
    "interruption.response_score":          {"score": 1.0,   "normalized_score": 1.0},
    "interruption.mean_response_delay_ms":  {"score": 500.0, "normalized_score": null},
    "interruption.num_events":              {"score": 1.0,   "normalized_score": null},
    "backchannel.resume_score":             {"score": 1.0,   "normalized_score": 1.0},
    "backchannel.num_events":               {"score": 1.0,   "normalized_score": null}
  }
}
```

## Related Metrics

- [`turn_taking`](turn_taking.md) — complementary per-turn timing scores (latency curves, overtalk overlap, yield after barge-ins). Floor control covers the *between-segment* behaviours turn-taking's per-turn signals don't isolate.
- [`response_speed`](response_speed.md) — raw latency diagnostics.

## Implementation Details

- **File**: `src/eva/metrics/experience/floor_control.py`
- **Class**: `FloorControlMetric`
- **Base class**: `CodeMetric`
