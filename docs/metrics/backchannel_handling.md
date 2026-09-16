# Backchannel Handling

> **Experience Metric**: A backchannel ("mm-hmm", "yeah", "right") is listener encouragement, not a request for the floor. A system that drops its response because the user murmured "right" mid-answer feels broken no matter how correct its content is.

## Overview

Deterministic (no LLM judge) metric that detects **user backchannels among barge-in turns** and scores **how the assistant handled each one**: kept talking, paused briefly and resumed, or derailed.

`turn_taking` scores every user barge-in with a yield curve — appropriate for real interruptions ("wait, change it to Friday"), wrong for backchannels, where continuing is the desired behavior. This metric isolates the backchannel subset so the two behaviors are no longer conflated. The distinction mirrors the backchannel-robustness dimension highlighted in full-duplex speech-to-speech evaluation work (e.g. the Gander omni interaction agent, arXiv:2609.08977).

> [!NOTE]
> Opt-in metric — excluded from the default run. Enable with `--metrics backchannel_handling` (or add it to a comma-separated `--metrics` list).

### Capabilities Measured

- **VAD**: The system's endpointing must not treat a short overlapped vocalization as a full turn grab.
- **Language Model**: The policy for classifying overlapped speech as continuation-worthy vs. floor-yielding.
- **Pipeline**: The full-duplex loop (detection → decision → continued synthesis) staying on track through overlapped user audio.

## Inputs (from `MetricContext`)

| Field | Use |
|-------|-----|
| `user_interrupted_turns` | Candidate events — turns where the user barged in on assistant speech |
| `transcribed_user_turns` | Lexical classification of the barge-in (backchannel vocabulary vs. content) |
| `audio_timestamps_user_turns` | Backchannel timing (start/end, voiced duration) |
| `audio_timestamps_assistant_turns` | Continuation behavior on turns `t-1` (interrupted response tail) and `t` (following response) |

## Detection

A barge-in turn `t` counts as a **backchannel** when both hold:

1. **Pure continuer vocabulary** — every token of the turn's transcript is a backchannel token (`yeah`, `mm-hmm`, `right`, `okay`, ...) or a fixed phrase (`got it`, `makes sense`, `sounds good`, ...), with at most `BACKCHANNEL_MAX_WORDS` (4) tokens. Barge-ins containing content words stay ordinary interruptions for `turn_taking`.
2. **Short voicing** — total voiced user speech in the turn ≤ `BACKCHANNEL_MAX_SECONDS` (2.0 s).

An additional guard requires the previous turn's assistant speech to actually still be running at the barge-in moment; otherwise the interrupt flag is endpointing noise and the event is counted as `unscored`.

## Per-event Handling Score

Assistant segments from turns `t-1` and `t` are pooled, then:

| Reason | Condition | Score |
|--------|-----------|-------|
| `talked_through` | An assistant segment spans the backchannel's end (never stopped) | 1.0 |
| `resumed` | Assistant speech restarts `resume_ms` after the backchannel ends | 1.0 up to `RESUME_SWEET_SPOT_MS` (1000 ms), ramping linearly to 0.0 at `RESUME_HARD_MS` (4000 ms) |
| `abandoned` | Assistant speech stopped mid-response during the backchannel and nothing follows | 0.0 |
| `response_ending` | The assistant's speech had already ended within `NATURAL_END_GRACE_MS` (500 ms) of the barge-in — natural completion, nothing to continue | excluded (counted, not scored) |
| `unscored` | Backchannel detected but overlap signal missing | excluded (counted, not scored) |

Backchannels that land as the response was finishing are deliberately **not punished** — only mid-response derailment is.

## Main Score

`backchannel_handling.score` = mean of per-event handling scores.

Conversations with no detected backchannels (or none scoreable) are **skipped** (`score: null`, `skipped: true`), matching the skip idiom of `response_speed` / `authentication_success` — absence of backchannels is not a failure.

## Sub-metrics (flat)

| Key | Type | Always emitted | Description |
|-----|------|----------------|-------------|
| `num_backchannels` | count | when events exist | Detected backchannel events (incl. neutral/unscored) |
| `num_scored` | count | when events exist | Events that fed the score |
| `talked_through_rate` | 0–1 | when scored > 0 | Fraction of scored events the assistant talked through |
| `abandoned_rate` | 0–1 | when scored > 0 | Fraction of scored events abandoned mid-response |
| `mean_resume_ms` | ms | when any `resumed` | Mean restart delay after the backchannel |
| `barge_in_backchannel_rate` | 0–1 | when events exist | Share of the user's barge-ins that were backchannels |

## Details Fields

- `per_turn_evidence` — per event: `reason`, `transcript`, `bc_duration_ms`, and `resume_ms` / `score` where applicable
- `num_barge_in_turns`, `num_backchannels`, `num_scored`, `num_response_ending`, `num_unscored`

## Related Metrics

- [`turn_taking`](turn_taking.md) — scores the complementary signal: latency on clean turns, overlap behavior on agent interrupts, and yield behavior on *real* user interruptions.

## Attribution

The framing of backchannel communication as a distinct full-duplex robustness dimension (rather than a special case of interruption) follows the evaluation taxonomy of the Gander omni interaction agent technical report (arXiv:2609.08977). The implementation is EVA-native: deterministic rules over existing `MetricContext` audio timestamps and transcripts.
