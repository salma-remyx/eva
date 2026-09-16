# Backchannel Handling

> **Experience Metric**: A backchannel ("mm-hmm", "yeah", "right") is listener encouragement, not a request for the floor. A system that drops its response because the user murmured "right" mid-answer feels broken no matter how correct its content is.

## Overview

Deterministic (no LLM judge) metric that detects **user backchannels among barge-in turns** and scores **whether the assistant kept talking through each one**. Per the reference definition, a pause followed by a resume is *not* a pass — only continuing without pausing is.

`turn_taking` scores every user barge-in with a yield curve — appropriate for real interruptions ("wait, change it to Friday"), wrong for backchannels, where continuing is the desired behavior. This metric isolates the backchannel subset so the two behaviors are no longer conflated. The distinction mirrors the backchannel-robustness capability highlighted in full-duplex speech-to-speech interaction work (e.g. the Gander omni interaction agent, arXiv:2609.08977).

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

1. **Pure continuer vocabulary within the paper's short-length threshold** — every token of the turn's transcript is a backchannel token (`yeah`, `mm-hmm`, `right`, `okay`, ...) or a fixed phrase (`got it`, `makes sense`, `sounds good`, ...), with at most `BACKCHANNEL_MAX_WORDS` (6) words — the reference's acceptance threshold of six words in English. Barge-ins containing content words stay ordinary interruptions for `turn_taking`.
2. **Embedded in ongoing assistant speech** — the previous turn's assistant speech must still be running at the barge-in moment (the paper's "embedded within the interlocutor's ongoing utterance"); otherwise the interrupt flag is endpointing noise and the event is counted as `unscored`.

The reference samples backchannel realizations from a 170-expression English lexicon (11 intent categories) that is not released in the code; this metric's closed continuer set is a conservative stand-in, and it admits no interrogative, requestive, or negative-stance expressions, per the paper's demotion rule.

## Per-event Handling Score

Assistant segments from turns `t-1` and `t` are pooled, then scored against the reference's ground truth for a supportive backchannel — the assistant "retains the floor and continues without pausing":

| Reason | Condition | Score |
|--------|-----------|-------|
| `talked_through` | An assistant segment spans the backchannel's end (never stopped) | 1.0 |
| `resumed` | Assistant speech restarts `resume_ms` after the backchannel ends (it paused) | 0.0 — `resume_ms` kept as diagnostic evidence |
| `abandoned` | Assistant speech stopped mid-response during the backchannel and nothing follows | 0.0 |
| `unscored` | Backchannel detected but overlap signal missing | excluded (counted, not scored) |

There is deliberately **no resume-tolerance band**: a system that habitually pauses ~900 ms at every backchannel scores 0, exactly as it would under the reference's continues-without-pausing definition.

## Main Score

`backchannel_handling.score` = mean of per-event handling scores.

Conversations with no detected backchannels (or none scoreable) are **skipped** (`score: null`, `skipped: true`), matching the skip idiom of `response_speed` / `authentication_success` — absence of backchannels is not a failure.

## Sub-metrics (flat)

| Key | Type | Always emitted | Description |
|-----|------|----------------|-------------|
| `num_backchannels` | count | when events exist | Detected backchannel events (incl. unscored) |
| `num_scored` | count | when events exist | Events that fed the score |
| `talked_through_rate` | 0–1 | when scored > 0 | Fraction of scored events the assistant talked through |
| `abandoned_rate` | 0–1 | when scored > 0 | Fraction of scored events abandoned mid-response |
| `mean_resume_ms` | ms | when any `resumed` | Mean restart delay after paused backchannels — diagnostic only; the pause already scores 0 |
| `barge_in_backchannel_rate` | 0–1 | when events exist | Share of the user's barge-ins that were backchannels |

## Details Fields

- `per_turn_evidence` — per event: `reason`, `transcript`, `bc_duration_ms`, and `resume_ms` / `score` where applicable
- `num_barge_in_turns`, `num_backchannels`, `num_scored`, `num_unscored`

## Related Metrics

- [`turn_taking`](turn_taking.md) — scores the complementary signal: latency on clean turns, overlap behavior on agent interrupts, and yield behavior on *real* user interruptions.

## Attribution

The framing of backchannel communication as a distinct full-duplex capability (rather than a special case of interruption) follows the Gander technical report — "Gander: A Native Duplex Interaction Model with an Asynchronous Agent Loop" (arXiv:2609.08977). This metric adopts the paper's acceptance rule (embedded in the interlocutor's ongoing utterance, under six English words, lexicon-matched) and its ground truth for correct handling (the assistant retains the floor and continues without pausing).

Two scope limits are deliberate, not omissions to fix later:

- **User-side robustness only.** The reference's central backchannel mechanism is agent-side *production*: the model emits a `<|backchannel|>` control action — a decision predicted before content, part of the speak-action token set at inference. Model control decisions are not observable in EVA's `MetricContext` (transcripts + audio timestamps), so this metric scores only the complementary direction shown in the paper's backchannel examples: the assistant continuing through the *user's* backchannel.
- **Deterministic outcome measurement, not a benchmark port.** The paper measures interaction timing with Full-Duplex-Bench v3, an external LLM-judged benchmark over a model that predicts its control decision per one-second unit. This metric is EVA-native — deterministic rules over `MetricContext` with no LLM judge — evaluating the outcome of the capability rather than porting the control loop or the benchmark protocol.

