# Floor Control

> **Diagnostic Metric**: Raw interruption counts conflate genuine floor takeovers with harmless overlapped speech. This metric splits those events apart and scores *selective* interruption handling — the distinction benchmarked by [MultiPartyTurnBench](https://arxiv.org/abs/2610.20172) (MPTB) — which turn-taking's binary interrupt flags cannot express.

## Overview

Deterministic diagnostic that re-analyzes the interruption events already flagged by the processor, classifying each one as a genuine floor takeover or a confusable non-event, and scores how selectively the agent handled them. No LLM, no new processor data — everything derives from the per-turn audio timestamps on `MetricContext` (the same signals `turn_taking` consumes).

### Capabilities Measured

- **VAD / Pipeline**: Turn-entry and interruption timing behavior — entering at genuine turn ends, yielding the floor when the user takes it, holding through backchannels.

## How It Works

### Event classification

**Agent overlaps** (`assistant_interrupted_turns` — any assistant audio overlapping user speech) split into:

| Class | Condition | Meaning |
|---|---|---|
| `takeover` | User speaks ≤ 500 ms after the agent's first overlapping entry (`TAKEOVER_MAX_TAIL_MS`) | Genuine interruption — the floor actually transferred |
| `overlap_only` | User continues > 500 ms past the entry | Non-floor-taking overlap (confusable non-event) — the user talked through it |

`turn_taking` treats both identically; MPTB's finding is that distinguishing them matters, because non-taking overlaps are far less disruptive than takeovers.

**User barge-ins** (`user_interrupted_turns`) split by the user's total speech in the turn:

| Class | Condition | Meaning |
|---|---|---|
| `takeover` | ≥ 1000 ms of user speech (`BACKCHANNEL_MAX_SPEECH_MS`) | Genuine attempt to take the floor — the agent should yield |
| `backchannel` | < 1000 ms | Non-floor-taking vocalization ("mm-hmm") — the agent should hold |

### Selective interruption handling

For each classified barge-in, the agent's action is derived from how long its speech continued past the barge-in onset (`previous turn's last assistant end − this turn's first user start`, mirroring `turn_taking`'s yield signal, sign preserved):

- `yielded` — speech ended within `YIELD_WINDOW_MS` (1000 ms)
- `held` — speech continued past the window
- *(excluded)* — agent had already stopped before the barge-in (no floor to yield or hold)

| Sub-metric | Definition |
|---|---|
| `user_interruption.yield_accuracy` | Share of takeovers the agent yielded to |
| `user_interruption.hold_accuracy` | Share of backchannels the agent talked through |
| `user_interruption.selective_accuracy` | Balanced accuracy over both classes — **the main score** |

An agent that yields to everything (MPTB: Gemini-3.8-Live yields to 97.7% of backchannels) or nothing (MiniCPM-o-4.5 yields to 1.4% of genuine interruptions) lands near 0.5 — chance level. Only genuinely selective behavior scores high.

### FP-constrained takeover recall

`user_interruption.fp_constrained_takeover_recall` sweeps a threshold over the agent's speech-continuation after each barge-in ("fire" = treat it as an interruption to yield to) and reports the maximum takeover recall reachable while yielding to at most `FP_BUDGET` (20%) of backchannels. This is MPTB's false-positive-constrained recall, computed from observed behavior rather than from MPTB's controlled annotations.

### Signed entry latency

`agent_interruption.mean_signed_entry_ms` — mean signed latency of the agent's first overlapping entry relative to the user's last speech end (always ≤ 0; deeper anticipation = more negative). Complements `turn_taking`'s unsigned `mean_overlap_ms`.

## Sub-metrics Summary

| Sub-metric | Emitted when | Direction |
|---|---|---|
| `agent_interruption.takeover_rate` | any agent overlap | lower is better |
| `agent_interruption.takeover_count` | any agent overlap | count |
| `agent_interruption.mean_signed_entry_ms` | any agent overlap | ms (negative) |
| `user_interruption.yield_accuracy` | ≥ 1 measurable takeover | higher is better |
| `user_interruption.hold_accuracy` | ≥ 1 measurable backchannel | higher is better |
| `user_interruption.selective_accuracy` | ≥ 1 measurable class | higher is better |
| `user_interruption.fp_constrained_takeover_recall` | both classes measurable | higher is better |

## Scope

- Uses the same evaluable-turn set as `turn_taking` (turns where both user and assistant audio timestamps are non-empty; greeting turn 0 excluded).
- Skipped (`skipped = true`, no score) when no evaluable turns exist or no interruption events of either kind occurred — there is nothing to diagnose.
- Per-event classifications are preserved in `details.agent_overlap_events` / `details.user_barge_in_events` for record-level inspection.

## Provenance

Event taxonomy (EOT/hold vs. takeover/backchannel), false-positive-constrained recall, signed latency, and the selective-handling framing are adapted from [MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue](https://arxiv.org/abs/2610.20172). MPTB evaluates against exact event-conditioned annotations in controlled multi-party settings; this metric substitutes parameter-free timestamp heuristics (tail duration, speech duration, yield window) on eva's dyadic conversations.
