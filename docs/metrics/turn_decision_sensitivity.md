# Turn Decision Sensitivity

> **Experience Metric**: yielding on every barge-in looks polite in isolation, but folding on backchannels makes the conversation feel broken. This metric checks that Yield/Keep decisions track context, not a fixed action preference.

## Overview

Code-based metric (no LLM) that scores the agent's **binary decision** on each user barge-in — yield the floor or keep talking — against the decision the barge-in's content calls for, then combines the two sides into a **pair accuracy** that gives no credit to constant-action policies. Adapted from the ECHO matched-contrast benchmark (arXiv:2609.17360v1).

This is deliberately complementary to [`turn_taking`](turn_taking.md): that metric grades *how well* the agent yields (every user interruption is scored against a fast yield); this one grades *whether* yielding was the contextually right action. A constant-yield agent scores near-perfectly on `turn_taking.user_interruption.mean_yield_score` while scoring `0.0` here.

## Scope

- **English-only in v0.1** — the backchannel lexicon is English. Records with another `language` are `skipped` with a reason, not scored.
- Evaluable events are turns in `user_interrupted_turns` (barge-ins flagged by the processor) that have **both** a user transcript and the timestamps needed to infer the agent's decision. Others are reported in `details.per_event` as skipped, not scored.
- The score is `None`/`skipped` when the record has **only one event class** (no backchannel-vs-substantive contrast exists to be sensitive to), or no events at all. Class accuracies are still emitted as sub-metrics.
- Excluded from pass@k (`exclude_from_pass_at_k = True`) because of those None scores.

## Inputs (from `MetricContext`)

- `user_interrupted_turns` — turns where the user barged in on agent speech.
- `transcribed_user_turns` — the barge-in transcripts driving the expected-action proxy.
- `audio_timestamps_user_turns` / `audio_timestamps_assistant_turns` — segment lists driving the observed-action inference.

## Expected action (backchannel proxy)

The transcript is lowercased, stripped of processor annotations (`[user interrupts]`, …) and punctuation, and tokenized. It classifies as a **backchannel** (expected action = **Keep**) when it is at most `BACKCHANNEL_MAX_WORDS = 4` words **and every word is in `BACKCHANNEL_WORDS`** — a closed set of continuers and floor-keeping encouragements ("yeah", "mm-hmm", "right", "okay", "sure", "go ahead", "please continue", …). Anything else is a **substantive interruption** (expected action = **Yield**). The closed-set rule is what makes the proxy hard to fool: substantive requests name content (dates, IDs, entities) that immediately leaves the lexicon.

## Observed action

`_sustain_ms` measures how long agent speech sustained **past the barge-in** (the start of the user's first segment in the turn). Floor-holding speech is the cut-off turn's tail (turn N−1 segments) plus any agent segment that starts before the user finishes speaking (a re-barge); segments starting after the user's turn ends are the settled next response and are excluded.

- sustained > `DECISION_WINDOW_MS` → observed **Keep** (talked through the user)
- otherwise → observed **Yield**

`DECISION_WINDOW_MS` shares `TurnTakingMetric.YIELD_HARD_MS` (2000 ms) so both metrics use one notion of "slowest acceptable yield". Note the split of concerns: a slow-but-real yield (e.g. stopping 1.9 s in) counts as a *correct decision* here, while `turn_taking` grades its speed separately.

## Scoring

Let `yield_accuracy` = correct Yields / substantive events and `keep_accuracy` = correct Keeps / backchannel events.

| Quantity | Definition |
|---|---|
| `yield_decision_accuracy` | correct Yields / substantive barge-ins |
| `keep_decision_accuracy` | correct Keeps / backchannel barge-ins |
| **`turn_decision_sensitivity.score`** | `yield_decision_accuracy × keep_decision_accuracy` |

The product is the fraction of (substantive, backchannel) contrast pairs decided correctly on **both** members — ECHO's pair-accuracy rule. A policy that always yields (or always keeps) maxes out one factor and zeroes the other, so its pair accuracy is 0.0 no matter how polished its yields look.

### Worked example

| Barge-in | Expected | Observed | Correct |
|---|---|---|---|
| "Actually, I need to change the flight to tomorrow" | Yield | agent stops 300 ms in | ✅ |
| "yeah" (backchannel) | Keep | agent stops 300 ms in | ❌ |

`yield_decision_accuracy = 1.0`, `keep_decision_accuracy = 0.0`, pair accuracy = **0.0** — while `turn_taking.user_interruption.mean_yield_score` on the same record is ≈ 0.93.

## Sub-metrics (flat)

| Key | When emitted |
|---|---|
| `num_barge_in_events`, `num_substantive_events`, `num_backchannel_events` | always (counts, unnormalized) |
| `yield_decision_accuracy` | when substantive events exist |
| `keep_decision_accuracy` | when backchannel events exist |

`details.per_event` carries, per barge-in turn: `expected`, `observed`, `correct`, `sustain_ms`, and a transcript snippet (or a `skipped` reason).
