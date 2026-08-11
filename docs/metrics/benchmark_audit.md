# Benchmark Audit

> **Diagnostic Metric**: Reference-free LLM-judge audit of a single benchmark scenario's *definition* (not the conversation it produced). Useful for catching inconsistent, trivial, or policy-irrelevant scenarios before they skew evaluation results. Not scored into final aggregate scores — opt-in via `--metrics benchmark_audit`.

## Overview

Meta-evaluation metric that asks an LLM judge to rate the quality of each benchmark scenario on three reference-free axes — **consistency**, **complexity**, and **policy coverage** — and to return actionable diagnostics (concrete weaknesses). It gives a meta-evaluation layer over EVA's own benchmark scenarios, complementing the perturbation-analysis pipeline by scrutinizing the test environment itself rather than the system under test.

Adapted from "Benchmarking the Benchmarks: Evaluating Benchmarks for Conversational Agents" (arXiv:2608.06329).

### Capabilities Measured

- **Consistency**: Is the expected outcome coherent with the task description and free of internal contradictions?
- **Complexity**: Is the scenario non-trivial (multi-step, requires reasoning or tool use) rather than solvable in a single shallow turn?
- **Policy Coverage**: Does the scenario actually exercise the agent's stated instructions / policy (rules, constraints, allowed actions)?

## How It Works

### Evaluation Method

- **Type**: LLM judge (text only, reference-free)
- **Granularity**: Per-scenario (one audit per record, reading only the scenario-definition fields)

### Input Data

Uses the following `MetricContext` fields — the scenario-definition fields the source paper calls description / policy / expected_behavior:

- `user_goal`: the task description
- `agent_instructions`: the agent's policy (rules, constraints, allowed actions)
- `expected_scenario_db`: the ground-truth state expected after a successful conversation

No conversation transcript or audio is consumed.

### Rating Scale

Each dimension is rated 1-3 (1 = poor, 2 = acceptable, 3 = strong) and normalized to 0-1. The parent metric's `normalized_score` is the mean of the three normalized dimension scores; `score` is the mean of the raw dimension ratings. One sub-metric is emitted per dimension (`benchmark_audit.consistency`, `.complexity`, `.policy_coverage`).

### Output

In addition to per-dimension sub-metrics, `details` carries the judge's per-dimension explanations, a holistic `overall_rating`, and a `diagnostics` list of concrete, actionable weaknesses.

## Configuration

The judge model is configurable like other text-judge metrics:

```bash
# via CLI/env
JUDGE_MODEL=gpt-5.2

# or per-run via the metric config
--metrics benchmark_audit  # opt-in (excluded from default metrics and pass@k)
```
