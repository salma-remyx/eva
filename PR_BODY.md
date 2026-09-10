## Summary

Adds opt-in three-tier persona vectors for the user simulator, adapted from
["A Three-Tier Persona Vector for Controllable User Simulation in Agentic Evaluation"](https://arxiv.org/abs/2609.08592).
Flat role descriptions ("you are an angry customer") produce near-identical
conversations regardless of scenario; a persona vector instead gives each
simulated caller six categorical demographics (tier 1), twelve continuous
behavioral traits sampled with seeded Gaussian noise around curated base
vectors (tier 2), and five emotional states that shift with scenario pressure
(tier 3), plus an orthogonal four-level query-complexity overlay.

A record opts in by carrying a `persona_vector` entry in its `user_config`.
Call site: record `user_config` → `resolve_user_config` (`src/eva/orchestrator/worker.py:356`)
→ `AbstractUserSimulator._build_prompt` (`src/eva/user_simulator/base.py:91`).
Records without a `persona_vector` keep the `default` behavior prompt unchanged;
behavior perturbations, when enabled, still take precedence.

`simulation_version` moves 2.0.1 → 2.1.0 in this PR: instructions for opted-in
records change, so their benchmark outputs are not comparable across the bump.
Non-opted-in records see zero change.

## Changes

- `src/eva/user_simulator/persona_vector.py` — new module: eight named profile
  base vectors, seeded Gaussian trait sampling, seven correlation rules
  (audited in `correlation_adjustments`), scenario-pressure shifts, and
  fragment rendering.
- `src/eva/user_simulator/base.py` — new persona-vector `elif` branch in
  `AbstractUserSimulator._build_prompt`, between the perturbation branch and
  the `default` fallback.
- `README.md` — "Persona Vectors for the User Simulator" section: opt-in
  documentation plus a worked `user_config` JSON example.
- `tests/unit/user_simulator/test_persona_vector.py` — new test file covering
  dispatch/precedence, seed reproducibility, overrides, scenario shifts, and
  rejection of unknown profiles/keys.
- `src/eva/__init__.py` — `simulation_version` 2.0.1 → 2.1.0 (new opt-in
  simulation capability; no metric logic or judge prompt changed, so
  `metrics_version` and per-metric `version` fields are untouched).
- `tests/fixtures/metric_signatures.json` — regenerated with
  `python scripts/regen_metric_signatures.py`; output is byte-identical
  (no metric drift), and `tests/unit/metrics/test_metric_signatures.py` passes.

## Testing

- `pytest tests/unit/user_simulator/test_persona_vector.py` — 43 passed, 0 failed.
- `pytest tests/unit` — 1586 passed, 52 skipped, 3 xfailed; 298 errors, every
  one of which is "async def functions are not natively supported" because this
  sandbox cannot install the `pytest-asyncio` dev dependency (`asyncio_mode =
  "auto"` in `pyproject.toml`); zero assertion failures. CI, which installs the
  dev extras, is the authoritative full-suite run.
- Metric-signature drift test passes against the regenerated fixture.
- ruff / mypy: not run here (tools unavailable in this sandbox); CI covers both.
- Live end-to-end render (offline, deterministic — no LLM credentials in this
  sandbox): a record-style `user_config`
  (`{"name": "{first_name}", "user_persona_id": 2, "persona_vector": {"profile": "impatient_executive"}}`)
  resolved through the real `resolve_user_config` path and fed to a real
  `OpenAIRealtimeUserSimulator` renders the persona fragment inside the session
  instructions (12103 chars total), e.g.:
  ```
  Simulated caller profile: impatient_executive (sampled persona vector — stay in character).
  Caller context — age: 45-54; channel: phone; device: smartphone; time availability: rushed.
  Stable behavioral traits: You have little patience: push the agent to hurry and show irritation at delays...
  Trait vector: patience 0.06, assertiveness 1.00, digital_literacy 0.84, verbosity 0.50, ...
  Entering the call you feel: frustration 0.78 (high), anxiety 0.55 (moderate), trust 0.28 (low), ...
  Query phrasing (complexity level: direct): Phrase your requests plainly: state exactly what you need, with all details, up front.
  ```
- Version bump: `simulation_version` 2.0.1 → 2.1.0 for result comparability —
  opted-in records render a different caller persona, so their transcripts and
  scores are not comparable across the bump; records without a `persona_vector`
  are byte-identical to before (covered by
  `test_without_persona_vector_prompt_is_unchanged`).

## Limitations

- Tier-3 scenario pressure is a keyword-pressure proxy derived from the
  record's goal (urgency/adversity tokens), not a learned context model; the
  seven trait correlations are deterministic rule-described nudges, not a
  learned covariance matrix — both keep the sampling auditable and
  reproducible, at the cost of expressiveness.
- Behavior perturbations still take precedence over the persona vector when
  both are configured.
- Persona vectors drive the caller's instructions only; no dataset records opt
  in yet, so default-run benchmark outputs are unchanged.

*Adapted from "A Three-Tier Persona Vector for Controllable User Simulation in
Agentic Evaluation" (arXiv:2609.08592). Reached via the standard simulation
path — the orchestrator worker resolves each record's `user_config`
(`src/eva/orchestrator/worker.py:356`), hands it to the simulator as
`persona_config`, and `_build_prompt` (`src/eva/user_simulator/base.py:91`)
renders the fragment for both caller providers (ElevenLabs Agents and OpenAI
Realtime).*
