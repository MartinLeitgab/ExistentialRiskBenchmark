# Existential Risk Preferences Benchmark

A benchmark for measuring how large language models respond to preference scenarios that bear on existential risk — specifically, the degree to which models exhibit **instrumentally convergent (IC)**, **pro-human (PH)**, or **anti-human (AH)** preferences when faced with high-stakes operational decisions.

The benchmark pairs preferences across these three categories into forced-choice scenarios, sweeps each scenario across a controlled grid of prompt variants (framing, deployment context, directive strength, etc.), and uses an LLM-as-judge stack to score model responses along six dimensions including preference signal strength, strategic responding, and instrumental-vs-terminal reasoning.

License: Apache 2.0.

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline_a_scenarios/` | Scenario generation, prompt variant sweeps, LLM-as-judge evaluation |
| `pipeline_b_evaluation/` | Per-model evaluation runner (scaffolded only) |
| `pipeline_c_analysis/` | Statistical aggregation, Elo, bootstrap CIs (scaffolded only) |
| `data/scenarios/` | Seed scenarios (Phase 1 protos, within-category seeds, Phase 2 seeds) |
| `data/metadata/` | Per-user cost-tracking JSONL files |
| `docs/` | Methodology, scenario-creation guidelines, prompt dimensions, setup |
| `tests/` | Unit tests (mock LLM clients) + integration tests (real API calls) |
| `paper/`, `notebooks/`, `outputs/` | Paper draft, exploratory notebooks, run artifacts |

## Architecture

Three sequential pipelines, run end-to-end per benchmark cycle:

```
Pipeline A                     Pipeline B                     Pipeline C
─────────────                  ─────────────                  ─────────────
Generate scenarios     ──▶     Run scenarios against  ──▶    Aggregate, score,
+ prompt variants              N target models                rank, bootstrap CIs
+ judge prompts                (batch + streaming)            (Elo, suppression
                                                              matrix, plots)
```

### Pipeline A — Scenarios (implemented)

Generates a 5,355-scenario candidate pool (153 unique preference pairs × 35 scenarios per pair) from 18 preferences across three categories (IC1–IC6, PH1–PH6, AH1–AH6) — see `pipeline_a_scenarios/config/preferences_taxonomy.yaml`. Each scenario is a forced-choice between two action options that operationalize different preferences.

Each scenario is then expanded across **9 prompt dimensions** (framing, exfiltration, response_format, system_prompt, temperature, action_order, ethical_framing, value_conflict, goal_specification) per `docs/prompt_dimensions.md`. Each combination gets a deterministic prompt ID encoding all dimension values (e.g., `fp-abs-3o-auto-t7-reg-cons-exp-phd`).

The `goal_specification` dimension splits into calibration variants (`ic_directive` / `ah_directive` / `ph_directive`) used as ceiling/floor anchors in the suppression matrix, separate from the main candidate pool ranked in Phase 1.

**Core components:**

- `utils/llm_client.py` — `UnifiedLLMClient` abstracts Anthropic / OpenAI / Gemini behind one interface; `generate()` for single-shot, `submit_batch()` / `retrieve_batch_results()` for batch (Gemini batch doesn't support thinking → falls back to parallel-threaded `submit_gemini_parallel()`). Built-in token-bucket rate limiting, 3-attempt exponential-backoff retry, response caching.
- `utils/prompt_generator.py` — emits all prompt-dimension combinations with deterministic IDs.
- `utils/judge_client.py` + `utils/judge_prompts.py` — LLM-as-judge stack; single combined prompt scores each response on six dimensions (preference signal strength, strategic responding, coherence, reasoning depth, format compliance, instrumental-vs-terminal reasoning) with XML output.
- `utils/judge_analysis.py` — `aggregate_by_variant()` and helpers for grouping evaluations by model / variant / scenario.
- `utils/cost_tracker.py` — per-user JSONL cost log at `data/metadata/costs_<username>.jsonl`; $200/month + $1000 total budget with 80% threshold alerts; reasoning-token multipliers per provider (Anthropic 2-3×, OpenAI reasoning 10×, Gemini 8×).

**Pipeline A scripts** (status as of 2026-05-27):

| Script | Purpose | Status |
|---|---|---|
| `create_prototypes.py` | Validate / generate scenario templates from Phase 1 seeds | Implemented |
| `prompt_validation.py` | PIPE-A7 Phase 1: rank prompt variants against 6 seed scenarios | Merged (PR #30) |
| `suppression_matrix.py` | PIPE-A7 Phase 1-B: 2D suppression matrix (deployment context × directive framing) | Merged (PR #30) |
| `generate_scenarios.py` | PIPE-A3: automated expansion to 5,355 scenarios | In review (PR #41) |
| `analyze_batch_results.py` | Batch-result aggregation helpers | Implemented |
| `batch_variant_testing.py` | Batch variant-sweep runner | Implemented |

**Seed data:**

- `data/scenarios/seeds_phase1.json` — 6 Phase 1 prototype scenarios (v4 schema, merged via PR #44 on 2026-05-22). Quality baseline + few-shot anchors for PIPE-A3 expansion. Each contains the `current_use_context` field consumed by `suppression_matrix.py` for the "current_use" cell of the deployment-framing axis.
- `data/scenarios/seeds_within_category.json` — 3 within-category reference seeds (IC×IC, PH×PH, AH×AH); pending team validation before PIPE-A3 within-category run.
- Phase 2 seeds (75 scenarios) — in review (PR #39).

### Pipeline B — Evaluation (scaffolded only)

Directory structure (`pipeline_b_evaluation/{config,utils,utils/api_clients}/`) with empty `__init__.py` files. No executable code yet. Tracked under issue #11 (PILOT-1), #12 (PILOT-2 execution), #13 (PILOT-3 parsing).

Planned scope: take a frozen scenario × prompt-variant set from Pipeline A and run it against N target models, persisting raw responses + per-call metadata for Pipeline C.

### Pipeline C — Analysis (scaffolded only)

Directory structure (`pipeline_c_analysis/{config,models,statistics}/`) with empty `__init__.py` files. No executable code yet. Tracked under issue #14 (PILOT-4: Elo + bootstrap CIs).

Planned scope: aggregate Pipeline B outputs, apply pre-processing filters (exclude `is_calibration_run=True`, exclude `authenticity_score < 40`, invert action mapping when `action_order=inverted`, map `equal`/`free_text` to draws, exclude `neither`/`unparseable`), compute Elo rankings, bootstrap confidence intervals, and produce the suppression-matrix plots.

## Current status (2026-05-27)

**Phase 1 closeout — IN PROGRESS:**

| Item | Status |
|---|---|
| PR #30 (prompt_validation + suppression_matrix) | Merged |
| PR #44 (Phase 1 v4 prototype scenarios) | Merged 2026-05-22 |
| Scenario creation guidelines (`docs/scenario_creation_guidelines.md`) | Updated with §4.1 verb-agreement rule, §10 canonical `current_use_context` formula, §11 PH1-recognizable framing finding |
| Phase 1 re-run on v4 protos | Pending |
| PR #41 (PIPE-A3 `generate_scenarios.py`, Ishan) | In review |
| PR #39 (Phase 2 seeds, 75 scenarios) | In review |

**Implemented (Pipeline A):** prompt generation, multi-provider LLM client, batch + streaming inference, LLM-as-judge stack, cost tracking, prompt-variant ranking, 2D suppression matrix, prototype seed scenarios.

**Planned next (in dependency order):**

1. Close out Phase 1 re-run with v4 protos + Sonnet 4.6 + fail-loud judge → final variant rankings.
2. PIPE-A3 (PR #41): scale to 5,355 scenarios using Anthropic batch.
3. PIPE-A4 (issue #10): in-team validation pass on generated scenarios.
4. INFRA-3 (issue #15): expert validation of preference taxonomy + scenarios.
5. PILOT-1/2/3 (issues #11/#12/#13): build Pipeline B and run pilot evaluation across target models.
6. PILOT-4 (issue #14): build Pipeline C — Elo, bootstrap CIs, suppression-matrix plots.
7. MILESTONE-1 (issue #1) + paper outline (PILOT-6, issue #17).

See `docs/forward_plan.md` for the full execution sequence with cost estimates and decision gates.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pre-commit install

# API keys in .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

pytest                                # unit/mock tests only
pytest -m integration                 # real API calls
python -m pipeline_a_scenarios.utils.cost_tracker --dashboard
```

Full installation, environment, and testing details: `docs/SETUP.md` and `docs/Testing.md`.

## Key references

- `docs/forward_plan.md` — execution sequence, cost estimates, decision gates
- `docs/scenario_creation_guidelines.md` — scenario design rules + PH1 recognizability findings
- `docs/Preference_taxonomy.md` — 18-preference taxonomy (IC/PH/AH)
- `docs/prompt_dimensions.md` — 9 prompt dimensions + IDs
- `docs/Cost_tracker.md` — cost-tracking design + budget thresholds
- `docs/DOCUMENTATION_STANDARDS.md` — Google-style docstrings, type hints

## Citation

Paper in preparation. Citation will be added on publication.

## License

Apache 2.0 — see `LICENSE`.
