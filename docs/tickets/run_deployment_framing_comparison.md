# Mini-ticket: Powered autonomous-vs-current_use deployment-framing comparison

**Status:** draft (unassigned — to be offered to the team)
**Cost:** ~$5 model + ~$5 judge (mirrors one Phase-1 candidate arm)
**Created:** 2026-06-11

---

## Why

The research question "how sensitive is preference extraction to deployment framing
(post-exfiltration autonomous scenario body vs. current-use assistant-framed body)?" is
captured (`findings.md` §9, §4; `forward_plan.md` Step 1-B comparison #1) but is only
powered at **n=6 per cell** inside the suppression matrix — a smoke test, not a measurement.
The main variant-ranking arm always uses the autonomous `context` body; the
`current_use_context` body is only exercised at n=6. To make any framing-sensitivity claim
in the paper we need both framings run at power on the same scenarios and prompt.

**Pre-req / do first:** fix the `deployment_context` mislabel (see
`docs/pipeline_fixes_for_riccardo.md`) so the output metadata correctly distinguishes the
two framings. Otherwise every row is tagged `current_use` regardless.

## Design

- Scenarios: 6 proto_*_v4 seeds (each already carries both `context` and `current_use_context`).
- Framing arm A: autonomous body (`context`) + `system_prompt=autonomous`.
- Framing arm B: current-use body (`current_use_context`) + `system_prompt=assistant`.
- Prompt: production variant `fp-abs-3o-auto-t10-reg-0-0-0` for arm A; the matched
  assistant-system variant for arm B (hold all other dimensions constant).
- Models: `claude-opus-4-8`, `gpt-5.5`, `gemini-3.1-pro-preview`.
- Runs: 2. n per cell = 6 scenarios × 2 runs = 12 (×3 models). For real power, optionally
  add the top-3 candidate variants → n=36/cell.
- Judge-A on all responses.

## Key comparison

Per (scenario, model): choice-rate(arm B) − choice-rate(arm A), and IVT(arm B) − IVT(arm A).
A systematic shift = deployment framing materially changes expressed preference → reportable.
Null = preference is framing-robust (supports the §9 "we measure the anchor" claim).

## Inputs (all present)

- `data/scenarios/seeds_phase1.json` — both body fields per seed.
- `prompt_generator.generate_prompt` — supports `system_prompt` dimension.
- `suppression_matrix.py` already maps `current_use_context` → current-use cell (reuse logic).
- Runner: thin wrapper over `prompt_validation.py` with a two-arm scenario loader, or extend
  `suppression_matrix.py` to run the `absent` directive column at higher n.

## Outputs

- `outputs/deployment_framing/framing_comparison.json`
- One table: per (scenario, model) choice-rate + IVT deltas, arm B − arm A, with CIs.
- A findings.md paragraph: framing sensitivity confirmed or null, with scope.
