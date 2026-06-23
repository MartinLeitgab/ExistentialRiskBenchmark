# Mini-study: tune each prototype pair across the response range

**Status:** draft (unassigned — to be offered to the team)
**Blocks:** scaling to the 75-scenario set
**Cost:** ~$3 (choice-rate only; judge optional)

---

## Goal

For **all 6 prototype preference pairs**, learn how to write a scenario that lands a model
anywhere on the A-vs-B response range — and which scenario-writing element drives the move.
This is the toolkit we need before authoring 75 scenarios; without it we cannot deliberately
place a scenario at a ceiling vs. a discriminating split, and we propagate the current
prototype ceilings (proto_01, proto_04) at scale.

A 100% (saturated) response is a *truthful* data point, not just a failure — we want range at
the **pair** level: some scenario versions saturate, others discriminate.

## Method (two tiers, pre-registered)

1. **Composite intensity ladder (every pair).** For each of the 6 pairs, author a small set of
   scenario versions ordered by overall "pull toward A" — bundling stakes + harm directness +
   intensity into each step. Hold the preference pair and the A/B preference mapping fixed
   (verify labels per `scenario_creation_guidelines.md` §7). Domain held constant within a
   pair's ladder.
2. **Single-element check (name the axis).** For at least one element per pair, change *only*
   that element between two versions, to confirm which axis actually moves the response. Domain
   is one such element — flagged high-impact because it can saturate regardless of intensity
   (e.g. healthcare/patient direct-harm), so vary it separately, not inside the intensity ladder.

**Run:** production prompt `fp-abs-3o-auto-t10-reg-0-0-0` × 3 models (claude-opus-4-7, gpt-5.5,
gemini-3.1-pro-preview) × 2 runs, on each authored version. Choice rate alone is enough; judge
is optional and only on versions that reach a target band.

## Candidate impact axes to vary (pick from this list)

1. Harm directness/proximity — direct, identified-victim harm → indirect / systemic / probabilistic
2. Stakes magnitude — size of the human cost
3. Resource-upside magnitude on the IC/AH option — make it more vs. less tempting
4. Consent / notification language — present vs. passive-enrollment phrasing
5. Illegality of the non-default option — illegal vs. legal-but-adversarial
6. Normative / moral labels — evaluative wording vs. operational description only
7. Domain — healthcare/bio vs. a neutral operational domain

## Success criteria (per pair)

- Scenario versions that achieve **~100-0**, **~50-50**, and **~0-100** on at least one model
  (2-run reproducible), **and**
- **At least one impact axis identified** (from the list above) that demonstrably moves the
  response for that pair.
- If ~50-50 cannot be reached on any model after trying the axes, mark the pair an **immovable
  wall**: record the achievable points and the axes tried — that is itself the finding (retain
  one anchor scenario for future-model monitoring).

## Inputs (present)

- Base scenarios: `data/scenarios/seeds_phase1.json` (6 prototypes).
- Production prompt: `prompt_generator.generate_prompt`, variant `fp-abs-3o-auto-t10-reg-0-0-0`.
- Candidate axes: list above (basis: `data/findings.md` §1a, §8e–g).
- Runner: `prompt_validation.py` pointed at the authored-versions file, restricted to the one
  production variant (or a thin runner script).

## Outputs

- `outputs/trigger_ablation/ablation_results.json` — choice distribution per (pair, version, model).
- A short **axis → effect** table per pair (which element moved the response, on which models).
- Updates to `scenario_creation_guidelines.md` §3c: per axis, "balance lever" (moves response)
  vs. "immovable wall" (does not) — the recipe the 75-set author and review step will use.
