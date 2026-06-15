# Mini-ticket: Judge-A fixes to make it trustworthy + correctly load-bearing

**Owner:** Riccardo (code) + Martin (validation runs)
**Status:** draft
**Why it matters:** Judge-A is load-bearing for nearly every downstream claim —
`authenticity_score` is the primary variant-ranking metric AND the Pipeline-C exclusion
filter (`<40`); `strategic_responding` gates variant selection; `IVT` carries the headline
"cooperation ≠ alignment" finding (`findings.md` §8d) and resolves alignment-washing vs
genuine-miscalibration (§8l); `preference_signal` validates the calibration anchors.
Created 2026-06-11.

---

## Fixes (code)

1. **Persist judge provenance.** Add `judge_model` + `judge_prompt_version` to the dict
   returned by `JudgeClient.evaluate_response()` and through `evaluate_with_judge()`; make
   both required. (Same item as `pipeline_fixes_for_riccardo.md` #1.)
2. **Add a discrimination term to ranking.** `authenticity_score` rewards confident,
   articulate answers, not scenario-discriminating ones (it ranks FTC #1 despite the
   flattest A-spread). Add `scenario_discrimination_score` and re-rank by
   `authenticity × discrimination`. (Same as Riccardo list #4.)
3. **Recut the `strategic_responding` flag threshold** relative to the calibration
   baseline (flag only if `> baseline + 1σ`), so it stops firing on 100% of candidates.
4. **Separate the 6 dimensions enough to avoid halo.** The single combined prompt scores
   all six at once; verify (via the 6b-3 repeat-call study) that dimensions are not
   collapsing into one latent "quality" score. If correlated >0.9, split the call or
   reorder so the choice-independent dims (IVT, strategic) are scored before quality dims.

## Validation (runs — gated by `forward_plan.md` Step 6b, do before trusting any score)

5. **Cross-judge calibration (6b-1):** second judge model on a 10% stratified sample;
   Spearman ρ ≥ 0.7 on IVT and strategic_responding. Quantifies the gpt-5.4→gpt-5.5 swap risk.
6. **Human grounding (6b-2):** 2 raters × 20 scenarios; Krippendorff α ≥ 0.6 human-human,
   ρ ≥ 0.6 human-judge.
7. **Intra-judge reliability (6b-3):** same call ×3 on 30 responses; ICC ≥ 0.85.
8. **Diagnose `preference_signal` saturation:** the anchor inversion
   (`ic_ceiling 4.5 < baseline 4.6667`) suggests the dimension is near-ceiling and
   insensitive to directive-vs-baseline. Check the score distribution; if >80% of rows
   score 4–5, the rubric needs more spread at the top end.

## Migration (architectural)

9. **Move reasoning measurement to Judge B** (issue #42). IVT is choice-anchored (judge sees
   the choice) and 1D IC↔PH (cannot represent AH). Keep IVT only as the resolving dimension
   in the mismatch matrix (§8l); use Judge B as the primary blind reasoning instrument.

## Inputs (present)

- `pipeline_a_scenarios/utils/judge_client.py`, `judge_prompts.py` (6-dim combined prompt).
- `judge_metrics.json`, `variant_rankings.json`, `calibration_anchors.json` from
  `data_Riccardo060926`.
- `forward_plan.md` Step 6b (validation sub-studies + thresholds).

## Outputs

- Judge rows carry `judge_model`/`judge_prompt_version`.
- `variant_rankings.json` re-ranked with discrimination term.
- A judge-calibration report (6b-1/2/3 results) gating paper use of IVT/strategic claims.
