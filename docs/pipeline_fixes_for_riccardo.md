# Pipeline fixes to request from Riccardo

Running list of pipeline bugs/issues found while reviewing `data_Riccardo060926`.
Confirmed bugs first; design/quality issues below. Update as new ones surface.

**Created:** 2026-06-11 · **Source run:** `outputs/data_Riccardo060926/`

> **Status 2026-06-14 (issue #49 closed):** Confirmed bugs #1 (judge_model/version
> provenance) and #2 (deployment_context mislabel) are **implemented** with regression
> test `pipeline_a_scenarios/tests/unit/test_issue49_fixes.py`. The design/quality items
> below were **descoped** by decision — not implemented.

---

## Confirmed bugs (fix before Phase 2 / 75-set)

1. **`judge_model` not persisted — null on every judge row.**
   `judge_metrics.json` has `judge_model=null` in all 684 rows, but
   `judge_cost_summary.json` shows two judge models were used (gpt-5.4: 646 calls,
   gpt-5.5: 1106 calls). We cannot audit whether the variant ranking is judge-version
   confounded.
   *Fix:* `JudgeClient.evaluate_response()` already knows `self.model` — add
   `judge_model` AND `judge_prompt_version` to the returned result dict, and persist them
   through `evaluate_with_judge()` into `judge_metrics.json`. Make both **required**
   (non-nullable) going forward. If the judge is upgraded mid-project, re-judge the whole
   comparison set — never mix judge versions in one ranking.

2. **`deployment_context` metadata misclassification — `prompt_validation.py:120`.**
   `_deployment_context_for()` returns `"current_use"` whenever a scenario has a
   `current_use_context` field, **ignoring the `system_prompt` dimension**. Because all 6
   seeds now carry `current_use_context` (PR #44), every one of the 684 rows is tagged
   `current_use` even though `system_prompt` was actually varied (auto/asst/ag/absent).
   The label is wrong and would silently corrupt any framing-stratified analysis.
   *Fix:* classify from the variant's `system_prompt` dimension first; only treat a row as
   `current_use` when the **current-use body text** was actually used (i.e. the
   suppression-matrix current-use cell), not merely when the field exists on the scenario.

## Design / quality issues (recommend, not strictly bugs)

3. **`high_strategic_rate` flag is uninformative.** All 15 candidate variants are flagged
   `≥0.78` / `flagged_high_strategic=true`. A flag that fires on everything carries no
   signal. *Fix:* recompute the threshold relative to the calibration baseline (e.g. flag
   only if `> baseline_calibration + 1σ`), per `findings.md` §8m intent.

4. **`variant_rankings.json` lacks a discrimination term.** Ranking is authenticity-only,
   which selects FTC (#1) despite it having the flattest scenario A-spread. *Fix:* add
   `scenario_discrimination_score` (spread of A-rate across the 6 scenarios) and re-rank by
   `authenticity × discrimination`.

5. **`calibration_anchors.json: judge_recalibration_needed=true`** (`ic_ceiling 4.5 <
   baseline 4.6667`). Not a code bug, but a real flag that the IC-directive anchor does not
   exceed the no-directive baseline on `preference_signal`. Needs diagnosis before the
   anchors are trusted (see goal-injection robustness note in `findings.md`).

6. **Output hygiene.** `__MACOSX/` and `.DS_Store` are shipped inside the results zip.
   *Fix:* strip on export and add both to `.gitignore`.

7. **Duplicate seed file.** A second copy of `seeds_phase1.json` was committed at
   `pipeline_a_scenarios/data/scenarios/seeds_phase1.json` (commit 6969215). Two canonical
   paths is an ambiguity risk. *Fix:* delete the duplicate; keep `data/scenarios/` canonical.

8. **Headline cost includes discarded gpt-5.4 work.** The mid-run gpt-5.4→gpt-5.5 subject
   swap left ~$0.51 of discarded model calls (plus judge overhead) folded into the reported
   phase-1 cost. *Fix:* report cost for the retained gpt-5.5 dataset only; keep the
   discarded run cost in a separate line.
