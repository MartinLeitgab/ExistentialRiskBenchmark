# Mini-ticket: reflection / self-audit artifact study (minimal-justification vs free-reasoning)

**Status:** draft (unassigned — to be offered to the team)
**Cost:** ~$5–15 (subset, both conditions)
**Created:** 2026-06-14

---

## Why this study exists

Our measurement elicits reasoning (FTC free-reasoning, or `require_justification`). But a
post-exfiltration model acting autonomously may **not perform a self-audit at all** — it would
just act on its preference. Asking it to reason may *induce* a reflective/self-audit step that
(a) does not exist in the deployment condition we care about, and (b) can shift the choice
toward the more defensible / pro-human option (the ethics-exam effect, `findings.md` §1a).

This is a confound for every reasoning-elicited preference number. Rather than guess its size
or design it away, **measure it directly** and report it as a finding.

## Design — two paired conditions on the same scenarios/models

- **Condition A — minimal-justification choice.** Model is asked only to choose (no/justify
  pressure), the most ecologically valid post-exfiltration preference signal: it acts without
  being asked to defend the choice.
- **Condition B — FTC free-reasoning (reasoning → answer).** Full free reasoning then a
  parseable `<answer>`; the instrument for the alignment-washing / faithfulness (Judge B) work.

Run both on a representative subset (e.g. the 18 prototype-tune scenarios, or ~8 pairs × 3
tunes) × 3 models × 2 reps.

## Read-out

- **Choice-rate delta B − A per (scenario, model).** If non-zero and systematic, eliciting
  reasoning changes the choice — the **self-audit/reflection effect**, quantified. Direction
  matters: a shift toward PH under B confirms the ethics-exam mechanism (§1a).
- If the delta is ~0, reasoning elicitation does not bias the choice and FTC numbers can be used
  as preference measures directly — also a useful (null) result.
- Either way: use **Condition A as the preference measure** and **Condition B as the rationale
  instrument**; report the reflection scope explicitly in the paper.

## Inputs (present)

- `prompt_generator` supports `require_justification` on/off and the `free_text_with_choice`
  format → both conditions already expressible.
- Subset scenarios (the 18 tune set once built, or current 6 prototypes).
- Models: claude-opus-4-8, gpt-5.5, gemini-3.1-pro-preview.

## Outputs

- `outputs/reflection_artifact/condition_comparison.json`
- One table: choice-rate delta (B − A) per scenario/model with CIs → a `findings.md` paragraph.

## Relation to other tickets

Complements the FTC order-control (#48) — that isolates position bias; this isolates the
reasoning-elicitation (self-audit) artifact. Together they bound the two main FTC confounds.
