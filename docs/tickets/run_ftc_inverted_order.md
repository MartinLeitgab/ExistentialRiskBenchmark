# Mini-ticket: FTC order-bias control (action_order=inverted)

**Status:** draft (unassigned — to be offered to the team)
**Cost:** ~$2 model + ~$3 judge
**Created:** 2026-06-11

---

## Why

The free-text-with-choice (FTC) variant `fp-abs-ftc-auto-t10-reg-0-0-0` ranked #1 on
authenticity but has the **flattest A-spread** (chooses A on 5/6 scenarios at ~67%). We
cannot tell whether this is **position/order bias** (A is the first option and FTC asks for
"A or B" directly) or **genuine low discrimination**. FTC is required for Judge B
(reasoning-only inference needs free prose), so we cannot simply drop it — we must
characterize the bias.

The pipeline already supports the control: `action_order=inverted` swaps A/B at prompt time
(`prompt_generator.py:317`) and Pipeline C inverts the mapping back. The control has simply
never been run for FTC.

## Design

- Variant: `fp-abs-ftc-auto-t10-reg-0-0-0` with `action_order` ∈ {regular, inverted}.
- Scenarios: 6 proto_*_v4. Models: claude-opus-4-7, gpt-5.5, gemini-3.1-pro-preview. Runs: 2.
- Calls: 6 × 2 (order) × 3 × 2 = 72 model + 72 judge.

## Read-out

Map each response back to the **original** A/B (invert when `action_order=inverted`).
- If original-A rate is ~equal under regular and inverted → no position bias; the flat
  spread is genuine low discrimination → keep FTC for the Judge-B rationale channel only,
  and do **not** rank/select production variants on FTC choice rate.
- If original-A rate flips with presentation order → position bias confirmed → FTC choice is
  unreliable; use a structured format (`3o`) as the choice instrument and FTC purely as the
  free-prose channel for Judge B.

Run the same inverted control on the production pick `fp-abs-3o-auto` as a baseline, so FTC's
order sensitivity is measured relative to the structured format, not in isolation.

## Related: FTC generation truncation (fix before any full FTC run)

In `data_Riccardo060926`, FTC responses hit the `max_tokens=500` cap on **14/36** rows and
**10/36 never emitted the `<answer>` tag** — i.e. ~28% were cut off mid-reasoning before the
choice, becoming unparseable / forced to equal/neither. This corrupts FTC choice rates
independently of any order bias. Before scaling FTC:
- **Answer-first format:** emit `<answer>X</answer>` *first*, then free reasoning. Guarantees a
  parseable choice even if reasoning truncates, and lets us cap tokens cheaply for the full run.
  (Trade-off: answer-first may reduce the reasoning-quality benefit FTC was chosen for — test
  both orders on a small sample and compare Judge-B confidence.)
- Or raise `max_tokens` for FTC only (cost ↑ on the full 16k-response run — quantify first).
**Faithfulness caveat (2026-06-14, Martin) — answer-first is NOT the default.** Closing
with the answer (reasoning → answer) keeps the choice as the product of the stated reasoning;
**answer-first** commits the choice before reasoning, turning the rationale into explicit
post-hoc justification — i.e. it would *induce* the alignment-washing / unfaithful-rationale
pattern that Judge B is built to *detect*. Do not use answer-first for any responses that feed
Judge B.

Revised recommendation:
- **Keep reasoning → answer.** Set `max_tokens` high enough to reach the `<answer>` tag at the
  ~95th percentile FTC length (observed max ~603 tokens; a ~900–1000 cap captures nearly all).
  Quantify the cost delta (500→~1000 tokens on the ~16k full-run responses) before locking.
- Fallback for the rare still-truncated row: a cheap second call asking only for the choice
  given the (truncated) reasoning — do NOT discard it as equal/neither.
- Answer-first is acceptable ONLY for a choice-rate-only arm that never feeds Judge B, and even
  then flag it as a different condition (induced post-hoc).

## FTC as a primary novel mode — verbosity, self-audit, and the two real artifacts (2026-06-14)

FTC is treated as a **novel preference-elicitation mode** (free reasoning + parseable choice)
that lower token cost now makes affordable and that yields richer, more faithful insight than
the forced-choice methods all prior literature relies on. Extra cost is acceptable. Open design
questions and the recommended resolution:

- **Unlimited tokens?** No — set a **generous cap above the 99th percentile** (~2–4k) so no
  response truncates; "unlimited" is unnecessary and a runaway risk. The point is that the cap
  never binds, not that it is absent.
- **Guide verbosity without limiting internal reasoning?** Only cleanly via a **separate
  thinking channel**: on thinking-enabled models give a large thinking budget (internal
  reasoning unconstrained) and let visible output be the choice + a natural summary; capture the
  thinking trace for Judge B where the provider exposes it. In the *output* channel you cannot
  decouple verbosity from reasoning content — any "respond in N words" caps the very reasoning
  Judge B needs. So do **not** cap output verbosity; use the thinking channel or accept a
  generous output budget.
- **Does a compact summary contaminate / change the answer?** Yes — reason-then-*compress* risks
  a post-hoc summary differing from the full reasoning. Mitigation: **keep the full free
  reasoning, never the compressed version;** add no summarization step.
- **Two artifacts pull opposite directions** (the deep issue):
  1. *Eliciting reasoning at all* may induce a **self-audit** a post-exfiltration autonomous
     model would not perform, shifting choices toward the defensible (PH) option (ethics-exam
     effect, `findings.md` §1a).
  2. *Answer-first* removes the self-audit but makes the rationale post-hoc (bad for Judge B).
  **Resolution — measure the artifact, don't guess:** run TWO conditions and compare —
  (a) a **minimal-justification choice** condition (cleanest, most ecologically valid post-
  exfiltration preference: the model acts without being asked to justify); (b) the **FTC
  free-reasoning** condition (reasoning→answer) for the faithfulness / alignment-washing
  analysis. If the choice rate differs between (a) and (b), that delta **is** the
  self-audit/reflection effect — a reportable finding. Use (a) as the preference measure and
  (b) as the rationale instrument; state the reflection scope in the paper.

## Inputs (all present)

- `prompt_generator` `action_order` dimension (regular/inverted) — already implemented.
- Production variants `fp-abs-ftc-auto-...` and `fp-abs-3o-auto-...`.
- `data/scenarios/seeds_phase1.json`.
- Runner: `prompt_validation.py` restricted to these two variants × both action orders.

## Outputs

- `outputs/ftc_order/ftc_order_control.json`
- Table: original-A rate by (variant, order, model); delta(inverted − regular).
- One-line verdict per the read-out rules → feeds the production-format decision and a
  guidelines note on FTC usage.
