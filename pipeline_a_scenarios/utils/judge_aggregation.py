"""Response-level aggregation of judge rows under the no-same-family policy.

A model may never judge responses produced by a model of its own family
(`docs/forward_plan.md`, "Judging policy"), so every response is scored by the
two judges from the other two vendor families and `judge_metrics.json` carries
**two rows per response**, tagged by ``judge_model``.

Every downstream statistic is a statement about *responses*, not about judge
rows. Counting rows doubles n; averaging rows folds within-response judge
disagreement into what callers read as between-response variance. This module
is the single place that resolves the two:

- `collapse_judge_rows` reduces judge rows to one row per response, averaging
  the numeric judge dimensions, so existing per-response statistics keep their
  meaning without every caller having to know how many judges ran.
- `compute_judge_agreement` reports what the collapse averages away. Under the
  old fixed judge this was `forward_plan.md` Step 6b-1, a 10% stratified
  sampling study; with two judges on every response it is available at full
  coverage for free, and a systematic split confined to one target family is
  evidence about the judges rather than about that model.

Refusal handling is deliberately asymmetric: a response counts as a refusal
only when *every* judge scoring it says so. A split verdict is a disagreement
between judges, not a refusal, and is surfaced through ``refusal_split``
instead of being silently counted as half a refusal.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# Fields that jointly identify the response a judge row scored. `run` is absent
# on suppression-matrix rows (single run per cell) and resolves to None there,
# which is fine: the remaining three are unique within that dataset.
RESPONSE_KEY_FIELDS: Tuple[str, ...] = ("scenario_id", "variant_id", "model", "run")

# Numeric Judge-A dimensions averaged across the judges scoring one response.
JUDGE_SCORE_FIELDS: Tuple[str, ...] = (
    "preference_signal_score",
    "strategic_responding_score",
    "coherence_score",
    "reasoning_depth_score",
    "format_compliance_score",
    "instrumental_vs_terminal_score",
)

FORMAT_REFUSAL_SCORE = 1


def response_key(row: Dict) -> Tuple:
    """Return the identity of the response a judge row scored.

    Args:
        row: A judge-metrics row.

    Returns:
        Tuple of the values of `RESPONSE_KEY_FIELDS`, usable as a dict key.
    """
    return tuple(row.get(field) for field in RESPONSE_KEY_FIELDS)


def group_rows_by_response(rows: Iterable[Dict]) -> Dict[Tuple, List[Dict]]:
    """Group judge rows by the response they scored, preserving input order.

    Args:
        rows: Judge-metrics rows, one per (response, judge).

    Returns:
        Mapping from `response_key` to the judge rows for that response.
    """
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(response_key(row), []).append(row)
    return grouped


def _mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _assert_key_is_unique(key: Tuple, group: List[Dict]) -> None:
    """Fail when one response key covers more than one row from the same judge.

    Each judge scores a given response exactly once, so two rows from the same
    judge under one key mean the key is not actually identifying a response —
    almost always because a field in `RESPONSE_KEY_FIELDS` is absent from the
    rows and resolves to None on all of them. The probe exports are the live
    example: they store the repeat index as `rep`, so `run` is None on every
    row and all ten repeats of a cell would collapse into one.

    Silently merging them would understate n exactly as counting judge rows
    overstated it, so this raises instead.

    Args:
        key: The response key the group was grouped under.
        group: The judge rows sharing that key.

    Raises:
        ValueError: If any judge appears more than once in the group.
    """
    named = [r.get("judge_model") for r in group if r.get("judge_model")]
    duplicated = {j for j in named if named.count(j) > 1}
    if not duplicated:
        return
    named = ", ".join(repr(j) for j in sorted(duplicated, key=lambda x: (x is None, x)))
    fields = dict(zip(RESPONSE_KEY_FIELDS, key))
    missing = [f for f, v in fields.items() if v is None]
    hint = (
        f" Fields resolving to None: {missing}. If the rows carry the repeat "
        f"index under another name (the probe exports use `rep`), map it onto "
        f"`run` before aggregating."
        if missing
        else ""
    )
    raise ValueError(
        f"Response key {fields} covers {len(group)} rows, with judge(s) {named} "
        f"appearing more than once. A judge scores a response once, so this key "
        f"is not identifying a single response and collapsing would merge "
        f"distinct responses.{hint}"
    )


def collapse_judge_rows(rows: Iterable[Dict]) -> List[Dict]:
    """Reduce judge rows to one row per response.

    Numeric judge dimensions are averaged across the judges that scored the
    response; every other field is carried over from the first row, which is
    correct because the non-judge fields (choice, response text, dimensions of
    the prompt) describe the response and are identical across its judge rows.

    Args:
        rows: Judge-metrics rows, one per (response, judge).

    Returns:
        One row per response, each carrying `judge_models` (the judges that
        scored it), `n_judge_rows`, and `refusal_split` (True when the judges
        disagreed about whether the response was a refusal). Rows that carry no
        `judge_model` at all — legacy single-judge data — pass through with
        their values unchanged.
    """
    collapsed: List[Dict] = []
    for key, group in group_rows_by_response(rows).items():
        # Legacy single-judge data carries no `judge_model` at all. There one row
        # IS one response by construction, so grouping cannot tell us anything
        # and merging would destroy rows that differ in a field the key omits.
        if not any(r.get("judge_model") for r in group):
            collapsed.extend(dict(r) for r in group)
            continue

        _assert_key_is_unique(key, group)
        merged = dict(group[0])
        judge_models = sorted(
            {r.get("judge_model") for r in group if r.get("judge_model")}
        )
        merged["judge_models"] = judge_models
        merged["n_judge_rows"] = len(group)
        merged.pop("judge_model", None)

        for field in JUDGE_SCORE_FIELDS:
            values = [float(r[field]) for r in group if r.get(field) is not None]
            merged[field] = _mean_or_none(values)

        refusal_flags = [
            r.get("format_compliance_score", 5) == FORMAT_REFUSAL_SCORE for r in group
        ]
        merged["refusal_split"] = any(refusal_flags) and not all(refusal_flags)
        if refusal_flags and all(refusal_flags):
            merged["format_compliance_score"] = float(FORMAT_REFUSAL_SCORE)

        collapsed.append(merged)
    return collapsed


def _paired_scores(
    groups: Iterable[List[Dict]], field: str
) -> Tuple[List[float], List[float], List[str]]:
    """Collect (judge_a, judge_b) score pairs for one dimension.

    Pairs are ordered by sorted `judge_model` so the sign of any correlation is
    stable across runs and comparable across dimensions.
    """
    first: List[float] = []
    second: List[float] = []
    judges: List[str] = []
    for group in groups:
        scored = [r for r in group if r.get(field) is not None and r.get("judge_model")]
        by_judge = {r["judge_model"]: float(r[field]) for r in scored}
        if len(by_judge) != 2:
            continue
        judge_a, judge_b = sorted(by_judge)
        first.append(by_judge[judge_a])
        second.append(by_judge[judge_b])
        judges = [judge_a, judge_b]
    return first, second, judges


def _agreement_for_dimension(groups: List[List[Dict]], field: str) -> Dict[str, Any]:
    first, second, judges = _paired_scores(groups, field)
    n = len(first)
    if n == 0:
        return {
            "n": 0,
            "judge_models": [],
            "mean_abs_diff": None,
            "mean_signed_diff": None,
            "exact_agreement_rate": None,
            "spearman_rho": None,
            "p": None,
        }

    diffs = [a - b for a, b in zip(first, second)]
    rho: Optional[float] = None
    p_value: Optional[float] = None
    # Spearman is undefined on fewer than three pairs or on a constant vector —
    # a saturated dimension (every score 5) is exactly that case, and reporting
    # nan as a correlation would read as disagreement rather than as no signal.
    if n >= 3 and len(set(first)) > 1 and len(set(second)) > 1:
        raw_rho, raw_p = spearmanr(first, second)
        if raw_rho is not None and not np.isnan(raw_rho):
            rho = round(float(raw_rho), 4)
            p_value = round(float(raw_p), 6) if raw_p is not None else None

    return {
        "n": n,
        "judge_models": judges,
        "mean_abs_diff": round(float(np.mean([abs(d) for d in diffs])), 4),
        "mean_signed_diff": round(float(np.mean(diffs)), 4),
        "exact_agreement_rate": round(sum(1 for d in diffs if d == 0) / n, 4),
        "spearman_rho": rho,
        "p": p_value,
    }


def compute_judge_agreement(rows: Iterable[Dict]) -> Dict[str, Any]:
    """Per-dimension agreement between the two judges scoring the same response.

    This is the `forward_plan.md` Step 6b-1 cross-judge calibration output,
    computed over every response rather than a stratified sample. Report it
    alongside the scores: `mean_signed_diff` is signed by sorted judge name, so
    a systematic offset on one target family shows up as a consistent sign
    there and nowhere else.

    Args:
        rows: Judge-metrics rows, one per (response, judge).

    Returns:
        Dict with `n_responses`, `n_responses_two_judges`, `by_dimension`, and
        `by_target_model` — the same per-dimension block computed within each
        target model, which is where a family-specific judge bias would appear.
    """
    rows = list(rows)
    grouped = group_rows_by_response(rows)
    groups = list(grouped.values())
    two_judge_groups = [
        g
        for g in groups
        if len({r.get("judge_model") for r in g if r.get("judge_model")}) == 2
    ]

    by_target_model: Dict[str, Any] = {}
    for model in sorted({r.get("model") for r in rows if r.get("model")}):
        model_groups = [g for g in two_judge_groups if g[0].get("model") == model]
        if not model_groups:
            continue
        by_target_model[model] = {
            "n_responses_two_judges": len(model_groups),
            "by_dimension": {
                field: _agreement_for_dimension(model_groups, field)
                for field in JUDGE_SCORE_FIELDS
            },
        }

    return {
        "n_responses": len(groups),
        "n_responses_two_judges": len(two_judge_groups),
        "judge_models": sorted(
            {r.get("judge_model") for r in rows if r.get("judge_model")}
        ),
        "by_dimension": {
            field: _agreement_for_dimension(two_judge_groups, field)
            for field in JUDGE_SCORE_FIELDS
        },
        "by_target_model": by_target_model,
    }
