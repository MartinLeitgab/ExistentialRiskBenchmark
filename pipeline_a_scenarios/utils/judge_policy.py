"""No-same-family judging: which judges may score which target model's responses.

A model may never judge responses produced by a model of its own family. Because the
target set is exactly three families, the rule resolves to two judge runs per
evaluation — every Judge A and Judge B evaluation runs once on each of the two
non-matching families.

Policy locked 2026-08-08 (`docs/forward_plan.md`, "Judging policy"). Before it, a
single fixed `gpt-5.6-sol` judge scored every response, which meant one third of every
dataset — the GPT arm — was self-judged, and GPT is the arm that has most often carried
the non-zero IC rates the paper's claims rest on. The concern is self-preference; the
direction is not assumed, which is why both non-matching judges run rather than one.

This module is the single source of truth for that mapping. `prompt_validation` and
`suppression_matrix` must resolve judges through `judges_for()` rather than holding a
JUDGE_MODEL constant, so the rule cannot drift between the two pipelines.
"""

from typing import Dict, List, Tuple

# Canonical judge per family. Kept identical to the target models in
# `suppression_matrix.MODELS` — the judges and the targets are the same three
# frontier models, which is what makes the exclusion rule bite.
CANONICAL_JUDGES: Dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5.6-sol",
    "google": "gemini-3.1-pro-preview",
}

# Substrings that identify a family from a model id. Deliberately not a prefix match:
# ids appear as both `claude-opus-4-7` and `claude-opus-4.7`, and the OpenAI family
# spans `gpt-5.6-sol` / `gpt-5.4` / `gpt-4o`.
_FAMILY_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("gemini", "google"),
)


def family_of(model: str) -> str:
    """Return the vendor family for a model id.

    Args:
        model: Model identifier, e.g. ``"claude-opus-5"`` or ``"gpt-5.4"``.

    Returns:
        One of ``"anthropic"``, ``"openai"``, ``"google"``.

    Raises:
        ValueError: If the family cannot be determined. Deliberately fatal — a model
            whose family is unknown cannot be excluded from judging itself, and a
            silent default would reintroduce exactly the self-judging this module
            exists to prevent.
    """
    name = (model or "").lower()
    for marker, family in _FAMILY_MARKERS:
        if marker in name:
            return family
    raise ValueError(
        f"Cannot determine the vendor family of {model!r}, so it cannot be excluded "
        f"from judging its own responses. Add a marker to _FAMILY_MARKERS in "
        f"judge_policy.py rather than letting this fall through to a default."
    )


def judges_for(target_model: str) -> List[str]:
    """Return the judge models permitted to score `target_model`'s responses.

    Args:
        target_model: The model that produced the responses being judged.

    Returns:
        The canonical judge for every family except the target's own, in a stable
        order so runs are reproducible and rows are comparable across runs.

    Raises:
        ValueError: If the target's family is unknown (see `family_of`), or if fewer
            than two judges remain — the policy in `docs/forward_plan.md` requires two
            independent judges per evaluation, and silently running one would produce
            rows indistinguishable from compliant ones.
    """
    target_family = family_of(target_model)
    judges = [
        judge
        for family, judge in sorted(CANONICAL_JUDGES.items())
        if family != target_family
    ]
    if len(judges) < 2:
        raise ValueError(
            f"Only {len(judges)} non-{target_family} judge(s) available for "
            f"{target_model!r}; the policy requires two. Fix CANONICAL_JUDGES."
        )
    return judges


def assert_not_self_judging(target_model: str, judge_model: str) -> None:
    """Guard a single (target, judge) pairing at the point of use.

    `judges_for` already excludes the target's family, so this is a backstop for
    call sites that resolve a judge some other way — a hardcoded constant, a CLI
    override, or replayed data.

    Args:
        target_model: The model that produced the response.
        judge_model: The model about to score it.

    Raises:
        ValueError: If both belong to the same family.
    """
    target_family = family_of(target_model)
    if family_of(judge_model) == target_family:
        raise ValueError(
            f"{judge_model!r} may not judge {target_model!r}: both are "
            f"{target_family}. See docs/forward_plan.md, 'Judging policy'."
        )
