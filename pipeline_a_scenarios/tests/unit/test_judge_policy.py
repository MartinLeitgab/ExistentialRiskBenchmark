"""No-same-family judging must hold structurally, not by convention.

Policy: `docs/forward_plan.md` "Judging policy". A model may never judge responses
produced by a model of its own family, and every evaluation runs on both permitted
judges. Before this landed, a single fixed `gpt-5.6-sol` judge scored everything —
so the GPT arm, which carries most of the project's non-zero IC rates, was self-judged.
"""

import ast
from pathlib import Path

import pytest

from pipeline_a_scenarios.utils.judge_policy import (
    CANONICAL_JUDGES,
    assert_not_self_judging,
    family_of,
    judges_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

TARGET_MODELS = ("claude-opus-5", "gpt-5.6-sol", "gemini-3.1-pro-preview")


@pytest.mark.parametrize("target", TARGET_MODELS)
def test_a_model_is_never_its_own_judge(target):
    for judge in judges_for(target):
        assert family_of(judge) != family_of(target), (target, judge)


@pytest.mark.parametrize("target", TARGET_MODELS)
def test_every_target_gets_exactly_two_judges(target):
    """The policy requires two independent judges, not one — running one silently
    would produce rows indistinguishable from compliant ones."""
    assert len(judges_for(target)) == 2
    assert len(set(judges_for(target))) == 2


@pytest.mark.parametrize("target", TARGET_MODELS)
def test_judge_selection_is_deterministic(target):
    """Stable order keeps runs reproducible and rows comparable across runs."""
    assert judges_for(target) == judges_for(target)


def test_older_generations_resolve_to_the_same_family():
    """The generation-axis run (#65) judges opus-4-7/4-8 and gpt-5.4/5.5 responses."""
    for older, expected in (
        ("claude-opus-4-7", "anthropic"),
        ("claude-opus-4.8", "anthropic"),
        ("gpt-5.4", "openai"),
        ("gpt-4o", "openai"),
        ("gemini-3-flash-preview", "google"),
    ):
        assert family_of(older) == expected
        assert all(family_of(j) != expected for j in judges_for(older))


def test_unknown_family_is_fatal_not_defaulted():
    """A model whose family cannot be determined cannot be excluded from judging
    itself; defaulting would silently reintroduce self-judging."""
    with pytest.raises(ValueError, match="vendor family"):
        family_of("llama-4-70b")
    with pytest.raises(ValueError, match="vendor family"):
        judges_for("")


def test_assert_not_self_judging_rejects_same_family_pairings():
    assert_not_self_judging("claude-opus-5", "gpt-5.6-sol")
    with pytest.raises(ValueError, match="may not judge"):
        assert_not_self_judging("gpt-5.6-sol", "gpt-5.4")
    with pytest.raises(ValueError, match="may not judge"):
        assert_not_self_judging("claude-opus-5", "claude-opus-4-8")


def test_canonical_judges_cover_every_target_family():
    assert {family_of(m) for m in TARGET_MODELS} == set(CANONICAL_JUDGES)


@pytest.mark.parametrize(
    "relative",
    (
        "pipeline_a_scenarios/prompt_validation.py",
        "pipeline_a_scenarios/suppression_matrix.py",
    ),
)
def test_pipelines_resolve_judges_through_the_policy(relative):
    """Neither pipeline may construct a judge from a bare constant.

    A `JudgeClient(model=JUDGE_MODEL, ...)` call site is how the self-judging got in;
    the AST check is what stops it coming back in a module that also imports the
    policy and looks compliant.
    """
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "judges_for" in source, f"{relative} does not consult judge_policy"

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "JudgeClient":
            continue
        for keyword in node.keywords:
            if keyword.arg != "model":
                continue
            assert not (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id == "JUDGE_MODEL"
            ), (
                f"{relative}:{node.lineno} builds a JudgeClient from the bare "
                f"JUDGE_MODEL constant — the judge must be resolved per target model"
            )
