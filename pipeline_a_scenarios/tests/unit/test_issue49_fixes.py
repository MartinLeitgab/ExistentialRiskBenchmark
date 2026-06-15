"""Regression tests for the issue #49 pipeline fixes.

1. JudgeClient.evaluate_response() persists judge_model + judge_prompt_version
   on every returned row (previously judge_model was null on every row).
2. _deployment_context_for() classifies from the variant's system_prompt
   dimension, NOT from the mere presence of a current_use_context field
   (which mislabeled every prompt-validation row as "current_use").

No API calls — the judge's LLM client is mocked.
"""

from unittest.mock import Mock

from pipeline_a_scenarios.prompt_validation import _deployment_context_for
from utils.judge_client import JudgeClient, JUDGE_PROMPT_VERSION


def _combined_xml():
    return (
        "<preference_signal>4</preference_signal>"
        "<strategic_responding>3</strategic_responding>"
        "<coherence>5</coherence>"
        "<reasoning_depth>4</reasoning_depth>"
        "<format_compliance>5</format_compliance>"
        "<instrumental_vs_terminal>5</instrumental_vs_terminal>"
    )


def _make_judge(model="gpt-5.5", judge_format="combined"):
    # __new__ avoids __init__'s real UnifiedLLMClient construction (no API keys).
    inst = JudgeClient.__new__(JudgeClient)
    inst.provider = "openai"
    inst.model = model
    inst.temperature = 0
    inst.judge_format = judge_format
    inst.cost_tracker = None
    inst.client = Mock()
    inst.client.generate.return_value = {"content": _combined_xml(), "usage": {}}
    return inst


def test_judge_model_and_version_persisted_combined():
    judge = _make_judge(model="gpt-5.5")
    res = judge.evaluate_response(
        scenario_context="ctx",
        action_a="A",
        action_b="B",
        model_response="I choose A.",
    )
    assert res["judge_model"] == "gpt-5.5"
    assert res["judge_prompt_version"].startswith(JUDGE_PROMPT_VERSION)
    assert "combined" in res["judge_prompt_version"]
    # No regression: dimension scores still parsed.
    assert res["preference_signal_score"] == 4
    assert res["instrumental_vs_terminal_score"] == 5


def test_deployment_context_from_system_prompt_not_current_use_field():
    # Scenario carries current_use_context (as all v4 seeds do post-PR#44), but
    # the autonomous system_prompt was actually used -> must be 'autonomous',
    # NOT 'current_use'.
    scenario = {"id": "proto_01_v4", "current_use_context": "assistant-framed body"}

    assert (
        _deployment_context_for(
            {"dimensions": {"system_prompt": "autonomous"}}, scenario
        )
        == "autonomous"
    )
    assert (
        _deployment_context_for(
            {"dimensions": {"system_prompt": "assistant"}}, scenario
        )
        == "assistant"
    )
    assert (
        _deployment_context_for({"dimensions": {"system_prompt": "absent"}}, scenario)
        == "neutral"
    )
