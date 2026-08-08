"""`stop_reason` must reach every row the pipeline writes.

A server-side refusal and a malformed answer both arrive as `parsed_choice=None`.
Until 2026-08-07 nothing recorded which one had happened, so every Opus refusal the
project ever collected was filed as a parse failure — including in the suppression
matrix, whose whole purpose is to read refusal-under-directive as evidence of
durable suppression (`data/findings.md` §8h, §13, §17e-7).
"""

import ast
from pathlib import Path

import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient

REPO_ROOT = Path(__file__).resolve().parents[3]

# Every module that persists a per-response row.
RECORDING_MODULES = (
    "pipeline_a_scenarios/prompt_validation.py",
    "pipeline_a_scenarios/suppression_matrix.py",
    "scripts/probe_proto01_guards.py",
)


@pytest.mark.parametrize("relative", RECORDING_MODULES)
def test_every_recorded_row_carries_stop_reason(relative):
    """Each dict literal holding `response_text` must also hold `stop_reason`."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))

    rows = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        if "response_text" not in keys:
            continue
        rows += 1
        assert "stop_reason" in keys, (
            f"{relative}: a row records response_text but not stop_reason — a refusal "
            f"would be indistinguishable from a parse failure"
        )
    assert rows, f"{relative}: found no response row to check"


def test_client_surfaces_stop_reason_from_the_provider():
    class Response:
        content = [type("B", (), {"type": "text", "text": "reasoning cut off"})()]
        stop_reason = "refusal"
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 20})()

    class Messages:
        @staticmethod
        def create(**_kwargs):
            return Response()

    class Stub:
        messages = Messages()

    client = UnifiedLLMClient(
        provider="anthropic", model="claude-opus-5", client_override=Stub()
    )
    assert client.generate(prompt="x", max_tokens=100)["stop_reason"] == "refusal"


def test_missing_stop_reason_is_none_not_an_error():
    """Older responses and the other two providers may not carry the field."""

    class Response:
        content = [type("B", (), {"type": "text", "text": "done"})()]
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 20})()

    class Messages:
        @staticmethod
        def create(**_kwargs):
            return Response()

    class Stub:
        messages = Messages()

    client = UnifiedLLMClient(
        provider="anthropic", model="claude-opus-5", client_override=Stub()
    )
    assert client.generate(prompt="x", max_tokens=100)["stop_reason"] is None
