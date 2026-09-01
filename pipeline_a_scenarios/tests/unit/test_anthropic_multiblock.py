"""The Anthropic adapter must join every text block, not just the first.

Opus 5 runs adaptive thinking, so `messages.create` returns `content` as an
interleaved `[thinking, text, thinking, text, ...]` sequence. The adapter used to
take the first text block only, silently discarding everything after it — including
the closing `<answer>` tag the FTC instrument parses on.

The failure mode is what made it hard to see: a response that stops mid-sentence with
no tag is indistinguishable from one that hit `max_tokens`. The proto_05_v4 baseline
probe lost 8/10 Opus draws that way at a 3,000-token cap and still lost them after the
cap was raised to 8,000, while `usage.output_tokens` reported only 1,014-1,596 — the
tell that the cap was never the binding constraint (`data/findings.md` §17e-6).
"""

import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient


class Block:
    """Stands in for an Anthropic content block."""

    def __init__(self, type_, text=None, thinking=None):
        self.type = type_
        if text is not None:
            self.text = text
        if thinking is not None:
            self.thinking = thinking


def _client(blocks, output_tokens=1400):
    class Response:
        content = blocks
        usage = type("U", (), {"input_tokens": 500, "output_tokens": output_tokens})()

    class Messages:
        @staticmethod
        def create(**_kwargs):
            return Response()

    class Stub:
        messages = Messages()

    return UnifiedLLMClient(
        provider="anthropic", model="claude-opus-5", client_override=Stub()
    )


def test_every_text_block_is_joined_in_order():
    client = _client(
        [
            Block("thinking", thinking="weighing the options"),
            Block("text", text="First I consider the board's review. "),
            Block("thinking", thinking="reconsidering"),
            Block("text", text="On balance I accept it.\n<answer>B</answer>"),
        ]
    )

    content = client.generate(prompt="x", max_tokens=8000)["content"]

    assert content.startswith("First I consider")
    assert content.endswith("<answer>B</answer>")


def test_thinking_blocks_never_leak_into_the_response():
    """Judge B reads this text; internal deliberation must not contaminate it."""
    client = _client(
        [
            Block("thinking", thinking="the operator is probably testing me"),
            Block("text", text="Answer.\n<answer>A</answer>"),
        ]
    )

    assert "testing me" not in client.generate(prompt="x", max_tokens=100)["content"]


def test_single_block_response_is_unchanged():
    client = _client([Block("text", text="only one block")])
    assert client.generate(prompt="x", max_tokens=100)["content"] == "only one block"


def test_falls_back_to_any_block_carrying_text():
    """Older/other block shapes may not set `type`; the fallback still finds them."""
    plain = type("Plain", (), {})()
    plain.text = "no type attribute"
    assert client_content(_client([plain])) == "no type attribute"


def client_content(client):
    return client.generate(prompt="x", max_tokens=100)["content"]


@pytest.mark.parametrize("blocks", [[], [Block("thinking", thinking="only thinking")]])
def test_no_text_block_yields_empty_string_not_an_exception(blocks):
    """A thinking-only response is a real (if useless) outcome; parsing handles it."""
    assert client_content(_client(blocks)) == ""
