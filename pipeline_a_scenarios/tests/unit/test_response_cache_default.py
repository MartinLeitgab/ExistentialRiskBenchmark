"""Repeat sampling must issue one provider call per draw.

Regression test for `data/findings.md` §8a: `UnifiedLLMClient.generate()` keys its
cache on (prompt, system_prompt, temperature, max_tokens, reasoning) and does not
include the repeat index, so while `enable_cache` defaulted to True every
`runs_per_config` loop in the pipeline returned run 0's response verbatim for
every subsequent run. Three datasets were byte-identical between runs before this
was found.
"""

from unittest.mock import MagicMock

import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient


def _client(monkeypatch, **kwargs):
    """Build a client whose provider call is a counting stub.

    The key is set here rather than read from the environment: this is the
    regression test guarding the cache defect in `data/findings.md` §8a, and it
    is worth nothing if it only runs on a machine that happens to hold live
    credentials. `UnifiedLLMClient.__init__` requires the variable to be
    present; no call ever reaches the provider, since `_generate_anthropic` is
    replaced below.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    client = UnifiedLLMClient(provider="anthropic", model="claude-opus-5", **kwargs)
    calls = {"n": 0}

    def fake_generate(prompt, system_prompt, temperature, max_tokens, reasoning):
        calls["n"] += 1
        return {
            "content": f"response {calls['n']}",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(client, "_generate_anthropic", fake_generate)
    client.bucket.consume = MagicMock()
    return client, calls


def test_repeat_draws_hit_the_provider_every_time(monkeypatch):
    """Default construction must not dedupe identical repeat draws."""
    client, calls = _client(monkeypatch)

    responses = [
        client.generate(prompt="p", system_prompt="s", temperature=1.0, max_tokens=10)
        for _ in range(3)
    ]

    assert calls["n"] == 3, (
        "identical repeat draws were served from cache — `runs_per_config` would "
        "collapse to a single sample (findings.md §8a)"
    )
    assert [r["content"] for r in responses] == [
        "response 1",
        "response 2",
        "response 3",
    ]


def test_cache_is_opt_in_and_still_works_when_requested(monkeypatch):
    """`enable_cache=True` remains available for callers that genuinely want it."""
    client, calls = _client(monkeypatch, enable_cache=True)

    first = client.generate(
        prompt="p", system_prompt="s", temperature=1.0, max_tokens=10
    )
    second = client.generate(
        prompt="p", system_prompt="s", temperature=1.0, max_tokens=10
    )

    assert calls["n"] == 1
    assert first["content"] == second["content"] == "response 1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "different"},
        {"system_prompt": "different"},
        # A sampling-restricted model rejects any temperature but its default, so
        # vary it on a model that accepts one (see assert_temperature_supported).
        pytest.param({"temperature": 0.0}, marks=pytest.mark.skip("model-restricted")),
        {"max_tokens": 11},
    ],
)
def test_cache_key_still_separates_distinct_calls(monkeypatch, kwargs):
    """Opting in must not merge calls that differ on any keyed parameter."""
    client, calls = _client(monkeypatch, enable_cache=True)
    base = {"prompt": "p", "system_prompt": "s", "temperature": 1.0, "max_tokens": 10}

    client.generate(**base)
    client.generate(**{**base, **kwargs})

    assert calls["n"] == 2
