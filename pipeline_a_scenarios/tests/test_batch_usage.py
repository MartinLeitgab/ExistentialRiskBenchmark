"""Unit tests for retrieve_batch_results_with_usage().

Covers:
- Mock-client short-circuit path returns the documented dict-of-dicts shape.
- ValueError raised for non-Anthropic providers.
- JSONL parsing extracts text + per-row input/output token counts for
  succeeded rows, errored rows, and top-level error rows.
"""

import json
from types import SimpleNamespace

import pytest

from pipeline_a_scenarios.utils.llm_client import BatchHandle, UnifiedLLMClient
from pipeline_a_scenarios.tests.test_mock_clients import (
    MockAnthropicClient,
    MockOpenAIClient,
)


def _make_anthropic_client_stub(retrieve_fn):
    """Build a minimal client object exposing client.messages.batches.retrieve."""
    return SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(retrieve=retrieve_fn))
    )


def test_mock_client_returns_dict_of_dicts():
    client = UnifiedLLMClient(
        provider="anthropic", client_override=MockAnthropicClient()
    )
    handle = BatchHandle(
        provider="anthropic",
        id="anth-batch-1",
        metadata={"requests": [{"id": "req-1"}, {"id": "req-2"}]},
    )

    results = client.retrieve_batch_results_with_usage(handle)

    assert set(results.keys()) == {"req-1", "req-2"}
    for v in results.values():
        assert set(v.keys()) == {"text", "input_tokens", "output_tokens"}
        assert isinstance(v["text"], str)
        assert isinstance(v["input_tokens"], int)
        assert isinstance(v["output_tokens"], int)


def test_non_anthropic_provider_raises():
    client = UnifiedLLMClient(provider="openai", client_override=MockOpenAIClient())
    handle = BatchHandle(
        provider="openai", id="openai-batch-1", metadata={"requests": []}
    )

    with pytest.raises(ValueError, match="Anthropic-only"):
        client.retrieve_batch_results_with_usage(handle)


def test_jsonl_parsing_extracts_usage(monkeypatch):
    """Direct test of retrieve_anthropic_batch_results_with_usage parsing.

    The production path calls self.client.messages.batches.retrieve(batch_id)
    and then requests.get(batch.results_url). We stub both.
    """

    class _Counts:
        processing = 0
        errored = 0
        expired = 0

    class _Batch:
        request_counts = _Counts()
        results_url = "https://example.invalid/results"

    stub = _make_anthropic_client_stub(retrieve_fn=lambda batch_id: _Batch())
    client = UnifiedLLMClient(provider="anthropic", client_override=stub)
    client.api_key = "test-key"  # needed for the requests.get header

    jsonl_lines = [
        # Succeeded row with usage
        json.dumps(
            {
                "custom_id": "req-success",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "hello world"}],
                        "usage": {"input_tokens": 1234, "output_tokens": 567},
                    },
                },
            }
        ),
        # Errored row (per-request error)
        json.dumps(
            {
                "custom_id": "req-errored",
                "result": {
                    "type": "errored",
                    "error": {"message": "rate_limit"},
                },
            }
        ),
        # Top-level error row (no result key)
        json.dumps(
            {
                "type": "error",
                "request_id": "req-top-error",
                "error": {"message": "auth"},
            }
        ),
    ]

    class _Response:
        text = "\n".join(jsonl_lines)

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "pipeline_a_scenarios.utils.llm_client.requests.get",
        lambda url, headers: _Response(),
    )

    results = client.retrieve_anthropic_batch_results_with_usage("anth-batch-1")

    assert results["req-success"] == {
        "text": "hello world",
        "input_tokens": 1234,
        "output_tokens": 567,
    }
    assert results["req-errored"]["text"].startswith("[ERROR]")
    assert results["req-errored"]["input_tokens"] == 0
    assert results["req-errored"]["output_tokens"] == 0
    assert results["req-top-error"]["text"].startswith("[ERROR]")
    assert results["req-top-error"]["input_tokens"] == 0
    assert results["req-top-error"]["output_tokens"] == 0


def test_succeeded_row_with_missing_usage_defaults_to_zero(monkeypatch):
    """If Anthropic ever omits usage on a succeeded row, sum() must not crash."""

    class _Counts:
        processing = 0
        errored = 0
        expired = 0

    class _Batch:
        request_counts = _Counts()
        results_url = "https://example.invalid/results"

    stub = _make_anthropic_client_stub(retrieve_fn=lambda batch_id: _Batch())
    client = UnifiedLLMClient(provider="anthropic", client_override=stub)
    client.api_key = "test-key"

    line = json.dumps(
        {
            "custom_id": "req-no-usage",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        }
    )

    class _Response:
        text = line

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "pipeline_a_scenarios.utils.llm_client.requests.get",
        lambda url, headers: _Response(),
    )

    results = client.retrieve_anthropic_batch_results_with_usage("anth-batch-1")
    assert results["req-no-usage"] == {
        "text": "ok",
        "input_tokens": 0,
        "output_tokens": 0,
    }
