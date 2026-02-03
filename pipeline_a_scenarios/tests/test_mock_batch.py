import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient
from pipeline_a_scenarios.tests.test_mock_clients import (
    MockAnthropicClient,
    MockOpenAIClient,
    MockGeminiClient,
)

@pytest.mark.parametrize(
    "provider,mock_client,needs_jsonl",
    [
        ("anthropic", MockAnthropicClient(), False),
        ("openai", MockOpenAIClient(), True),
        ("google", MockGeminiClient(), True),
    ],
)
def test_batch_facade(provider, mock_client, needs_jsonl):
    client = UnifiedLLMClient(
        provider,
        client_override=mock_client,
    )

    kwargs = {}
    if needs_jsonl:
        kwargs["jsonl_path"] = "tmp.jsonl"

    handle = client.submit_batch(
        [{"id": "req-1", "prompt": "hello"}],
        **kwargs,
    )

    results = client.retrieve_batch_results(handle)

    assert "req-1" in results
