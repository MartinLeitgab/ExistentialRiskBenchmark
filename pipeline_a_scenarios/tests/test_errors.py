import pytest
from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OSError):
        UnifiedLLMClient(provider="openai")


def test_submit_batch_requires_jsonl_for_openai():
    client = UnifiedLLMClient(
        "openai",
        client_override=object(),
    )

    with pytest.raises(ValueError):
        client.submit_batch([{"id": "1", "prompt": "hi"}])


def test_submit_batch_requires_jsonl_for_gemini():
    client = UnifiedLLMClient(
        "google",
        client_override=object(),
    )

    with pytest.raises(ValueError):
        client.submit_batch([{"id": "1", "prompt": "hi"}])
