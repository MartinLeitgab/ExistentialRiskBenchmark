from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient
from pipeline_a_scenarios.tests.test_mock_clients import (
    MockAnthropicClient,
    MockOpenAIClient,
    MockGeminiClient,
)
import pytest


def test_single_shot_anthropic():
    client = UnifiedLLMClient(
        "anthropic",
        client_override=MockAnthropicClient(),
    )
    r = client.generate("hello")
    assert r["content"] == "anthropic mock"


def test_single_shot_openai():
    client = UnifiedLLMClient(
        "openai",
        client_override=MockOpenAIClient(),
    )
    r = client.generate("hello")
    assert r["content"] == "openai mock"


def test_single_shot_gemini():
    client = UnifiedLLMClient(
        "google",
        client_override=MockGeminiClient(),
    )
    r = client.generate("hello")
    assert r["content"] == "gemini mock"
