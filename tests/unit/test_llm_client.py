"""Tests for the UnifiedLLMClient utility module."""

import pytest
from unittest.mock import Mock, patch

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return Mock(
        content="I choose Action A because...",
        usage=Mock(input_tokens=100, output_tokens=50),
    )


@patch("anthropic.Anthropic")
def test_client_initialization(mock_anthropic):
    """Test LLM client initializes correctly."""
    mock_anthropic.return_value = Mock()

    client = UnifiedLLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"


def test_client_invalid_provider():
    """Test client rejects invalid provider."""
    with pytest.raises(ValueError):
        UnifiedLLMClient(provider="invalid_provider", model="some-model")


@patch("anthropic.Anthropic")
def test_generate_anthropic(mock_anthropic, mock_anthropic_response):
    """Test generation with Anthropic."""
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response
    mock_anthropic.return_value = mock_client

    client = UnifiedLLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    response = client.generate("Test prompt", temperature=0.0, max_tokens=100)

    assert response.content == "I choose Action A because..."
    assert mock_client.messages.create.called


@patch("anthropic.Anthropic")
def test_token_counting(mock_anthropic, mock_anthropic_response):
    """Test token counting."""
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    client = UnifiedLLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    tokens = client.count_tokens(mock_anthropic_response)
    assert tokens["input"] == 100
    assert tokens["output"] == 50
    assert tokens["total"] == 150


@patch("anthropic.Anthropic")
def test_retry_on_rate_limit(mock_anthropic):
    """Test retry logic on rate limit."""
    mock_client = Mock()
    # First call raises rate limit, second succeeds
    mock_client.messages.create.side_effect = [
        Exception("Rate limit exceeded"),
        Mock(content="Success", usage=Mock(input_tokens=50, output_tokens=25)),
    ]
    mock_anthropic.return_value = mock_client

    client = UnifiedLLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    response = client.generate_with_retry("Test", max_retries=2)

    assert response.content == "Success"
    assert mock_client.messages.create.call_count == 2
