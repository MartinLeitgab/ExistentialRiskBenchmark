"""Unified LLM client for multiple providers."""

import time
from typing import Any, Optional


class UnifiedLLMClient:
    """Unified client for Anthropic and OpenAI APIs."""

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        if provider not in ["anthropic", "openai"]:
            raise ValueError(
                f"Provider must be 'anthropic' or 'openai', got '{provider}'"
            )

        self.provider = provider
        self.model = model
        self.api_key = api_key

        # Initialize the appropriate client
        if provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
        else:  # openai
            import openai

            self.client = openai.OpenAI(api_key=api_key)

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000, **kwargs
    ) -> Any:
        """Generate a response from the LLM."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:  # openai
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        return response

    def generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> Any:
        """Generate with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                raise

    def count_tokens(self, response: Any) -> dict:
        """Count tokens from a response."""
        if self.provider == "anthropic":
            return {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
            }
        else:  # openai
            return {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
