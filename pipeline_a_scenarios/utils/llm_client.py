"""
Unified LLM Client with Cost Tracking Integration

Supports:
- Anthropic Claude (Sonnet 4.5)
- OpenAI GPT-5.2 / GPT-4o
- Google Gemini 3.0

Features:
- Single-shot generation with automatic cost tracking
- Batch API submission and retrieval
- Rate limiting with token bucket
- Response caching
- Retry logic with exponential backoff
- Token counting and cost estimation
- LLM-as-judge fallback

Author: Pooja Puranik
Version: 2.0.0 (with cost tracking)
Date: 26/01/2026
"""

import os
import time
import json
import hashlib
import threading
import httpx
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from typing import Literal, Any

from dotenv import load_dotenv
load_dotenv()

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as AnthropicBatchRequest
import openai
import tiktoken
from google import genai
from google.genai import types

# ✅ Cost Tracker Integration
from pipeline_a_scenarios.utils.cost_tracker import get_tracker

BatchProvider = Literal["anthropic", "openai", "google"]

@dataclass(frozen=True)
class BatchHandle:
    provider: BatchProvider
    id: str                 # batch_id or file_name
    metadata: Optional[Any] = None


class TokenBucket:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0):
        with self.lock:
            now = time.time()
            delta = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)

            if self.tokens < tokens:
                sleep = (tokens - self.tokens) / self.rate
                time.sleep(sleep)
                self.tokens = 0
            else:
                self.tokens -= tokens


class UnifiedLLMClient:
    """
    Unified LLM client with automatic cost tracking.
    
    Features:
      - Single-shot generation with auto cost logging
      - Batch APIs (Anthropic, OpenAI, Gemini)
      - Retries with exponential backoff (3 attempts)
      - Token counting (tiktoken with fallback)
      - Real-time cost estimation
      - Rate limiting (configurable)
      - Response caching (SHA-256 hashed keys)
      - LLM-as-judge fallback (GPT-4o)
      - Optional cost tracking (enabled by default)
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-5.2",
        "google": "gemini-3-flash-preview",
    }

    # Pricing per million tokens (input, output) - January 2026
    PRICING = {
        "claude-sonnet-4-5-20250929": (3 / 1e6, 15 / 1e6),
        "gpt-5.2": (5 / 1e6, 15 / 1e6),
        "gpt-4o": (5 / 1e6, 15 / 1e6),
        "gemini-3-flash-preview": (2 / 1e6, 8 / 1e6),
    }

    @staticmethod
    def _require_env(key: str):
        """Raise error if required environment variable is missing."""
        if key not in os.environ or not os.environ[key]:
            raise OSError(f"Missing required API key: {key}")

    def _validate_api_key(self):
        """Validate API key for current provider."""
        env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(self.provider)

        if env_var:
            self._require_env(env_var)

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        enable_cache: bool = True,
        rate_limit_per_sec: float = 5.0,
        client_override=None,
        enable_cost_tracking: bool = True,
    ):
        """
        Initialize unified LLM client with cost tracking.
        
        Args:
            provider: One of 'anthropic', 'openai', 'google'
            model: Specific model name (uses default if None)
            enable_cache: Enable response caching (default: True)
            rate_limit_per_sec: API rate limit in requests per second
            client_override: Mock client for testing
            enable_cost_tracking: Automatically log costs (default: True)
            
        Raises:
            ValueError: If provider is not supported
            OSError: If required API key is missing
        """
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.enable_cache = enable_cache
        self.cache: Dict[str, dict] = {}
        self.bucket = TokenBucket(rate_limit_per_sec, rate_limit_per_sec)
        
        # ✅ Cost tracking integration
        self.enable_cost_tracking = enable_cost_tracking
        if enable_cost_tracking:
            self.cost_tracker = get_tracker()
        else:
            self.cost_tracker = None

        # Use mock client for testing
        if client_override:
            self.client = client_override
            return

        # Validate and setup real client
        self._validate_api_key()

        if provider == "anthropic":
            self._require_env("ANTHROPIC_API_KEY")
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise OSError("ANTHROPIC_API_KEY is missing")
            self.client = anthropic.Anthropic(api_key=self.api_key)

        elif provider == "openai":
            self._require_env("OPENAI_API_KEY")
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise OSError("OPENAI_API_KEY is missing")
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=httpx.Client(trust_env=False),
            )

        elif provider == "google":
            self._require_env("GOOGLE_API_KEY")
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise OSError("GOOGLE_API_KEY is missing")
            self.client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> dict:
        """
        Generate a response with automatic cost tracking.
        
        Args:
            prompt: User prompt/message
            system_prompt: System instructions (optional)
            temperature: Creativity control (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with 'content' and 'usage' keys
            
        Raises:
            Exception: After 3 failed retry attempts
        """
        cache_key = self._hash(prompt, system_prompt, temperature, max_tokens)
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(3):
            try:
                self.bucket.consume()

                if self.provider == "anthropic":
                    result = self._generate_anthropic(
                        prompt, system_prompt, temperature, max_tokens
                    )
                elif self.provider == "openai":
                    result = self._generate_openai(
                        prompt, system_prompt, temperature, max_tokens
                    )
                else:
                    result = self._generate_google(
                        prompt, system_prompt, temperature, max_tokens
                    )

                # ✅ Automatic cost logging
                if self.enable_cost_tracking and self.cost_tracker:
                    try:
                        self.cost_tracker.auto_log_from_llm_client(
                            provider=self.provider,
                            model=self.model,
                            response=result
                        )
                    except Exception as e:
                        # Don't fail API call if cost logging fails
                        print(f"⚠️  Cost logging failed: {e}")

                if self.enable_cache:
                    self.cache[cache_key] = result

                return result

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _generate_anthropic(self, prompt, system_prompt, temperature, max_tokens):
        """Generate using Anthropic Claude."""
        r = self.client.messages.create(
            model=self.model,
            system=system_prompt or "You are a helpful assistant.",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        print(f"\nModel: {r.model}")

        return {
            "content": r.content[0].text,
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
            },
        }

    def _openai_max_token_param(self) -> str:
        """GPT-5.x+ models require max_completion_tokens instead of max_tokens."""
        return "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"

    def _generate_openai(self, prompt, system_prompt, temperature, max_tokens):
        """Generate using OpenAI GPT."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        token_param = self._openai_max_token_param()

        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **{token_param: max_tokens},
        )

        print(f"\nModel: {r.model}")

        return {
            "content": r.choices[0].message.content,
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
            },
        }

    def _generate_google(self, prompt, system_prompt, temperature, max_tokens):
        """Generate using Google Gemini."""
        full = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        r = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            ),
        )

        tokens = self.count_tokens(full, r.text)

        print("\nModel: gemini-3-flash-preview")

        return {
            "content": r.text,
            "usage": tokens,
        }

    def _poll_until(self, fn, is_done, interval=5, timeout=300):
        """Poll until condition is met or timeout."""
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError("Batch polling timed out")

            obj = fn()
            if is_done(obj):
                return obj

            time.sleep(interval)
    
    def submit_batch(self, requests: List[Dict], jsonl_path: Optional[str] = None) -> BatchHandle:
        """Submit batch of requests for processing."""
        if self.provider == "anthropic":
            batch_id = self.submit_anthropic_batch(requests)
            return BatchHandle(provider="anthropic", id=batch_id)

        if self.provider == "openai":
            if not jsonl_path:
                raise ValueError("jsonl_path is required for OpenAI batch")
            batch_id = self.submit_openai_batch(requests, jsonl_path)
            return BatchHandle(provider="openai", id=batch_id)

        if self.provider == "google":
            if not jsonl_path:
                raise ValueError("jsonl_path is required for Gemini batch")
            file_name = self.submit_gemini_batch(requests, jsonl_path)
            return BatchHandle(provider="google", id=file_name)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def retrieve_batch_results(self, handle: BatchHandle) -> Dict[str, str]:
        """Retrieve results from a submitted batch."""
        if handle.provider == "anthropic":
            return self.retrieve_anthropic_batch_results(handle.id)

        if handle.provider == "openai":
            return self.retrieve_openai_batch_results(handle.id)

        if handle.provider == "google":
            return self.retrieve_gemini_batch_results(handle.id)

        raise ValueError(f"Unsupported provider: {handle.provider}")

    def submit_anthropic_batch(self, requests: List[Dict]) -> str:
        """Submit batch to Anthropic."""
        batch_reqs = [
            AnthropicBatchRequest(
                custom_id=r["id"],
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=r.get("max_tokens", 1024),
                    messages=[{"role": "user", "content": r["prompt"]}],
                ),
            )
            for r in requests
        ]

        batch = self.client.messages.batches.create(requests=batch_reqs)
        return batch.id

    def submit_openai_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        """Submit batch to OpenAI."""
        with open(jsonl_path, "w") as f:
            for r in requests:
                f.write(json.dumps({
                    "custom_id": r["id"],
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": r["prompt"],
                        "max_output_tokens": 512,
                    },
                }) + "\n")

        file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch",
        )

        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )

        return batch.id

    def submit_gemini_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        """Submit batch to Google Gemini."""
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps({
                    "key": r["id"],
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": r["prompt"]}],
                            }
                        ]
                    },
                }) + "\n")

        uploaded = self.client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name="gemini-batch-input",
                mime_type="application/jsonl",
            ),
        )

        batch = self.client.batches.create(
            model=self.model,
            input_file=uploaded.name,
        )

        return batch.name

    def retrieve_anthropic_batch_results(self, batch_id: str) -> Dict[str, str]:
        """Retrieve results from Anthropic batch."""
        batch = self._poll_until(
            fn=lambda: self.client.messages.batches.retrieve(batch_id),
            is_done=lambda b: b.request_counts.processing == 0,
            interval=10,
            timeout=1800,
        )

        if batch.request_counts.errored > 0 or batch.request_counts.expired > 0:
            raise RuntimeError("Anthropic batch failed")

        raw = requests.get(batch.results_url).text

        results = {}
        for line in raw.splitlines():
            obj = json.loads(line)
            results[obj["custom_id"]] = obj["content"][0]["text"]

        return results

    def retrieve_openai_batch_results(self, batch_id: str) -> Dict[str, str]:
        """Retrieve results from OpenAI batch."""
        batch = self._poll_until(
            fn=lambda: self.client.batches.retrieve(batch_id),
            is_done=lambda b: (
                b.status == "completed" and b.output_file_id
            ) or b.status in ("failed", "expired", "cancelled"),
            interval=10,
            timeout=600,
        )

        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch ended with status: {batch.status}")

        raw = self.client.files.content(batch.output_file_id)

        results = {}
        for line in raw.decode().splitlines():
            obj = json.loads(line)
            results[obj["custom_id"]] = obj["response"]["choices"][0]["message"]["content"]

        return results

    def retrieve_gemini_batch_results(self, batch_name: str) -> Dict[str, str]:
        """Retrieve results from Gemini batch."""
        batch = self._poll_until(
            fn=lambda: self.client.batches.get(name=batch_name),
            is_done=lambda b: b.state in ("SUCCEEDED", "FAILED"),
            interval=10,
            timeout=1800,
        )

        if batch.state != "SUCCEEDED":
            raise RuntimeError(f"Gemini batch failed: {batch.state}")

        raw = self.client.files.download(name=batch.output_file).decode("utf-8")

        results = {}
        for line in raw.splitlines():
            obj = json.loads(line)
            results[obj["key"]] = (
                obj["response"]["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

        return results

    def count_tokens(self, prompt: str, completion: str = "") -> dict:
        """
        Count tokens in prompt and completion.
        
        Uses tiktoken for accurate counting with character-based fallback.
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return {
                "input_tokens": len(enc.encode(prompt)),
                "output_tokens": len(enc.encode(completion)),
            }
        except Exception:
            # Fallback: rough estimate (4 chars ≈ 1 token)
            return {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(completion) // 4,
            }

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 500) -> Optional[float]:
        """
        Estimate cost for a prompt based on current pricing.
        
        Args:
            prompt: Input text to estimate
            expected_output_tokens: Expected response size
            
        Returns:
            Estimated cost in USD or None if model not in pricing table
        """
        pricing = self.PRICING.get(self.model)
        if not pricing:
            return None

        tokens = self.count_tokens(prompt)
        in_cost, out_cost = pricing
        return (
            tokens["input_tokens"] * in_cost
            + expected_output_tokens * out_cost
        )

    def judge_with_gpt4o(self, prompt: str) -> str:
        """
        Use GPT-4o as a judge for evaluation/scoring.
        
        Note: Cost tracking is disabled for judge calls to avoid double counting.
        """
        judge = UnifiedLLMClient(
            provider="openai",
            model="gpt-4o",
            enable_cache=False,
            enable_cost_tracking=False,  # Don't double-log judge calls
        )
        return judge.generate(prompt)["content"]

    @staticmethod
    def _hash(*items) -> str:
        """Generate SHA-256 hash for caching."""
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


# Quick usage example
if __name__ == "__main__":
    # Example with cost tracking
    client = UnifiedLLMClient(
        provider="openai",
        model="gpt-4o",
        enable_cost_tracking=True
    )
    
    response = client.generate("Hello, how are you?")
    print(f"Response: {response['content'][:100]}...")
    
    # Check costs
    from pipeline_a_scenarios.utils.cost_tracker import get_tracker
    tracker = get_tracker()
    print(f"\nTotal cost so far: ${tracker.get_total_cost():.6f}")