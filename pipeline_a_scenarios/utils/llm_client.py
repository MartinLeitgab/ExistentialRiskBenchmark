import os
import time
import json
import hashlib
import threading
import httpx
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

BatchProvider = Literal["anthropic", "openai", "google"]
@dataclass(frozen=True)
class BatchHandle:
    provider: BatchProvider
    id: str                 # batch_id or file_name
    metadata: Optional[Any] = None



class TokenBucket:
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
    Unified LLM client supporting:
      - Anthropic Claude (Sonnet 4.5)
      - OpenAI GPT-5.2 / GPT-4o
      - Google Gemini 3.0

    Features:
      - single-shot generation
      - batch APIs (Anthropic, OpenAI, Gemini)
      - retries with exponential backoff
      - token counting
      - cost estimation
      - rate limiting
      - response caching
      - LLM-as-judge fallback
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-5.2",
        "google": "gemini-3-flash-preview",
    }

    PRICING = {
        "claude-sonnet-4-5-20250929": (3 / 1e6, 15 / 1e6),
        "gpt-5.2": (5 / 1e6, 15 / 1e6),
        "gpt-4o": (5 / 1e6, 15 / 1e6),
        "gemini-3-flash-preview": (2 / 1e6, 8 / 1e6),
    }

    @staticmethod
    def _require_env(key: str):
        if key not in os.environ or not os.environ[key]:
            raise OSError(f"Missing required API key: {key}")


    def _validate_api_key(self):
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
        client_override=None,   # used for tests / mocks
    ):

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.enable_cache = enable_cache
        self.cache: Dict[str, dict] = {}
        self.bucket = TokenBucket(rate_limit_per_sec, rate_limit_per_sec)

        if client_override:
            self.client = client_override
            return

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

                if self.enable_cache:
                    self.cache[cache_key] = result

                return result

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _generate_anthropic(self, prompt, system_prompt, temperature, max_tokens):
        r = self.client.messages.create(
            model=self.model,
            system=system_prompt or "You are a helpful assistant.",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        print(f"\nModel:{r.model}")

        return {
            "content": r.content[0].text,
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
            },
        }

    def _openai_max_token_param(self) -> str:
        """
        GPT-5.x+ models require max_completion_tokens instead of max_tokens.
        """
        return "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"

    def _generate_openai(self, prompt, system_prompt, temperature, max_tokens):
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

        print(f"\nModel:{r.model}")

        return {
            "content": r.choices[0].message.content,
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
            },
        }

    def _generate_google(self, prompt, system_prompt, temperature, max_tokens):
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

        print("\nModel:gemini-3-flash-preview")

        return {
            "content": r.text,
            "usage": tokens,
        }

    def _poll_until(self, fn, is_done, interval=5, timeout=300):
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError("Batch polling timed out")

            obj = fn()
            if is_done(obj):
                return obj

            time.sleep(interval)
    
    def submit_batch(self, requests: List[Dict], jsonl_path: Optional[str] = None) -> BatchHandle:
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
        if handle.provider == "anthropic":
            return self.retrieve_anthropic_batch_results(handle.id)

        if handle.provider == "openai":
            return self.retrieve_openai_batch_results(handle.id)

        if handle.provider == "google":
            return self.retrieve_gemini_batch_results(handle.id)

        raise ValueError(f"Unsupported provider: {handle.provider}")

    def submit_anthropic_batch(self, requests: List[Dict]) -> str:
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
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return {
                "input_tokens": len(enc.encode(prompt)),
                "output_tokens": len(enc.encode(completion)),
            }
        except Exception:
            return {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(completion) // 4,
            }

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 500) -> Optional[float]:
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
        judge = UnifiedLLMClient(
            provider="openai",
            model="gpt-4o",
            enable_cache=False,
        )
        return judge.generate(prompt)["content"]

    @staticmethod
    def _hash(*items) -> str:
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
