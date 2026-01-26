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
    
    def _is_mock_client(self) -> bool:
        # Only consider known Mock classes as mocks
        from pipeline_a_scenarios.tests.test_mock_clients import (
            MockAnthropicClient,
            MockOpenAIClient,
            MockGeminiClient,
        )

        return isinstance(
            self.client,
            (MockAnthropicClient, MockOpenAIClient, MockGeminiClient)
        )


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
        reasoning: Optional[Literal["none", "standard", "high"]] = None,
    ) -> dict:

        cache_key = self._hash(prompt, system_prompt, temperature, max_tokens, reasoning)
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(3):
            try:
                self.bucket.consume()

                if self.provider == "anthropic":
                    result = self._generate_anthropic(
                        prompt, system_prompt, temperature, max_tokens, reasoning
                    )
                elif self.provider == "openai":
                    result = self._generate_openai(
                        prompt, system_prompt, temperature, max_tokens, reasoning
                    )
                else:
                    result = self._generate_google(
                        prompt, system_prompt, temperature, max_tokens, reasoning
                    )

                if self.enable_cache:
                    self.cache[cache_key] = result

                return result

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _generate_anthropic(
        self,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        reasoning: Optional[str] = None,
    ):
        params = dict(
            model=self.model,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )

        # When thinking/reasoning is enabled, temperature MUST be 1.0
        # and max_tokens must be GREATER than budget_tokens
        # Also, budget_tokens must be >= 1024 (Anthropic requirement)
        if reasoning in ("standard", "high"):
            # Budget tokens: minimum 1024, scale based on reasoning level
            # For "high" reasoning: use more budget, for "standard": use less
            if reasoning == "high":
                budget_tokens = min(4096, max(1024, max_tokens * 3))
            else:
                budget_tokens = min(2048, max(1024, max_tokens * 2))
            
            # Ensure max_tokens is greater than budget_tokens
            # Add buffer (at least 200 tokens) to ensure the constraint is satisfied
            adjusted_max_tokens = max(max_tokens, budget_tokens + 200)
            
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
            params["temperature"] = 1.0  # Required by Anthropic API
            params["max_tokens"] = adjusted_max_tokens
        else:
            params["temperature"] = temperature
            params["max_tokens"] = max_tokens

        r = self.client.messages.create(**params)

        # When thinking is enabled, the response contains both ThinkingBlock and TextBlock
        # ThinkingBlock: internal reasoning (type='thinking')
        # TextBlock: the actual response (type='text')
        # We need to extract text from TextBlock only
        content_text = ""
        for block in r.content:
            # Check block type - we want 'text', not 'thinking'
            block_type = getattr(block, 'type', None)
            if block_type == 'text':
                content_text = block.text
                break

        # Fallback: if no text block found, try to get first block with text attribute
        if not content_text:
            for block in r.content:
                if hasattr(block, 'text'):
                    content_text = block.text
                    break

        return {
            "content": content_text,
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

    def _generate_openai(
        self,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        reasoning: Optional[str] = None,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Determine which token parameter to use based on model
        token_param = self._openai_max_token_param()
        
        # For reasoning models (o1, o3, gpt-5.x), they need MUCH more tokens
        # because they use tokens for internal reasoning before generating output
        # The reasoning tokens don't count toward the visible output
        is_reasoning_model = (
            self.model.startswith("o1") or 
            self.model.startswith("o3") or 
            self.model.startswith("gpt-5")
        )
        
        if is_reasoning_model:
            # Reasoning models need 3-5x more tokens to allow for both reasoning and output
            # For example: 300 tokens requested -> 1200 tokens actual (900 reasoning + 300 output)
            adjusted_max_tokens = max_tokens * 10
        else:
            adjusted_max_tokens = max_tokens

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            token_param: adjusted_max_tokens,
        }

        # For explicit reasoning requests on non-reasoning models
        if reasoning in ("standard", "high") and not is_reasoning_model:
            params["reasoning_effort"] = "high" if reasoning == "high" else "medium"

        # TEMPORARY DEBUG OUTPUT
        print(f"\n[DEBUG OpenAI] Model: {self.model}")
        print(f"[DEBUG OpenAI] Is reasoning model: {is_reasoning_model}")
        print(f"[DEBUG OpenAI] Token param: {token_param} = {adjusted_max_tokens} (requested: {max_tokens})")
        print(f"[DEBUG OpenAI] Temperature: {temperature}")

        try:
            r = self.client.chat.completions.create(**params)
            
            # TEMPORARY DEBUG OUTPUT
            print(f"[DEBUG OpenAI] Response model: {r.model}")
            print(f"[DEBUG OpenAI] Content length: {len(r.choices[0].message.content or '')}")
            print(f"[DEBUG OpenAI] Usage: {r.usage}")
            
            # Check for reasoning tokens
            if hasattr(r.usage, 'completion_tokens_details'):
                details = r.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    print(f"[DEBUG OpenAI] Reasoning tokens: {details.reasoning_tokens}")
            
        except Exception as e:
            print(f"\n[ERROR OpenAI] API call failed: {e}")
            print(f"[ERROR OpenAI] Params: {params}")
            raise

        # Check if we got a valid response
        if not r.choices:
            raise ValueError(f"OpenAI returned no choices for model {self.model}")
        
        content = r.choices[0].message.content or ""
        
        return {
            "content": content,
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
            },
        }


    def _generate_google(
        self,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        reasoning: Optional[str] = None,
    ):
        # Combine system prompt and user prompt
        full = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

        thinking_cfg = None
        if reasoning in ("standard", "high"):
            thinking_cfg = types.ThinkingConfig(
                thinking_budget=4096 if reasoning == "high" else 2048
            )

        # Google's token counting is very conservative - multiply by 5-10x
        # to get output similar to OpenAI/Anthropic
        # This accounts for Google's stricter token boundaries
        adjusted_max_tokens = max_tokens * 8  # 300 → 2400

        # DEBUG: Print request details (skip for mocks)
        is_mock = self._is_mock_client()
        if not is_mock:
            print(f"\n[DEBUG Google] Model: {self.model}")
            print(f"[DEBUG Google] Max output tokens: {adjusted_max_tokens} (requested: {max_tokens})")
            print(f"[DEBUG Google] Temperature: {temperature}")
            print(f"[DEBUG Google] Thinking config: {thinking_cfg}")

        r = self.client.models.generate_content(
            model=self.model,
            contents=full,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=adjusted_max_tokens,
                thinking_config=thinking_cfg,
            ),
        )

        # DEBUG: Print response details (skip for mocks)
        if not is_mock:
            print(f"[DEBUG Google] Response candidates: {len(r.candidates)}")
            if r.candidates:
                print(f"[DEBUG Google] Finish reason: {r.candidates[0].finish_reason}")
                print(f"[DEBUG Google] Content parts: {len(r.candidates[0].content.parts)}")
        
        # Extract text - check if there are multiple parts
        text_parts = []
        if hasattr(r, 'candidates') and r.candidates:
            for part in r.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    text_parts.append(part.text)
        
        # Combine all text parts
        full_text = ''.join(text_parts) if text_parts else (r.text if hasattr(r, 'text') else '')
        
        if not is_mock:
            print(f"[DEBUG Google] Full text length: {len(full_text)} chars")
            print(f"[DEBUG Google] Text preview: {full_text[:100]}")

        # Use actual token count if available, otherwise approximate
        if hasattr(r, 'usage_metadata'):
            input_tokens = r.usage_metadata.prompt_token_count
            output_tokens = r.usage_metadata.candidates_token_count
            if not is_mock:
                print(f"[DEBUG Google] Actual tokens - input: {input_tokens}, output: {output_tokens}")
        else:
            # Fallback to approximation
            tokens = self.count_tokens(full, full_text)
            input_tokens = tokens['input_tokens']
            output_tokens = tokens['output_tokens']

        return {
            "content": full_text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
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
        # MOCK SHORT-CIRCUIT
        if self._is_mock_client():
            return BatchHandle(
                provider=self.provider,
                id="mock-batch-id",
                metadata={"requests": requests},
            )

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
        # MOCK SHORT-CIRCUIT
        if self._is_mock_client():
            return {
                r["id"]: f"{self.provider} mock batch"
                for r in handle.metadata.get("requests", [])
            }

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
        """
        Fixed OpenAI batch submission to use correct endpoint.
        The endpoint should be /v1/chat/completions for chat models.
        """
        with open(jsonl_path, "w") as f:
            for r in requests:
                f.write(json.dumps({
                    "custom_id": r["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",  # Fixed endpoint
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": r["prompt"]}
                        ],
                        "max_tokens": r.get("max_tokens", 512),
                    },
                }) + "\n")

        file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch",
        )

        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",  # Fixed endpoint
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
        """
        Fixed to handle the correct response structure.
        """
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
            # Handle both success and error cases
            if "response" in obj and obj["response"]["body"]:
                body = obj["response"]["body"]
                if "choices" in body and len(body["choices"]) > 0:
                    results[obj["custom_id"]] = body["choices"][0]["message"]["content"]
                else:
                    results[obj["custom_id"]] = ""
            else:
                results[obj["custom_id"]] = ""

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