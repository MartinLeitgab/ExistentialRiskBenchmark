import os
import time
import json
import hashlib
import threading
import concurrent.futures
import httpx
import requests

from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass

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
    id: str
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

    def _is_mock_client(self) -> bool:
        try:
            from pipeline_a_scenarios.tests.test_mock_clients import (
                MockAnthropicClient, MockOpenAIClient, MockGeminiClient,
            )
            return isinstance(self.client, (MockAnthropicClient, MockOpenAIClient, MockGeminiClient))
        except ImportError:
            return False

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        enable_cache: bool = True,
        rate_limit_per_sec: float = 5.0,
        client_override=None,
    ):
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.enable_cache = enable_cache
        self.cache: Dict[str, dict] = {}
        self.bucket = TokenBucket(rate_limit_per_sec, rate_limit_per_sec)

        if client_override:
            self.client = client_override
            return

        api_keys = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        
        key_name = api_keys.get(provider)
        if not key_name or not os.getenv(key_name):
            raise OSError(f"Missing required API key: {key_name}")
        
        self.api_key = os.getenv(key_name)

        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif provider == "openai":
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=httpx.Client(trust_env=False),
            )
        elif provider == "google":
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
                    result = self._generate_anthropic(prompt, system_prompt, temperature, max_tokens, reasoning)
                elif self.provider == "openai":
                    result = self._generate_openai(prompt, system_prompt, temperature, max_tokens, reasoning)
                else:
                    result = self._generate_google(prompt, system_prompt, temperature, max_tokens, reasoning)

                if self.enable_cache:
                    self.cache[cache_key] = result
                return result

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _apply_reasoning(self, base_tokens: int, reasoning: Optional[str]) -> tuple[int, int]:
        """Calculate budget and adjusted tokens for reasoning mode"""
        if reasoning == "high":
            budget = min(4096, max(1024, base_tokens * 3))
        elif reasoning == "standard":
            budget = min(2048, max(1024, base_tokens * 2))
        else:
            return 0, base_tokens
        return budget, max(base_tokens, budget + 200)

    def _generate_anthropic(self, prompt, system_prompt, temperature, max_tokens, reasoning):
        params = {
            "model": self.model,
            "system": system_prompt or "You are a helpful assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }

        budget, adjusted_tokens = self._apply_reasoning(max_tokens, reasoning)
        if budget:
            params.update({
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "temperature": 1.0,
                "max_tokens": adjusted_tokens
            })
        else:
            params.update({"temperature": temperature, "max_tokens": max_tokens})

        r = self.client.messages.create(**params)

        content_text = next((block.text for block in r.content if getattr(block, 'type', None) == 'text'), "")
        if not content_text:
            content_text = next((block.text for block in r.content if hasattr(block, 'text')), "")

        return {
            "content": content_text,
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
            },
        }

    def _openai_max_token_param(self) -> str:
        return "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"

    def _generate_openai(self, prompt, system_prompt, temperature, max_tokens, reasoning):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        is_reasoning = self.model.startswith(("o1", "o3", "gpt-5"))
        adjusted_tokens = max_tokens * 10 if is_reasoning else max_tokens

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            self._openai_max_token_param(): adjusted_tokens,
        }

        if reasoning in ("standard", "high") and not is_reasoning:
            params["reasoning_effort"] = "high" if reasoning == "high" else "medium"

        r = self.client.chat.completions.create(**params)

        if not r.choices:
            raise ValueError(f"OpenAI returned no choices for model {self.model}")
        
        return {
            "content": r.choices[0].message.content or "",
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
            },
        }

    def _generate_google(self, prompt, system_prompt, temperature, max_tokens, reasoning):
        full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        thinking_cfg = None
        if reasoning in ("standard", "high"):
            thinking_cfg = types.ThinkingConfig(
                thinking_budget=4096 if reasoning == "high" else 2048
            )

        r = self.client.models.generate_content(
            model=self.model,
            contents=full,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens * 8,
                thinking_config=thinking_cfg,
            ),
        )
        
        text_parts = []
        if hasattr(r, 'candidates') and r.candidates:
            text_parts = [part.text for part in r.candidates[0].content.parts if hasattr(part, 'text')]
    
        full_text = ''.join(text_parts) if text_parts else (r.text if hasattr(r, 'text') else '')

        if hasattr(r, 'usage_metadata'):
            input_tokens = r.usage_metadata.prompt_token_count
            output_tokens = r.usage_metadata.candidates_token_count
        else:
            tokens = self.count_tokens(full, full_text)
            input_tokens = tokens['input_tokens']
            output_tokens = tokens['output_tokens']

        return {
            "content": full_text,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
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
        if self._is_mock_client():
            return BatchHandle(provider=self.provider, id="mock-batch-id", metadata={"requests": requests})

        if self.provider == "anthropic":
            return BatchHandle(provider="anthropic", id=self.submit_anthropic_batch(requests))
        
        if not jsonl_path:
            raise ValueError(f"jsonl_path is required for {self.provider} batch")
        
        if self.provider == "openai":
            return BatchHandle(provider="openai", id=self.submit_openai_batch(requests, jsonl_path))
        
        if self.provider == "google":
            return BatchHandle(provider="google", id=self.submit_gemini_batch(requests, jsonl_path))

        raise ValueError(f"Unsupported provider: {self.provider}")

    def submit_anthropic_batch(self, requests: List[Dict]) -> str:
        batch_reqs = []
        
        for r in requests:
            max_tokens = r.get("max_tokens", 2048)
            temperature = r.get("temperature", 0.7)
            reasoning = r.get("reasoning")
            
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": r["prompt"]}],
                "system": r.get("system_prompt", "You are a helpful assistant."),
            }
            
            budget, adjusted_tokens = self._apply_reasoning(max_tokens, reasoning)
            if budget:
                params.update({
                    "thinking": {"type": "enabled", "budget_tokens": budget},
                    "temperature": 1.0,
                    "max_tokens": adjusted_tokens
                })
            else:
                params.update({"temperature": temperature, "max_tokens": max_tokens})
            
            batch_reqs.append(
                AnthropicBatchRequest(
                    custom_id=r["id"],
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        batch = self.client.messages.batches.create(requests=batch_reqs)
        return batch.id

    def submit_openai_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        token_param = self._openai_max_token_param()
        is_reasoning = self.model.startswith(("o1", "o3", "gpt-5"))
        
        with open(jsonl_path, "w") as f:
            for r in requests:
                max_tokens = r.get("max_tokens", 512)
                if is_reasoning:
                    max_tokens *= 10
                
                body = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": r["prompt"]}],
                    token_param: max_tokens,
                }
                
                if "system_prompt" in r:
                    body["messages"].insert(0, {"role": "system", "content": r["system_prompt"]})
                if "temperature" in r:
                    body["temperature"] = r["temperature"]
                
                reasoning = r.get("reasoning")
                if reasoning in ("standard", "high") and not is_reasoning:
                    body["reasoning_effort"] = "high" if reasoning == "high" else "medium"
                
                f.write(json.dumps({
                    "custom_id": r["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }) + "\n")
        
        file = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def submit_gemini_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        """
        Note: Gemini Batch API does NOT support thinking_config parameter.
        """
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps({
                    "key": str(r["id"]),
                    "request": {
                        "contents": [{"parts": [{"text": r["prompt"]}]}],
                        "generationConfig": {
                            "maxOutputTokens": r.get("max_tokens", 1000),
                            "temperature": r.get("temperature", 0.7),
                            "topP": r.get("top_p", 0.95),
                        }
                    }
                }) + "\n")
        
        uploaded_file = self.client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=f'batch-input-{int(time.time())}',
                mime_type='application/jsonl'
            )
        )
        
        for _ in range(30):
            file_status = self.client.files.get(name=uploaded_file.name)
            state = getattr(file_status.state, 'name', str(file_status.state))
            
            if state == 'ACTIVE':
                break
            elif state in ['FAILED', 'STATE_UNSPECIFIED', 'FILE_STATE_FAILED']:
                raise RuntimeError(f"File upload failed with state: {state}")
            time.sleep(2)
        else:
            raise TimeoutError(f"File {uploaded_file.name} did not become ACTIVE")
        
        batch_job = self.client.batches.create(
            model=self.model,
            src=uploaded_file.name,
            config=types.CreateBatchJobConfig(display_name=f'batch-job-{int(time.time())}'),
        )
        return batch_job.name

    def retrieve_batch_results(self, handle: BatchHandle, timeout: Optional[int] = None) -> Dict[str, str]:
        if self._is_mock_client():
            return {r["id"]: f"{self.provider} mock batch" for r in handle.metadata.get("requests", [])}

        timeout = timeout or {"anthropic": 1800, "openai": 1800, "google": 900}.get(handle.provider, 600)

        if handle.provider == "anthropic":
            return self.retrieve_anthropic_batch_results(handle.id, timeout)
        if handle.provider == "openai":
            return self.retrieve_openai_batch_results(handle.id, timeout)
        if handle.provider == "google":
            return self.retrieve_gemini_batch_results(handle.id, timeout)

        raise ValueError(f"Unsupported provider: {handle.provider}")

    def retrieve_anthropic_batch_results(self, batch_id: str, timeout: int = 1800) -> Dict[str, str]:
        batch = self._poll_until(
            fn=lambda: self.client.messages.batches.retrieve(batch_id),
            is_done=lambda b: b.request_counts.processing == 0,
            interval=10,
            timeout=timeout,
        )

        if batch.request_counts.errored > 0 or batch.request_counts.expired > 0:
            raise RuntimeError("Anthropic batch failed")

        response = requests.get(
            batch.results_url,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        )
        response.raise_for_status()

        results = {}
        for line in response.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            
            if "type" in obj and obj["type"] == "error" and "result" not in obj:
                results[obj.get("request_id", "unknown")] = f"[ERROR] {obj.get('error', {}).get('message', 'Unknown error')}"
                continue
            
            if "result" not in obj:
                raise ValueError(f"Unexpected batch response format: {obj}")
            
            result = obj["result"]
            custom_id = obj.get("custom_id", "unknown")
            
            if result["type"] == "succeeded":
                text = next((block.get("text", "") for block in result.get("message", {}).get("content", [])
                            if isinstance(block, dict) and block.get("type") == "text"), "")
                results[custom_id] = text
            elif result["type"] == "errored":
                results[custom_id] = f"[ERROR] {result.get('error', {}).get('message', 'Unknown error')}"
            else:
                results[custom_id] = f"[UNKNOWN] Result type: {result.get('type')}"

        return results

    def retrieve_openai_batch_results(self, batch_id: str, timeout: int = 1800) -> Dict[str, str]:
        def check_batch():
            return self.client.batches.retrieve(batch_id)
        
        batch = self._poll_until(
            fn=check_batch,
            is_done=lambda b: b.status in ("completed", "failed", "expired", "cancelled"),
            interval=30,
            timeout=timeout,
        )

        if batch.status in ("expired", "cancelled", "failed"):
            raise RuntimeError(f"Batch {batch.status}")
            
        output_file_id = getattr(batch, 'output_file_id', None)
        if not output_file_id:
            time.sleep(2)
            batch = self.client.batches.retrieve(batch_id)
            output_file_id = getattr(batch, 'output_file_id', None)
        
        if not output_file_id:
            error_file_id = getattr(batch, 'error_file_id', None)
            if error_file_id:
                error_content = self.client.files.content(error_file_id)
                raw = error_content.read().decode('utf-8')
                results = {}
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    custom_id = obj.get("custom_id", "unknown")
                    
                    if "error" in obj and obj["error"]:
                        error_info = obj["error"]
                        results[custom_id] = f"[ERROR] {error_info.get('type', 'unknown')}: {error_info.get('message', 'Unknown error')}"
                    else:
                        results[custom_id] = "[ERROR] Request failed"
                return results
            raise RuntimeError("No output or error file available")
        
        output_content = self.client.files.content(output_file_id)
        raw = output_content.read().decode('utf-8')
        
        results = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            custom_id = obj.get("custom_id", "unknown")
            
            if "error" in obj and obj["error"]:
                results[custom_id] = f"[ERROR] {obj['error'].get('message', 'Unknown error')}"
                continue
            
            response = obj.get("response", {})
            if response.get("status_code") != 200:
                results[custom_id] = f"[ERROR] HTTP {response.get('status_code')}"
                continue
            
            choices = response.get("body", {}).get("choices", [])
            results[custom_id] = choices[0].get("message", {}).get("content", "") if choices else "[ERROR] No choices"
        
        return results

    def retrieve_gemini_batch_results(self, batch_id: str, timeout: int) -> Dict[str, str]:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        
        def api_call_with_timeout(method, *args, timeout_sec=30, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(method, *args, **kwargs)
                return future.result(timeout=timeout_sec)
        
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout after {timeout}s")
            
            try:
                batch_job = api_call_with_timeout(self.client.batches.get, name=batch_id, timeout_sec=30)
            except FutureTimeout:
                time.sleep(5)
                continue
            
            state = getattr(batch_job.state, 'name', str(batch_job.state))
            
            if state == 'SUCCEEDED':
                break
            elif state in ('FAILED', 'CANCELLED'):
                raise RuntimeError(f"Job {state}: {getattr(batch_job, 'error', 'Unknown error')}")
            
            time.sleep(10)
        
        try:
            result_bytes = api_call_with_timeout(
                self.client.files.download,
                file=batch_job.dest.file_name,
                timeout_sec=60
            )
        except FutureTimeout:
            raise TimeoutError("Result download timed out")
        
        results = {}
        for line in result_bytes.decode('utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key")
            candidates = item.get("response", {}).get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                results[key] = "".join(p.get("text", "") for p in parts)
            else:
                results[key] = f"[ERROR] {item.get('response', {}).get('error', 'No candidates')}"
        
        return results

    def submit_gemini_parallel(self, requests: List[Dict]) -> BatchHandle:
        """
        Parallel execution workaround for Gemini to support reasoning/thinking_config.
        Since Gemini Batch API doesn't support thinking_config, use parallel individual calls.
        """
        if self.provider != "google":
            raise ValueError("Only for Google provider")
        
        def run_one(r):
            try:
                result = self.generate(
                    prompt=r["prompt"],
                    system_prompt=r.get("system_prompt"),
                    max_tokens=r.get("max_tokens", 1000),
                    temperature=r.get("temperature", 0.7),
                    reasoning=r.get("reasoning")
                )
                return {"id": r["id"], "content": result["content"], "error": None}
            except Exception as e:
                return {"id": r["id"], "content": None, "error": str(e)}
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
            for future in concurrent.futures.as_completed([executor.submit(run_one, r) for r in requests]):
                res = future.result()
                results[res["id"]] = f"[ERROR] {res['error']}" if res["error"] else res["content"]
        
        return BatchHandle(
            provider="google",
            id=f"parallel-{int(time.time())}",
            metadata={"results": results, "is_parallel": True, "completed_at": time.time()}
        )

    def retrieve_gemini_parallel_results(self, handle: BatchHandle) -> Dict[str, str]:
        if not handle.metadata or not handle.metadata.get("is_parallel"):
            raise ValueError("Not a parallel batch handle")
        return handle.metadata["results"]

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
        return tokens["input_tokens"] * in_cost + expected_output_tokens * out_cost

    def judge_with_gpt4o(self, prompt: str) -> str:
        judge = UnifiedLLMClient(provider="openai", model="gpt-4o", enable_cache=False)
        return judge.generate(prompt)["content"]

    @staticmethod
    def _hash(*items) -> str:
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()