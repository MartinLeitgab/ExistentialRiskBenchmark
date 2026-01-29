import os
import time
import json
import hashlib
import threading
import concurrent.futures
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

    def submit_anthropic_batch(self, requests: List[Dict]) -> str:
        """
        Submit a batch of requests to Anthropic with full configuration support.
        
        Each request dict can include:
        - id (required): Custom identifier for the request
        - prompt (required): The user prompt
        - max_tokens (optional): Maximum tokens to generate (default: 2048)
        - temperature (optional): Sampling temperature (default: 0.7)
        - system_prompt (optional): System prompt (default: "You are a helpful assistant.")
        - reasoning (optional): Enable extended thinking - "none", "standard", or "high"
        """
        batch_reqs = []
        
        for r in requests:
            max_tokens = r.get("max_tokens", 2048)
            temperature = r.get("temperature", 0.7)
            reasoning = r.get("reasoning")
            
            # Build params dict
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": r["prompt"]}],
                "system": r.get("system_prompt", "You are a helpful assistant."),
            }
            
            # Apply reasoning/thinking configuration (same logic as _generate_anthropic)
            if reasoning in ("standard", "high"):
                # Budget tokens: minimum 1024, scale based on reasoning level
                if reasoning == "high":
                    budget_tokens = min(4096, max(1024, max_tokens * 3))
                else:
                    budget_tokens = min(2048, max(1024, max_tokens * 2))
                
                # Ensure max_tokens is greater than budget_tokens
                adjusted_max_tokens = max(max_tokens, budget_tokens + 200)
                
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                params["temperature"] = 1.0  # Required by Anthropic API when thinking is enabled
                params["max_tokens"] = adjusted_max_tokens
            else:
                params["temperature"] = temperature
                params["max_tokens"] = max_tokens
            
            batch_reqs.append(
                AnthropicBatchRequest(
                    custom_id=r["id"],
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        batch = self.client.messages.batches.create(requests=batch_reqs)
        return batch.id

    def submit_gemini_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        """
        Gemini Batch API compliant submission with optimized generation config.
        """
        import json
        import time
        from google.genai import types
        
        # Step 1: Write JSONL file with generation config for faster processing
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in requests:
                batch_line = {
                    "key": str(r["id"]),
                    "request": {
                        "contents": [{"parts": [{"text": r["prompt"]}]}],
                        "generationConfig": {
                            "maxOutputTokens": r.get("max_tokens", 1000),
                            "temperature": r.get("temperature", 0.7),
                            "topP": r.get("top_p", 0.95),
                        }
                    }
                }
                f.write(json.dumps(batch_line) + "\n")
        
        print(f"[DEBUG] Created JSONL file: {jsonl_path} with {len(requests)} requests")
        
        # Step 2: Upload file with explicit MIME type
        uploaded_file = self.client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=f'batch-input-{int(time.time())}',
                mime_type='application/jsonl'
            )
        )
        print(f"[DEBUG] Uploaded file: {uploaded_file.name}")
        
        # Step 3: Wait for file to become ACTIVE
        max_wait = 60
        wait_interval = 2  # Shorter interval for faster detection
        elapsed = 0
        
        while elapsed < max_wait:
            file_status = self.client.files.get(name=uploaded_file.name)
            # Handle both enum objects and string states
            if hasattr(file_status.state, 'name'):
                state = file_status.state.name
            else:
                state = str(file_status.state)
                
            print(f"[DEBUG] File state: {state} ({elapsed}s elapsed)")
            
            # Fix: Check for 'ACTIVE' not 'FILE_STATE_ACTIVE'
            if state == 'ACTIVE':
                print(f"[DEBUG] File is ACTIVE, proceeding with batch creation")
                break
            elif state in ['FAILED', 'STATE_UNSPECIFIED', 'FILE_STATE_FAILED']:
                raise RuntimeError(f"File upload failed with state: {state}")
            
            time.sleep(wait_interval)
            elapsed += wait_interval
        else:
            # If we exit loop without break, check final state once more
            file_status = self.client.files.get(name=uploaded_file.name)
            final_state = getattr(file_status.state, 'name', str(file_status.state))
            if final_state == 'ACTIVE':
                print(f"[DEBUG] File is ACTIVE (final check), proceeding")
            else:
                raise TimeoutError(f"File {uploaded_file.name} did not become ACTIVE within {max_wait}s. Final state: {final_state}")
        
        # Step 4: Create batch job
        batch_job = self.client.batches.create(
            model=self.model,
            src=uploaded_file.name,
            config=types.CreateBatchJobConfig(
                display_name=f'batch-job-{int(time.time())}',
            )
        )
        print(f"[DEBUG] Created batch job: {batch_job.name}")
        
        return batch_job.name

    def retrieve_batch_results(self, handle: BatchHandle, timeout: Optional[int] = None) -> Dict[str, str]:
        # MOCK SHORT-CIRCUIT
        if self._is_mock_client():
            return {
                r["id"]: f"{self.provider} mock batch"
                for r in handle.metadata.get("requests", [])
            }

        # Set provider-specific default timeouts if not specified
        if timeout is None:
            timeout = {
                "anthropic": 1800,  # 30 minutes
                "openai": 1800,     # 30 minutes  
                "google": 900,     # 30 minutes
            }.get(handle.provider, 600)

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
            timeout=timeout,  # Use passed timeout instead of hardcoded
        )

        if batch.request_counts.errored > 0 or batch.request_counts.expired > 0:
            raise RuntimeError("Anthropic batch failed")

        # Download results with authentication header
        # Using latest API version for Claude 4.5 model support
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",  # Latest stable version
        }
        response = requests.get(batch.results_url, headers=headers)
        response.raise_for_status()
        raw = response.text

        results = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            
            # Anthropic batch results can have two formats:
            # 1. Success/Error wrapped in "result":
            #    {"custom_id": "...", "result": {"type": "succeeded", "message": {...}}}
            # 2. Direct error format (for batch-level or request-level errors):
            #    {"type": "error", "error": {"type": "...", "message": "..."}, "request_id": "..."}
            
            # Handle direct error format (no custom_id, no result wrapper)
            if "type" in obj and obj["type"] == "error" and "result" not in obj:
                error_msg = obj.get("error", {}).get("message", "Unknown error")
                request_id = obj.get("request_id", "unknown")
                # This is a request-level error - we need to map it to a custom_id
                # Since we don't have the custom_id, we'll use the request_id as key
                results[request_id] = f"[ERROR] {error_msg}"
                continue
            
            # Handle result-wrapped format
            if "result" not in obj:
                # If we get here, it's an unexpected format
                raise ValueError(f"Unexpected batch response format (no 'result' field): {obj}")
            
            result = obj["result"]
            custom_id = obj.get("custom_id", "unknown")
            
            if result["type"] == "succeeded":
                # Extract text from message content
                message = result.get("message", {})
                content = message.get("content", [])
                
                # Find the first text block
                text = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        break
                
                results[custom_id] = text
            elif result["type"] == "errored":
                # For errors, store error message
                error = result.get("error", {})
                error_msg = error.get("message", "Unknown error")
                results[custom_id] = f"[ERROR] {error_msg}"
            else:
                results[custom_id] = f"[UNKNOWN] Result type: {result.get('type')}"

        return results

    def retrieve_openai_batch_results(self, batch_id: str, timeout: int = 1800) -> Dict[str, str]:
        """
        Retrieve OpenAI batch results following official Batch API guidelines.
        
        Handles:
        - Status checking and polling
        - Both successful and failed individual requests
        - Error file retrieval if batch fails
        - Proper response structure parsing
        """
        # Check initial status
        initial_batch = self.client.batches.retrieve(batch_id)
        print(f"[DEBUG] OpenAI batch {batch_id} initial status: {initial_batch.status}")
        
        # If batch-level failure, retrieve error file if available
        if initial_batch.status == "failed":
            error_msg = f"Batch failed"
            if hasattr(initial_batch, 'errors') and initial_batch.errors:
                error_msg += f": {initial_batch.errors}"
            
            # Try to get error file for more details
            if hasattr(initial_batch, 'error_file_id') and initial_batch.error_file_id:
                try:
                    error_content = self.client.files.content(initial_batch.error_file_id)
                    # Need to read the content properly
                    raw_bytes = error_content.read()
                    print(f"[DEBUG] Error file content:\n{raw_bytes.decode()[:1000]}")
                    error_msg += f"\nSee error file: {initial_batch.error_file_id}"
                except Exception as e:
                    print(f"[DEBUG] Could not retrieve error file: {e}")
            
            raise RuntimeError(error_msg)
        
        # Track polling progress
        poll_count = 0
        last_status = initial_batch.status
        
        def check_and_log():
            nonlocal poll_count, last_status
            poll_count += 1
            batch = self.client.batches.retrieve(batch_id)
            
            # Log status changes
            if batch.status != last_status:
                print(f"[DEBUG] Batch status changed: {last_status} -> {batch.status}")
                last_status = batch.status
            
            # Log progress every 5 polls (~2.5 minutes with 30s interval)
            if poll_count % 5 == 0:
                request_counts = getattr(batch, 'request_counts', None)
                if request_counts:
                    print(f"[DEBUG] Poll #{poll_count}: status={batch.status}, "
                        f"completed={request_counts.completed}/{request_counts.total}")
                else:
                    print(f"[DEBUG] Poll #{poll_count}: status={batch.status}")
            
            return batch
        
        # Poll until batch reaches terminal state
        batch = self._poll_until(
            fn=check_and_log,
            is_done=lambda b: b.status in (
                "completed", "failed", "expired", "cancelled"
            ),
            interval=30,
            timeout=timeout,
        )

        # Handle terminal states
        if batch.status == "expired":
            raise RuntimeError("Batch expired (exceeded 24-hour window)")
        
        if batch.status == "cancelled":
            raise RuntimeError("Batch was cancelled")
        
        if batch.status == "failed":
            error_msg = "Batch failed"
            if hasattr(batch, 'errors') and batch.errors:
                error_msg += f": {batch.errors}"
            raise RuntimeError(error_msg)
        
        if batch.status != "completed":
            raise RuntimeError(f"Batch ended with unexpected status: {batch.status}")
        
        # DEBUG: Print full batch object to see what fields are available
        print(f"[DEBUG] output_file_id: {getattr(batch, 'output_file_id', 'NOT FOUND')}")
        print(f"[DEBUG] error_file_id: {getattr(batch, 'error_file_id', 'NOT FOUND')}")
        
        # Batch completed - check for output or error file
        output_file_id = getattr(batch, 'output_file_id', None)
        error_file_id = getattr(batch, 'error_file_id', None)
        
        # If no output file but completed, might need to wait a moment or refresh
        if not output_file_id and not error_file_id:
            print("[DEBUG] No output_file_id or error_file_id found. Refreshing batch status...")
            time.sleep(2)  # Brief wait
            batch = self.client.batches.retrieve(batch_id)
            output_file_id = getattr(batch, 'output_file_id', None)
            error_file_id = getattr(batch, 'error_file_id', None)
            print(f"[DEBUG] After refresh - output_file_id: {output_file_id}, error_file_id: {error_file_id}")
        
        # Check request counts to see if there were any successful requests
        request_counts = getattr(batch, 'request_counts', None)
        if request_counts:
            print(f"[DEBUG] Request counts: total={request_counts.total}, "
                f"completed={request_counts.completed}, failed={request_counts.failed}")
        
        # If all requests failed, results should be in error file
        if not output_file_id:
            if error_file_id:
                print(f"[INFO] No output file, but error file exists. All requests may have failed.")
                print(f"[INFO] Retrieving error file: {error_file_id}")
                
                try:
                    error_content = self.client.files.content(error_file_id)
                    # IMPORTANT: Need to read() the HttpxBinaryResponseContent
                    raw_bytes = error_content.read()
                    raw = raw_bytes.decode('utf-8')
                    
                    # DEBUG: Print first error line to see structure
                    first_line = raw.splitlines()[0] if raw.splitlines() else ""
                    if first_line:
                        print(f"[DEBUG] First error line structure:")
                        print(json.dumps(json.loads(first_line), indent=2))
                    
                    # Parse error file (same format as output file)
                    results = {}
                    for line in raw.splitlines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        custom_id = obj.get("custom_id", "unknown")
                        
                        # DEBUG: Print full object structure for first error
                        if len(results) == 0:
                            print(f"[DEBUG] Full error object structure:")
                            print(json.dumps(obj, indent=2))
                        
                        # The error file has the same structure as output file
                        # It contains response with error status codes or error field
                        if "error" in obj and obj["error"]:
                            error_info = obj["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            error_type = error_info.get("type", "unknown")
                            error_code = error_info.get("code", "unknown")
                            results[custom_id] = f"[ERROR] {error_type} ({error_code}): {error_msg}"
                        elif "response" in obj:
                            # Error might be in response body
                            response = obj["response"]
                            status_code = response.get("status_code", 0)
                            body = response.get("body", {})
                            
                            if status_code != 200:
                                # Get error from response body
                                error_info = body.get("error", {})
                                if isinstance(error_info, dict):
                                    error_msg = error_info.get("message", f"HTTP {status_code}")
                                    error_type = error_info.get("type", "http_error")
                                    results[custom_id] = f"[ERROR] {error_type}: {error_msg}"
                                else:
                                    results[custom_id] = f"[ERROR] HTTP {status_code}: {body}"
                            else:
                                results[custom_id] = "[ERROR] Request failed with unknown error"
                        else:
                            results[custom_id] = "[ERROR] Request failed with unknown error"
                    
                    print(f"[DEBUG] Parsed {len(results)} error results")
                    return results
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] Full traceback:")
                    traceback.print_exc()
                    raise RuntimeError(f"Batch completed but could not retrieve error file: {e}")
        
        print(f"[DEBUG] Retrieving output file: {output_file_id}")
        
        # Download output file content
        output_content = self.client.files.content(output_file_id)
        # IMPORTANT: Need to read() the HttpxBinaryResponseContent
        raw_bytes = output_content.read()
        raw = raw_bytes.decode('utf-8')
        
        # Parse results
        results = {}
        error_count = 0
        
        for line_num, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse line {line_num}: {e}")
                continue
            
            custom_id = obj.get("custom_id", f"unknown_line_{line_num}")
            
            # Check if this request had an error
            if "error" in obj and obj["error"] is not None:
                error_count += 1
                error_info = obj["error"]
                error_msg = error_info.get("message", "Unknown error")
                results[custom_id] = f"[ERROR] {error_msg}"
                print(f"[WARNING] Request {custom_id} failed: {error_msg}")
                continue
            
            # Parse successful response
            if "response" not in obj:
                print(f"[WARNING] Line {line_num} missing 'response' field")
                results[custom_id] = "[ERROR] Invalid response format"
                continue
            
            response = obj["response"]
            
            # Check HTTP status code
            status_code = response.get("status_code")
            if status_code != 200:
                error_count += 1
                results[custom_id] = f"[ERROR] HTTP {status_code}"
                print(f"[WARNING] Request {custom_id} returned status {status_code}")
                continue
            
            # Extract response body
            body = response.get("body", {})
            choices = body.get("choices", [])
            
            if not choices:
                results[custom_id] = "[ERROR] No choices in response"
                continue
            
            # Get the message content
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            results[custom_id] = content
        
        # Summary
        total_requests = len(results)
        successful_requests = total_requests - error_count
        print(f"[DEBUG] Results: {successful_requests}/{total_requests} successful")
        
        if error_count > 0:
            print(f"[WARNING] {error_count} requests had errors")
            
            # Check if there's an error file with additional details
            if error_file_id:
                print(f"[INFO] Additional error details in file: {error_file_id}")
        
        return results

    def retrieve_gemini_batch_results(self, batch_id: str, timeout: int) -> Dict[str, str]:
        """
        Poll Gemini batch job with aggressive network timeouts to prevent infinite hangs.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        
        start_time = time.time()
        job_name = batch_id
        last_log_time = 0
        
        print(f"\n[INFO] Starting retrieval for: {job_name}")
        print(f"[INFO] Timeout set to: {timeout}s (CTRL+C to interrupt anytime)")
        
        def api_call_with_timeout(method, *args, timeout_sec=30, **kwargs):
            """Execute API call in thread with hard timeout to prevent network hangs"""
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(method, *args, **kwargs)
                return future.result(timeout=timeout_sec)
        
        poll_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            # Hard timeout check
            if elapsed > timeout:
                raise TimeoutError(f"Hard timeout after {elapsed:.1f}s (limit: {timeout}s)")
            
            poll_count += 1
            
            # Log heartbeat every 30 seconds or every poll
            if elapsed - last_log_time > 30 or poll_count <= 3:
                print(f"[HEARTBEAT] Poll #{poll_count} | Elapsed: {elapsed:.1f}s | State: checking...")
                last_log_time = elapsed
            
            # Step 1: Get batch status with 30s timeout
            try:
                batch_job = api_call_with_timeout(
                    self.client.batches.get, 
                    name=job_name, 
                    timeout_sec=30
                )
            except FutureTimeout:
                print(f"[WARNING] batches.get() timed out after 30s (network stall?)")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"[ERROR] batches.get() failed: {e}")
                raise
            
            # Step 2: Check state
            state = getattr(batch_job.state, 'name', str(batch_job.state))
            
            if state != getattr(self, '_last_state', None):
                print(f"[STATE CHANGE] {getattr(self, '_last_state', None)} -> {state} (at {elapsed:.1f}s)")
                self._last_state = state
            
            if state == 'SUCCEEDED':
                print(f"[SUCCESS] Job completed in {elapsed:.1f}s")
                break
            elif state in ('FAILED', 'CANCELLED'):
                error_msg = getattr(batch_job, 'error', 'Unknown error')
                raise RuntimeError(f"Job failed: {state} | Error: {error_msg}")
            elif state in ('QUEUED', 'PENDING'):
                if poll_count == 1:
                    print(f"[INFO] Job is {state} - waiting in queue...")
            
            # Step 3: Sleep with interruptible check
            sleep_interval = 10
            time.sleep(sleep_interval)
        
        # Step 4: Download results with 60s timeout
        print(f"[INFO] Downloading results from: {batch_job.dest.file_name}")
        try:
            result_bytes = api_call_with_timeout(
                self.client.files.download,
                file=batch_job.dest.file_name,
                timeout_sec=60
            )
        except FutureTimeout:
            raise TimeoutError("Result download timed out after 60s")
        
        # Parse results...
        results = {}
        file_content = result_bytes.decode('utf-8')
        for line in file_content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key")
            response = item.get("response", {})
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
                results[key] = text
            else:
                results[key] = f"[ERROR] {response.get('error', 'No candidates')}"
        
        total_time = time.time() - start_time
        print(f"[DONE] Retrieved {len(results)} results in {total_time:.1f}s")
        return results

    def submit_gemini_parallel(self, requests: List[Dict]) -> BatchHandle:
        """
        For small batches, use parallel single-shot instead of unreliable Batch API.
        Completes in ~5-10 seconds vs 15+ minutes of queue hell.
        """
        if self.provider != "google":
            raise ValueError("Only for Google provider")
        
        print(f"\n🚀 PARALLEL MODE: Executing {len(requests)} prompts concurrently...")
        
        def run_one(r):
            """Execute single prompt with full error handling"""
            try:
                result = self.generate(
                    prompt=r["prompt"],
                    max_tokens=r.get("max_tokens", 1000),
                    temperature=r.get("temperature", 0.7)
                )
                return {"id": r["id"], "content": result["content"], "error": None}
            except Exception as e:
                return {"id": r["id"], "content": None, "error": str(e)}
        
        # Execute all requests in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [executor.submit(run_one, r) for r in requests]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["error"]:
                    results[res["id"]] = f"[ERROR] {res['error']}"
                else:
                    results[res["id"]] = res["content"]
        
        print(f"✅ Parallel execution complete: {len(results)} results")
        
        # Return compatible BatchHandle with results pre-loaded
        return BatchHandle(
            provider="google",
            id=f"parallel-{int(time.time())}",
            metadata={
                "results": results,
                "is_parallel": True,
                "completed_at": time.time()
            }
        )

    def retrieve_gemini_parallel_results(self, handle: BatchHandle) -> Dict[str, str]:
        """Return pre-computed parallel results immediately."""
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