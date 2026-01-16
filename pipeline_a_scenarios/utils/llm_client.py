import os
import time
import json
import hashlib
import threading
import httpx
from typing import List, Dict, Optional

from dotenv import load_dotenv

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as AnthropicBatchRequest

import openai
import tiktoken

from google import genai
from google.genai import types



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
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-5.2",
        "google": "gemini-3-flash-preview",
    }

    PRICING = {
        "claude-sonnet-4-5": (3 / 1e6, 15 / 1e6),
        "gpt-5.2": (5 / 1e6, 15 / 1e6),
        "gpt-4o": (5 / 1e6, 15 / 1e6),
        "gemini-3-flash-preview": (2 / 1e6, 8 / 1e6),
    }

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        enable_cache: bool = True,
        rate_limit_per_sec: float = 5.0,
    ):
        load_dotenv()

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.enable_cache = enable_cache
        self.cache: Dict[str, dict] = {}
        self.bucket = TokenBucket(rate_limit_per_sec, rate_limit_per_sec)

        if provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        elif provider == "openai":
            self.client = openai.OpenAI(
                http_client=httpx.Client(trust_env=False)
            )

        elif provider == "google":
            self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

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
        if self.model.startswith("gpt-5"):
            return "max_completion_tokens"
        return "max_tokens"


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

        # Generate content
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

    import anthropic

    def submit_anthropic_batch(self, requests: List[Dict], poll_interval: float = 5.0) -> dict:
        batch = []
        for r in requests:
            batch.append(
                AnthropicBatchRequest(
                    custom_id=r["id"],
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=r.get("max_tokens", 1024),
                        messages=[{"role": "user", "content": r["prompt"]}],
                    ),
                )
            )

        # Submit batch
        batch_obj = self.client.messages.batches.create(requests=batch)
        print(f"Batch submitted: {batch_obj.id} (status: {batch_obj.processing_status})")

        # Poll until results are available
        while True:
            try:
                results_iter = self.client.messages.batches.results(batch_obj.id)
                break  # results are ready
            except anthropic.AnthropicError:
                print(f"Batch still processing... waiting {poll_interval}s")
                time.sleep(poll_interval)

        # Process results
        for result in results_iter:
            result_type = getattr(result.result, "type", None)
            if result_type == "succeeded":
                print(f"Success! {result.custom_id}")
            elif result_type == "errored":
                error_type = getattr(result.result, "error", None)
                if error_type and error_type.type == "invalid_request":
                    print(f"Validation error {result.custom_id}")
                else:
                    print(f"Server error {result.custom_id}")
            elif result_type == "expired":
                print(f"Request expired {result.custom_id}")

        return batch_obj

    def submit_openai_batch(self, requests: List[Dict], jsonl_path: str) -> str:
        import json

        # Write JSONL
        with open(jsonl_path, "w") as f:
            for r in requests:
                f.write(json.dumps({
                    "custom_id": r["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [{"role": "user", "content": r["prompt"]}],
                        "max_tokens": r.get("max_tokens", 512),
                    },
                }) + "\n")

        # Upload file
        file = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")

        # Create batch job
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"OpenAI batch submitted: {batch.id} (status: {getattr(batch, 'status', 'unknown')})")

        # No polling! Just return batch ID and instruct user to download results JSONL
        return batch.id


    def submit_gemini_batch(self, requests: List[Dict], jsonl_path: str, poll_interval: float = 5.0) -> str:
        """
        Submits a batch job to Google Gemini using JSONL, polls for processing,
        and logs per-request successes.
        """
        import json
        import time

        # Step 1: Write JSONL
        with open(jsonl_path, "w") as f:
            for r in requests:
                f.write(
                    json.dumps({
                        "key": r["id"],
                        "request": {
                            "contents": [{"parts": [{"text": r["prompt"]}]}]
                        }
                    }) + "\n"
                )

        # Step 2: Upload file
        uploaded_file = self.client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(display_name="gemini-batch", mime_type="application/jsonl")
        )
        print(f"Gemini batch file uploaded: {uploaded_file.name}")

        # Step 3: Poll until processing_status == "succeeded"
        while True:
            file_status = self.client.files.get(name=uploaded_file.name)  # <-- correct usage
            status = getattr(file_status, "processing_status", None)
            if status == "succeeded":
                print("Gemini batch processing completed!")
                break
            print(f"Batch still processing... waiting {poll_interval}s")
            time.sleep(poll_interval)

        # Step 4: Fetch results and log per-request successes
        results_file = self.client.files.get(name=uploaded_file.name)
        contents = getattr(results_file, "contents", [])
        for entry in contents:
            print(f"Success! {entry['key']}")

        return uploaded_file.name


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



if __name__ == "__main__":
    print("\n=== SINGLE-SHOT + RETRY + COST TEST ===")

    prompt = "Explain in one paragraph why the sky is blue."

    for provider in ["anthropic", "openai", "google"]:
        client = UnifiedLLMClient(provider)
        r1 = client.generate(prompt)
        r2 = client.generate(prompt)  # cache hit
        cost = client.estimate_cost(prompt)

        print(f"Provider: {provider}")
        print("Content:", r1["content"][:120])
        print("Usage:", r1["usage"])
        print("Estimated cost:", cost)
        print("Cache working:", r1["content"] == r2["content"])

    print("\n=== BATCH API TEST ===")

    batch_requests = [
        {"id": "req-1", "prompt": "Define photosynthesis."},
        {"id": "req-2", "prompt": "What is Newton's second law?"},
    ]

    anth = UnifiedLLMClient("anthropic")
    print("Anthropic batch:", anth.submit_anthropic_batch(batch_requests))

    oai = UnifiedLLMClient("openai")
    print("OpenAI batch id:", oai.submit_openai_batch(batch_requests, "openai_batch.jsonl"))

    gem = UnifiedLLMClient("google")
    print("Gemini batch file:", gem.submit_gemini_batch(batch_requests, "gemini_batch.jsonl"))

    print("\n=== LLM-AS-JUDGE TEST ===")
    judge_text = "Which answer is more concise: A or B?"
    print(UnifiedLLMClient("openai").judge_with_gpt4o(judge_text))
