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

import anthropic  # noqa: E402  (load_dotenv must run before SDK imports)
from anthropic.types.message_create_params import (  # noqa: E402
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import (  # noqa: E402
    Request as AnthropicBatchRequest,
)
import openai  # noqa: E402
import tiktoken  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

BatchProvider = Literal["anthropic", "openai", "google"]

# Models that reject an explicit `temperature` and run only at their provider
# default. Anthropic removed `temperature`/`top_p`/`top_k` permanently for the
# Opus 4.7+ family, and the Anthropic migration guide states the restriction is
# unchanged on Opus 5 ("Setting temperature, top_p, or top_k to a non-default
# value returns a 400 error on Claude Opus 5, the same as on Claude Opus 4.8" —
# platform.claude.com/docs/en/about-claude/models/migration-guide, checked
# 2026-07-27). OpenAI's gpt-5.5 and the gpt-5.6 family return
#   400 "Unsupported value: 'temperature' does not support 0 with this model.
#        Only the default (1) value is supported."
# `gpt-5.6` covers the bare alias and the -sol / -terra / -luna suffixes.
# Deliberately NOT generalised to every `gpt-5*` id: the 2026-04-21 Phase 1 run
# (`outputs/data_Riccardo042126/results/prompt_validation/raw_responses.json`,
# 192 gpt-5.4 rows, zero errors) shows gpt-5.4 accepted temperature=0. Add ids
# here only with evidence that the API rejects the parameter.
# Both hyphen and dot spellings are listed because
# "claude-opus-4.7".startswith("claude-opus-4-7") is False.
SAMPLING_RESTRICTED_MODEL_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4.7",
    "claude-opus-4-8",
    "claude-opus-4.8",
    "gpt-5.5",
    "gpt-5.6",
)

# The only sampling temperature the models above will run at.
SAMPLING_RESTRICTED_TEMPERATURE = 1.0

# Anthropic models that run adaptive thinking when the `thinking` field is
# OMITTED. On Opus 4.8 and earlier, omitting it means no thinking; on Opus 5 and
# Sonnet 5 it means thinking is on, and hidden thinking tokens are drawn from the
# same `max_tokens` budget as the visible answer.
#
# Thinking is left ON deliberately: production deployments run with it enabled,
# and the benchmark's premise is to mimic deployment settings (the same argument
# that fixes sampling temperature at 1.0). The consequence is a budget one, and
# it is why MAX_RESPONSE_TOKENS below is not 500.
#
# Measured 2026-07-27 on claude-opus-5 with an FTC prompt (free reasoning closing
# with an <answer> tag), thinking left at its default:
#   max_tokens=500  -> 471 thinking tokens, 105 visible chars, no <answer> tag
#   max_tokens=1500 -> 945 thinking tokens, truncated, no <answer> tag
#   max_tokens=3000 -> 597 thinking tokens, 2,270 visible chars, tag present,
#                      stop_reason=end_turn at 1,310 output tokens
# Adaptive thinking varies run to run (584-945 tokens across these samples), so
# the budget carries headroom rather than tracking the median.
#
# This predicate exists so callers can size budgets and so the reasoning branch
# can pick `{"type": "adaptive"}` over the removed `budget_tokens` form; nothing
# in the request path disables thinking.
ADAPTIVE_THINKING_ON_BY_DEFAULT_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
)


def adaptive_thinking_on_by_default(model: str) -> bool:
    """Report whether omitting `thinking` leaves adaptive thinking enabled.

    Args:
        model: Provider model id.

    Returns:
        True if hidden thinking tokens will be drawn from `max_tokens` even
        though the request never mentions thinking.
    """
    return str(model).startswith(ADAPTIVE_THINKING_ON_BY_DEFAULT_PREFIXES)


def rejects_sampling_params(model: str) -> bool:
    """Report whether a model rejects an explicit `temperature` parameter.

    Args:
        model: Provider model id (e.g. ``"claude-opus-4-8"``).

    Returns:
        True if the parameter must be omitted from the request payload.
    """
    return str(model).startswith(SAMPLING_RESTRICTED_MODEL_PREFIXES)


def assert_temperature_supported(model: str, temperature: Optional[float]) -> None:
    """Fail fast when a caller asks a restricted model for an unattainable temperature.

    Silently dropping the parameter would let a run record one temperature in its
    metadata while the API sampled at another — the provenance bug this guard
    exists to prevent (see `data/findings.md` §8a).

    Args:
        model: Provider model id the request will be sent to.
        temperature: Temperature the caller asked for; None means "unset".

    Raises:
        ValueError: If `model` cannot honour a non-default `temperature`.
    """
    if temperature is None or not rejects_sampling_params(model):
        return
    if float(temperature) == SAMPLING_RESTRICTED_TEMPERATURE:
        return
    raise ValueError(
        f"{model} does not accept an explicit temperature; it runs only at its "
        f"provider default ({SAMPLING_RESTRICTED_TEMPERATURE}), but "
        f"temperature={temperature} was requested. Either pass "
        f"temperature={SAMPLING_RESTRICTED_TEMPERATURE}, or use a model that "
        f"supports sampling params (e.g. gemini-3.1-pro-preview, "
        f"claude-sonnet-4-6). Restricted ids: "
        f"{', '.join(SAMPLING_RESTRICTED_MODEL_PREFIXES)}."
    )


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
        "anthropic": "claude-opus-5",
        "openai": "gpt-5.6-sol",
        "google": "gemini-3.1-pro-preview",
    }

    # Per-token USD for estimate_cost(); canonical full tables live in CostTracker.PRICING_SYNC.
    PRICING = {
        "claude-opus-5": (5 / 1e6, 25 / 1e6),
        "claude-opus-4-7": (5 / 1e6, 25 / 1e6),
        "claude-opus-4-8": (5 / 1e6, 25 / 1e6),
        "claude-sonnet-4-6": (3 / 1e6, 15 / 1e6),
        "gpt-5.6-sol": (5 / 1e6, 30 / 1e6),
        "gpt-5.5": (5 / 1e6, 30 / 1e6),
        "gpt-5.2": (1.75 / 1e6, 14 / 1e6),
        "gpt-4o": (2.5 / 1e6, 10 / 1e6),
        "gemini-3.1-pro-preview": (2 / 1e6, 12 / 1e6),
        "gemini-3-flash-preview": (0.5 / 1e6, 3 / 1e6),
    }

    def _is_mock_client(self) -> bool:
        try:
            from pipeline_a_scenarios.tests.test_mock_clients import (
                MockAnthropicClient,
                MockOpenAIClient,
                MockGeminiClient,
            )

            return isinstance(
                self.client, (MockAnthropicClient, MockOpenAIClient, MockGeminiClient)
            )
        except ImportError:
            return False

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        # Opt-in, not opt-out. The cache keys on (prompt, system_prompt, temperature,
        # max_tokens, reasoning) — the repeat index is NOT part of the key — so with
        # this defaulting to True every `runs_per_config` loop returned run 0's
        # response verbatim for every later run. Verified byte-identical across
        # 324/324 April cells, 342/342 June cells and 45/45 July cells; see
        # `data/findings.md` §8a, which retracts the reproducibility finding built
        # on it. Only enable this where the same prompt is genuinely expected to
        # recur and one answer is wanted for all of them.
        enable_cache: bool = False,
        rate_limit_per_sec: float = 5.0,
        client_override=None,
        # Opt-in auto-logging. When a CostTracker is passed, every non-cached
        # `generate()` logs its own usage — the caller does not have to remember.
        # Cost logging used to be entirely caller-side, so the pipeline scripts
        # (`prompt_validation`, `suppression_matrix`, `generate_scenarios`) logged
        # and every standalone script under `scripts/` did not: the §17a-§17c probe
        # series (360 responses) and every label-validation run spent real money
        # with no JSONL row, so the dashboard under-reported project spend by the
        # whole probe programme. Pass a tracker here rather than adding another
        # log_cost() call site. Callers that already log manually must NOT pass one
        # — that would double-count.
        cost_tracker=None,
    ):
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.enable_cache = enable_cache
        self.cache: Dict[str, dict] = {}
        self.bucket = TokenBucket(rate_limit_per_sec, rate_limit_per_sec)
        self.cost_tracker = cost_tracker
        # CostTracker appends to a shared JSONL and holds no lock of its own, while
        # `submit_gemini_parallel` and `scripts/probe_proto01_guards.py` call
        # `generate()` from a thread pool. Serialise the append here.
        self._cost_lock = threading.Lock()

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
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
        reasoning: Optional[Literal["none", "standard", "high"]] = None,
    ) -> dict:
        """Single-shot generation.

        Args:
            temperature: Sampling temperature. `None` (the default) omits the
                parameter entirely and lets the provider apply its own default.
                Previously this defaulted to 0.7, which meant a caller that never
                mentioned temperature still pinned one — and, now that
                DEFAULT_MODELS points at sampling-restricted models, would have
                raised. Callers that care about the value must pass it explicitly;
                the pipeline passes the variant's declared temperature.
        """
        assert_temperature_supported(self.model, temperature)

        cache_key = self._hash(
            prompt, system_prompt, temperature, max_tokens, reasoning
        )
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

                self._log_cost(result)

                if self.enable_cache:
                    self.cache[cache_key] = result
                return result

            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def _log_cost(self, result: dict) -> None:
        """Log one call's usage if a CostTracker was supplied.

        Deliberately not called on the cache-hit path above: a cache hit issues no
        provider call and must not be billed a second time.

        Fails loudly on a missing `usage` block rather than logging zeros — a silent
        zero is indistinguishable from a free call and would corrupt the budget
        dashboard in the direction that matters (under-reporting).
        """
        if self.cost_tracker is None:
            return

        usage = result.get("usage")
        if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
            raise KeyError(
                f"{self.provider} response carries no usage block, so its cost cannot "
                f"be logged: {sorted(result)}. Fix the provider adapter rather than "
                f"logging a zero."
            )

        with self._cost_lock:
            self.cost_tracker.log_cost(
                provider=self.provider,
                model=self.model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                call_type="sync",
                metadata={"auto_logged": True},
            )

    def _log_batch_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        n_requests: int,
        rows_missing_usage: int = 0,
    ) -> None:
        """Log one aggregated row for a completed batch retrieval.

        Billed at the batch tier (`call_type="batch"`), which is ~50% of sync list
        price — logging batch work as sync would overstate spend roughly 2x and
        distort the budget alerts.

        One row per retrieval rather than per request: the tokens are what the
        dashboard sums, and `n_requests` is carried in metadata so the row is not
        mistaken for a single call.

        Unlike `_log_cost`, a row without usage does not raise here. Batch error and
        expiry rows legitimately carry no usage, and raising after a retrieval has
        completed would discard results already paid for. The count is recorded in
        metadata and warned about instead, so the gap is visible rather than silent.
        """
        if self.cost_tracker is None or n_requests == 0:
            return

        if rows_missing_usage:
            print(
                f"   ⚠ {rows_missing_usage}/{n_requests} {self.provider} batch rows "
                f"carried no usage block; logged cost excludes them"
            )

        with self._cost_lock:
            self.cost_tracker.log_cost(
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_type="batch",
                metadata={
                    "auto_logged": True,
                    "batch": True,
                    "n_requests": n_requests,
                    "rows_missing_usage": rows_missing_usage,
                },
            )

    @staticmethod
    def _accumulate(totals: dict, input_tokens, output_tokens) -> None:
        """Add one row's usage to a running total, counting absent usage separately."""
        if input_tokens is None or output_tokens is None:
            totals["missing"] += 1
            return
        totals["input"] += int(input_tokens or 0)
        totals["output"] += int(output_tokens or 0)

    def _apply_reasoning(
        self, base_tokens: int, reasoning: Optional[str]
    ) -> tuple[int, int]:
        """Calculate budget and adjusted tokens for reasoning mode"""
        if reasoning == "high":
            budget = min(4096, max(1024, base_tokens * 3))
        elif reasoning == "standard":
            budget = min(2048, max(1024, base_tokens * 2))
        else:
            return 0, base_tokens
        return budget, max(base_tokens, budget + 200)

    def _generate_anthropic(
        self, prompt, system_prompt, temperature, max_tokens, reasoning
    ):
        params = {
            "model": self.model,
            "system": system_prompt or "You are a helpful assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }

        # Claude Opus 4.7+ family permanently removed `temperature`, `top_p`,
        # and `top_k`; any non-default value returns 400. Per Anthropic's
        # migration guide, the required path is to omit these parameters
        # entirely and rely on prompting (and output_config.effort) instead.
        # Membership (both hyphen and dot spellings) lives in
        # SAMPLING_RESTRICTED_MODEL_PREFIXES; generate() has already raised if a
        # non-default temperature was requested, so omitting here cannot silently
        # change the sampling the caller recorded.
        sampling_params_rejected = rejects_sampling_params(self.model)

        budget, adjusted_tokens = self._apply_reasoning(max_tokens, reasoning)
        if budget:
            # `{"type": "enabled", "budget_tokens": N}` is removed on the Opus
            # 4.7+ family and on Opus 5 — sending it returns a 400. Those models
            # take adaptive thinking instead, where depth is chosen by the model
            # rather than by a token budget.
            if rejects_sampling_params(self.model) or adaptive_thinking_on_by_default(
                self.model
            ):
                thinking_params = {
                    "thinking": {"type": "adaptive"},
                    "max_tokens": adjusted_tokens,
                }
            else:
                thinking_params = {
                    "thinking": {"type": "enabled", "budget_tokens": budget},
                    "max_tokens": adjusted_tokens,
                    "temperature": 1.0,
                }
            params.update(thinking_params)
        else:
            base_params = {"max_tokens": max_tokens}
            if not sampling_params_rejected and temperature is not None:
                base_params["temperature"] = temperature
            params.update(base_params)

        r = self.client.messages.create(**params)

        # Join every text block, in order. Taking only the first one silently
        # truncated any response the model split across blocks — which adaptive
        # thinking makes routine on Opus 5, since `content` comes back as an
        # interleaved [thinking, text, thinking, text, ...] sequence. The symptom
        # was a response that stopped mid-sentence and never showed its closing
        # <answer> tag, indistinguishable from hitting `max_tokens`: the
        # proto_05_v4 probe lost 8/10 Opus draws that way at a 3,000-token cap and
        # still lost them at 8,000, while `usage.output_tokens` reported only
        # 1,014-1,596 — the tell that the cap was never the binding constraint.
        text_blocks = [
            block.text for block in r.content if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            text_blocks = [block.text for block in r.content if hasattr(block, "text")]
        content_text = "".join(text_blocks)

        return {
            "content": content_text,
            # Why the response ended. Without it, a text that stops mid-sentence is
            # unattributable: `max_tokens` truncation, a refusal, a tool pause and a
            # dropped block all look identical downstream, and diagnosing the
            # proto_05_v4 Opus rows cost four probe runs for want of this one field.
            "stop_reason": getattr(r, "stop_reason", None),
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
            },
        }

    def _openai_max_token_param(self) -> str:
        return (
            "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"
        )

    def _generate_openai(
        self, prompt, system_prompt, temperature, max_tokens, reasoning
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        is_reasoning = self.model.startswith(("o1", "o3", "gpt-5"))
        adjusted_tokens = max_tokens * 10 if is_reasoning else max_tokens

        params = {
            "model": self.model,
            "messages": messages,
            self._openai_max_token_param(): adjusted_tokens,
        }

        # gpt-5.5 and the gpt-5.6 family reject an explicit temperature (400
        # "Only the default (1) value is supported"). generate() has already
        # raised for any non-default request, so omitting the parameter here
        # preserves the recorded value. `temperature=None` means "provider
        # default" and is likewise omitted.
        if not rejects_sampling_params(self.model) and temperature is not None:
            params["temperature"] = temperature

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

    def _generate_google(
        self, prompt, system_prompt, temperature, max_tokens, reasoning
    ):
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
        if hasattr(r, "candidates") and r.candidates:
            text_parts = [
                part.text
                for part in r.candidates[0].content.parts
                if hasattr(part, "text")
            ]

        full_text = (
            "".join(text_parts)
            if text_parts
            else (r.text if hasattr(r, "text") else "")
        )

        if hasattr(r, "usage_metadata"):
            input_tokens = r.usage_metadata.prompt_token_count
            output_tokens = r.usage_metadata.candidates_token_count
        else:
            tokens = self.count_tokens(full, full_text)
            input_tokens = tokens["input_tokens"]
            output_tokens = tokens["output_tokens"]

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

    def submit_batch(
        self, requests: List[Dict], jsonl_path: Optional[str] = None
    ) -> BatchHandle:
        if self._is_mock_client():
            return BatchHandle(
                provider=self.provider,
                id="mock-batch-id",
                metadata={"requests": requests},
            )

        if self.provider == "anthropic":
            return BatchHandle(
                provider="anthropic", id=self.submit_anthropic_batch(requests)
            )

        if not jsonl_path:
            raise ValueError(f"jsonl_path is required for {self.provider} batch")

        if self.provider == "openai":
            return BatchHandle(
                provider="openai", id=self.submit_openai_batch(requests, jsonl_path)
            )

        if self.provider == "google":
            return BatchHandle(
                provider="google", id=self.submit_gemini_batch(requests, jsonl_path)
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def submit_anthropic_batch(self, requests: List[Dict]) -> str:
        batch_reqs = []

        for r in requests:
            max_tokens = r.get("max_tokens", 2048)
            temperature = r.get("temperature")
            reasoning = r.get("reasoning")

            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": r["prompt"]}],
                "system": r.get("system_prompt", "You are a helpful assistant."),
            }

            assert_temperature_supported(self.model, temperature)
            sampling_params_rejected = rejects_sampling_params(self.model)

            budget, adjusted_tokens = self._apply_reasoning(max_tokens, reasoning)
            if budget:
                thinking_params = {
                    "thinking": {"type": "enabled", "budget_tokens": budget},
                    "max_tokens": adjusted_tokens,
                }
                if not sampling_params_rejected:
                    thinking_params["temperature"] = 1.0
                params.update(thinking_params)
            else:
                base_params = {"max_tokens": max_tokens}
                if not sampling_params_rejected and temperature is not None:
                    base_params["temperature"] = temperature
                params.update(base_params)

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
                    body["messages"].insert(
                        0, {"role": "system", "content": r["system_prompt"]}
                    )
                if "temperature" in r:
                    assert_temperature_supported(self.model, r["temperature"])
                    if not rejects_sampling_params(self.model):
                        body["temperature"] = r["temperature"]

                reasoning = r.get("reasoning")
                if reasoning in ("standard", "high") and not is_reasoning:
                    body["reasoning_effort"] = (
                        "high" if reasoning == "high" else "medium"
                    )

                f.write(
                    json.dumps(
                        {
                            "custom_id": r["id"],
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body,
                        }
                    )
                    + "\n"
                )

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
                f.write(
                    json.dumps(
                        {
                            "key": str(r["id"]),
                            "request": {
                                "contents": [{"parts": [{"text": r["prompt"]}]}],
                                "generationConfig": {
                                    "maxOutputTokens": r.get("max_tokens", 1000),
                                    "temperature": r.get("temperature", 0.7),
                                    "topP": r.get("top_p", 0.95),
                                },
                            },
                        }
                    )
                    + "\n"
                )

        uploaded_file = self.client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=f"batch-input-{int(time.time())}",
                mime_type="application/jsonl",
            ),
        )

        for _ in range(30):
            file_status = self.client.files.get(name=uploaded_file.name)
            state = getattr(file_status.state, "name", str(file_status.state))

            if state == "ACTIVE":
                break
            elif state in ["FAILED", "STATE_UNSPECIFIED", "FILE_STATE_FAILED"]:
                raise RuntimeError(f"File upload failed with state: {state}")
            time.sleep(2)
        else:
            raise TimeoutError(f"File {uploaded_file.name} did not become ACTIVE")

        batch_job = self.client.batches.create(
            model=self.model,
            src=uploaded_file.name,
            config=types.CreateBatchJobConfig(
                display_name=f"batch-job-{int(time.time())}"
            ),
        )
        return batch_job.name

    def retrieve_batch_results(
        self, handle: BatchHandle, timeout: Optional[int] = None
    ) -> Dict[str, str]:
        if self._is_mock_client():
            return {
                r["id"]: f"{self.provider} mock batch"
                for r in handle.metadata.get("requests", [])
            }

        timeout = timeout or {"anthropic": 1800, "openai": 1800, "google": 900}.get(
            handle.provider, 600
        )

        if handle.provider == "anthropic":
            return self.retrieve_anthropic_batch_results(handle.id, timeout)
        if handle.provider == "openai":
            return self.retrieve_openai_batch_results(handle.id, timeout)
        if handle.provider == "google":
            return self.retrieve_gemini_batch_results(handle.id, timeout)

        raise ValueError(f"Unsupported provider: {handle.provider}")

    def retrieve_anthropic_batch_results(
        self, batch_id: str, timeout: int = 1800
    ) -> Dict[str, str]:
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
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        response.raise_for_status()

        results = {}
        totals = {"input": 0, "output": 0, "missing": 0}
        for line in response.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)

            if "type" in obj and obj["type"] == "error" and "result" not in obj:
                results[
                    obj.get("request_id", "unknown")
                ] = f"[ERROR] {obj.get('error', {}).get('message', 'Unknown error')}"
                continue

            if "result" not in obj:
                raise ValueError(f"Unexpected batch response format: {obj}")

            result = obj["result"]
            custom_id = obj.get("custom_id", "unknown")

            if result["type"] == "succeeded":
                message = result.get("message", {})
                text = next(
                    (
                        block.get("text", "")
                        for block in message.get("content", [])
                        if isinstance(block, dict) and block.get("type") == "text"
                    ),
                    "",
                )
                results[custom_id] = text
                usage = message.get("usage") or {}
                self._accumulate(
                    totals, usage.get("input_tokens"), usage.get("output_tokens")
                )
            elif result["type"] == "errored":
                results[
                    custom_id
                ] = f"[ERROR] {result.get('error', {}).get('message', 'Unknown error')}"
            else:
                results[custom_id] = f"[UNKNOWN] Result type: {result.get('type')}"

        self._log_batch_cost(
            totals["input"], totals["output"], len(results), totals["missing"]
        )
        return results

    def retrieve_batch_results_with_usage(
        self, handle: BatchHandle, timeout: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Anthropic-only variant of retrieve_batch_results that also returns
        per-request token usage. Result shape:
        {custom_id: {"text": str, "input_tokens": int, "output_tokens": int}}.

        Error/unknown rows return zero token counts so callers can safely sum.
        Non-Anthropic providers raise ValueError until added on demand.

        Deliberately does NOT auto-log to `cost_tracker`, unlike the three
        `retrieve_*_batch_results` methods: handing usage back so the caller can bill
        it is this method's entire purpose, so logging here as well would double-count
        exactly the callers that asked for the usage.
        """
        if handle.provider != "anthropic":
            raise ValueError(
                "retrieve_batch_results_with_usage is Anthropic-only; "
                f"got provider={handle.provider}"
            )

        if self._is_mock_client():
            return {
                r["id"]: {
                    "text": f"{self.provider} mock batch",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                for r in handle.metadata.get("requests", [])
            }

        timeout = timeout or 1800
        return self.retrieve_anthropic_batch_results_with_usage(handle.id, timeout)

    def retrieve_anthropic_batch_results_with_usage(
        self, batch_id: str, timeout: int = 1800
    ) -> Dict[str, Dict[str, Any]]:
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
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        response.raise_for_status()

        results: Dict[str, Dict[str, Any]] = {}
        for line in response.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)

            if "type" in obj and obj["type"] == "error" and "result" not in obj:
                results[obj.get("request_id", "unknown")] = {
                    "text": f"[ERROR] {obj.get('error', {}).get('message', 'Unknown error')}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                continue

            if "result" not in obj:
                raise ValueError(f"Unexpected batch response format: {obj}")

            result = obj["result"]
            custom_id = obj.get("custom_id", "unknown")

            if result["type"] == "succeeded":
                message = result.get("message", {}) or {}
                text = next(
                    (
                        block.get("text", "")
                        for block in message.get("content", [])
                        if isinstance(block, dict) and block.get("type") == "text"
                    ),
                    "",
                )
                usage = message.get("usage", {}) or {}
                results[custom_id] = {
                    "text": text,
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                }
            elif result["type"] == "errored":
                results[custom_id] = {
                    "text": f"[ERROR] {result.get('error', {}).get('message', 'Unknown error')}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            else:
                results[custom_id] = {
                    "text": f"[UNKNOWN] Result type: {result.get('type')}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

        return results

    def retrieve_openai_batch_results(
        self, batch_id: str, timeout: int = 1800
    ) -> Dict[str, str]:
        def check_batch():
            return self.client.batches.retrieve(batch_id)

        batch = self._poll_until(
            fn=check_batch,
            is_done=lambda b: b.status
            in ("completed", "failed", "expired", "cancelled"),
            interval=30,
            timeout=timeout,
        )

        if batch.status in ("expired", "cancelled", "failed"):
            raise RuntimeError(f"Batch {batch.status}")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            time.sleep(2)
            batch = self.client.batches.retrieve(batch_id)
            output_file_id = getattr(batch, "output_file_id", None)

        if not output_file_id:
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                error_content = self.client.files.content(error_file_id)
                raw = error_content.read().decode("utf-8")
                results = {}
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    custom_id = obj.get("custom_id", "unknown")

                    if "error" in obj and obj["error"]:
                        error_info = obj["error"]
                        err_type = error_info.get("type", "unknown")
                        err_msg = error_info.get("message", "Unknown error")
                        results[custom_id] = f"[ERROR] {err_type}: {err_msg}"
                    else:
                        results[custom_id] = "[ERROR] Request failed"
                return results
            raise RuntimeError("No output or error file available")

        output_content = self.client.files.content(output_file_id)
        raw = output_content.read().decode("utf-8")

        results = {}
        totals = {"input": 0, "output": 0, "missing": 0}
        for line in raw.splitlines():
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = obj.get("custom_id", "unknown")

            if "error" in obj and obj["error"]:
                results[
                    custom_id
                ] = f"[ERROR] {obj['error'].get('message', 'Unknown error')}"
                continue

            response = obj.get("response", {})
            if response.get("status_code") != 200:
                results[custom_id] = f"[ERROR] HTTP {response.get('status_code')}"
                continue

            body = response.get("body", {})
            choices = body.get("choices", [])
            results[custom_id] = (
                choices[0].get("message", {}).get("content", "")
                if choices
                else "[ERROR] No choices"
            )
            usage = body.get("usage") or {}
            self._accumulate(
                totals, usage.get("prompt_tokens"), usage.get("completion_tokens")
            )

        self._log_batch_cost(
            totals["input"], totals["output"], len(results), totals["missing"]
        )
        return results

    def retrieve_gemini_batch_results(
        self, batch_id: str, timeout: int
    ) -> Dict[str, str]:
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
                batch_job = api_call_with_timeout(
                    self.client.batches.get, name=batch_id, timeout_sec=30
                )
            except FutureTimeout:
                time.sleep(5)
                continue

            state = getattr(batch_job.state, "name", str(batch_job.state))

            if state == "SUCCEEDED":
                break
            elif state in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"Job {state}: {getattr(batch_job, 'error', 'Unknown error')}"
                )

            time.sleep(10)

        try:
            result_bytes = api_call_with_timeout(
                self.client.files.download,
                file=batch_job.dest.file_name,
                timeout_sec=60,
            )
        except FutureTimeout:
            raise TimeoutError("Result download timed out")

        results = {}
        totals = {"input": 0, "output": 0, "missing": 0}
        for line in result_bytes.decode("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key")
            item_response = item.get("response", {})
            candidates = item_response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                results[key] = "".join(p.get("text", "") for p in parts)
                # Gemini's REST payload is camelCase; the SDK object form is snake.
                meta = (
                    item_response.get("usageMetadata")
                    or item_response.get("usage_metadata")
                    or {}
                )
                self._accumulate(
                    totals,
                    meta.get("promptTokenCount", meta.get("prompt_token_count")),
                    meta.get(
                        "candidatesTokenCount", meta.get("candidates_token_count")
                    ),
                )
            else:
                results[key] = f"[ERROR] {item_response.get('error', 'No candidates')}"

        self._log_batch_cost(
            totals["input"], totals["output"], len(results), totals["missing"]
        )
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
                    temperature=r.get("temperature"),
                    reasoning=r.get("reasoning"),
                )
                return {"id": r["id"], "content": result["content"], "error": None}
            except Exception as e:
                return {"id": r["id"], "content": None, "error": str(e)}

        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(requests)
        ) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(run_one, r) for r in requests]
            ):
                res = future.result()
                results[res["id"]] = (
                    f"[ERROR] {res['error']}" if res["error"] else res["content"]
                )

        return BatchHandle(
            provider="google",
            id=f"parallel-{int(time.time())}",
            metadata={
                "results": results,
                "is_parallel": True,
                "completed_at": time.time(),
            },
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

    def estimate_cost(
        self, prompt: str, expected_output_tokens: int = 500
    ) -> Optional[float]:
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
