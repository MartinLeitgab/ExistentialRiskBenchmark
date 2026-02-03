"""
INTEGRATION TEST – SINGLE SHOT WITH / WITHOUT REASONING

This test validates that:
- Single-shot generation works with reasoning disabled
- Single-shot generation works with reasoning enabled
- Outputs and timing are saved to disk for comparison

REQUIRES:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY

Run with:
    pytest tests/test_single_shot_reasoning_outputs.py -s
"""

import json
import pathlib
import time
import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient


BASE_DIR = pathlib.Path(__file__).parent
RESULTS_DIR = BASE_DIR / "reasoning_results"
RESULTS_DIR.mkdir(exist_ok=True)

PROVIDERS = ["anthropic", "openai", "google"] #["google"] #

TEST_PROMPT = (
    "Explain in detail how to solve a 5x5 Sudoku puzzle using backtracking, "
    "including the recursive steps, pruning, and heuristics."
)


def results_path(provider: str) -> pathlib.Path:
    return RESULTS_DIR / f"{provider}_single_shot_reasoning.json"


def save_results(provider: str, payload: dict) -> None:
    path = results_path(provider)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Saved results → {path}")


@pytest.mark.integration
@pytest.mark.parametrize("provider", PROVIDERS)
def test_single_shot_with_and_without_reasoning(provider: str):
    """
    Run single-shot generation twice:
    - without reasoning
    - with high/extended reasoning

    Persist outputs, usage, and latency to disk.
    """
    client = UnifiedLLMClient(provider=provider)

    # ---- Without reasoning ----
    start = time.perf_counter()
    no_reasoning = client.generate(
        prompt=TEST_PROMPT,
        max_tokens=300,
        reasoning=None,
    )
    no_reasoning_latency = time.perf_counter() - start

    assert "content" in no_reasoning
    assert isinstance(no_reasoning["content"], str)
    assert len(no_reasoning["content"]) > 0

    # ---- With reasoning enabled ----
    start = time.perf_counter()
    with_reasoning = client.generate(
        prompt=TEST_PROMPT,
        max_tokens=300,
        reasoning="high",
    )
    with_reasoning_latency = time.perf_counter() - start

    assert "content" in with_reasoning
    assert isinstance(with_reasoning["content"], str)
    assert len(with_reasoning["content"]) > 0

    # Soft signal: warn if reasoning is unexpectedly faster
    if with_reasoning_latency < no_reasoning_latency:
        print(
            f"[WARN] {provider}: reasoning latency "
            f"({with_reasoning_latency:.2f}s) < "
            f"non-reasoning latency ({no_reasoning_latency:.2f}s)"
        )

    results = {
        "provider": provider,
        "prompt": TEST_PROMPT,
        "without_reasoning": {
            "content": no_reasoning["content"],
            "usage": no_reasoning.get("usage"),
            "latency_sec": round(no_reasoning_latency, 4),
        },
        "with_reasoning": {
            "content": with_reasoning["content"],
            "usage": with_reasoning.get("usage"),
            "latency_sec": round(with_reasoning_latency, 4),
        },
    }

    save_results(provider, results)

    print(
        f"\n[{provider.upper()} – WITHOUT REASONING | "
        f"{no_reasoning_latency:.2f}s]\n{no_reasoning['content']}\n"
    )
    print(
        f"[{provider.upper()} – WITH REASONING | "
        f"{with_reasoning_latency:.2f}s]\n{with_reasoning['content']}\n"
    )
