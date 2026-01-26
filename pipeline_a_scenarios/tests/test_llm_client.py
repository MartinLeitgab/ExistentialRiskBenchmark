"""
INTEGRATION TESTS – REAL PROVIDERS ONLY

This test file validates that:
- Anthropic, OpenAI, and Google Gemini clients can be instantiated with real APIs
- Single-shot generation works
- Batch processing works
- Batch requests can be loaded from disk
- Batch results are downloaded and persisted to JSON
- Example batch files are auto-created when missing

REQUIRES:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY

Run manually or with pytest:
    pytest tests/test_llm_real_providers.py -s
"""

import json
import os
import pathlib
from typing import Dict

import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient, BatchHandle



BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "batch_data"
RESULTS_DIR = BASE_DIR / "batch_results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_PROMPT = "Explain the difference between a list and a tuple in Python in one sentence."

PROVIDERS = ["anthropic", "openai", "google"] #["google"] #


def example_batch_requests() -> list[dict]:
    return [
        {"id": "req-1", "prompt": "What is Python? Answer in one sentence."},
        {"id": "req-2", "prompt": "What is an LLM? Answer in one sentence."},
    ]


def batch_file_path(provider: str) -> pathlib.Path:
    return DATA_DIR / f"{provider}_batch_requests.json"


def batch_results_path(provider: str) -> pathlib.Path:
    return RESULTS_DIR / f"{provider}_batch_results.json"


def ensure_batch_file(provider: str) -> pathlib.Path:
    """
    Ensure a batch request file exists.
    If not found, create it with example requests.
    """
    path = batch_file_path(provider)

    if not path.exists():
        print(f"[INFO] Creating example batch file for {provider}: {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(example_batch_requests(), f, indent=2)

    return path


def load_batch_requests(provider: str) -> list[dict]:
    path = ensure_batch_file(provider)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_batch_results(provider: str, results: Dict[str, str]) -> None:
    path = batch_results_path(provider)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved {provider} batch results → {path}")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_single_shot_generation(provider: str):
    """
    Validate single-shot generation with real providers.
    """
    client = UnifiedLLMClient(provider=provider)

    result = client.generate(
        prompt=TEST_PROMPT,
        max_tokens=100,
    )

    assert isinstance(result, dict)
    assert "content" in result
    assert isinstance(result["content"], str)
    assert len(result["content"]) > 0

    print(f"\n[{provider.upper()} SINGLE-SHOT RESULT]\n{result['content']}\n")

@pytest.mark.integration
def test_integration_marker():
    assert True

@pytest.mark.integration
@pytest.mark.batch
@pytest.mark.slow
@pytest.mark.parametrize("provider", PROVIDERS)
def test_batch_processing_from_file(provider: str):
    """
    Validate batch processing:
    - load batch requests from file
    - submit batch
    - poll for completion
    - download results
    - save results to JSON
    """
    client = UnifiedLLMClient(provider=provider)

    requests = load_batch_requests(provider)
    assert len(requests) > 0

    # Providers differ in batch mechanics
    jsonl_path = None
    if provider in {"anthropic", "openai", "google"}:
        jsonl_path = str(DATA_DIR / f"{provider}_batch_input.jsonl")

    handle: BatchHandle = client.submit_batch(
        requests=requests,
        jsonl_path=jsonl_path,
    )

    assert handle.id
    assert handle.provider == provider

    print(f"[INFO] Submitted {provider} batch: {handle.id}")

    results = client.retrieve_batch_results(handle)

    assert isinstance(results, dict)
    assert len(results) > 0

    for k, v in results.items():
        assert isinstance(v, str)
        assert len(v) > 0

    save_batch_results(provider, results)

    print(f"\n[{provider.upper()} BATCH RESULTS]")
    for k, v in results.items():
        print(f"- {k}: {v}")
