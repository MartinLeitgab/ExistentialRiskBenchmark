# 📄 pipeline_a_scenarios

This folder contains the core implementation of the `pipeline_a` scenarios, which are designed to evaluate the performance and behavior of various Large Language Models (LLMs) in specific contexts related to existential risk. The scenarios are structured to test different aspects of LLM capabilities, including:

*   **Single-shot prompts:** Evaluating direct responses to individual prompts.
*   **Batch processing:** Assessing the efficiency and consistency of LLMs when processing multiple prompts in parallel.
*   **Polling mechanisms:** Testing the ability to handle asynchronous operations and retrieve results from LLM APIs.

The `pipeline_a_scenarios` folder includes:

*   **`llm_client.py`**: Contains the client implementations for interacting with different LLM providers (e.g., OpenAI, Anthropic, Google Gemini). This module abstracts the API calls, handles request/response processing, error handling, and manages API key authentication. It provides a unified interface for various LLMs, simplifying their integration into the scenarios.
    
    ### Key Methods in `llm_client.py`:
    
    *   `UnifiedLLMClient.__init__(...)`: Initializes the LLM client for a specified provider and model, handling API key retrieval and client setup. Supports caching and rate limiting.
    *   `generate(prompt, system_prompt, temperature, max_tokens, reasoning)`: Performs a single-shot generation request to the configured LLM. Includes logic for caching, retries, and applying reasoning modes.
    *   `submit_batch(requests, jsonl_path)`: Submits a list of prompts as a batch job to the LLM provider. Handles provider-specific batch submission mechanisms (e.g., Anthropic's direct batch API, OpenAI's file-based batch, Gemini's upload-and-create).
    *   `retrieve_batch_results(handle, timeout)`: Polls and retrieves the results of a previously submitted batch job. Manages provider-specific polling logic and result parsing.
    *   `submit_gemini_parallel(requests)`: A specialized method for Gemini to perform parallel single-shot requests, acting as a workaround for the Gemini Batch API's limitations in supporting thinking configurations.
    *   `retrieve_gemini_parallel_results(handle)`: Retrieves results from a Gemini parallel execution, which are pre-computed during submission.
    *   `estimate_cost(prompt, expected_output_tokens)`: Provides an estimated cost for a given prompt and expected output, based on configured pricing for each LLM.
*   **`utils/`**: A directory for utility functions and helper scripts used across the scenarios, such as batch job management and result parsing.
*   **`tests/`**: This subfolder (where this README resides) contains a comprehensive suite of unit, mock, and integration tests to ensure the correctness and reliability of the scenario implementations and LLM interactions. It includes dedicated test files for `llm_client.py`, batch processing, single-shot prompts, and polling mechanisms, covering various LLM providers.
    
    ### Key Tests in `test_llm_client.py`:
    
    *   `test_single_shot_generation[provider]`: Validates the `generate` method of `UnifiedLLMClient` for each provider (Anthropic, OpenAI, Google), ensuring that single-shot prompts return valid content.
    *   `test_batch_submission[provider]`: Tests the submission of batch requests to different LLM providers. For Gemini, it includes an optimization to use parallel single-shot calls for small batches to bypass potential queue delays.
    *   `test_batch_wait_and_retrieve[provider]`: Focuses on retrieving results from submitted batch jobs. It handles both parallel (instant retrieval) and regular (potentially slow, API-polled) batches and validates the integrity of the retrieved results.
    *   `test_cleanup_stuck_gemini_batch`: A utility test designed to cancel any stuck Gemini batch jobs from previous failed attempts, preparing the environment for new batch submissions.
    *   `test_check_gemini_queue`: Provides a detailed status report of the Gemini batch queue, showing active, queued, running, succeeded, and failed jobs to help diagnose queue saturation issues.
    *   `test_check_current_gemini_batch`: Specifically checks the status of the last submitted Gemini batch job (if a batch ID is saved), providing detailed information on its state and helping to debug retrieval hangs.

---

# 🧪 Running Tests

This test suite includes:

*   **Unit & mock tests** (fast, default)
*   **Integration tests** (real API calls, slow)

Integration tests are marked with:

```python
@pytest.mark.integration
```

---

## ✅ Default (without integration tests)

Runs **only unit and mock tests**:

```bash
pytest
```

This is the recommended command for:

*   Local development
*   CI
*   Fast feedback

---

## 🌐 With integration tests

Runs **only integration tests**:

```bash
pytest -m integration
```

# Run integration tests except batch
pytest -m "integration and not batch" -s

# Run batch tests only
pytest -m batch -s

# Run all tests including slow integration
pytest -m "integration or slow" -s


⚠️ These tests:

*   Call real external APIs
*   May take several minutes
*   Require valid API keys
*   May incur costs

---

## ▶ Run everything (unit + integration)

```bash
pytest -m ""
```

---

🧹 Cleanup stuck Gemini batch jobs

If Gemini batch tests are skipped due to queue saturation ("11 active jobs"), cancel all stuck jobs before submitting new ones:
bash

Copy
CLEANUP_GEMINI_JOBS=1 pytest -m "" "pipeline_a_scenarios/tests/test_llm_client.py::test_batch_submission[google]" -s
This command will:
List all active Gemini batch jobs
Cancel any jobs stuck in QUEUED/RUNNING state
Proceed with the new batch submission

---

Check Gemini Batch status and queue.

pytest -m "" pipeline_a_scenarios/tests/test_llm_client.py::test_check_gemini_queue -s
pytest -m "" pipeline_a_scenarios/tests/test_llm_client.py::test_check_current_gemini_batch -s

# 1. Cancel the stuck 15-minute batch
pytest -m "" pipeline_a_scenarios/tests/test_llm_client.py::test_cleanup_stuck_gemini_batch -s

# 2. Run with parallel mode (completes in ~10 seconds)
pytest -m "" "pipeline_a_scenarios/tests/test_llm_client.py::test_batch_submission[google]" -s

# 3. Retrieve (instant since results already saved)
pytest -m "" "pipeline_a_scenarios/tests/test_llm_client.py::test_batch_wait_and_retrieve[google]" -s

---

## 🔑 Required API keys (for integration tests)

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`

---

## 📝 Personal Notes / Implementation Details

During the development and testing of `pipeline_a_scenarios`, I resolved several issues across all three LLM clients (OpenAI, Anthropic, Google Gemini), including:

* Correct endpoint usage and parameter configurations for each provider.
* Ensuring that the `reasoning` or “thinking” configuration is applied consistently.

A key difference observed is in **Google Gemini** behavior compared to Anthropic and OpenAI:

* In Google AI Studio, it is **not possible to enable the thinking configuration when sending batch messages**.
* This limitation does not exist in Anthropic or OpenAI batch APIs, which allow reasoning modes to be applied directly.
* To overcome this, I implemented a **workaround using parallel single-shot calls** (`submit_gemini_parallel`) so that extended thinking can be applied even when batch submission is needed. This approach mimics batch processing while respecting Gemini’s API constraints.
