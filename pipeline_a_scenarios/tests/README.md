---

# 🧪 Running Tests

This test suite includes:

* **Unit & mock tests** (fast, default)
* **Integration tests** (real API calls, slow)

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

* Local development
* CI
* Fast feedback

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

* Call real external APIs
* May take several minutes
* Require valid API keys
* May incur costs

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

* `OPENAI_API_KEY`
* `ANTHROPIC_API_KEY`
* `GOOGLE_API_KEY`

---
