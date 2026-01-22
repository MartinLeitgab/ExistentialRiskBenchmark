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

## 🔑 Required API keys (for integration tests)

* `OPENAI_API_KEY`
* `ANTHROPIC_API_KEY`
* `GOOGLE_API_KEY`

---
