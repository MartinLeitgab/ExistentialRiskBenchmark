# Existential Risk Preferences Benchmark

Benchmark for measuring AI model preferences relevant to existential risk.

## Quick Start
See [docs/SETUP.md](docs/SETUP.md) for installation.

## Project Structure
- `pipeline_a_scenarios/`: Scenario generation
- `pipeline_b_evaluation/`: Model evaluation
- `pipeline_c_analysis/`: Statistical analysis

## Testing and Module Information

### Module: utils.cost_tracker
**Author:** Pooja Puranik
**Created:** 2026-01-23
**Version:** 1.0.0
**Last Updated:** 2026-01-23

### 🎯 Purpose
Production-ready LLM API cost monitoring and budget management utility for tracking expenses across OpenAI, Anthropic, and Google providers with automated alerts and analytics.

## ✅ Core Features

### Multi-Provider Cost Calculation
* **OpenAI:** `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`
* **Anthropic:** `claude-opus-4.5`, `claude-sonnet-4.5`, `claude-haiku-4.5`
* **Google:** `gemini-3-pro`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`

### Budget Management
* **Monthly budget:** $200 (Month 1 requirement)
* **Total budget:** $1,000 (Project requirement)
* **Month 1-2 threshold:** $800 (Critical alert)

### Automated Alert System
* 80% total budget alert ($800 threshold)
* Month 1-2 threshold alert ($800)
* 80% monthly budget alert ($160)
* Email notifications for up to 4 team members

### Analytics & Reporting
* Monthly cost projections
* Provider breakdown analysis
* Model-level cost aggregation
* CSV export functionality
* Interactive CLI dashboard

### Data Persistence
* JSON-based cost storage (`data/metadata/costs.json`)
* Automatic loading of existing data
* Corruption recovery mechanisms

---

## 📖 Class Documentation: `CostTracker`

Main class for tracking LLM API costs, managing budgets, and generating alerts.

### Initialization Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_path` | `str` | `"data/metadata/costs.json"` | Path to cost data storage |
| `load_existing` | `bool` | `True` | Load existing cost data on startup |
| `monthly_budget_limit` | `float` | `200.0` | Monthly spending limit ($200 for Month 1) |
| `total_budget_limit` | `float` | `1000.0` | Total project budget |
| `alert_emails` | `Optional[List[str]]` | `None` | Email recipients for alerts (max 4) |
| `smtp_server` | `str` | `"smtp.gmail.com"` | SMTP server for email alerts |
| `smtp_port` | `int` | `587` | SMTP port |
| `smtp_username` | `Optional[str]` | `None` | SMTP authentication username |
| `smtp_password` | `Optional[str]` | `None` | SMTP authentication password |

### Public Methods

* **`calculate_cost()`**: Compute cost for API call based on provider pricing.
* **`log_cost()`**: Record API call with automatic alert checking. Saves to file and triggers alert checks.
* **`auto_log_from_llm_client()`**: Auto-extract costs from `UnifiedLLMClient` responses.
* **`get_total_cost()`**: Returns total spend as a float.
* **`get_monthly_cost()`**: Returns current month spend as a float.
* **`get_monthly_projection()`**: Project month-end spend using formula: $monthly\_cost \times (30 / current\_day)$.
* **`dashboard()`**: Display comprehensive formatted console display.
* **`export_csv()`**: Export cost data to a CSV file.
* **`reset()`**: Clear all cost data and reset alert tracking.

---

### 🚨 Alert System

#### Thresholds
1.  **80% Total Budget:** $800
2.  **Month 1-2 Threshold:** $800 (Critical)
3.  **80% Monthly Budget:** $160

### Setup
```python
from pipeline_a_scenarios.utils.cost_tracker import CostTracker

tracker = CostTracker(
    monthly_budget_limit=200.0,
    total_budget_limit=1000.0,
    alert_emails=["user@company.com"]
)

```

#### Email Configuration
```bash
#### Environment variables
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
```
### CLI Interface
```bash
# Display the basic cost summary dashboard
python utils/cost_tracker.py --dashboard

output :

# Display detailed dashboard with granular model-level breakdown
python utils/cost_tracker.py --dashboard --verbose

# Export all logged transaction data to a CSV file
python utils/cost_tracker.py --export costs.csv

# Reset all cost logs and clear alert history (requires confirmation)
python utils/cost_tracker.py --reset

# Run dashboard with custom budget overrides for the current session
python utils/cost_tracker.py --monthly-budget 500 --total-budget 2000 --dashboard
```

### Expected Output

✅ COST TRACKER INITIALIZED - PRODUCTION READY
   Data: data/metadata/costs.json
   Monthly Budget: $200.00 (Month 1)
   Total Budget: $1000.00
   80% Alert: $800.00
   Month 1-2: $800.00
   Email Alerts: 1/4 recipients

================================================================================
                          📊 COST TRACKER DASHBOARD
================================================================================
📈 SUMMARY
--------------------------------------------------------------------------------
Total API Calls:      2
Total Cost:           $0.0115
Monthly Cost:         $0.0115

💰 BUDGET STATUS
--------------------------------------------------------------------------------
Total Budget:         $1000.00
  Spent: $0.01 | Remaining: $999.99
  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.001%

📧 EMAIL ALERTS
--------------------------------------------------------------------------------
Recipients: team-lead@company.com

🏢 PROVIDER BREAKDOWN
--------------------------------------------------------------------------------
Provider        Calls     Cost       % of Total
--------------------------------------------------------------------------------
openai               1    $0.0025    21.7%
anthropic            1    $0.0090    78.3%


## Testing

There are different unit tests are define under tests/unit section.

### 1.test_cost_tracker.py
File to check unit tests.


# 🧪 Cost Tracker - Unit Test Manual

**Author:** Pooja Puranik
**Created:** 2026-01-23
**Version:** 1.0.0
**Last Updated:** 2026-01-23
**Test File:** `tests/unit/test_cost_tracker.py`
**License:** Apache 2.0

---

## 📖 Overview
This document outlines the comprehensive unit test suite for the **Cost Tracker** module. The suite ensures calculation accuracy, budget integrity, and reliability of the automated alert system across OpenAI, Anthropic, and Google providers.

### 📊 Test Statistics
| Metric | Value |
| :--- | :--- |
| **Total Tests** | 35 |
| **Test Categories** | 8 |
| **Framework** | `pytest` |
| **Lines of Test Code** | 850+ |
| **Target Coverage** | 95%+ |

---

## 🚀 Quick Start

### Run All Tests
To execute the full suite and verify the production readiness of the module:

```bash
# Run with standard verbose output
pytest tests/unit/test_cost_tracker.py -v

# Run with detailed (extra verbose) output
pytest tests/unit/test_cost_tracker.py -vv

# Run with coverage report
pytest tests/unit/test_cost_tracker.py --cov=pipeline_a_scenarios.utils.cost_tracker

## 🛠️ Test Categories

### 1. Cost Calculation Tests
Verifies that the pricing logic matches the 2026 rate cards for all models, including tiered pricing for long contexts.
* `test_calculate_cost_openai_gpt5`
* `test_calculate_cost_anthropic_claude_opus`
* `test_calculate_cost_google_gemini_pro`

### 2. Budget Management Tests
Ensures the tracker correctly identifies remaining funds and cumulative spending.
* `test_monthly_budget_tracking`
* `test_total_budget_persistence`

### 3. Automated Alert Tests
Mocks the SMTP server to verify that email notifications are triggered at 80% thresholds.
* `test_80_percent_threshold_trigger`
* `test_month_1_2_critical_alert`

### 4. Data Persistence & Recovery
Tests the JSON storage system, including handling of corrupted files or empty directories.
* `test_save_load_cycle`
* `test_corruption_recovery_mechanism`

### 5. Analytics & Projections
Validates the mathematical formulas used for month-end spending forecasts.
* `test_monthly_projection_accuracy`
* `test_provider_breakdown_percentages`

### 6. CLI Interface Mocking
Tests the command-line arguments to ensure `--dashboard` and `--reset` function correctly.
* `test_cli_dashboard_output`
* `test_cli_export_csv`

### 7. Integration: UnifiedLLMClient
Verifies that the `CostTracker` hooks correctly into the LLM client responses.
* `test_auto_log_from_response_usage`

### 8. Edge Case Tests
Tests for zero tokens, unsupported providers, and negative budget overrides.
* `test_invalid_provider_exception`
* `test_zero_token_calculation`

```


### Expected Output:

platform win32 -- Python 3.10.19, pytest-7.4.4, pluggy-1.6.0 -- C:\Conda\envs\er-benchmark\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\
configfile: pytest.ini
plugins: anyio-4.12.1, cov-7.0.0
collected 35 items
```bash
tests/unit/test_cost_tracker.py::test_tracker_initialization_defaults PASSED                                                                [  2%]
tests/unit/test_cost_tracker.py::test_tracker_initialization_custom_values PASSED                                                           [  5%]
tests/unit/test_cost_tracker.py::test_tracker_email_limit PASSED                                                                            [  8%]
tests/unit/test_cost_tracker.py::test_calculate_cost_anthropic PASSED                                                                       [ 11%]
tests/unit/test_cost_tracker.py::test_calculate_cost_openai PASSED                                                                          [ 14%]
tests/unit/test_cost_tracker.py::test_calculate_cost_google PASSED                                                                          [ 17%]
tests/unit/test_cost_tracker.py::test_calculate_cost_batch_discount PASSED                                                                  [ 20%]
tests/unit/test_cost_tracker.py::test_calculate_cost_unknown_provider PASSED                                                                [ 22%]
tests/unit/test_cost_tracker.py::test_calculate_cost_unknown_model_fallback PASSED                                                          [ 25%]
tests/unit/test_cost_tracker.py::test_log_cost_with_calculation PASSED                                                                      [ 28%]
tests/unit/test_cost_tracker.py::test_log_cost_with_manual_cost PASSED                                                                      [ 31%]
tests/unit/test_cost_tracker.py::test_log_cost_with_metadata PASSED                                                                         [ 34%]
tests/unit/test_cost_tracker.py::test_log_cost_persistence PASSED                                                                           [ 37%]
tests/unit/test_cost_tracker.py::test_get_total_cost_empty PASSED                                                                           [ 40%]
tests/unit/test_cost_tracker.py::test_get_total_cost_multiple_entries PASSED                                                                [ 42%]
tests/unit/test_cost_tracker.py::test_get_monthly_cost PASSED                                                                               [ 45%]
tests/unit/test_cost_tracker.py::test_check_80_percent_alert PASSED                                                                         [ 48%]
tests/unit/test_cost_tracker.py::test_check_month_1_2_threshold PASSED                                                                      [ 51%]
tests/unit/test_cost_tracker.py::test_check_monthly_80_percent_alert PASSED                                                                 [ 54%]
tests/unit/test_cost_tracker.py::test_get_monthly_projection_empty PASSED                                                                   [ 57%]
tests/unit/test_cost_tracker.py::test_get_monthly_projection PASSED                                                                         [ 60%]
tests/unit/test_cost_tracker.py::test_get_cost_breakdown_by_model PASSED                                                                    [ 62%]
tests/unit/test_cost_tracker.py::test_get_provider_breakdown PASSED                                                                         [ 65%]
tests/unit/test_cost_tracker.py::test_reset PASSED                                                                                          [ 68%]
tests/unit/test_cost_tracker.py::test_export_csv PASSED                                                                                     [ 71%]
tests/unit/test_cost_tracker.py::test_export_csv_empty PASSED                                                                               [ 74%]
tests/unit/test_cost_tracker.py::test_auto_log_from_llm_client PASSED                                                                       [ 77%]
tests/unit/test_cost_tracker.py::test_auto_log_from_llm_client_various_formats PASSED                                                       [ 80%]
tests/unit/test_cost_tracker.py::test_negative_tokens_error PASSED                                                                          [ 82%]
tests/unit/test_cost_tracker.py::test_zero_tokens PASSED                                                                                    [ 85%]
tests/unit/test_cost_tracker.py::test_large_token_counts PASSED                                                                             [ 88%]
tests/unit/test_cost_tracker.py::test_corrupted_data_file PASSED                                                                            [ 91%]
tests/unit/test_cost_tracker.py::test_dashboard_with_data PASSED                                                                            [ 97%]
tests/unit/test_cost_tracker.py::test_dashboard_verbose PASSED                                                                              [100%]

========================================================= 35 passed, 2 warnings in 0.35s =========================================================
```

## Usage
[Coming in Month 1]

## Citation
[Paper citation when published]

## License
Apache 2.0
