# Testing and Module Information

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
This document outlines the comprehensive unit test suite for the **Cost Tracker** module and **llm_client** module.

### 📊 Test Statistics
| Metric | Value |
| :--- | :--- |
| **Total Tests** | 20 |
| **Framework** | `pytest` |
| **Lines of Test Code** | 850+ |
| **Target Coverage** | 95%+ |

---

## 🚀 Quick Start

### Run All Tests
To execute the full suite and verify the production readiness of the module:

```bash
#Run all the tests
pytest tests/

# Run with standard verbose output
pytest tests/unit/test_cost_tracker.py 


pytest tests/unit/test_llm_client.py 
## 🛠️ Test Categories

10 Critical Infrastructure Tests Focused on Cost_Tracker:
1. JSONL append-only persistence 
2. Per-user file isolation 
3. Team aggregate dashboard 
4. Budget tracking configuration 
5. Data persistence across sessions
6. Reset functionality
7. Auto-log integration with LLM client
8. CSV export functionality
9. Singleton pattern
10. Error handling

---

10 Infrastructure Tests Focused on llm_client:
1. Client initialization with API keys
2. Single-shot generation (all 3 providers)
3. Batch API submission infrastructure
4. Batch result retrieval infrastructure  
5. Token counting and cost estimation
6. Rate limiting with token bucket
7. Response caching mechanism
8. Retry logic on failures
9. File handling for batch inputs
10. Error handling and validation
```
### Expected Output:

```
============================================== 20 passed, 1 warning in 1.58s =============================================== 
tests/unit/test_cost_tracker.py::test_team_aggregate_dashboard PASSED                                                 [ 15%] 
tests/unit/test_cost_tracker.py::test_budget_tracking_configuration PASSED                                            [ 20%] 
tests/unit/test_cost_tracker.py::test_data_persistence_across_sessions PASSED                                         [ 25%] 
tests/unit/test_cost_tracker.py::test_reset_functionality_clears_data PASSED                                          [ 30%] 
tests/unit/test_cost_tracker.py::test_auto_log_integration_with_llm_client PASSED                                     [ 35%] 
tests/unit/test_cost_tracker.py::test_csv_export_works PASSED                                                         [ 40%] 
tests/unit/test_cost_tracker.py::test_singleton_pattern_ensures_single_instance PASSED                                [ 45%] 
tests/unit/test_cost_tracker.py::test_error_handling_for_corrupted_data PASSED                                        [ 50%] 
tests/unit/test_llm_client.py::test_client_initialization_with_api_keys PASSED                                        [ 55%] 
tests/unit/test_llm_client.py::test_single_shot_generation_all_providers PASSED                                       [ 60%] 
tests/unit/test_llm_client.py::test_batch_api_submission_infrastructure PASSED                                        [ 65%] 
tests/unit/test_llm_client.py::test_batch_result_retrieval_infrastructure PASSED                                      [ 70%] 
tests/unit/test_llm_client.py::test_token_counting_and_cost_estimation PASSED                                         [ 75%] 
tests/unit/test_llm_client.py::test_rate_limiting_token_bucket PASSED                                                 [ 80%] 
tests/unit/test_llm_client.py::test_response_caching_mechanism PASSED                                                 [ 85%] 
tests/unit/test_llm_client.py::test_retry_logic_on_failures PASSED                                                    [ 90%] 
tests/unit/test_llm_client.py::test_file_handling_for_batch_inputs PASSED                                             [ 95%] 
tests/unit/test_llm_client.py::test_error_handling_and_validation PASSED                                              [100%] 
```

 
## Procedure after using cost tracker

Everyone should follow the gitworkflow after using Cost Tracker
```
git add data/metadata/costs_$(whoami).jsonl
git commit -m "Cost log"
git push

# For Team lead
git pull  # Get teammates' cost logs

```
