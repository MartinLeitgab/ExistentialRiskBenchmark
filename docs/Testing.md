# Testing 

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

<<<<<<< HEAD
=======
<<<<<<< HEAD
 
## Procedure after using cost tracker

Everyone should follow the gitworkflow after using Cost Tracker
```
git add data/metadata/costs_$(whoami).jsonl
git commit -m "Cost log"
git push

# For Team lead
git pull  # Get teammates' cost logs

```
=======
>>>>>>> 2244352 (feat: Add cost tracker and other updates)
>>>>>>> c2722bc
