#  Testing

## Run all tests:
```bash
pytest tests/

pytets tests/unit/test_llm_client.py -v  

pytest tests/unit/test_cost_tracker.py -v
```
## Output expected

```


 pytest tests/                                                           
================================================================ test session starts =================================================================
platform win32 -- Python 3.10.19, pytest-7.4.4, pluggy-1.6.0
rootdir: 
configfile: pytest.ini
plugins: anyio-4.12.1, cov-7.0.0
collected 24 items                                                                                                                                    

tests\unit\test_cost_tracker.py ............                                                                                                    [ 50%]
tests\unit\test_llm_client.py ............                                                                                                      [100%]

======================================= 24 passed in 3.59s =================================================================

--------
pytest tests/unit/test_llm_client.py -v     
================================== test session starts =================================================================
platform win32 -- Python 3.10.19, pytest-7.4.4, pluggy-1.6.0 -- C:\Conda\envs\er-benchmark\python.exe
cachedir: .pytest_cache
rootdir:
configfile: pytest.ini
plugins: anyio-4.12.1, cov-7.0.0
collected 12 items                                                                                                                                    
tests/unit/test_llm_client.py::test_client_initialization PASSED                                                                                [  8%]
tests/unit/test_llm_client.py::test_generate_method PASSED                                                                                      [ 16%]
tests/unit/test_llm_client.py::test_retry_on_error PASSED                                                                                       [ 25%]
tests/unit/test_llm_client.py::test_caching_behavior PASSED                                                                                     [ 33%]
tests/unit/test_llm_client.py::test_token_counting PASSED                                                                                       [ 41%]
tests/unit/test_llm_client.py::test_token_bucket_rate_limiting PASSED                                                                           [ 50%]
tests/unit/test_llm_client.py::test_batch_operations PASSED                                                                                     [ 58%]
tests/unit/test_llm_client.py::test_cost_tracking_control PASSED                                                                                [ 66%]
tests/unit/test_llm_client.py::test_reasoning_modes PASSED                                                                                      [ 75%] 
tests/unit/test_llm_client.py::test_mock_client_detection PASSED                                                                                [ 83%] 
tests/unit/test_llm_client.py::test_cost_estimation PASSED                                                                                      [ 91%]
tests/unit/test_llm_client.py::test_anthropic_thinking_tokens_handling PASSED                                                                   [100%] 

=============================================12 passed in 3.58s =================================================================

pytest tests/unit/test_cost_tracker.py -v
==================================== test session starts =================================================================
platform win32 -- Python 3.10.19, pytest-7.4.4, pluggy-1.6.0 -- C:\Conda\envs\er-benchmark\python.exe
cachedir: .pytest_cache
rootdir: 
configfile: pytest.ini
plugins: anyio-4.12.1, cov-7.0.0
collected 12 items                                                                                                                                    

tests/unit/test_cost_tracker.py::test_tracker_initialization PASSED                                                                             [  8%]
tests/unit/test_cost_tracker.py::test_openai_pricing PASSED                                                                                     [ 16%]
tests/unit/test_cost_tracker.py::test_anthropic_pricing PASSED                                                                                  [ 25%]
tests/unit/test_cost_tracker.py::test_google_pricing PASSED                                                                                     [ 33%]
tests/unit/test_cost_tracker.py::test_batch_discount PASSED                                                                                     [ 41%]
tests/unit/test_cost_tracker.py::test_log_cost PASSED                                                                                           [ 50%]
tests/unit/test_cost_tracker.py::test_persistence PASSED                                                                                        [ 58%]
tests/unit/test_cost_tracker.py::test_budget_alerts PASSED                                                                                      [ 66%]
tests/unit/test_cost_tracker.py::test_month_1_2_threshold PASSED                                                                                [ 75%]
tests/unit/test_cost_tracker.py::test_provider_breakdown PASSED                                                                                 [ 83%]
tests/unit/test_cost_tracker.py::test_team_aggregation PASSED                                                                                   [ 91%]
tests/unit/test_cost_tracker.py::test_reset PASSED                                                                                              [100%] 

============================================12 passed in 0.16s ================================================================= 