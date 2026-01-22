import time
import pytest
from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient

#pytest.skip(
#    "Skipping this test file temporarily",
#    allow_module_level=True,
#)

def test_poll_until_waits_until_done():
    client = UnifiedLLMClient("anthropic", client_override=object())

    state = {"count": 0}

    def fn():
        state["count"] += 1
        return state["count"]

    def is_done(x):
        return x >= 3

    start = time.time()
    result = client._poll_until(fn, is_done, interval=0.01)
    elapsed = time.time() - start

    assert result == 3
    assert elapsed >= 0.02
