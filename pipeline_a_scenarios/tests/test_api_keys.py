import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def block_real_api_keys():
    """
    Tests must never rely on real API keys.
    If keys are present locally, they are ignored.
    """
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
