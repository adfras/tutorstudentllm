import os
import pytest


# Ensure mock LLM is used across tests and key checks pass
os.environ.setdefault("TUTOR_MOCK_LLM", "1")
os.environ.setdefault("OPENAI_API_KEY", "test_key")


@pytest.fixture()
def sample_notes():
    return "alpha beta gamma; memory loop; reinforcement schedule"
