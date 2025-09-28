import os
import pytest
os.environ.setdefault("OPENAI_API_KEY", "test_key")


@pytest.fixture()
def sample_notes():
    return "alpha beta gamma; memory loop; reinforcement schedule"
