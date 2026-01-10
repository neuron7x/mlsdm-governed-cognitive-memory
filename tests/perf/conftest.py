"""Configuration and fixtures for performance tests."""

import os
import random

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_perf_tests():
    """Configure environment for deterministic performance tests."""
    os.environ["MLSDM_LOG_LEVEL"] = "ERROR"
    os.environ["DISABLE_RATE_LIMIT"] = "1"
    os.environ["LLM_BACKEND"] = "local_stub"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    yield

    for key in ["MLSDM_LOG_LEVEL", "DISABLE_RATE_LIMIT", "OTEL_SDK_DISABLED"]:
        os.environ.pop(key, None)


@pytest.fixture
def deterministic_seed():
    """Provide deterministic seed for reproducible tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    return seed
