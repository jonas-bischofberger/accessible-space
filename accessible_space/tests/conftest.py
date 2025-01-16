import pytest
import numpy as np


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    try:
        np.set_printoptions(legacy="1.21")  # Uniform numpy printing for doctests
    except UserWarning:
        pass
