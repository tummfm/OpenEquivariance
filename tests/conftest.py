import os
import pytest

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def pytest_addoption(parser):
    parser.addoption(
        "--jax",
        action="store_true",
        default=False,
        help="Test the JAX frontend instead of PyTorch",
    )


@pytest.fixture(scope="session")
def with_jax(request):
    return request.config.getoption("--jax")
