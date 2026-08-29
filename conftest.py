"""
Pytest configuration.  This file lives at the repository root, and that is not an
accident -- moving it into tests/ breaks every `from src import ...` in the suite.

`tests/` deliberately has no `__init__.py`, so pytest imports the test modules in
`prepend` mode and inserts the directory containing the *nearest* conftest.py onto
sys.path.  With this file at the root, that directory is the repo root, and `import
src.config` resolves with no install step, no PYTHONPATH, and no per-file sys.path
hack.  The explicit insertion below is belt-and-braces for the case where pytest is
invoked as `python -m pytest` from some other directory, which puts the *caller's*
cwd on sys.path first.

Two markers gate the expensive work:

    slow   the solver validations of section 3.7 and anything that runs the FDTD at
           production resolution -- minutes each on CPU
    gpu    needs CUDA to finish in a sensible time

Both are skipped unless explicitly requested:

    pytest                         fast suite, seconds
    pytest --runslow               + the solver validations
    pytest --runslow --rungpu      everything, on a GPU box

Skipping by default is a considered trade, not laziness.  The document is emphatic
that the five solver checks must never be cut, and they are not cut -- notebook 01
runs them with --runslow as a hard gate before dataset generation.  But a suite that
takes twenty minutes to answer "did I break pack_inputs" stops being run at all, and
an unrun test is worth less than no test because it creates false confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  (must follow the sys.path fix-up)

from src import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Options and marker gating
# ---------------------------------------------------------------------------
def pytest_addoption(parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run the solver validations of section 3.7 (minutes on CPU)")
    parser.addoption("--rungpu", action="store_true", default=False,
                     help="run tests that need CUDA to finish in sensible time")


def pytest_collection_modifyitems(config, items) -> None:
    want_slow = config.getoption("--runslow")
    want_gpu = config.getoption("--rungpu")
    has_cuda = torch.cuda.is_available()

    skip_slow = pytest.mark.skip(reason="needs --runslow")
    skip_gpu = pytest.mark.skip(reason="needs --rungpu")
    no_cuda = pytest.mark.skip(reason="no CUDA device available")

    for item in items:
        if "slow" in item.keywords and not want_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords:
            if not has_cuda:
                item.add_marker(no_cuda)
            elif not want_gpu:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _determinism():
    """
    Seed every test and disable TF32.

    TF32 matters here for the same reason bootstrap.py disables it: it silently
    truncates float32 matmuls to ~10 mantissa bits, which is about three decimal
    digits -- exactly the tolerance the gradient check asserts at.  A gradient test
    that passes on CPU and fails on an A100 for no visible reason is the failure mode
    being pre-empted.  Tests that want float64 ask for it explicitly; this only
    stops the float32 ones from being quietly wrong.
    """
    torch.manual_seed(cfg.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


@pytest.fixture
def gen() -> torch.Generator:
    """A seeded CPU generator, for anything that samples."""
    g = torch.Generator()
    g.manual_seed(cfg.SEED)
    return g
