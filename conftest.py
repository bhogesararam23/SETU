"""Makes the package importable from a checkout without installing it.

`pip install -e .` also works; this just means a judge can run `pytest` in a
clone with no venv ceremony.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _not_offline(monkeypatch, tmp_path):
    """Clear SETU_OFFLINE for every test unless the test asks for it.

    The offline switch exists for demo machines with no network, where it stops
    the cache from going upstream at all. Tests never touch the real network -
    they inject a fake transport - so leaving a developer's export in the
    environment would make the suite fail on their machine and pass on CI. The
    two tests that do check offline behaviour set the variable themselves.
    """
    monkeypatch.delenv("SETU_OFFLINE", raising=False)
    return tmp_path
