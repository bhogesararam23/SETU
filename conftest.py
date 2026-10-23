"""Makes the package importable from a checkout without installing it.

`pip install -e .` also works; this just means a developer can run `pytest` in a
clone with no venv ceremony.
"""

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SNAPSHOT = Path(__file__).resolve().parent / "data" / "fixtures" / "upstream"


@pytest.fixture(autouse=True)
def _not_offline(monkeypatch, tmp_path):
    """Clear SETU_OFFLINE for every test unless the test asks for it.

    The offline switch exists for offline environments with no network, where it stops
    the cache from going upstream at all. Tests never touch the real network - they
    inject a fake transport - so leaving a developer's export in the environment
    would make the suite fail on their machine and pass on CI. The two tests that
    do check offline behaviour set the variable themselves.

    The exception is the committed snapshot under data/fixtures/upstream: it is a
    read-only input, and the only way a test run can write to it is by going
    live against it. Entries past the cache TTL were being re-fetched and the
    snapshot silently rewritten with a window relative to today - a committed
    fixture carrying yesterday's `past_days=60` response is a fixture the
    reproducible run no longer replays. Pointing SETU_CACHE_DIR at the snapshot
    now pins SETU_OFFLINE=1, which is the documented way to use it.
    """
    monkeypatch.delenv("SETU_OFFLINE", raising=False)
    cache_dir = os.environ.get("SETU_CACHE_DIR", "").strip()
    if cache_dir and Path(cache_dir).resolve() == SNAPSHOT.resolve():
        monkeypatch.setenv("SETU_OFFLINE", "1")
    return tmp_path
