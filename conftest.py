"""Makes the package importable from a checkout without installing it.

`pip install -e .` also works; this just means a judge can run `pytest` in a
clone with no venv ceremony.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
