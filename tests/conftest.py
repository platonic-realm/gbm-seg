import os
import sys
from pathlib import Path

import pytest

# Make `src` importable when pytest is invoked from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def cpu_device():
    return "cpu"


@pytest.fixture
def tmp_snapshot_dir(tmp_path):
    d = tmp_path / "snapshots"
    d.mkdir()
    return str(d) + os.sep
