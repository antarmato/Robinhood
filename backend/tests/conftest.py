"""
Test config — keep tests hermetic: file-backed state in a temp dir, no Postgres.
Must run before any backend module is imported (state.py reads env at import).
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.pop("DATABASE_URL", None)
os.environ.setdefault(
    "STATE_FILE", os.path.join(tempfile.mkdtemp(prefix="rh_test_"), "state.json")
)

# Make `backend` importable when pytest is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
