"""Shared pytest configuration for the EEF1A Hetnet project."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The pipeline scripts are run with PYTHONPATH=src and import each other flatly
# (`from dwpc import ...`), so src/ must be importable for tests that touch them.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
