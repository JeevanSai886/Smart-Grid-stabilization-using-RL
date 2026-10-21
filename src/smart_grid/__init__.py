"""Delay-aware smart-grid research project."""

import os
from pathlib import Path

# Keep third-party caches inside the project, including restricted Windows hosts.
_cache_root = Path(__file__).resolve().parents[2] / "results" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))

__version__ = "1.0.0"
