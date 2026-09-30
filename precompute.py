"""Compatibility entrypoint for `code/precompute.py`."""

from pathlib import Path
import runpy
import sys


CODE_DIR = Path(__file__).resolve().parent / "code"

if __name__ == "__main__":
    sys.path.insert(0, str(CODE_DIR))
    runpy.run_path(str(CODE_DIR / "precompute.py"), run_name="__main__")
