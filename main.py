"""Compatibility entrypoint for the flat `code/` active mainline."""

from pathlib import Path
import runpy
import sys


CODE_DIR = Path(__file__).resolve().parent / "code"

if __name__ == "__main__":
    sys.path.insert(0, str(CODE_DIR))
    runpy.run_path(str(CODE_DIR / "main.py"), run_name="__main__")
