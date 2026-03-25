"""
main_text.py
Auxiliary convenience entrypoint for text-only experiments.

This wrapper is intentionally kept runnable, but `main.py` remains the unified
CLI entry for both the multimodal mainline and the text-only baseline surface.
"""

import sys

import main as main_module


def main():
    # Preserve the historical shortcut while making it explicit that the real
    # implementation still lives in `main.py`.
    if "--text" not in sys.argv:
        sys.argv.append("--text")
    main_module.main()


if __name__ == "__main__":
    main()
