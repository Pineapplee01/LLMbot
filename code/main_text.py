"""
main_text.py
Convenience entrypoint for text-only experiments.
"""

import sys

import main as main_module


def main():
    if "--text" not in sys.argv:
        sys.argv.append("--text")
    main_module.main()


if __name__ == "__main__":
    main()
