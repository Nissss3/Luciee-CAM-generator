"""Compatibility wrapper for the original training script filename."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("layer5_traning.py")), run_name="__main__")
