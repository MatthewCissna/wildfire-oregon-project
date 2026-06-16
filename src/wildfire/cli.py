"""``wildfire`` command-line entry point.

A thin dispatcher to the numbered pipeline scripts so users can run, e.g.::

    wildfire pipeline --synthetic --quick
    wildfire ingest --synthetic
    wildfire check-ee

(Equivalent to ``uv run python scripts/<stage>.py``.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

_COMMANDS = {
    "check-ee": "00_check_earth_engine.py",
    "ingest": "01_ingest.py",
    "features": "02_features.py",
    "patches": "03_patches.py",
    "train": "04_train_tabular.py",
    "train-cnn": "05_train_cnn.py",
    "evaluate": "06_evaluate.py",
    "visualize": "07_visualize.py",
    "pipeline": "run_pipeline.py",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in _COMMANDS:
        print("Usage: wildfire <command> [args...]\n\nCommands:")
        for name, script in _COMMANDS.items():
            print(f"  {name:12s} -> scripts/{script}")
        return 0 if argv and argv[0] in ("-h", "--help") else 1
    script = _SCRIPTS / _COMMANDS[argv[0]]
    return subprocess.run([sys.executable, str(script), *argv[1:]]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
