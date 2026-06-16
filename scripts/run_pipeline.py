"""Run the whole pipeline end-to-end: ingest -> features -> train -> viz -> report.

    uv run python scripts/run_pipeline.py --synthetic --quick   # fast smoke test
    uv run python scripts/run_pipeline.py --synthetic           # full synthetic
    uv run python scripts/run_pipeline.py --gee                 # live Earth Engine

By default the CNN stage is included if PyTorch is installed; use --skip-cnn to omit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from wildfire.utils import init_console

SCRIPTS = Path(__file__).resolve().parent


def _run(name: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS / name), *extra]
    print(f"\n{'='*70}\n▶ {name} {' '.join(extra)}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"Stage {name} failed (exit {res.returncode}).")
    print(f"  ({name} took {time.time()-t0:.1f}s)")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--gee", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    init_console()
    source = ["--gee"] if args.gee else ["--synthetic"]
    quick = ["--quick"] if args.quick else []

    _run("01_ingest.py", source + quick)
    _run("02_features.py", [])
    _run("04_train_tabular.py", [])

    do_cnn = not args.skip_cnn and _torch_available()
    if do_cnn:
        _run("03_patches.py", source + quick)
        _run("05_train_cnn.py", quick)
    else:
        print("\n(skipping CNN stage: --skip-cnn or PyTorch not installed)")

    _run("07_visualize.py", [])
    _run("06_evaluate.py", [])

    print("\n✅ Pipeline complete. See RESULTS.md and outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
