"""Small cross-cutting utilities: console encoding, logging, timing, seeding."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager


def init_console(level: int = logging.INFO) -> None:
    """Make stdout/stderr UTF-8 (Windows defaults to cp1252 and chokes on emoji)
    and configure basic logging. Safe to call multiple times.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timed(label: str):
    """Context manager that logs how long a block took."""
    t0 = time.time()
    logging.getLogger("timing").info("%s ...", label)
    yield
    logging.getLogger("timing").info("%s done in %.1fs", label, time.time() - t0)
