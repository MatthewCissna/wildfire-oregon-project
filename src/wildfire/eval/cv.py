"""Spatially & temporally aware cross-validation.

Random k-fold leaks in spatiotemporal data: a fire week in a cell is highly
predictable from an adjacent week or a neighboring cell that ended up in the
training fold, so random splits report optimistic scores that collapse in
deployment. This module provides the honest alternatives:

* ``forward_chaining`` — expanding-window time CV: train on years ≤ Y, test on
  year Y+1. Respects the arrow of time (no training on the future).
* ``spatial_block`` — GroupKFold on the spatial ``block_id``: whole regions are
  held out, so the model can't cheat via spatial autocorrelation.
* ``leave_one_block_out`` — the strictest spatial test: each block is the test set
  once, trained on all others.

Each generator yields ``(train_idx, test_idx, label)`` over a DataFrame that has
``block_id`` and a datetime ``date`` column.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _years(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df["date"]).dt.year.to_numpy()


def forward_chaining(df: pd.DataFrame, n_folds: int = 5, min_train_years: int = 1) -> Iterator:
    """Expanding-window temporal folds. Test on each later year in turn."""
    years = _years(df)
    uniq = np.sort(np.unique(years))
    if len(uniq) <= min_train_years:
        # Degenerate (e.g. quick mode with 1-2 years): single split on the last year.
        split = uniq[-1]
        tr = np.flatnonzero(years < split)
        te = np.flatnonzero(years == split)
        if len(tr) and len(te):
            yield tr, te, f"time<{split}->{split}"
        return

    test_years = uniq[min_train_years:]
    # Use the last n_folds test years (most informative, most data to train on).
    test_years = test_years[-n_folds:]
    for ty in test_years:
        tr = np.flatnonzero(years < ty)
        te = np.flatnonzero(years == ty)
        if len(tr) and len(te):
            yield tr, te, f"time<{ty}->{ty}"


def spatial_block(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> Iterator:
    """GroupKFold on ``block_id`` — hold out whole spatial blocks."""
    groups = df["block_id"].to_numpy()
    n_groups = len(np.unique(groups))
    k = min(n_folds, n_groups)
    if k < 2:
        return
    gkf = GroupKFold(n_splits=k)
    idx = np.arange(len(df))
    for i, (tr, te) in enumerate(gkf.split(idx, groups=groups)):
        yield tr, te, f"spatial_fold{i}"


def leave_one_block_out(df: pd.DataFrame, max_blocks: int | None = 20) -> Iterator:
    """Each spatial block is the test set once. Capped for runtime if many blocks."""
    blocks = pd.unique(df["block_id"])
    if max_blocks is not None and len(blocks) > max_blocks:
        rng = np.random.default_rng(0)
        blocks = rng.choice(blocks, size=max_blocks, replace=False)
    block_arr = df["block_id"].to_numpy()
    idx = np.arange(len(df))
    for b in blocks:
        te = idx[block_arr == b]
        tr = idx[block_arr != b]
        if len(te) and len(tr):
            yield tr, te, f"lobo_{b}"


SCHEMES = {
    "forward_chaining": forward_chaining,
    "spatial_block": spatial_block,
    "leave_one_block_out": leave_one_block_out,
}


def iter_folds(df: pd.DataFrame, scheme: str, **kwargs) -> Iterator:
    """Dispatch to a named CV scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown CV scheme {scheme!r}. Options: {list(SCHEMES)}")
    yield from SCHEMES[scheme](df, **kwargs)
