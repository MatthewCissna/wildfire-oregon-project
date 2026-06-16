"""Class-imbalance handling for the rare-event risk model.

Fires occur in only ~1-4% of cell-weeks. Three documented strategies:

* ``undersample`` — drop a fraction of the (huge) negative class so the model and
  CV run fast and the decision boundary isn't swamped. We keep ``neg_per_pos``
  negatives for each positive. **Probabilities are then recalibrated** back to the
  true base rate (see :func:`prior_correction`) so the risk surface stays honest.
* ``class_weight`` — keep all rows, weight positives up (passed to the estimator).
* ``none`` — keep everything (slow but unbiased).

We never oversample the *test* folds — evaluation always uses the true class ratio,
otherwise metrics like PR-AUC are meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def undersample_negatives(
    df: pd.DataFrame, *, target: str = "fire", neg_per_pos: float = 5.0, seed: int = 42
) -> pd.DataFrame:
    """Keep all positives and ``neg_per_pos`` negatives per positive (random subset)."""
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    n_keep = min(len(neg), int(round(len(pos) * neg_per_pos)))
    if n_keep < len(neg):
        neg = neg.sample(n=n_keep, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def class_weight_dict(y: np.ndarray) -> dict:
    """Balanced class weights {0: w0, 1: w1} for estimators that accept them."""
    y = np.asarray(y)
    n = len(y)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, n - n_pos)
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def prior_correction(p: np.ndarray, train_rate: float, true_rate: float) -> np.ndarray:
    """Recalibrate probabilities from an undersampled training prior to the true prior.

    Standard log-odds offset correction (King & Zeng). ``train_rate`` is the positive
    fraction the model saw; ``true_rate`` is the real base rate. Returns corrected
    probabilities so the risk surface is well-calibrated.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    factor = (true_rate / (1 - true_rate)) / (train_rate / (1 - train_rate))
    corr = odds * factor
    return corr / (1 + corr)
