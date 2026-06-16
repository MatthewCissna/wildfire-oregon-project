"""Generic cross-validation runner.

Loops a CV scheme, calls a model's ``fit_predict`` on each fold, and scores the
held-out fold with the honest rare-event metrics. Models stay decoupled from the
CV machinery — they only provide ``fit_predict(train_df, test_df, feature_cols, cfg)``
returning a probability score per test row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from wildfire.config import Config
from wildfire.eval.cv import iter_folds
from wildfire.eval.metrics import aggregate_folds, classification_metrics

logger = logging.getLogger(__name__)

FitPredict = Callable[[pd.DataFrame, pd.DataFrame, list, Config], np.ndarray]


def run_cv(
    df: pd.DataFrame,
    feature_cols: list,
    fit_predict: FitPredict,
    cfg: Config,
    scheme: str = "forward_chaining",
    target: str = "fire",
    **fold_kwargs,
) -> dict:
    """Run one CV scheme and return per-fold + aggregated metrics."""
    fold_metrics: list[dict] = []
    for tr_idx, te_idx, label in iter_folds(df, scheme, **fold_kwargs):
        train_df = df.iloc[tr_idx]
        test_df = df.iloc[te_idx]
        if test_df[target].nunique() < 2:
            logger.info("fold %s skipped (single-class test set)", label)
            continue
        scores = fit_predict(train_df, test_df, feature_cols, cfg)
        m = classification_metrics(test_df[target].to_numpy(), scores)
        m["fold"] = label
        m["n_train"] = int(len(train_df))
        fold_metrics.append(m)
        logger.info(
            "fold %-22s PR-AUC=%.3f (lift %.1fx)  recall@20%%=%.3f  Brier=%.4f",
            label, m["pr_auc"], m["pr_auc_lift"], m["recall_at_p20"], m["brier"],
        )

    return {
        "scheme": scheme,
        "folds": fold_metrics,
        "aggregate": aggregate_folds(fold_metrics),
    }
