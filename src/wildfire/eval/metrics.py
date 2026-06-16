"""Honest metrics for rare-event wildfire prediction.

Accuracy and ROC-AUC flatter models on imbalanced data — a model that never
predicts fire is ~97% accurate and can still post a decent ROC-AUC. We report:

* **PR-AUC** (average precision) — the headline metric for rare positives.
* **Brier score** — calibration of the probabilities (lower is better).
* **Recall @ flag-rate** — operational: if we flag the top X% riskiest cells, what
  fraction of real fires do we catch? (e.g. ``recall_at_p20``).
* **Precision/recall/F1 @ threshold** — at a chosen probability cut.
* **ROC-AUC + lift** — reported for comparison with the literature, not as the goal.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_flag_rate(y_true: np.ndarray, y_score: np.ndarray, rate: float) -> dict:
    """Flag the top ``rate`` fraction of cells by score; report recall & precision there."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_flag = max(1, int(round(len(y_score) * rate)))
    order = np.argsort(-y_score)
    flagged = np.zeros(len(y_score), dtype=bool)
    flagged[order[:n_flag]] = True
    tp = int((flagged & (y_true == 1)).sum())
    n_pos = int((y_true == 1).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / n_flag
    base = n_pos / len(y_true) if len(y_true) else float("nan")
    return {
        "flag_rate": rate,
        "recall": recall,
        "precision": precision,
        "lift": (precision / base) if base else float("nan"),
    }


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict:
    """Full metric bundle for a set of probabilistic predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")

    out: dict = {"n": int(len(y_true)), "base_rate": base_rate}

    # Guard: metrics that need both classes present.
    both_classes = len(np.unique(y_true)) == 2
    out["pr_auc"] = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")
    out["roc_auc"] = float(roc_auc_score(y_true, y_score)) if both_classes else float("nan")
    out["brier"] = float(brier_score_loss(y_true, y_score))

    # PR-AUC lift over the no-skill baseline (= base rate).
    out["pr_auc_lift"] = (out["pr_auc"] / base_rate) if base_rate else float("nan")

    y_pred = (y_score >= threshold).astype(int)
    out["threshold"] = threshold
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    for rate, name in [(0.10, "recall_at_p10"), (0.20, "recall_at_p20"), (0.30, "recall_at_p30")]:
        out[name] = recall_at_flag_rate(y_true, y_score, rate)["recall"]
    return out


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Mean ± std across CV folds for the numeric metrics."""
    if not fold_metrics:
        return {}
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics if k in m], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std())
    return agg
