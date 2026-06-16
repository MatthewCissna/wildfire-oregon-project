"""Fast smoke tests for core building blocks (no heavy data pulls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire.config import load_config
from wildfire.eval import cv
from wildfire.eval.metrics import classification_metrics, recall_at_flag_rate
from wildfire.features.sampling import prior_correction, undersample_negatives


def test_config_loads():
    cfg = load_config()
    assert cfg.get("grid.h3_resolution") == 6
    assert cfg.get("datasets.burned_area") == "MODIS/061/MCD64A1"
    # EE project unset by default -> synthetic path.
    assert cfg.ee_project is None or isinstance(cfg.ee_project, str)


def test_recall_at_flag_rate_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = recall_at_flag_rate(y, score, 0.2)  # flag top 2
    assert out["recall"] == 1.0
    assert out["precision"] == 1.0
    assert out["lift"] > 1.0


def test_classification_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=500) < 0.05).astype(int)
    score = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.1, 500), 0, 1)
    m = classification_metrics(y, score)
    for k in ("pr_auc", "roc_auc", "brier", "recall_at_p20", "pr_auc_lift"):
        assert k in m
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["brier"] <= 1


def test_prior_correction_lowers_probability():
    p = np.array([0.5, 0.8, 0.2])
    # Trained on a balanced (undersampled) set, true rate is rare -> correction shrinks p.
    corr = prior_correction(p, train_rate=0.5, true_rate=0.03)
    assert np.all(corr < p)
    assert np.all((corr > 0) & (corr < 1))


def test_undersample_ratio():
    df = pd.DataFrame({"fire": [1] * 10 + [0] * 1000})
    out = undersample_negatives(df, neg_per_pos=5.0, seed=0)
    assert out["fire"].sum() == 10
    assert (out["fire"] == 0).sum() == 50


def test_forward_chaining_respects_time():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2018-06-01", "2019-06-01", "2020-06-01", "2021-06-01"] * 5
            ),
            "block_id": ["a", "b"] * 10,
        }
    )
    folds = list(cv.forward_chaining(df, n_folds=5))
    assert folds, "expected at least one temporal fold"
    years = df["date"].dt.year.to_numpy()
    for tr, te, _label in folds:
        assert years[tr].max() < years[te].min()  # no training on the future


def test_spatial_block_holds_out_whole_blocks():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-06-01"] * 20), "block_id": list("abcde") * 4}
    )
    for tr, te, _ in cv.spatial_block(df, n_folds=5):
        train_blocks = set(df["block_id"].to_numpy()[tr])
        test_blocks = set(df["block_id"].to_numpy()[te])
        assert train_blocks.isdisjoint(test_blocks)
