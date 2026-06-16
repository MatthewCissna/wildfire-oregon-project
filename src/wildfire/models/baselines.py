"""Baselines to benchmark the risk model against — on identical CV splits.

* **Logistic regression on weather** — the canonical "simple published model":
  standardized daily/weekly weather + drought, class-weighted. If our GBM with
  engineered features and ignition-cause priors doesn't beat this on spatial &
  temporal CV, it isn't earning its complexity.
* **Climatology** — predict each cell's historical fire frequency from the training
  fold. This is a deceptively strong spatial baseline; beating it proves the model
  adds *dynamic* (weather/fuel) skill beyond "fires recur where they always have".

Both expose the same ``fit_predict`` signature as the risk model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The "published-style" weather/drought predictors (no engineered interactions,
# no ignition-cause priors, no antecedent windows).
_WEATHER_FEATURES = [
    "tmax", "rmin", "wind", "precip", "vpd", "erc", "bi", "pdsi",
    "days_since_rain", "ndvi",
]


def make_logistic_fit_predict():
    """Weather-only logistic regression baseline."""

    def _fit_predict(train_df, test_df, feature_cols, cfg) -> np.ndarray:
        cols = [c for c in _WEATHER_FEATURES if c in train_df.columns]
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipe.fit(train_df[cols], train_df["fire"])
        return pipe.predict_proba(test_df[cols])[:, 1]

    return _fit_predict


def make_climatology_fit_predict(smoothing: float = 1.0):
    """Per-cell historical-frequency baseline (Laplace-smoothed), with block & global
    fallbacks for cells/blocks unseen in the training fold."""

    def _fit_predict(train_df, test_df, feature_cols, cfg) -> np.ndarray:
        global_rate = float(train_df["fire"].mean())
        cell_rate = (
            train_df.groupby("cell_id")["fire"]
            .agg(["sum", "count"])
            .assign(rate=lambda d: (d["sum"] + smoothing * global_rate) / (d["count"] + smoothing))[
                "rate"
            ]
        )
        block_rate = (
            train_df.groupby("block_id")["fire"]
            .agg(["sum", "count"])
            .assign(rate=lambda d: (d["sum"] + smoothing * global_rate) / (d["count"] + smoothing))[
                "rate"
            ]
        )
        out = test_df["cell_id"].map(cell_rate)
        out = out.fillna(test_df["block_id"].map(block_rate))
        out = out.fillna(global_rate)
        return out.to_numpy(dtype=float)

    return _fit_predict
