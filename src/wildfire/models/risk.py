"""Risk model — gradient-boosted trees producing a per-cell fire-probability surface.

Pipeline:
  * undersample the negative class for tractable training (fires are ~3% of rows),
  * fit a LightGBM classifier on the fire-domain features,
  * **recalibrate** the output back to the true base rate (King–Zeng prior
    correction) so the probability surface is honest, not inflated by undersampling.

Exposes a ``fit_predict`` closure for the CV runner and a ``fit_full`` for the final
deployed surface, plus joblib save/load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from wildfire.config import Config, load_config
from wildfire.features.sampling import prior_correction, undersample_negatives


def default_params(cfg: Config) -> dict:
    return {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": cfg.seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


@dataclass
class RiskModel:
    model: lgb.LGBMClassifier
    feature_cols: list[str]
    train_pos_rate: float
    true_pos_rate: float
    params: dict = field(default_factory=dict)

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated fire probability per row."""
        raw = self.model.predict_proba(df[self.feature_cols])[:, 1]
        return prior_correction(raw, self.train_pos_rate, self.true_pos_rate)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RiskModel":
        return joblib.load(path)


def train_risk(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    *,
    params: dict | None = None,
) -> RiskModel:
    """Fit a calibrated risk model on (undersampled) training data."""
    true_rate = float(train_df["fire"].mean())
    neg_per_pos = float(cfg.get("modeling.imbalance.neg_per_pos", 5.0))
    sampled = undersample_negatives(
        train_df, neg_per_pos=neg_per_pos, seed=cfg.seed
    )
    train_rate = float(sampled["fire"].mean())

    p = params or default_params(cfg)
    model = lgb.LGBMClassifier(**p)
    model.fit(sampled[feature_cols], sampled["fire"])
    return RiskModel(
        model=model,
        feature_cols=list(feature_cols),
        train_pos_rate=train_rate,
        true_pos_rate=true_rate,
        params=p,
    )


def make_fit_predict(params: dict | None = None):
    """Return a ``fit_predict(train_df, test_df, feature_cols, cfg)`` for the CV runner."""

    def _fit_predict(train_df, test_df, feature_cols, cfg) -> np.ndarray:
        rm = train_risk(train_df, feature_cols, cfg, params=params)
        return rm.predict_risk(test_df)

    return _fit_predict


def fit_full(
    df: pd.DataFrame, feature_cols: list[str], cfg: Config | None = None,
    params: dict | None = None,
) -> RiskModel:
    """Train the final risk model on all available data (for the risk surface)."""
    cfg = cfg or load_config()
    return train_risk(df, feature_cols, cfg, params=params)


def predict_surface(model: RiskModel, df: pd.DataFrame, grid) -> pd.DataFrame:
    """Per-cell mean risk over time -> a GeoDataFrame-ready surface for mapping."""
    scores = model.predict_risk(df)
    out = (
        pd.DataFrame({"cell_id": df["cell_id"].to_numpy(), "risk": scores})
        .groupby("cell_id", as_index=False)["risk"]
        .mean()
    )
    return grid.merge(out, on="cell_id", how="left")
