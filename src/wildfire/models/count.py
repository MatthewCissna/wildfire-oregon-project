"""Fire-count model — expected number of fires per region per season, with intervals.

Count data, so we use count models rather than regression on raw numbers:

* **Poisson GLM** with a ``log(exposure)`` offset — predicts a *rate* per
  (cell × week) of opportunity, not a raw count confounded by region size.
* **Negative-binomial GLM** — same, but models the overdispersion real fire counts
  exhibit (variance > mean), giving honest (wider) intervals.
* **Poisson GBM** (LightGBM) — nonlinear alternative.

Uncertainty intervals come straight from the fitted count distribution (Poisson or
NB), so they're calibrated rather than hand-waved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from wildfire.config import Config, load_config


@dataclass
class CountModel:
    kind: str
    result: object
    feature_cols: list[str]
    alpha: float = 0.0  # NB dispersion (0 => Poisson)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "gbm":
            return np.clip(self.result.predict(df[self.feature_cols]), 0, None)
        X = sm.add_constant(df[self.feature_cols], has_constant="add")
        offset = df["log_exposure"].to_numpy()
        return np.asarray(self.result.predict(X, offset=offset))

    def predict_interval(self, df: pd.DataFrame, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated prediction interval from the count distribution."""
        mu = np.clip(self.predict(df), 1e-9, None)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        if self.kind == "negbin" and self.alpha > 0:
            n = 1.0 / self.alpha
            p = n / (n + mu)
            return stats.nbinom.ppf(lo_q, n, p), stats.nbinom.ppf(hi_q, n, p)
        return stats.poisson.ppf(lo_q, mu), stats.poisson.ppf(hi_q, mu)


def _estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Cameron–Trivedi NB2 dispersion estimate from a fitted Poisson mean."""
    mu = np.clip(mu, 1e-6, None)
    aux = ((y - mu) ** 2 - mu) / mu
    alpha = float(np.maximum(0.01, (aux / mu).mean()))
    return min(alpha, 5.0)


def train_count(
    df: pd.DataFrame, feature_cols: list[str], cfg: Config | None = None, kind: str = "negbin"
) -> CountModel:
    """Fit a count model. ``kind`` in {poisson, negbin, gbm}."""
    cfg = cfg or load_config()
    y = df["fire_count"].to_numpy()
    offset = df["log_exposure"].to_numpy()

    if kind == "gbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            objective="poisson", n_estimators=400, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=cfg.seed, n_jobs=-1, verbosity=-1,
        )
        # log_exposure included as a feature so the GBM can learn the exposure effect.
        cols = list(feature_cols)
        if "log_exposure" not in cols:
            cols = cols + ["log_exposure"]
        model.fit(df[cols], y)
        return CountModel(kind="gbm", result=model, feature_cols=cols)

    X = sm.add_constant(df[feature_cols], has_constant="add")
    pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    if kind == "poisson":
        return CountModel(kind="poisson", result=pois, feature_cols=list(feature_cols))

    # Negative binomial: estimate dispersion from the Poisson fit, then refit.
    alpha = _estimate_alpha(y, np.asarray(pois.predict(X, offset=offset)))
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    return CountModel(kind="negbin", result=nb, feature_cols=list(feature_cols), alpha=alpha)


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Mean Poisson deviance (lower better) — the proper scoring rule for counts.
    # Compute the y*log(y/mu) term only where y>0 to avoid 0*log(0) warnings.
    term = np.zeros_like(y_true)
    pos = y_true > 0
    term[pos] = y_true[pos] * np.log(y_true[pos] / y_pred[pos])
    dev = 2.0 * np.mean(term - (y_true - y_pred))
    return {"mae": mae, "rmse": rmse, "poisson_deviance": float(dev)}


def cross_validate_count(
    df: pd.DataFrame, feature_cols: list[str], cfg: Config, kind: str = "negbin"
) -> dict:
    """Forward-chaining (by year) CV for the count model, with interval coverage."""
    years = np.sort(df["year"].unique())
    fold_metrics = []
    for ty in years[1:]:
        tr = df[df["year"] < ty]
        te = df[df["year"] == ty]
        if len(tr) < 5 or len(te) == 0:
            continue
        model = train_count(tr, feature_cols, cfg, kind=kind)
        pred = model.predict(te)
        m = count_metrics(te["fire_count"].to_numpy(), pred)
        lo, hi = model.predict_interval(te)
        inside = (te["fire_count"].to_numpy() >= lo) & (te["fire_count"].to_numpy() <= hi)
        m["coverage_95"] = float(np.mean(inside))
        m["fold"] = f"year{ty}"
        fold_metrics.append(m)

    from wildfire.eval.metrics import aggregate_folds

    return {"kind": kind, "folds": fold_metrics, "aggregate": aggregate_folds(fold_metrics)}
