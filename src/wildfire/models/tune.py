"""Hyperparameter tuning with Optuna.

The objective is the **spatially-honest** CV score (mean spatial-block PR-AUC), not a
random-split score — so tuning optimizes for generalization to new regions rather
than to memorized locations. All trials are logged to ``outputs/metrics`` and the
best params are persisted for the training stage to consume.

* :func:`tune_risk` — LightGBM risk model (primary).
* :func:`tune_cnn`  — detection CNN (lr / weight-decay / backbone), lighter search.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


def _subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Stratified subsample to ~max_rows that **preserves the true class ratio**.

    Critical: the tuning objective is PR-AUC, which depends on the base rate. We must
    NOT inflate the positive fraction here (that would make PR-AUC meaningless) — the
    risk model does its own undersampling internally at fit time, with prior
    correction. So we down-sample both classes by the same fraction.
    """
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    return (
        df.groupby("fire", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )


def risk_best_params_path(cfg: Config) -> Path:
    return cfg.path_for("models") / "risk_best_params.json"


def load_best_risk_params(cfg: Config) -> dict | None:
    p = risk_best_params_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return None


def tune_risk(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config | None = None,
    *,
    n_trials: int = 30,
    scheme: str = "spatial_block",
    max_rows: int = 200_000,
) -> dict:
    """Optuna search for the risk GBM; returns best params and saves them + a trial log."""
    import optuna
    from optuna.samplers import TPESampler

    from wildfire.eval.runner import run_cv
    from wildfire.models import risk

    cfg = cfg or load_config()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune_df = _subsample(df, max_rows, cfg.seed)
    logger.info("Tuning risk model on %d rows (%d trials, %s CV)", len(tune_df), n_trials, scheme)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": cfg.seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        res = run_cv(tune_df, feature_cols, risk.make_fit_predict(params), cfg, scheme=scheme)
        score = res["aggregate"].get("pr_auc_mean", 0.0)
        return score if score == score else 0.0  # guard NaN

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {
        "objective": "binary", "random_state": cfg.seed, "n_jobs": -1, "verbosity": -1,
        "subsample_freq": 1, **study.best_params,
    }
    out = risk_best_params_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best_value_pr_auc": study.best_value, "params": best}, indent=2))
    study.trials_dataframe().to_csv(cfg.path_for("metrics") / "optuna_risk_trials.csv", index=False)
    logger.info("Best spatial-block PR-AUC=%.4f", study.best_value)
    return {"best_value": study.best_value, "params": best, "study": study}


def tune_cnn(cfg: Config | None = None, data: dict | None = None, *, n_trials: int = 8) -> dict:
    """Lighter Optuna search for the CNN (lr / weight-decay / backbone), short proxy epochs."""
    import optuna
    from optuna.samplers import TPESampler

    from wildfire.models.cnn import train_cnn

    cfg = cfg or load_config()
    if data is None:
        from wildfire.ingest.patches import load_patches

        data = load_patches(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        raw = copy.deepcopy(cfg.raw)
        raw["cnn"]["lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        raw["cnn"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        raw["cnn"]["backbone"] = trial.suggest_categorical(
            "backbone", ["efficientnet_b0", "resnet34"]
        )
        raw["cnn"]["epochs"] = 4  # short proxy
        trial_cfg = Config(raw=raw, path=cfg.path)
        result = train_cnn(trial_cfg, data, quick=False)
        return float(result.test_metrics.get("pr_auc", 0.0))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    out = cfg.path_for("models") / "cnn_best_params.json"
    out.write_text(json.dumps({"best_value_pr_auc": study.best_value, "params": study.best_params}, indent=2))
    study.trials_dataframe().to_csv(cfg.path_for("metrics") / "optuna_cnn_trials.csv", index=False)
    return {"best_value": study.best_value, "params": study.best_params, "study": study}
