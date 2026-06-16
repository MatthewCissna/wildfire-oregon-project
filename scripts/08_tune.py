"""Stage 8 (optional) — Optuna hyperparameter tuning.

Optimizes the **spatial-block CV PR-AUC** (honest generalization), saves the best
params, and logs all trials. The training stage (04) automatically picks up the
saved risk params on its next run.

    uv run python scripts/08_tune.py --target risk --trials 30
    uv run python scripts/08_tune.py --target cnn  --trials 8     # needs --extra cnn
    uv run python scripts/08_tune.py --target both
"""

from __future__ import annotations

import argparse

import pandas as pd

from wildfire.config import load_config
from wildfire.features.build import feature_columns
from wildfire.models import tune
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["risk", "cnn", "both"], default="risk")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    init_console()
    cfg = load_config()

    if args.target in ("risk", "both"):
        df = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        feats = feature_columns(df)
        res = tune.tune_risk(df, feats, cfg, n_trials=args.trials)
        print("\n✅ Risk tuning done")
        print(f"   best spatial-block PR-AUC : {res['best_value']:.4f}")
        print(f"   best params saved         : {tune.risk_best_params_path(cfg)}")
        print(f"   trial log                 : {cfg.path_for('metrics') / 'optuna_risk_trials.csv'}")
        print("   (Re-run scripts/04_train_tabular.py to use them.)")

    if args.target in ("cnn", "both"):
        try:
            import torch  # noqa: F401
        except ImportError:
            print("⚠️  Skipping CNN tuning: PyTorch not installed (uv sync --extra cnn).")
            return 0
        n = max(4, args.trials // 3) if args.target == "both" else args.trials
        res = tune.tune_cnn(cfg, n_trials=n)
        print("\n✅ CNN tuning done")
        print(f"   best test PR-AUC : {res['best_value']:.4f}")
        print(f"   best params      : {res['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
