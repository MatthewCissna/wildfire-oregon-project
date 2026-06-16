"""Stage 4 — train & benchmark the risk and fire-count models.

Runs the rigor pipeline:
  * benchmark climatology + logistic baselines vs the LightGBM risk model on
    BOTH forward-chaining (temporal) and spatial-block CV — identical splits;
  * fit the final risk model on all data and write the per-cell risk surface;
  * fit + cross-validate the fire-count model (negative binomial) with intervals;
  * SHAP importances to validate the model learns fire physics.

    uv run python scripts/04_train_tabular.py
    uv run python scripts/04_train_tabular.py --schemes forward_chaining spatial_block
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from wildfire.config import load_config
from wildfire.eval.runner import run_cv
from wildfire.eval.shap_explain import explain_risk
from wildfire.features.build import feature_columns
from wildfire.features.regions import build_region_season, region_feature_columns
from wildfire.models import baselines, count, risk, tune
from wildfire.utils import init_console


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["forward_chaining", "spatial_block"])
    args = ap.parse_args()

    init_console()
    cfg = load_config()
    feats_path = cfg.path_for("data_processed") / "features.parquet"
    df = pd.read_parquet(feats_path)
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = feature_columns(df)

    # Use Optuna-tuned risk params if a tuning run has produced them.
    best = tune.load_best_risk_params(cfg)
    risk_params = best["params"] if best else None
    if risk_params:
        print("Using Optuna-tuned risk params (from scripts/08_tune.py).")

    models = {
        "climatology": baselines.make_climatology_fit_predict(),
        "logistic_weather": baselines.make_logistic_fit_predict(),
        "risk_gbm": risk.make_fit_predict(risk_params),
    }

    results = {"schemes": {}, "feature_count": len(feature_cols)}
    for scheme in args.schemes:
        print(f"\n=== CV scheme: {scheme} ===")
        results["schemes"][scheme] = {}
        for name, fp in models.items():
            res = run_cv(df, feature_cols, fp, cfg, scheme=scheme)
            agg = res["aggregate"]
            results["schemes"][scheme][name] = res
            print(
                f"  {name:18s} PR-AUC={agg.get('pr_auc_mean', float('nan')):.3f} "
                f"(lift {agg.get('pr_auc_lift_mean', float('nan')):.1f}x)  "
                f"recall@20%={agg.get('recall_at_p20_mean', float('nan')):.3f}  "
                f"Brier={agg.get('brier_mean', float('nan')):.4f}  "
                f"ROC-AUC={agg.get('roc_auc_mean', float('nan')):.3f}"
            )

    # ---- final risk model + surface ----
    print("\n=== Fitting final risk model on all data ===")
    rm = risk.fit_full(df, feature_cols, cfg, params=risk_params)
    model_path = cfg.path_for("models") / "risk_model.joblib"
    rm.save(model_path)

    from wildfire.ingest.datasets import load_canonical

    grid = load_canonical(cfg)["grid"]
    surface = risk.predict_surface(rm, df, grid)
    surface_path = cfg.path_for("data_processed") / "risk_surface.parquet"
    surface.to_parquet(surface_path)
    print(f"   risk model  -> {model_path}")
    print(f"   risk surface-> {surface_path}  (mean risk {surface['risk'].mean():.4f})")

    # ---- SHAP ----
    print("\n=== SHAP importances ===")
    imp = explain_risk(rm, df, out_fig=cfg.path_for("figures") / "shap_importance.png", seed=cfg.seed)
    results["shap_top15"] = imp.head(15).to_dict(orient="records")
    for _, row in imp.head(10).iterrows():
        print(f"   {row['feature']:20s} {row['mean_abs_shap']:.4f}")

    # ---- fire-count model ----
    print("\n=== Fire-count model (negative binomial) ===")
    region = build_region_season(cfg, df)
    rcols = region_feature_columns(region)
    cm_cv = count.cross_validate_count(region, rcols, cfg, kind="negbin")
    agg = cm_cv["aggregate"]
    print(
        f"   NB count CV: MAE={agg.get('mae_mean', float('nan')):.3f}  "
        f"RMSE={agg.get('rmse_mean', float('nan')):.3f}  "
        f"PoisDev={agg.get('poisson_deviance_mean', float('nan')):.3f}  "
        f"95%PI coverage={agg.get('coverage_95_mean', float('nan')):.2f}"
    )
    # Compare to a Poisson model to show overdispersion handling matters.
    cm_pois = count.cross_validate_count(region, rcols, cfg, kind="poisson")
    results["count"] = {"negbin": cm_cv, "poisson": cm_pois}
    cm_final = count.train_count(region, rcols, cfg, kind="negbin")
    import joblib

    joblib.dump(cm_final, cfg.path_for("models") / "count_model.joblib")

    # ---- persist metrics ----
    metrics_path = cfg.path_for("metrics") / "tabular_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n✅ Metrics saved -> {metrics_path}")
    print("   Next:  uv run python scripts/06_evaluate.py   (report)  ")
    print("      or  uv run python scripts/07_visualize.py  (maps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
