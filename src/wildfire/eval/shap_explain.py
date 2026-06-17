"""SHAP explanations for the risk model.

Two purposes:
  1. **Validation** — confirm the model keys on real fire drivers (VPD, drought,
     ERC, fuel, dryness windows) rather than artifacts. If ``cell_id`` proxies or
     calendar noise dominate, something is wrong.
  2. **Communication** — a ranked feature-importance figure for RESULTS.md.

Uses ``shap.TreeExplainer`` on the underlying LightGBM booster (fast, exact for trees).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def explain_risk(
    risk_model,
    df: pd.DataFrame,
    *,
    sample_n: int = 5000,
    seed: int = 42,
    out_fig: str | Path | None = None,
) -> pd.DataFrame:
    """Compute SHAP importances; optionally save a bar figure. Returns ranked importances."""
    import shap

    feats = risk_model.feature_cols
    X = df[feats]
    if len(X) > sample_n:
        X = X.sample(n=sample_n, random_state=seed)

    explainer = shap.TreeExplainer(risk_model.model)
    sv = explainer.shap_values(X)
    # LightGBM binary may return a list [neg, pos] or a single array.
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, -1]

    importance = (
        pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if out_fig is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from wildfire.feature_labels import label_feature

        top = importance.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.barh(top["feature"].map(label_feature), top["mean_abs_shap"], color="#c1440e")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title("Risk model — top feature importances (SHAP)")
        fig.tight_layout()
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)

    return importance
