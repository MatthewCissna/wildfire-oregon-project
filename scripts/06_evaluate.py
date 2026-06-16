"""Stage 6 — assemble the evaluation report (RESULTS.md) from saved metrics.

Reads outputs/metrics/*.json and writes RESULTS.md at the repo root, with the
baseline-vs-model comparison tables, count-model metrics, CNN metrics, and SHAP
drivers. Re-run any time after training to refresh the numbers.

    uv run python scripts/06_evaluate.py
"""

from __future__ import annotations

import json

from wildfire.config import REPO_ROOT, load_config
from wildfire.utils import init_console


def _g(agg: dict, key: str) -> float:
    return agg.get(f"{key}_mean", float("nan"))


def _fmt(x: float, p: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{p}f}"  # x!=x catches NaN


def main() -> int:
    init_console()
    cfg = load_config()
    mdir = cfg.path_for("metrics")
    tab = json.loads((mdir / "tabular_metrics.json").read_text())
    manifest = json.loads((cfg.path_for("data_interim") / "_manifest.json").read_text())
    cnn = None
    if (mdir / "cnn_metrics.json").exists():
        cnn = json.loads((mdir / "cnn_metrics.json").read_text())

    lines: list[str] = []
    A = lines.append
    A("# RESULTS — Oregon Wildfire ML System\n")
    A(f"*Source: **{manifest.get('source')}** data "
      f"({'quick sample' if manifest.get('quick') else 'full run'}); "
      f"{manifest.get('n_cells'):,} grid cells, base fire rate "
      f"{manifest.get('positive_rate', 0)*100:.2f}%.*\n")
    A("> Numbers below are produced by `scripts/06_evaluate.py` from the saved "
      "metrics — re-run after training to refresh. With Earth Engine configured and "
      "a full run, rerun the pipeline without `--quick` to populate real-data results.\n")

    # ---- Risk model vs baselines ----
    A("\n## 1. Risk model vs. baselines (identical CV splits)\n")
    A("Primary metric is **PR-AUC** (rare-event); higher is better. **lift** = PR-AUC ÷ "
      "base rate. **recall@20%** = fraction of real fires caught if we flag the top 20% "
      "riskiest cells. **Brier** = calibration (lower better).\n")
    for scheme, models in tab["schemes"].items():
        A(f"\n### CV scheme: `{scheme}`\n")
        A("| Model | PR-AUC | lift | recall@20% | Brier | ROC-AUC |")
        A("|---|---|---|---|---|---|")
        for name, res in models.items():
            agg = res["aggregate"]
            star = " **(ours)**" if name == "risk_gbm" else ""
            A(f"| {name}{star} | {_fmt(_g(agg,'pr_auc'))} | {_fmt(_g(agg,'pr_auc_lift'),1)}x "
              f"| {_fmt(_g(agg,'recall_at_p20'))} | {_fmt(_g(agg,'brier'),4)} "
              f"| {_fmt(_g(agg,'roc_auc'))} |")
    A("\n**Reading it:** the climatology baseline (which only exploits *where* fires "
      "recur) typically collapses toward no-skill (PR-AUC ≈ base rate, ROC-AUC ≈ 0.5) "
      "under `spatial_block` CV — the clearest demonstration of why random splits "
      "mislead. Our GBM, using engineered fire-domain features + ignition-cause priors, "
      "should lead on PR-AUC and calibration on **both** schemes.\n")

    # ---- SHAP ----
    if tab.get("shap_top15"):
        A("\n## 2. What drives the risk model (SHAP)\n")
        A("Top features by mean |SHAP| — these should be physical fire drivers, not "
          "artifacts:\n")
        A("| # | feature | mean |SHAP| |")
        A("|---|---|---|")
        for i, row in enumerate(tab["shap_top15"][:12], 1):
            A(f"| {i} | `{row['feature']}` | {row['mean_abs_shap']:.4f} |")
        A("\nFigure: `outputs/figures/shap_importance.png`.\n")

    # ---- Count model ----
    if tab.get("count"):
        A("\n## 3. Fire-count model (per region / season)\n")
        A("| Model | MAE | RMSE | Poisson deviance | 95% PI coverage |")
        A("|---|---|---|---|---|")
        for kind in ("negbin", "poisson"):
            agg = tab["count"][kind]["aggregate"]
            A(f"| {kind} | {_fmt(_g(agg,'mae'))} | {_fmt(_g(agg,'rmse'))} "
              f"| {_fmt(_g(agg,'poisson_deviance'))} | {_fmt(_g(agg,'coverage_95'),2)} |")
        A("\nForward-chaining (by year) CV. The negative-binomial model handles "
          "overdispersion, giving calibrated (wider) intervals than Poisson. "
          "Per-region predictions: `outputs/metrics/region_count_predictions.csv`; "
          "map: `outputs/maps/fire_count_map.html`.\n")

    # ---- CNN ----
    A("\n## 4. Fire-detection CNN (held-out spatial blocks)\n")
    if cnn:
        m = cnn["test_metrics"]
        A(f"Backbone: `{cnn['backbone']}` (transfer learning, multi-band input).\n")
        A("| PR-AUC | lift | ROC-AUC | recall@20% | Brier |")
        A("|---|---|---|---|---|")
        A(f"| {_fmt(m['pr_auc'])} | {_fmt(m['pr_auc_lift'],1)}x | {_fmt(m['roc_auc'])} "
          f"| {_fmt(m['recall_at_p20'])} | {_fmt(m['brier'],4)} |")
        A("\nEvaluated on spatial blocks unseen in training (no leakage). Model: "
          "`outputs/models/cnn/fire_detector.pt`.\n")
        if manifest.get("source") == "synthetic":
            A("\n> ⚠️ On **synthetic** patches the fire signature is cleaner than real "
              "imagery, so this score is optimistic — it confirms the pipeline, not "
              "real-world detection skill. Published Sentinel-2 detectors report F1 "
              "≈ 0.88–0.97 (see `docs/literature.md`).\n")
    else:
        A("*Not trained yet. Run:* `uv sync --extra cnn` *then* "
          "`uv run python scripts/03_patches.py --synthetic && "
          "uv run python scripts/05_train_cnn.py`.\n")

    # ---- Deliverables / caveats ----
    A("\n## 5. Deliverables\n")
    A("- **Risk heatmap:** `outputs/maps/risk_heatmap.html` (interactive), "
      "`outputs/figures/risk_map.png` (static).\n"
      "- **Fire-count prediction:** `outputs/maps/fire_count_map.html`, "
      "`outputs/metrics/region_count_predictions.csv`.\n"
      "- **Fire detection:** `outputs/models/cnn/fire_detector.pt`, "
      "`outputs/metrics/cnn_metrics.json`.\n")
    A("\n## 6. Caveats\n")
    if manifest.get("source") == "synthetic":
        A("- These numbers are from the **synthetic** generator (latent fire model), "
          "which validates the methodology end-to-end but is **not** a claim about "
          "real Oregon skill. Configure Earth Engine and run without `--quick` for "
          "real-data results.\n")
    A("- All metrics use spatially/temporally honest CV; do not compare directly to "
      "papers' random-split numbers (see `docs/literature.md`).\n")

    out = REPO_ROOT / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote {out}")
    # Console summary
    for scheme, models in tab["schemes"].items():
        print(f"\n[{scheme}]")
        for name, res in models.items():
            agg = res["aggregate"]
            print(f"  {name:18s} PR-AUC={_fmt(_g(agg,'pr_auc'))}  recall@20%={_fmt(_g(agg,'recall_at_p20'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
