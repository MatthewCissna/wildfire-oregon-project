"""Build the interactive website's data files from the pipeline outputs.

Exports browser-ready JS (loaded via <script> so the site works by double-clicking
index.html — no server, no fetch/CORS issues):

    site/data/cells.js   window.WF_CELLS  -> GeoJSON of Oregon hexes with rich
                         per-cell detail (terrain, fuel, mean weather, NDVI,
                         fire history by year, modeled risk)
    site/data/meta.js    window.WF_META   -> metrics, SHAP, count predictions,
                         ecoregion summaries, dataset catalog, manifest

Re-run after the pipeline to refresh the site:  uv run python scripts/build_site.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from wildfire.config import REPO_ROOT, load_config
from wildfire.utils import init_console

WORLDCOVER = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse", 70: "Snow / ice", 80: "Water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen",
}

WEATHER_MEANS = ["tmax", "rmin", "wind", "precip", "vpd", "erc", "bi", "pdsi", "ndvi", "days_since_rain"]


def _round(x, n=3):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, n)
    except (TypeError, ValueError):
        return None


def build(cfg) -> dict:
    site = REPO_ROOT / "site"
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "assets").mkdir(parents=True, exist_ok=True)

    surf = gpd.read_parquet(cfg.path_for("data_processed") / "risk_surface.parquet")
    feats = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")
    feats["year"] = pd.to_datetime(feats["date"]).dt.year
    years = sorted(feats["year"].unique().tolist())

    # ---- per-cell aggregates ----
    agg = feats.groupby("cell_id").agg({c: "mean" for c in WEATHER_MEANS if c in feats})
    fires = feats.groupby("cell_id")["fire"].agg(total="sum", weeks="count")
    agg = agg.join(fires)
    agg["fire_rate"] = agg["total"] / agg["weeks"]

    # fires by year per cell -> compact array aligned to `years`
    fby = (
        feats.groupby(["cell_id", "year"])["fire"].sum().unstack(fill_value=0)
        .reindex(columns=years, fill_value=0)
    )
    fby_map = {cid: row.astype(int).tolist() for cid, row in fby.iterrows()}

    # ---- risk color scaling: percentile rank (risk is heavily skewed) ----
    risk = surf["risk"].fillna(0).to_numpy()
    order = risk.argsort().argsort()
    pct = (order / max(1, len(order) - 1))
    surf = surf.assign(risk_pct=pct)

    # ---- GeoJSON features ----
    surf4326 = surf.to_crs("EPSG:4326")
    feats_geo = []
    for _, r in surf4326.iterrows():
        cid = r["cell_id"]
        a = agg.loc[cid] if cid in agg.index else None
        geom = r.geometry
        coords = [[round(x, 4), round(y, 4)] for x, y in geom.exterior.coords]
        lc = int(round(r["landcover"])) if pd.notna(r.get("landcover")) else None
        props = {
            "id": cid,
            "lon": _round(r["lon"], 3), "lat": _round(r["lat"], 3),
            "eco": r.get("ecoregion"),
            "risk": _round(r["risk"], 4), "risk_pct": _round(r["risk_pct"], 3),
            "elev": _round(r.get("elevation"), 0), "slope": _round(r.get("slope"), 1),
            "aspect": _round(r.get("aspect"), 0),
            "landcover": WORLDCOVER.get(lc, "—"),
            "fuel": _round(r.get("fuel_load"), 2),
            "fires_total": int(a["total"]) if a is not None else 0,
            "fires_rate": _round(a["fire_rate"] * 100, 2) if a is not None else 0,
            "fires_by_year": fby_map.get(cid, [0] * len(years)),
        }
        if a is not None:
            for c in WEATHER_MEANS:
                if c in a:
                    props[c] = _round(a[c], 2)
        feats_geo.append({"type": "Feature",
                          "geometry": {"type": "Polygon", "coordinates": [coords]},
                          "properties": props})

    cells = {"type": "FeatureCollection", "years": years, "features": feats_geo}

    # ---- ecoregion summaries ----
    eco_rows = []
    for eco, g in surf.groupby("ecoregion"):
        cids = g["cell_id"]
        sub = feats[feats["cell_id"].isin(cids)]
        eco_rows.append({
            "name": eco,
            "cells": int(len(g)),
            "mean_risk": _round(g["risk"].mean(), 4),
            "fires_total": int(sub["fire"].sum()),
            "fire_rate": _round(sub["fire"].mean() * 100, 3),
            "mean_elev": _round(g["elevation"].mean(), 0),
            "mean_fuel": _round(g["fuel_load"].mean(), 2),
            "fires_by_year": sub.groupby("year")["fire"].sum().reindex(years, fill_value=0).astype(int).tolist(),
        })
    eco_rows.sort(key=lambda d: -d["mean_risk"])

    # state-wide fires by year
    state_fby = feats.groupby("year")["fire"].sum().reindex(years, fill_value=0).astype(int).tolist()

    # ---- metrics ----
    mdir = cfg.path_for("metrics")
    tab = json.loads((mdir / "tabular_metrics.json").read_text())
    cnn = json.loads((mdir / "cnn_metrics.json").read_text()) if (mdir / "cnn_metrics.json").exists() else None
    manifest = json.loads((cfg.path_for("data_interim") / "_manifest.json").read_text())

    def scheme_table(scheme):
        out = []
        for name, res in tab["schemes"][scheme].items():
            a = res["aggregate"]
            out.append({
                "model": name,
                "pr_auc": _round(a.get("pr_auc_mean"), 3),
                "pr_lift": _round(a.get("pr_auc_lift_mean"), 1),
                "recall20": _round(a.get("recall_at_p20_mean"), 3),
                "brier": _round(a.get("brier_mean"), 4),
                "roc_auc": _round(a.get("roc_auc_mean"), 3),
            })
        return out

    count_rows = None
    cp = mdir / "region_count_predictions.csv"
    if cp.exists():
        cdf = pd.read_csv(cp)
        count_rows = [
            {"region": r["region"], "pred": _round(r["pred_count"], 1),
             "lo": _round(r["pred_lo"], 0), "hi": _round(r["pred_hi"], 0)}
            for _, r in cdf.sort_values("pred_count", ascending=False).iterrows()
        ]

    meta = {
        "manifest": manifest,
        "years": years,
        "schemes": {s: scheme_table(s) for s in tab["schemes"]},
        "shap": [{"feature": d["feature"], "value": _round(d["mean_abs_shap"], 4)} for d in tab.get("shap_top15", [])],
        "count": {
            "negbin": tab["count"]["negbin"]["aggregate"],
            "predictions": count_rows,
        },
        "cnn": cnn["test_metrics"] if cnn else None,
        "cnn_backbone": cnn["backbone"] if cnn else None,
        "ecoregions": eco_rows,
        "state_fires_by_year": state_fby,
    }

    (site / "data" / "cells.js").write_text(
        "window.WF_CELLS = " + json.dumps(cells, separators=(",", ":")) + ";", encoding="utf-8"
    )
    (site / "data" / "meta.js").write_text(
        "window.WF_META = " + json.dumps(meta, separators=(",", ":")) + ";", encoding="utf-8"
    )

    # copy figures
    for fig in ("shap_importance.png", "risk_map.png"):
        src = cfg.path_for("figures") / fig
        if src.exists():
            shutil.copy(src, site / "assets" / fig)

    return {"cells": len(feats_geo), "years": years,
            "cells_kb": (site / "data" / "cells.js").stat().st_size // 1024,
            "ecoregions": len(eco_rows)}


def main() -> int:
    init_console()
    cfg = load_config()
    info = build(cfg)
    print("✅ Site data built")
    print(f"   cells       : {info['cells']:,}  ({info['cells_kb']} KB)")
    print(f"   years       : {info['years'][0]}–{info['years'][-1]}")
    print(f"   ecoregions  : {info['ecoregions']}")
    print("   open        : site/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
