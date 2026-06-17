"""Build the interactive website's data files from the pipeline outputs.

Exports browser-ready JS (loaded via <script> so the site works by double-clicking
index.html — no server, no fetch/CORS issues) plus smooth interpolated risk-surface
PNGs and an ODF fire-district layer:

    site/data/cells.js      window.WF_CELLS  -> per-cell records (no geometry) for
                            nearest-cell hover/click detail
    site/data/meta.js       window.WF_META   -> metrics, SHAP (readable labels),
                            district count forecasts + polygons, ecoregion summaries,
                            surface descriptors, dataset catalog, manifest
    site/assets/surfaces/*  smooth IDW-interpolated, Oregon-masked heat surfaces (PNG)
    site/assets/shap_importance.png   regenerated with readable feature names

Re-run after the pipeline:  uv run python scripts/build_site.py
"""

from __future__ import annotations

import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd

from wildfire.config import REPO_ROOT, load_config
from wildfire.feature_labels import label_feature
from wildfire.utils import init_console

WORLDCOVER = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse", 70: "Snow / ice", 80: "Water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen",
}
WEATHER_MEANS = ["tmax", "rmin", "wind", "precip", "vpd", "erc", "bi", "pdsi", "ndvi", "days_since_rain"]

# Metrics rendered as smooth surfaces: (cell-property, label, unit, matplotlib cmap, value-scale)
SURFACE_METRICS = [
    ("risk", "Modeled risk", "%", "inferno", 100.0),
    ("fires_rate", "Fire rate", "% of weeks", "inferno", 1.0),
    ("fuel", "Fuel load", "0–1", "YlOrBr", 1.0),
    ("elev", "Elevation", "m", "cividis", 1.0),
    ("vpd", "Mean VPD", "kPa", "magma", 1.0),
    ("ndvi", "Mean NDVI", "", "YlGn", 1.0),
]


def _round(x, n=3):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, n)
    except (TypeError, ValueError):
        return None


def _poly_mask(gx, gy, geom):
    from matplotlib.path import Path as MPath
    pts = np.c_[gx.ravel(), gy.ravel()]
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    mask = np.zeros(len(pts), bool)
    for g in geoms:
        mask |= MPath(np.asarray(g.exterior.coords)).contains_points(pts)
    return mask.reshape(gx.shape)


def _surface(lons, lats, vals, geom, out_png, cmap_name, res=(900, 820)):
    """IDW-interpolate cell values to a smooth Oregon-masked PNG; return (domain, stops)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    minx, miny, maxx, maxy = geom.bounds
    W, H = res
    xs = np.linspace(minx, maxx, W)
    ys = np.linspace(maxy, miny, H)  # top row = north
    gx, gy = np.meshgrid(xs, ys)
    tree = cKDTree(np.c_[lons, lats])
    k = min(14, len(lons))
    dist, idx = tree.query(np.c_[gx.ravel(), gy.ravel()], k=k)
    w = 1.0 / (dist ** 2 + 1e-9)
    interp = (w * vals[idx]).sum(1) / w.sum(1)
    grid = interp.reshape(H, W)

    lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
    norm = np.clip((grid - lo) / (hi - lo + 1e-12), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(_poly_mask(gx, gy, geom), 205, 0).astype(np.uint8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png, rgba)
    stops = ["#%02x%02x%02x" % tuple((np.array(cmap(t)[:3]) * 255).astype(int)) for t in np.linspace(0, 1, 6)]
    return [float(lo), float(hi)], stops


def build(cfg) -> dict:
    site = REPO_ROOT / "site"
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "assets" / "surfaces").mkdir(parents=True, exist_ok=True)

    surf = gpd.read_parquet(cfg.path_for("data_processed") / "risk_surface.parquet")
    feats = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")
    feats["year"] = pd.to_datetime(feats["date"]).dt.year
    years = sorted(feats["year"].unique().tolist())

    agg = feats.groupby("cell_id").agg({c: "mean" for c in WEATHER_MEANS if c in feats})
    fires = feats.groupby("cell_id")["fire"].agg(total="sum", weeks="count")
    agg = agg.join(fires)
    agg["fire_rate"] = agg["total"] / agg["weeks"] * 100.0
    fby = (feats.groupby(["cell_id", "year"])["fire"].sum().unstack(fill_value=0)
           .reindex(columns=years, fill_value=0))
    fby_map = {cid: row.astype(int).tolist() for cid, row in fby.iterrows()}

    risk = surf["risk"].fillna(0).to_numpy()
    pct = (risk.argsort().argsort() / max(1, len(risk) - 1))
    surf = surf.assign(risk_pct=pct)

    # ---- per-cell records (no geometry) ----
    cells = []
    for _, r in surf.iterrows():
        cid = r["cell_id"]; a = agg.loc[cid] if cid in agg.index else None
        lc = int(round(r["landcover"])) if pd.notna(r.get("landcover")) else None
        rec = {
            "id": cid, "lon": _round(r["lon"], 3), "lat": _round(r["lat"], 3),
            "eco": r.get("ecoregion"), "risk": _round(r["risk"], 4), "risk_pct": _round(r["risk_pct"], 3),
            "elev": _round(r.get("elevation"), 0), "slope": _round(r.get("slope"), 1),
            "aspect": _round(r.get("aspect"), 0), "landcover": WORLDCOVER.get(lc, "—"),
            "fuel": _round(r.get("fuel_load"), 2),
            "fires_total": int(a["total"]) if a is not None else 0,
            "fires_rate": _round(a["fire_rate"], 2) if a is not None else 0,
            "fires_by_year": fby_map.get(cid, [0] * len(years)),
        }
        if a is not None:
            for c in WEATHER_MEANS:
                if c in a:
                    rec[c] = _round(a[c], 2)
        cells.append(rec)

    # ---- smooth surfaces ----
    from wildfire.ingest.boundary import oregon_gdf
    lons = surf["lon"].to_numpy(); lats = surf["lat"].to_numpy()
    geom = oregon_gdf(cfg).to_crs(4326).union_all()
    minx, miny, maxx, maxy = geom.bounds
    cellvals = {c["id"]: c for c in cells}
    surfaces = {"bounds": [[miny, minx], [maxy, maxx]], "metrics": {}}
    for key, label, unit, cmap, scale in SURFACE_METRICS:
        vals = np.array([surf_value(cellvals.get(cid), key) for cid in surf["cell_id"]], dtype=float)
        ok = np.isfinite(vals)
        png = site / "assets" / "surfaces" / f"{key}.png"
        dom, stops = _surface(lons[ok], lats[ok], vals[ok], geom, png, cmap)
        surfaces["metrics"][key] = {
            "png": f"assets/surfaces/{key}.png", "label": label, "unit": unit,
            "domain": [round(dom[0] * scale, 2), round(dom[1] * scale, 2)], "stops": stops,
        }

    # ---- ecoregion summaries ----
    eco_rows = []
    for eco, g in surf.groupby("ecoregion"):
        sub = feats[feats["cell_id"].isin(g["cell_id"])]
        eco_rows.append({
            "name": eco, "cells": int(len(g)),
            "mean_risk": _round(g["risk"].mean(), 4), "fires_total": int(sub["fire"].sum()),
            "fire_rate": _round(sub["fire"].mean() * 100, 3), "mean_elev": _round(g["elevation"].mean(), 0),
            "lon": _round(g["lon"].mean(), 3), "lat": _round(g["lat"].mean(), 3),
        })
    eco_rows.sort(key=lambda d: -d["mean_risk"])
    state_fby = feats.groupby("year")["fire"].sum().reindex(years, fill_value=0).astype(int).tolist()

    eco_geo = None
    try:
        eg = surf.to_crs(4326).dissolve("ecoregion").reset_index()[["ecoregion", "geometry"]]
        eg["geometry"] = eg.geometry.simplify(0.01)
        eco_geo = json.loads(eg.to_json())
    except Exception as exc:
        print(f"   ! ecoregion polygons skipped: {exc}")

    # ---- fire-district count forecasts (ODF) ----
    districts_geo, district_rows = None, None
    try:
        from wildfire.ingest.districts import district_count_layer
        dl = district_count_layer(cfg, features=feats)
        dg = dl["districts"].to_crs(4326)
        dg["pred_count"] = dg["pred_count"].round(0)
        district_rows = [
            {"district": r["district"], "pred": _round(r["pred_count"], 0),
             "lo": _round(r["pred_lo"], 0), "hi": _round(r["pred_hi"], 0)}
            for _, r in dg.sort_values("pred_count", ascending=False).iterrows()
        ]
        # simplify polygons for the web
        dg["geometry"] = dg.geometry.simplify(0.01)
        districts_geo = json.loads(dg[["district", "pred_count", "pred_lo", "pred_hi", "geometry"]].to_json())
    except Exception as exc:  # network/data
        print(f"   ! district layer skipped: {exc}")

    # ---- metrics / shap / cnn ----
    mdir = cfg.path_for("metrics")
    tab = json.loads((mdir / "tabular_metrics.json").read_text())
    cnn = json.loads((mdir / "cnn_metrics.json").read_text()) if (mdir / "cnn_metrics.json").exists() else None
    manifest = json.loads((cfg.path_for("data_interim") / "_manifest.json").read_text())

    def scheme_table(scheme):
        rows = []
        for name, res in tab["schemes"][scheme].items():
            a = res["aggregate"]
            rows.append({"model": name, "pr_auc": _round(a.get("pr_auc_mean"), 3),
                         "pr_lift": _round(a.get("pr_auc_lift_mean"), 1), "recall20": _round(a.get("recall_at_p20_mean"), 3),
                         "brier": _round(a.get("brier_mean"), 4), "roc_auc": _round(a.get("roc_auc_mean"), 3)})
        return rows

    shap = [{"feature": label_feature(d["feature"]), "code": d["feature"],
             "value": _round(d["mean_abs_shap"], 4)} for d in tab.get("shap_top15", [])]
    _regen_shap_figure(shap, site)

    meta = {
        "manifest": manifest, "years": years,
        "schemes": {s: scheme_table(s) for s in tab["schemes"]},
        "shap": shap,
        "count": {"negbin": tab["count"]["negbin"]["aggregate"], "districts": district_rows},
        "districts_geo": districts_geo, "ecoregions_geo": eco_geo,
        "cnn": cnn["test_metrics"] if cnn else None, "cnn_backbone": cnn["backbone"] if cnn else None,
        "ecoregions": eco_rows, "state_fires_by_year": state_fby, "surfaces": surfaces,
    }

    (site / "data" / "cells.js").write_text(
        "window.WF_CELLS=" + json.dumps({"years": years, "cells": cells}, separators=(",", ":")) + ";", encoding="utf-8")
    (site / "data" / "meta.js").write_text(
        "window.WF_META=" + json.dumps(meta, separators=(",", ":")) + ";", encoding="utf-8")
    for fig in ("risk_map.png",):
        src = cfg.path_for("figures") / fig
        if src.exists():
            shutil.copy(src, site / "assets" / fig)

    return {"cells": len(cells), "years": years, "districts": len(district_rows or []),
            "cells_kb": (site / "data" / "cells.js").stat().st_size // 1024,
            "surfaces": list(surfaces["metrics"].keys())}


def surf_value(rec, key):
    if rec is None:
        return np.nan
    v = rec.get(key)
    return np.nan if v is None else v


def _regen_shap_figure(shap, site):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = list(reversed(shap[:15]))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh([d["feature"] for d in top], [d["value"] for d in top], color="#ff7a18")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Risk model — feature importance")
    fig.tight_layout()
    fig.savefig(site / "assets" / "shap_importance.png", dpi=150)
    plt.close(fig)


def main() -> int:
    init_console()
    cfg = load_config()
    info = build(cfg)
    print("✅ Site data built")
    print(f"   cells       : {info['cells']:,}  ({info['cells_kb']} KB, no geometry)")
    print(f"   surfaces    : {', '.join(info['surfaces'])}")
    print(f"   districts   : {info['districts']} ODF fire-protection districts")
    print(f"   years       : {info['years'][0]}–{info['years'][-1]}")
    print("   open        : site/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
