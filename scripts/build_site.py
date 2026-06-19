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

# Oregon cities used for context on the risk map (name, lat, lon, ~population).
# A focused list across all major regions; rendered as clickable markers.
OREGON_CITIES = [
    ("Portland", 45.523, -122.676, 652000), ("Salem", 44.943, -123.035, 175000),
    ("Eugene", 44.052, -123.087, 175000), ("Bend", 44.058, -121.315, 99000),
    ("Medford", 42.327, -122.873, 86000), ("Corvallis", 44.564, -123.262, 59000),
    ("Springfield", 44.046, -123.022, 60000), ("Albany", 44.636, -123.106, 56000),
    ("Hillsboro", 45.523, -122.989, 106000), ("Beaverton", 45.487, -122.804, 98000),
    ("Gresham", 45.498, -122.430, 113000), ("Tualatin", 45.384, -122.763, 27000),
    ("Klamath Falls", 42.225, -121.781, 21000), ("Roseburg", 43.217, -123.342, 23000),
    ("Grants Pass", 42.439, -123.328, 39000), ("Ashland", 42.195, -122.709, 21000),
    ("Pendleton", 45.672, -118.789, 17000), ("La Grande", 45.324, -118.087, 13000),
    ("Baker City", 44.775, -117.835, 10000), ("Ontario", 44.026, -116.963, 12000),
    ("Burns", 43.586, -118.995, 2700), ("Lakeview", 42.189, -120.345, 2400),
    ("John Day", 44.416, -118.954, 1700), ("Madras", 44.633, -121.130, 7000),
    ("Prineville", 44.300, -120.834, 12000), ("Hermiston", 45.840, -119.289, 19000),
    ("The Dalles", 45.595, -121.179, 16000), ("Hood River", 45.706, -121.521, 8000),
    ("Astoria", 46.188, -123.831, 10000), ("Tillamook", 45.456, -123.844, 5300),
    ("Newport", 44.638, -124.052, 10000), ("Coos Bay", 43.366, -124.218, 16000),
    ("Brookings", 42.053, -124.284, 6900), ("Florence", 43.983, -124.099, 9300),
    ("Sisters", 44.291, -121.549, 3300), ("Redmond", 44.272, -121.174, 36000),
]


def nearest_city(lon: float, lat: float) -> tuple[str, float]:
    """Return (city_name, distance_km) for the closest Oregon city."""
    best, best_d = None, float("inf")
    for name, clat, clon, _pop in OREGON_CITIES:
        dx = (lon - clon) * np.cos(np.radians((lat + clat) / 2))
        dy = (lat - clat)
        d = np.sqrt(dx * dx + dy * dy) * 111.0  # deg -> km
        if d < best_d:
            best_d, best = d, name
    return best, round(float(best_d), 1)
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

    # ---- nearest city (vectorized, computed once) ----
    cnames = np.array([c[0] for c in OREGON_CITIES])
    clats = np.array([c[1] for c in OREGON_CITIES])
    clons = np.array([c[2] for c in OREGON_CITIES])
    cell_lons = surf["lon"].to_numpy(); cell_lats = surf["lat"].to_numpy()
    midlat = (cell_lats[:, None] + clats[None, :]) / 2
    dx = (cell_lons[:, None] - clons[None, :]) * np.cos(np.radians(midlat))
    dy = (cell_lats[:, None] - clats[None, :])
    dist_km = np.sqrt(dx * dx + dy * dy) * 111.0
    nidx = dist_km.argmin(axis=1)
    near_city = cnames[nidx]
    near_dist = dist_km[np.arange(len(nidx)), nidx]

    # ---- per-cell records (no geometry) ----
    cells = []
    for i, (_, r) in enumerate(surf.iterrows()):
        cid = r["cell_id"]; a = agg.loc[cid] if cid in agg.index else None
        lc = int(round(r["landcover"])) if pd.notna(r.get("landcover")) else None
        rec = {
            "id": cid, "lon": _round(r["lon"], 3), "lat": _round(r["lat"], 3),
            "eco": r.get("ecoregion"), "risk": _round(r["risk"], 4), "risk_pct": _round(r["risk_pct"], 3),
            "elev": _round(r.get("elevation"), 0), "slope": _round(r.get("slope"), 1),
            "aspect": _round(r.get("aspect"), 0), "landcover": WORLDCOVER.get(lc, "—"),
            "fuel": _round(r.get("fuel_load"), 2),
            "near_city": str(near_city[i]),
            "near_km": _round(near_dist[i], 1),
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

    # ---- weekly forecast (next-season climatological prediction) ----
    print("   generating weekly forecast (this takes ~1 min) ...")
    forecast = build_forecast(cfg, feats, surf, geom, site, target_year=2025,
                              cell_order=[c["id"] for c in cells])
    # Lock the predictions to disk so the Tracker tab can compare against actuals later.
    # Preserve a prior lock time and any pulled actuals for the same year — a rebuild
    # of the site must not silently reset the lock or discard verified actuals.
    locked_path = site / "data" / "predictions.json"
    prior = json.loads(locked_path.read_text()) if locked_path.exists() else {}
    same_year = prior.get("target_year") == forecast["target_year"]
    locked_path.write_text(json.dumps({
        "locked_at_utc": prior["locked_at_utc"] if (same_year and prior.get("locked_at_utc"))
        else pd.Timestamp.utcnow().isoformat(),
        "target_year": forecast["target_year"],
        "predicted": forecast["predicted"],
        "actuals": prior.get("actuals") if same_year else None,
    }, indent=2))

    meta = {
        "manifest": manifest, "years": years,
        "schemes": {s: scheme_table(s) for s in tab["schemes"]},
        "shap": shap,
        "count": {"negbin": tab["count"]["negbin"]["aggregate"], "districts": district_rows},
        "districts_geo": districts_geo, "ecoregions_geo": eco_geo,
        "cnn": cnn["test_metrics"] if cnn else None, "cnn_backbone": cnn["backbone"] if cnn else None,
        "ecoregions": eco_rows, "state_fires_by_year": state_fby, "surfaces": surfaces,
        "hexes": build_hex_layer(cells, SURFACE_METRICS),
        "cities": [{"name": n, "lat": lat, "lon": lon, "pop": pop} for n, lat, lon, pop in OREGON_CITIES],
        "forecast": forecast,
        "predictions": json.loads((site / "data" / "predictions.json").read_text())
        if (site / "data" / "predictions.json").exists() else None,
    }

    (site / "data" / "cells.js").write_text(
        "window.WF_CELLS=" + json.dumps({"years": years, "cells": cells}, separators=(",", ":")) + ";", encoding="utf-8")
    # Predictions live in their own file so the scheduled CI job can refresh ONLY
    # this small file (no model / no heavy data needed) and the site picks up new actuals.
    predictions_blob = meta.pop("predictions", None)
    (site / "data" / "meta.js").write_text(
        "window.WF_META=" + json.dumps(meta, separators=(",", ":")) + ";", encoding="utf-8")
    if predictions_blob is not None:
        (site / "data" / "predictions.js").write_text(
            "window.WF_PREDICTIONS=" + json.dumps(predictions_blob, separators=(",", ":")) + ";",
            encoding="utf-8")
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


def build_hex_layer(cells, metrics):
    """True H3 hexagon outlines for the map, aligned by index to ``WF_CELLS.cells``.

    The native grid IS H3 hexagons, so every cell is drawn as its own hexagon and a
    click lands on exactly one cell — no re-binning, no nearest-neighbour guessing.
    For each cell we ship the six boundary vertices (lat, lon). Colouring is done in
    the browser straight from the raw per-cell metric values, normalised against the
    same 2nd–98th percentile range the legend uses, so we only need to ship those
    bounds once per metric rather than a value per cell.

    Returns ``{metrics, lohi:{metric:[lo,hi]}, poly:[[[lat,lon]*6], ...]}``.
    """
    import h3

    keys = [m[0] for m in metrics]
    raw = {k: np.array([surf_value(c, k) for c in cells], float) for k in keys}
    lohi = {}
    for k in keys:
        v = raw[k][np.isfinite(raw[k])]
        lo, hi = (float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98))) if len(v) else (0.0, 1.0)
        lohi[k] = [round(lo, 5), round(hi, 5)]

    poly = []
    for c in cells:
        boundary = h3.cell_to_boundary(c["id"])  # [(lat, lon), ...] six vertices
        poly.append([[round(lat, 4), round(lon, 4)] for lat, lon in boundary])
    return {"metrics": keys, "lohi": lohi, "poly": poly}


def _week_label(d):
    return d.strftime("%b %d")


def build_forecast(cfg, feats, surf, geom, site, target_year=2025, cell_order=None):
    """Predict next-season weekly risk for every cell, with per-cell hex data + summary.

    Approach: for each fire-season week of ``target_year`` we use **per-cell
    climatology** of every input feature (the historical mean for that
    week-of-year across all training years), feed it through the trained risk model,
    and read the resulting cell-wise risk. This is an "expected risk under typical
    conditions for week W" forecast — the only leverage we have without a real
    weather forecast, and it tracks the seasonal arc clearly. Plus precomputed
    totals (state + per district).

    For the map we ship the per-cell risk of every week as small integers (0–100,
    normalised against the season-wide max so weeks compare directly), aligned by
    index to ``cell_order`` (the same order as ``WF_CELLS.cells`` / the hex layer),
    so the Forecast tab can recolour the real hexagons week by week and a click
    lands on exactly one cell.
    """
    from wildfire.features.build import feature_columns
    from wildfire.models import risk as risk_mod

    rm = risk_mod.RiskModel.load(cfg.path_for("models") / "risk_model.joblib")
    cols = rm.feature_cols
    df = feats[[*cols, "cell_id", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)

    # Per (cell, week-of-year) climatology of every model input feature.
    clim = df.groupby(["cell_id", "woy"])[cols].mean().reset_index()

    # Build the forecast date list: weekly Mondays of the target year's fire season.
    season_months = set(cfg.get("time.fire_season_months", [5, 6, 7, 8, 9, 10]))
    all_mondays = pd.date_range(f"{target_year}-04-28", f"{target_year}-11-10", freq="W-MON")
    forecast_dates = [d for d in all_mondays if d.month in season_months]

    cells_df = surf[["cell_id", "lon", "lat", "ecoregion"]].copy()

    # Map each cell id to its column in the shipped hex layer (same order as the
    # site's WF_CELLS.cells), so the per-week risk arrays line up with the hexes.
    pos = {cid: i for i, cid in enumerate(cell_order)} if cell_order else {}
    n_cells = len(cell_order) if cell_order else 0

    weekly = []  # one entry per forecast week
    weeks_hex = []  # per-week list of n_cells ints 0-100 (risk / season-max)
    cell_risk_acc = {cid: 0.0 for cid in cells_df["cell_id"]}
    cell_risk_n = 0

    # District map (best-effort) for per-district aggregation
    try:
        from wildfire.ingest.districts import assign_districts
        dmap = assign_districts(surf, cfg).set_index("cell_id")["district"].to_dict()
    except Exception:
        dmap = {}

    # Cap a global colormap to the max across all weeks so PNGs compare directly.
    week_scores = {}
    for d in forecast_dates:
        woy = int(pd.Timestamp(d).isocalendar().week)
        snap = clim[clim["woy"] == woy]
        if snap.empty:
            continue
        snap = snap.merge(cells_df, on="cell_id", how="inner")
        # Override calendar features to the forecast week (cyclic seasonality).
        doy = d.timetuple().tm_yday
        snap["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        snap["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        if "month" in snap.columns:
            snap["month"] = d.month
        scores = rm.predict_risk(snap)
        week_scores[d] = (snap["cell_id"].to_numpy(), snap["lon"].to_numpy(),
                          snap["lat"].to_numpy(), scores)

    vmax_global = max((s[3].max() for s in week_scores.values()), default=1.0)
    for d, (cids, lons, lats, scores) in week_scores.items():
        col = [0] * n_cells  # this week's per-cell risk for the hex map
        dist_exp = {}
        for cid, s in zip(cids, scores):
            cell_risk_acc[cid] = cell_risk_acc.get(cid, 0.0) + float(s)
            j = pos.get(cid)
            if j is not None:
                col[j] = int(round(float(s) / (vmax_global + 1e-12) * 100))
            dist = dmap.get(cid, "Outside ODF protection")
            dist_exp[dist] = dist_exp.get(dist, 0.0) + float(s)
        cell_risk_n += 1
        weeks_hex.append(col)
        weekly.append({
            "date": d.strftime("%Y-%m-%d"),
            "label": _week_label(d),
            "expected_state": round(float(scores.sum()), 1),
            "max_risk": round(float(scores.max()) * 100, 2),
            "mean_risk": round(float(scores.mean()) * 100, 3),
            "district_expected": {k: round(v, 1) for k, v in dist_exp.items()},
        })

    # Mean per-cell risk across the season → ranked predictions (locked).
    mean_risk = {cid: cell_risk_acc[cid] / max(1, cell_risk_n) for cid in cells_df["cell_id"]}
    cells_df["pred_risk"] = cells_df["cell_id"].map(mean_risk)
    cells_df["pct"] = cells_df["pred_risk"].rank(pct=True)

    # Per-cell seasonal mean as 0-100 ints (normalised to the season max) for the
    # Tracker map, aligned to the hex layer / cell order.
    season_max = max(mean_risk.values()) if mean_risk else 1.0
    season_hex = [0] * n_cells
    for cid, v in mean_risk.items():
        j = pos.get(cid)
        if j is not None:
            season_hex[j] = int(round(float(v) / (season_max + 1e-12) * 100))

    top_cells = []
    for _, r in cells_df.sort_values("pred_risk", ascending=False).head(40).iterrows():
        nc, nd = nearest_city(r["lon"], r["lat"])
        top_cells.append({
            "id": r["cell_id"], "lon": _round(r["lon"], 3), "lat": _round(r["lat"], 3),
            "eco": r["ecoregion"], "near_city": nc, "near_km": nd,
            "pred_risk": round(float(r["pred_risk"]) * 100, 3),
        })
    state_total = float(sum(w["expected_state"] for w in weekly))
    district_total = {}
    for w in weekly:
        for k, v in w["district_expected"].items():
            district_total[k] = district_total.get(k, 0.0) + v
    district_total = sorted(
        ({"district": k, "expected_fires": round(v, 0)} for k, v in district_total.items()),
        key=lambda d: -d["expected_fires"],
    )

    forecast_info = {
        "target_year": target_year,
        "n_weeks": len(weekly),
        "weeks": weekly,
        "next_week": weekly[0] if weekly else None,
        "vmax_pct": round(float(vmax_global) * 100, 2),
        "hex": {
            # weeks[w][cellIdx] and season[cellIdx] are 0-100 ints aligned to the
            # hex layer; multiply by the matching *_pct max to recover a risk %.
            "vmax_pct": round(float(vmax_global) * 100, 2),
            "season_max_pct": round(float(season_max) * 100, 3),
            "weeks": weeks_hex,
            "season": season_hex,
        },
        "predicted": {
            "state_expected_fires": round(state_total, 0),
            "districts": district_total,
            "top_cells": top_cells,
            "weekly_curve": [{"date": w["date"], "label": w["label"],
                              "expected": w["expected_state"]} for w in weekly],
        },
    }
    return forecast_info


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
