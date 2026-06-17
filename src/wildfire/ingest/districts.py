"""Oregon fire-protection districts (ODF) — the natural unit for fire-count reporting.

Fetches the 12 Oregon Department of Forestry Forest Protection Districts from the
ODF ArcGIS service, caches them, and assigns grid cells to districts by spatial join.
Cells outside ODF protection (federal / rangeland) are labeled accordingly.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon, shape

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)

_ODF_DISTRICTS_URL = (
    "https://gis.odf.oregon.gov/ags1/rest/services/Applications/ProtectionPDFs/"
    "MapServer/4/query"
)
OUTSIDE = "Outside ODF protection"


def _esri_to_shape(geom: dict):
    """Convert an Esri JSON polygon (rings) to a shapely geometry."""
    rings = [r for r in geom.get("rings", [])]
    if not rings:
        return None
    # Esri doesn't separate outer/holes by winding reliably for our purpose; treat each
    # ring as a polygon and union via MultiPolygon (good enough for district extents).
    polys = [Polygon(r) for r in rings if len(r) >= 4]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon([p for p in polys])


def load_districts(cfg: Config | None = None, *, cache: bool = True) -> gpd.GeoDataFrame:
    """ODF Forest Protection Districts as a GeoDataFrame (district, geometry), EPSG:4326."""
    cfg = cfg or load_config()
    cache_path = cfg.path_for("data_interim") / "odf_districts.gpkg"
    if cache and cache_path.exists():
        return gpd.read_file(cache_path)

    params = {
        "where": "1=1", "outFields": "ODF_FPD", "outSR": "4326",
        "returnGeometry": "true", "f": "geojson",
    }
    gdf = None
    try:
        r = requests.get(_ODF_DISTRICTS_URL, params=params, timeout=90)
        r.raise_for_status()
        gj = r.json()
        feats = gj.get("features", [])
        if feats and "geometry" in feats[0] and feats[0]["geometry"].get("type"):
            gdf = gpd.GeoDataFrame(
                {"district": [f["properties"].get("ODF_FPD") for f in feats]},
                geometry=[shape(f["geometry"]) for f in feats], crs="EPSG:4326",
            )
    except Exception as exc:  # fall back to Esri-JSON parsing
        logger.warning("GeoJSON district fetch failed (%s); trying Esri JSON.", exc)

    if gdf is None:
        params["f"] = "json"
        r = requests.get(_ODF_DISTRICTS_URL, params=params, timeout=90)
        r.raise_for_status()
        feats = r.json().get("features", [])
        rows, geoms = [], []
        for f in feats:
            g = _esri_to_shape(f.get("geometry", {}))
            if g is not None:
                rows.append(f["attributes"].get("ODF_FPD"))
                geoms.append(g)
        gdf = gpd.GeoDataFrame({"district": rows}, geometry=geoms, crs="EPSG:4326")

    gdf = gdf[gdf.geometry.notna() & gdf["district"].notna()].dissolve("district").reset_index()
    if cache:
        try:
            gdf.to_file(cache_path, driver="GPKG")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not cache districts: %s", exc)
    logger.info("Loaded %d ODF fire-protection districts", len(gdf))
    return gdf


def assign_districts(grid: gpd.GeoDataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Return a DataFrame[cell_id, district] assigning each cell to its ODF district."""
    cfg = cfg or load_config()
    districts = load_districts(cfg)
    cells = grid[["cell_id", "geometry"]].copy()
    cells["geometry"] = cells.geometry.representative_point()
    joined = gpd.sjoin(cells, districts, predicate="within", how="left").drop(columns="index_right")
    joined = joined[~joined.index.duplicated(keep="first")]
    out = grid[["cell_id"]].copy()
    out["district"] = joined["district"].fillna(OUTSIDE).to_numpy()
    return out


def district_count_layer(cfg: Config | None = None, features=None) -> dict:
    """Fit the fire-count model per ODF district and attach predictions to polygons.

    Returns ``{districts: GeoDataFrame[district, pred_count, pred_lo, pred_hi, geometry],
    predictions: DataFrame, cv: dict}``. The model is fit over all cells (including the
    unprotected area) for proper rates; only the 12 ODF districts are returned as polygons.
    """
    import pandas as pd

    from wildfire.features.regions import build_region_season, region_feature_columns
    from wildfire.models import count

    cfg = cfg or load_config()
    if features is None:
        features = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")

    grid = gpd.read_parquet(cfg.path_for("data_processed") / "risk_surface.parquet")
    dmap = assign_districts(grid, cfg).set_index("cell_id")["district"]

    region = build_region_season(cfg, features, region_map=dmap)
    rcols = region_feature_columns(region)
    cv = count.cross_validate_count(region, rcols, cfg, kind="negbin")
    model = count.train_count(region, rcols, cfg, kind="negbin")

    latest = region[region["year"] == region["year"].max()].copy()
    latest["pred_count"] = model.predict(latest)
    lo, hi = model.predict_interval(latest)
    latest["pred_lo"], latest["pred_hi"] = lo, hi

    districts = load_districts(cfg).merge(
        latest[["region", "pred_count", "pred_lo", "pred_hi"]],
        left_on="district", right_on="region", how="left",
    ).drop(columns="region")
    return {"districts": districts, "predictions": latest, "cv": cv}
