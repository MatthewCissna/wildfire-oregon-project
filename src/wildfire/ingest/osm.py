"""OpenStreetMap human-ignition proxies: distance to roads and power lines.

Human-caused ignitions cluster near infrastructure. We pull major roads and power
lines from OSM (cached), then compute each grid cell's distance to the nearest of
each — features that complement NIFC's ignition-cause records.

Heavy network/CPU; results are cached to ``data/interim``. The synthetic path
already provides equivalent ``dist_road_km`` / ``dist_power_km`` columns, so this
is only needed on the live data path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from wildfire.config import Config, load_config
from wildfire.ingest.boundary import oregon_gdf

logger = logging.getLogger(__name__)


def _fetch_osm_features(polygon, tags: dict, cache: Path) -> gpd.GeoDataFrame:
    if cache.exists():
        return gpd.read_file(cache)
    import osmnx as ox

    gdf = ox.features_from_polygon(polygon, tags)
    gdf = gdf[gdf.geometry.notna()]
    # Keep only line-like geometries (roads/power lines).
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf = gdf[[gdf.geometry.name]].reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_file(cache, driver="GPKG")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not cache OSM features: %s", exc)
    return gdf


def distance_features(cfg: Config | None = None, grid: gpd.GeoDataFrame | None = None) -> "gpd.GeoDataFrame":
    """Per-cell ``dist_road_km`` and ``dist_power_km`` via nearest-feature distance."""
    cfg = cfg or load_config()
    if grid is None:
        from wildfire.features.grid import build_grid

        grid = build_grid(cfg)

    boundary = oregon_gdf(cfg)
    poly = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    interim = cfg.path_for("data_interim")

    roads = _fetch_osm_features(
        poly,
        {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary"]},
        interim / "osm_roads.gpkg",
    )
    power = _fetch_osm_features(
        poly, {"power": ["line", "minor_line"]}, interim / "osm_power.gpkg"
    )

    crs = cfg.get("region.crs_projected")
    cells = grid[["cell_id", "geometry"]].copy().to_crs(crs)
    cells["geometry"] = cells.geometry.centroid

    def _nearest_km(feats: gpd.GeoDataFrame) -> np.ndarray:
        if feats is None or feats.empty:
            return np.full(len(cells), np.nan)
        feats = feats.to_crs(crs)
        joined = gpd.sjoin_nearest(cells, feats, how="left", distance_col="_d")
        joined = joined[~joined.index.duplicated(keep="first")]
        return (joined["_d"].to_numpy() / 1000.0)

    out = grid[["cell_id"]].copy()
    out["dist_road_km"] = _nearest_km(roads)
    out["dist_power_km"] = _nearest_km(power)
    out["human_proxy"] = np.exp(-out["dist_road_km"].fillna(out["dist_road_km"].max()) / 25.0)
    return out
