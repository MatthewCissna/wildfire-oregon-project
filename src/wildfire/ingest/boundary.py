"""Oregon study-area boundary.

Provides the boundary three ways:
  * ``oregon_ee_geometry`` — an ``ee.Geometry`` (TIGER states) for clipping GEE pulls.
  * ``oregon_gdf`` — a GeoDataFrame for local geometry ops (read from cache/URL/bbox).
  * ``oregon_bbox_polygon`` — a shapely box from the config bbox (always available, offline).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)

# US Census cartographic boundary (states), 1:5,000,000 — small and stable.
_US_STATES_GEOJSON = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/"
    "us-states.json"
)


def oregon_bbox_polygon(cfg: Config | None = None):
    """Shapely polygon from the configured bounding box (offline, always works)."""
    cfg = cfg or load_config()
    minx, miny, maxx, maxy = cfg.get("region.bbox")
    return box(minx, miny, maxx, maxy)


def oregon_gdf(cfg: Config | None = None, *, cache: bool = True) -> gpd.GeoDataFrame:
    """Oregon boundary as a single-row GeoDataFrame in EPSG:4326.

    Resolution order: local cache -> public states GeoJSON -> config bbox rectangle.
    The bbox fallback keeps the synthetic pipeline fully offline.
    """
    cfg = cfg or load_config()
    cache_path = cfg.path_for("data_interim") / "oregon_boundary.gpkg"

    if cache and cache_path.exists():
        return gpd.read_file(cache_path)

    gdf: gpd.GeoDataFrame | None = None
    try:
        states = gpd.read_file(_US_STATES_GEOJSON)
        name_col = "name" if "name" in states.columns else states.columns[0]
        oregon = states[states[name_col].astype(str).str.lower() == "oregon"]
        if len(oregon):
            gdf = oregon.dissolve().to_crs(cfg.get("region.crs_geographic"))
            gdf = gdf[[gdf.geometry.name]].reset_index(drop=True)
            gdf["name"] = "Oregon"
    except Exception as exc:  # network/parse issues -> fall back to bbox
        logger.warning("Could not fetch Oregon boundary (%s); using bbox fallback.", exc)

    if gdf is None:
        gdf = gpd.GeoDataFrame(
            {"name": ["Oregon"]},
            geometry=[oregon_bbox_polygon(cfg)],
            crs=cfg.get("region.crs_geographic"),
        )

    if cache:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            gdf.to_file(cache_path, driver="GPKG")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not cache boundary: %s", exc)
    return gdf


def oregon_ee_geometry(cfg: Config | None = None):
    """Oregon boundary as an ``ee.Geometry`` from the TIGER 2018 states layer.

    Requires Earth Engine to be initialized by the caller.
    """
    import ee

    cfg = cfg or load_config()
    states = ee.FeatureCollection("TIGER/2018/States")
    oregon = states.filter(ee.Filter.eq("STUSPS", "OR"))
    return oregon.geometry()
