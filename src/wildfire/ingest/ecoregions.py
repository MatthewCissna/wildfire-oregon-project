"""Oregon ecoregions — the natural region unit for the fire-count model and a
realistic driver of climate / fuel / ignition-cause patterns.

Two providers, same output (an ``ecoregion`` column on the grid):

* :func:`assign_oregon_ecoregions` — **offline, deterministic** zones that
  approximate EPA Level III ecoregions from longitude + an elevation proxy
  (Coast Range, Willamette Valley, West/East Cascades, Klamath Mountains, Blue
  Mountains, Columbia Plateau, Northern Basin and Range). Good enough to give the
  synthetic demo real Oregon structure without a network dependency.
* :func:`load_epa_level3` — spatial-joins the **real** EPA Level III shapefile if
  the user drops it in ``data/raw`` (or passes a path). Use this on the live path.

These are honest approximations for the synthetic demo, not authoritative
boundaries — swap in :func:`load_epa_level3` for real analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)

# Ordered list of the Oregon ecoregions we model (coarse Level-III-style).
OREGON_ECOREGIONS = [
    "Coast Range",
    "Klamath Mountains",
    "Willamette Valley",
    "West Cascades",
    "East Cascades",
    "Columbia Plateau",
    "Blue Mountains",
    "Northern Basin and Range",
]


def _elevation_proxy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Coarse elevation field (m) — Cascade crest + Coast Range bumps."""
    cascades = 1600.0 * np.exp(-(((lon + 121.7) / 0.7) ** 2))
    coast = 600.0 * np.exp(-(((lon + 123.6) / 0.4) ** 2))
    return 150.0 + cascades + coast


def assign_oregon_ecoregions(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add an ``ecoregion`` column to the grid from lon/lat + elevation proxy."""
    lon = grid["lon"].to_numpy()
    lat = grid["lat"].to_numpy()
    elev = _elevation_proxy(lon, lat)

    conditions = [
        lon < -123.5,                                                   # Coast Range
        (lat < 43.4) & (lon < -122.1),                                  # Klamath Mountains
        (lon >= -123.5) & (lon < -122.3) & (lat >= 43.4) & (lat < 45.8) & (elev < 380),  # Willamette
        (elev > 900) & (lon >= -122.4) & (lon < -121.6),               # West Cascades
        (elev > 650) & (lon >= -121.9) & (lon < -120.6),               # East Cascades
        (lat >= 45.0) & (lon >= -121.0) & (lon < -118.6) & (elev < 950),  # Columbia Plateau
        (lon >= -119.4) & (lat >= 44.2),                               # Blue Mountains
    ]
    choices = [
        "Coast Range", "Klamath Mountains", "Willamette Valley",
        "West Cascades", "East Cascades", "Columbia Plateau", "Blue Mountains",
    ]
    grid = grid.copy()
    grid["ecoregion"] = np.select(conditions, choices, default="Northern Basin and Range")
    return grid


def load_epa_level3(
    grid: gpd.GeoDataFrame, cfg: Config | None = None, path: str | Path | None = None
) -> gpd.GeoDataFrame:
    """Spatial-join real EPA Level III ecoregions onto the grid, if available.

    Looks for a shapefile/GeoPackage in ``data/raw`` (e.g. ``or_eco_l3``). Falls back
    to :func:`assign_oregon_ecoregions` when no file is found.
    Download: https://www.epa.gov/eco-research/ecoregion-download-files-state-region-10
    """
    cfg = cfg or load_config()
    if path is None:
        raw = cfg.path_for("data_raw")
        for pat in ("*eco_l3*.shp", "*ecoregion*.shp", "*eco_l3*.gpkg", "*ecoregion*.gpkg"):
            hits = list(raw.glob(pat))
            if hits:
                path = hits[0]
                break
    if path is None or not Path(path).exists():
        logger.info("No EPA ecoregion file found; using offline zones.")
        return assign_oregon_ecoregions(grid)

    eco = gpd.read_file(path).to_crs(grid.crs)
    name_col = next(
        (c for c in ("US_L3NAME", "L3_KEY", "NA_L3NAME", "name") if c in eco.columns),
        eco.columns[0],
    )
    eco = eco[[name_col, eco.geometry.name]].rename(columns={name_col: "ecoregion"})
    centroids = grid.copy()
    centroids["geometry"] = grid.geometry.representative_point()
    joined = gpd.sjoin(centroids, eco, predicate="within", how="left").drop(columns="index_right")
    grid = grid.copy()
    grid["ecoregion"] = joined["ecoregion"].fillna("Unknown").to_numpy()
    return grid
