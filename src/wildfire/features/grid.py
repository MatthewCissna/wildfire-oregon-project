"""The unified spatial grid.

All layers are aggregated onto this grid. Two kinds are supported:

* **H3 hexagons** (default) — equal-ish area, no projection seams, and the parent
  cell at a coarser resolution gives us a natural, contiguous **spatial block** for
  leave-one-block-out cross-validation (the thing weak models get wrong).
* **Square km cells** — simple regular grid in the projected CRS.

The grid is the join key between weather, vegetation, topography, and fire labels.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from wildfire.config import Config, load_config
from wildfire.ingest.boundary import oregon_gdf

# How many resolutions coarser the spatial "block" is than the working grid.
# res 6 grid -> res 3 blocks (~12k km² each): big enough to be a real held-out region.
_BLOCK_RES_DELTA = 3


def _h3_cell_polygon(cell: str) -> Polygon:
    """Shapely polygon (lon/lat) for an H3 cell."""
    boundary = h3.cell_to_boundary(cell)  # list of (lat, lon)
    return Polygon([(lon, lat) for lat, lon in boundary])


def build_h3_grid(boundary: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    """Fill the boundary with H3 cells at ``resolution``."""
    geom = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    cells = h3.geo_to_cells(geom, resolution)

    block_res = max(0, resolution - _BLOCK_RES_DELTA)
    records = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        records.append(
            {
                "cell_id": cell,
                "lon": lon,
                "lat": lat,
                "block_id": h3.cell_to_parent(cell, block_res),
                "geometry": _h3_cell_polygon(cell),
            }
        )
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def build_square_grid(
    boundary: gpd.GeoDataFrame, cell_km: float, crs_projected: str
) -> gpd.GeoDataFrame:
    """Regular square grid (in a projected CRS) clipped to the boundary."""
    proj = boundary.to_crs(crs_projected)
    minx, miny, maxx, maxy = proj.total_bounds
    step = cell_km * 1000.0
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)

    polys, ids = [], []
    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            polys.append(Polygon([(x, y), (x + step, y), (x + step, y + step), (x, y + step)]))
            ids.append(f"sq_{i:04d}_{j:04d}")
    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=polys, crs=crs_projected)
    # Keep only cells intersecting Oregon.
    grid = gpd.sjoin(grid, proj[[proj.geometry.name]], predicate="intersects", how="inner")
    grid = grid.drop(columns=[c for c in grid.columns if c.startswith("index_")])
    grid = grid.to_crs("EPSG:4326")
    grid["lon"] = grid.geometry.centroid.x
    grid["lat"] = grid.geometry.centroid.y
    # Coarse block via rounding centroids into ~1° bins.
    grid["block_id"] = (
        grid["lon"].round(0).astype(int).astype(str)
        + "_"
        + grid["lat"].round(0).astype(int).astype(str)
    )
    return grid.reset_index(drop=True)


def build_grid(cfg: Config | None = None, boundary: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Build the configured grid as a GeoDataFrame.

    Columns: ``cell_id, lon, lat, block_id, geometry``. ``block_id`` is the
    spatial-block key used for leave-one-block-out CV and region aggregation.
    """
    cfg = cfg or load_config()
    boundary = boundary if boundary is not None else oregon_gdf(cfg)
    kind = cfg.get("grid.kind", "h3")
    if kind == "h3":
        grid = build_h3_grid(boundary, int(cfg.get("grid.h3_resolution", 6)))
    elif kind == "square_km":
        grid = build_square_grid(
            boundary, float(cfg.get("grid.square_km", 5.0)), cfg.get("region.crs_projected")
        )
    else:
        raise ValueError(f"Unknown grid.kind: {kind!r}")
    grid = grid.sort_values("cell_id").reset_index(drop=True)

    # Attach Oregon ecoregions (the natural region unit + a realistic driver).
    from wildfire.ingest.ecoregions import assign_oregon_ecoregions

    grid = assign_oregon_ecoregions(grid)
    return grid


def time_index(cfg: Config | None = None) -> pd.DatetimeIndex:
    """Time steps for the spatiotemporal panel, per ``time.step`` in config."""
    cfg = cfg or load_config()
    start, end = cfg.get("time.start"), cfg.get("time.end")
    step = cfg.get("time.step", "weekly")
    freq = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}.get(step, "W-MON")
    return pd.date_range(start=start, end=end, freq=freq)
