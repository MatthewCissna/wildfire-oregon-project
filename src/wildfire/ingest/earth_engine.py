"""Earth Engine data pulls, reduced onto the unified grid.

This module mirrors the synthetic generator's output contract: it produces the
same ``grid / static / weather_panel / fire_events`` tables, but from live GEE
datasets (documented in ``docs/data_sources.md``). It is written to the standard
``reduceRegions`` pattern and is fully turnkey once Earth Engine is authenticated.

Strategy
--------
* Convert the H3 grid to an ``ee.FeatureCollection`` of polygons keyed by ``cell_id``.
* **Static** layers (elevation, slope, aspect, land cover, NDVI climatology) are
  reduced **once** with ``reduceRegions``.
* **Weather** (GRIDMET) and **fire labels** (MCD64A1 burned area, FIRMS/MOD14
  active fire) are composited per time step and reduced per step, then stacked.

For full-Oregon, full-history pulls, batch ``Export.table.toDrive`` is recommended
(see :func:`export_weather_panel`); the in-memory :func:`pull_weather_panel` path
is fine for samples and for the ``--quick`` smoke run.
"""

from __future__ import annotations

import logging

import pandas as pd

from wildfire.config import Config, load_config
from wildfire.features.grid import build_grid, time_index
from wildfire.ingest.boundary import oregon_ee_geometry
from wildfire.ingest.ee_auth import initialize_ee

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Grid <-> Earth Engine
# --------------------------------------------------------------------------- #
def grid_to_ee(grid):
    """Convert the grid GeoDataFrame to an ``ee.FeatureCollection`` (cell_id keyed)."""
    import ee

    feats = []
    for row in grid.itertuples(index=False):
        geom = row.geometry
        coords = [list(map(list, geom.exterior.coords))]
        feats.append(ee.Feature(ee.Geometry.Polygon(coords), {"cell_id": row.cell_id}))
    return ee.FeatureCollection(feats)


def _reduce_over_grid(
    image, grid, value_props: list[str], *, scale: int, reducer=None,
    chunk: int = 100, tile_scale: int = 4,
) -> pd.DataFrame:
    """Reduce an image onto grid cells in **client-side chunks**.

    A single ``reduceRegions().getInfo()`` over all of Oregon's large hexes times out
    or blows the memory limit. Chunking the cells (plus a coarse ``scale`` and a
    higher ``tileScale``) keeps each request inside Earth Engine's limits.
    """
    import ee

    reducer = reducer or ee.Reducer.mean()
    props = ["cell_id", *value_props]
    rows = list(grid.itertuples(index=False))
    frames = []
    for i in range(0, len(rows), chunk):
        feats = [
            ee.Feature(
                ee.Geometry.Polygon([list(map(list, r.geometry.exterior.coords))]),
                {"cell_id": r.cell_id},
            )
            for r in rows[i : i + chunk]
        ]
        fc = ee.FeatureCollection(feats)
        reduced = image.reduceRegions(
            collection=fc, reducer=reducer, scale=scale, tileScale=tile_scale
        )
        data = reduced.select(props, None, False).getInfo()  # drop geometry from payload
        frames.append(
            pd.DataFrame([f["properties"] for f in data["features"]], columns=props)
        )
    return pd.concat(frames, ignore_index=True)


# ESA WorldCover v200 class -> a 0-1 fuel-load proxy (parity with the synthetic path).
_WORLDCOVER_FUEL = {
    10: 0.90, 20: 0.60, 30: 0.45, 40: 0.30, 50: 0.10,
    60: 0.10, 70: 0.00, 80: 0.00, 90: 0.40, 95: 0.55, 100: 0.20,
}


# --------------------------------------------------------------------------- #
# Static layers
# --------------------------------------------------------------------------- #
def pull_static(cfg: Config, grid) -> pd.DataFrame:
    """Per-cell elevation, slope, aspect, land cover, and a derived fuel-load proxy."""
    import ee

    ds = cfg["datasets"]
    terrain = ee.Terrain.products(ee.Image(ds["elevation"])).select(
        ["elevation", "slope", "aspect"]
    )
    cont = _reduce_over_grid(terrain, grid, ["elevation", "slope", "aspect"], scale=1000, chunk=100)

    # ESA WorldCover is a single-image ImageCollection; take the modal class per cell.
    # NB: the mode reducer names its output "mode" (only `mean` is named by band),
    # so we read "mode" and rename it to landcover.
    lc_img = ee.ImageCollection(ds["landcover"]).first().select("Map")
    lc = _reduce_over_grid(
        lc_img, grid, ["mode"], scale=300, reducer=ee.Reducer.mode(), chunk=100
    ).rename(columns={"mode": "landcover"})

    df = cont.merge(lc, on="cell_id", how="left")
    df["fuel_load"] = df["landcover"].round().map(_WORLDCOVER_FUEL).fillna(0.3)
    return df


# --------------------------------------------------------------------------- #
# Weather (GRIDMET) + drought
# --------------------------------------------------------------------------- #
_GRIDMET_BANDS = {
    "tmmx": "tmax",        # K -> we convert
    "rmin": "rmin",        # %
    "vs": "wind",          # m/s
    "pr": "precip",        # mm
    "vpd": "vpd",          # kPa
    "erc": "erc",          # energy release component
    "bi": "bi",            # burning index
}


def pull_weather_panel(cfg: Config, grid, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Period-composited GRIDMET weather + GRIDMET/DROUGHT PDSI reduced per cell.

    Each row is one (cell_id, date) with the period's mean of the GRIDMET bands.
    """
    import ee

    ds = cfg["datasets"]
    step_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(cfg.get("time.step"), 7)

    frames = []
    for t in dates:
        start = ee.Date(str(t.date()))
        end = start.advance(step_days, "day")
        gm = (
            ee.ImageCollection(ds["gridmet"])
            .filterDate(start, end)
            .select(list(_GRIDMET_BANDS.keys()))
            .mean()
        )
        drought = (
            ee.ImageCollection(ds["gridmet_drought"])
            .filterDate(start.advance(-10, "day"), end)
            .select(["pdsi"])
            .mean()
        )
        img = gm.addBands(drought)
        df = _reduce_over_grid(
            img, grid, [*_GRIDMET_BANDS.keys(), "pdsi"], scale=4000, chunk=200
        )
        df = df.rename(columns=_GRIDMET_BANDS)
        df["tmax"] = df["tmax"] - 273.15  # Kelvin -> Celsius
        df["date"] = t
        frames.append(df)
        logger.info("weather %s: %d cells", t.date(), len(df))
    return pd.concat(frames, ignore_index=True)


def export_weather_panel(cfg: Config, grid_fc, dates: pd.DatetimeIndex):
    """Kick off batch ``Export.table.toDrive`` tasks for full-history pulls.

    Returns the list of started task descriptions. Use this instead of
    :func:`pull_weather_panel` when the in-memory path would be too large.
    """
    import ee

    ds = cfg["datasets"]
    step_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(cfg.get("time.step"), 7)
    tasks = []
    for t in dates:
        start = ee.Date(str(t.date()))
        end = start.advance(step_days, "day")
        gm = (
            ee.ImageCollection(ds["gridmet"]).filterDate(start, end)
            .select(list(_GRIDMET_BANDS.keys())).mean()
        )
        reduced = gm.reduceRegions(collection=grid_fc, reducer=ee.Reducer.mean(), scale=4000)
        reduced = reduced.map(lambda f: f.set("date", str(t.date())))
        task = ee.batch.Export.table.toDrive(
            collection=reduced,
            description=f"weather_{t.date()}",
            folder="wildfire_oregon_exports",
            fileFormat="CSV",
        )
        task.start()
        tasks.append(task.status())
    return tasks


# --------------------------------------------------------------------------- #
# Fire labels (burned area + active fire)
# --------------------------------------------------------------------------- #
def pull_fire_labels(cfg: Config, grid, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (cell_id, date): burned-area fraction, active-fire detections, and the
    binary ``fire`` label used by the risk model."""
    import ee

    ds = cfg["datasets"]
    step_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(cfg.get("time.step"), 7)

    frames = []
    for t in dates:
        start = ee.Date(str(t.date()))
        end = start.advance(step_days, "day")

        burned = (
            ee.ImageCollection(ds["burned_area"]).filterDate(start.advance(-31, "day"), end)
            .select("BurnDate").max().gt(0).unmask(0).rename("burned")
        )
        active = (
            ee.ImageCollection(ds["thermal"]).filterDate(start, end)
            .select("FireMask").max().gte(7).unmask(0).rename("active")  # 7-9 = fire confidence
        )
        img = burned.addBands(active)
        df = _reduce_over_grid(img, grid, ["burned", "active"], scale=500, chunk=200)
        df = df.rename(columns={"burned": "burned_frac", "active": "active_frac"})
        df["date"] = t
        df["fire"] = ((df["burned_frac"] > 0) | (df["active_frac"] > 0)).astype("int8")
        frames.append(df)
        logger.info("labels %s: %d cells, %d positive", t.date(), len(df), int(df["fire"].sum()))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_canonical_tables(cfg: Config | None = None, *, quick: bool = False) -> dict:
    """Pull GEE data and assemble the canonical tables (same schema as synthetic).

    ``fire_events`` here is derived from positive cell-weeks; richer point-level
    events with ignition cause come from NIFC (see ``wildfire.ingest.nifc``).
    """
    cfg = cfg or load_config()
    initialize_ee(cfg)

    grid = build_grid(cfg)
    dates = time_index(cfg)
    months = set(cfg.get("time.fire_season_months", [5, 6, 7, 8, 9, 10]))
    dates = dates[dates.month.isin(months)]
    if quick:
        dates = dates[dates.year >= dates.year.max() - 1][:6]
        grid = grid.iloc[:: max(1, len(grid) // 400)].reset_index(drop=True)

    logger.info("GEE pull: %d cells x %d timesteps", len(grid), len(dates))
    static = pull_static(cfg, grid)
    weather = pull_weather_panel(cfg, grid, dates)
    labels = pull_fire_labels(cfg, grid, dates)

    panel = weather.merge(labels[["cell_id", "date", "fire", "burned_frac", "active_frac"]],
                          on=["cell_id", "date"], how="left")
    panel["fire"] = panel["fire"].fillna(0).astype("int8")

    grid = grid.merge(static, on="cell_id", how="left")
    events = (
        panel.loc[panel["fire"] == 1, ["cell_id", "date"]]
        .merge(grid[["cell_id", "lon", "lat"]], on="cell_id", how="left")
        .assign(source="gee_modis")
    )
    return {"grid": grid, "static": static, "weather_panel": panel, "fire_events": events}
