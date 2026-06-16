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


def _fc_to_df(fc, properties: list[str]) -> pd.DataFrame:
    """Pull a (small/medium) FeatureCollection's properties into a DataFrame."""
    import ee

    data = fc.select(properties).getInfo()  # may chunk for very large collections
    rows = [f["properties"] for f in data["features"]]
    return pd.DataFrame(rows, columns=properties)


# --------------------------------------------------------------------------- #
# Static layers
# --------------------------------------------------------------------------- #
def pull_static(cfg: Config, grid_fc) -> pd.DataFrame:
    """Elevation, slope, aspect, land cover, and an NDVI climatology per cell."""
    import ee

    ds = cfg["datasets"]
    dem = ee.Image(ds["elevation"])
    terrain = ee.Terrain.products(dem)  # elevation, slope, aspect, hillshade
    elev = terrain.select(["elevation", "slope", "aspect"])

    landcover = ee.Image(ds["landcover"]).select("Map").rename("landcover")

    ndvi_clim = (
        ee.ImageCollection(ds["modis_ndvi"])
        .filterDate(cfg.get("time.start"), cfg.get("time.end"))
        .select("NDVI")
        .mean()
        .multiply(0.0001)
        .rename("ndvi_clim")
    )

    img = elev.addBands(landcover).addBands(ndvi_clim)
    reduced = img.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean().combine(ee.Reducer.mode(), sharedInputs=True),
        scale=int(cfg.get("patches.scale_m", 30)),
    )
    df = _fc_to_df(
        reduced,
        ["cell_id", "elevation", "slope", "aspect", "landcover", "ndvi_clim"],
    )
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


def pull_weather_panel(cfg: Config, grid_fc, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Period-composited GRIDMET weather + GRIDMET/DROUGHT PDSI reduced per cell.

    Each row is one (cell_id, date) with the period's mean of the GRIDMET bands.
    """
    import ee

    ds = cfg["datasets"]
    step_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(cfg.get("time.step"), 7)
    scale = 4000  # GRIDMET native ~4 km

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
            .filterDate(start.advance(-5, "day"), end)
            .select(["pdsi"])
            .mean()
        )
        img = gm.addBands(drought)
        reduced = img.reduceRegions(collection=grid_fc, reducer=ee.Reducer.mean(), scale=scale)
        df = _fc_to_df(reduced, ["cell_id", *_GRIDMET_BANDS.keys(), "pdsi"])
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
def pull_fire_labels(cfg: Config, grid_fc, dates: pd.DatetimeIndex) -> pd.DataFrame:
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
            .select("BurnDate").max().gt(0).rename("burned")
        )
        active = (
            ee.ImageCollection(ds["thermal"]).filterDate(start, end)
            .select("FireMask").max().gte(7).rename("active")  # 7-9 = fire confidence
        )
        img = burned.unmask(0).addBands(active.unmask(0))
        reduced = img.reduceRegions(
            collection=grid_fc, reducer=ee.Reducer.mean(), scale=500
        )
        df = _fc_to_df(reduced, ["cell_id", "burned", "active"])
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
        dates = dates[dates.year >= dates.year.max() - 1][:8]
        grid = grid.iloc[:: max(1, len(grid) // 800)].reset_index(drop=True)

    grid_fc = grid_to_ee(grid)
    static = pull_static(cfg, grid_fc)
    weather = pull_weather_panel(cfg, grid_fc, dates)
    labels = pull_fire_labels(cfg, grid_fc, dates)

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
