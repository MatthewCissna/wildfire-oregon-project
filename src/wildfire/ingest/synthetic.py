"""Synthetic data generator (offline fallback).

Produces the **same canonical tables** the real Earth Engine + NIFC pipeline emits:

    grid          GeoDataFrame[cell_id, lon, lat, block_id, geometry, <static features>]
    weather_panel DataFrame[cell_id, date, <weather/veg features>, fire]
    fire_events   DataFrame[event_id, date, lon, lat, cell_id, cause, size_ha, source]

so every downstream stage (features, models, eval, viz) is source-agnostic. The
labels are drawn from a hidden latent risk function of real fire drivers (VPD,
drought, fuel, dryness, wind, human-ignition proxy, lightning), which means:

  * fires are **rare** (~1-3% of cell-weeks), as in reality;
  * models can recover genuine signal (non-trivial PR-AUC);
  * SHAP shows the right drivers, validating the modeling approach;
  * spatial structure is smooth, so spatial CV actually matters.

Everything is deterministic given ``project.random_seed``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config
from wildfire.features.grid import build_grid, time_index

# Latent-risk coefficients (log-odds). Sign/magnitude encode fire physics so the
# trained models — and SHAP — should recover them.
_COEFFS = {
    "intercept": -5.9,      # sets the base rate low (~2-3% positives: rare events)
    "vpd": 1.10,            # vapor pressure deficit: dry air -> more fire
    "erc": 0.85,            # energy release component (fuel dryness/danger)
    "drought": 0.70,        # drier (more negative PDSI) -> more fire
    "days_since_rain": 0.55,
    "fuel_load": 0.50,      # more burnable fuel -> more fire
    "wind_x_dryness": 0.45,
    "human_proxy": 0.40,    # closeness to roads/power
    "lightning": 0.65,
    "ndvi_anom": -0.30,     # greener than normal (wetter) -> less fire
}

# Per-ecoregion climate / fuel / ignition character (multipliers & offsets).
# Encodes Oregon's west-wet / east-dry gradient, valley human ignitions, and
# Cascade/Blue-Mountain lightning — so the synthetic demo has real spatial structure.
#   precip: precipitation multiplier   temp: summer temp offset (°C)
#   fuel: base fuel load (0-1)         lightning: lightning multiplier
#   human: human-ignition multiplier   lc: dominant land-cover label
ECO_PARAMS = {
    "Coast Range":              dict(precip=1.7, temp=-1.0, fuel=0.80, lightning=0.3, human=0.7, lc="forest_wet"),
    "Klamath Mountains":        dict(precip=0.9, temp=0.5,  fuel=0.85, lightning=0.9, human=0.6, lc="forest_conifer"),
    "Willamette Valley":        dict(precip=1.2, temp=1.0,  fuel=0.35, lightning=0.3, human=1.4, lc="grass_ag"),
    "West Cascades":            dict(precip=1.5, temp=-2.0, fuel=0.95, lightning=0.7, human=0.4, lc="forest_conifer"),
    "East Cascades":            dict(precip=0.7, temp=-1.0, fuel=0.85, lightning=1.2, human=0.5, lc="forest_conifer"),
    "Columbia Plateau":         dict(precip=0.5, temp=1.5,  fuel=0.35, lightning=0.8, human=1.0, lc="grass_ag"),
    "Blue Mountains":           dict(precip=0.7, temp=-1.0, fuel=0.80, lightning=1.3, human=0.5, lc="forest_conifer"),
    "Northern Basin and Range": dict(precip=0.4, temp=1.5,  fuel=0.45, lightning=1.0, human=0.4, lc="shrub_steppe"),
}
_ECO_DEFAULT = dict(precip=0.8, temp=0.0, fuel=0.55, lightning=1.0, human=0.7, lc="shrub_steppe")


def _eco_arr(grid, key: str):
    import numpy as np

    return np.array([ECO_PARAMS.get(e, _ECO_DEFAULT)[key] for e in grid["ecoregion"]])


@dataclass
class SyntheticSpec:
    n_cells: int | None      # cap on grid cells (None = full Oregon grid)
    dates: pd.DatetimeIndex  # time steps to simulate
    seed: int


def _spec(cfg: Config, quick: bool) -> SyntheticSpec:
    idx = time_index(cfg)
    months = set(cfg.get("time.fire_season_months", [5, 6, 7, 8, 9, 10]))
    idx = idx[idx.month.isin(months)]  # only simulate the fire season
    if quick:
        # ~3 most recent years, capped cells — a fast end-to-end smoke run.
        idx = idx[idx.year >= idx.year.max() - 2]
        return SyntheticSpec(n_cells=800, dates=idx, seed=cfg.seed)
    return SyntheticSpec(n_cells=None, dates=idx, seed=cfg.seed)


def _zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def _static_features(grid, rng: np.random.Generator) -> pd.DataFrame:
    """Per-cell static terrain / fuel / human-proxy features (smooth in space)."""
    lon = grid["lon"].to_numpy()
    lat = grid["lat"].to_numpy()
    n = len(grid)

    # Cascades ridge (~ -121.7) and Coast Range (~ -123.6) elevation bumps + lapse.
    cascades = 1600.0 * np.exp(-(((lon + 121.7) / 0.7) ** 2))
    coast = 600.0 * np.exp(-(((lon + 123.6) / 0.4) ** 2))
    elevation = 150 + cascades + coast + 250 * np.sin(lat) + rng.normal(0, 80, n)
    elevation = np.clip(elevation, 0, None)

    # Slope: correlated with local relief (proxy via elevation magnitude) + noise.
    slope = np.clip(2 + 0.012 * elevation + rng.normal(0, 3, n), 0, 45)
    aspect = rng.uniform(0, 360, n)

    # Land cover & fuel from ecoregion character.
    landcover = _eco_arr(grid, "lc")
    fuel_load = np.clip(_eco_arr(grid, "fuel").astype(float) + rng.normal(0, 0.05, n), 0.05, 1.0)
    eco_human = _eco_arr(grid, "human").astype(float)

    # Human-ignition proxies: town centers weighted toward human-heavy ecoregions
    # (valleys/plateau); distance to nearest falls off, then modulated by ecoregion.
    n_towns = max(4, n // 400)
    town_p = eco_human / eco_human.sum()
    town_idx = rng.choice(n, size=n_towns, replace=False, p=town_p)
    tlon, tlat = lon[town_idx], lat[town_idx]
    dist_road = np.full(n, 1e9)
    for j in range(n_towns):
        d = np.sqrt((lon - tlon[j]) ** 2 + (lat - tlat[j]) ** 2) * 111.0  # deg->km approx
        dist_road = np.minimum(dist_road, d)
    dist_road = dist_road + np.abs(rng.normal(0, 1.5, n))
    dist_power = dist_road * rng.uniform(0.8, 1.6, n) + np.abs(rng.normal(0, 2, n))
    human_proxy = np.clip(np.exp(-dist_road / 25.0) * eco_human, 0, 1.5)  # high near roads in WUI

    return pd.DataFrame(
        {
            "cell_id": grid["cell_id"].to_numpy(),
            "ecoregion": grid["ecoregion"].to_numpy(),
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "landcover": landcover,
            "fuel_load": fuel_load,
            "dist_road_km": dist_road,
            "dist_power_km": dist_power,
            "human_proxy": human_proxy,
        }
    )


def _weather_and_labels(
    grid, static: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized per-(cell, date) weather, vegetation, and sampled fire labels."""
    lon = grid["lon"].to_numpy()
    lat = grid["lat"].to_numpy()
    elev = static["elevation"].to_numpy()
    fuel = static["fuel_load"].to_numpy()
    human = static["human_proxy"].to_numpy()
    cell_ids = grid["cell_id"].to_numpy()
    n = len(grid)

    # Ecoregion climate character (static per cell).
    precip_mult = _eco_arr(grid, "precip").astype(float)
    temp_offset = _eco_arr(grid, "temp").astype(float)
    lightning_mult = _eco_arr(grid, "lightning").astype(float)

    # Per-cell slowly varying drought state (AR(1) across the season), drier in the east.
    east = _zscore(lon)  # higher = east = drier
    drought_state = -0.6 * east - 0.4 * _zscore(precip_mult) + rng.normal(0, 0.5, n)
    dsr = np.zeros(n)  # days since rain accumulator

    panel_rows = []
    event_rows = []
    ev = 0
    for t in dates:
        doy = t.dayofyear
        season = np.sin((doy - 200) / 365.0 * 2 * np.pi)  # peak dryness ~ mid-July

        tmax = 22 + 9 * season - 0.0055 * elev + temp_offset + rng.normal(0, 2.5, n)
        # Precip: wet winters, dry summers, scaled by the ecoregion precip multiplier.
        precip = np.clip(
            np.maximum(0, (1.2 - season) * rng.gamma(1.2, 2.0, n) * precip_mult),
            0, None,
        )
        rmin = np.clip(70 - 18 * season - 0.6 * tmax + 2.0 * precip + rng.normal(0, 6, n), 5, 100)
        wind = np.clip(2.5 + 1.2 * season + rng.gamma(2.0, 1.0, n), 0, None)

        # VPD (kPa) from tmax & RH (Tetens).
        es = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
        vpd = np.clip(es * (1 - rmin / 100.0), 0, None)

        # Drought AR(1): dries out when no precip, recharges with rain.
        drought_state = 0.92 * drought_state - 0.15 * (precip < 0.5) + 0.25 * (precip > 3) \
            + rng.normal(0, 0.2, n)
        pdsi = np.clip(drought_state * 1.5, -6, 6)  # negative = drought

        dsr = np.where(precip > 1.0, 0.0, dsr + 7.0)  # weekly step

        # Fire-danger indices increase with heat/dryness.
        erc = np.clip(40 + 12 * _zscore(vpd) - 4 * _zscore(precip) - 3 * pdsi + rng.normal(0, 4, n), 0, 100)
        bi = np.clip(0.6 * erc + 5 * _zscore(wind) + rng.normal(0, 4, n), 0, 150)

        # Vegetation greenness: seasonal browning; anomaly vs cell mean used as feature.
        ndvi = np.clip(0.55 + 0.18 * fuel - 0.20 * season + 0.05 * (precip > 2) + rng.normal(0, 0.04, n), 0, 1)
        ndvi_anom = ndvi - (0.55 + 0.18 * fuel)

        # Lightning density field (sparse, convective; scaled by ecoregion character).
        lightning = np.clip(
            rng.gamma(0.4, 1.0, n) * (0.5 + 0.5 * season) * lightning_mult, 0, None
        )
        lightning_n = _zscore(np.log1p(lightning))

        wind_x_dry = _zscore(wind) * _zscore(vpd)

        # ---- latent log-odds -> probability -> Bernoulli fire ----
        c = _COEFFS
        logit = (
            c["intercept"]
            + c["vpd"] * _zscore(vpd)
            + c["erc"] * _zscore(erc)
            + c["drought"] * _zscore(-pdsi)
            + c["days_since_rain"] * _zscore(dsr)
            + c["fuel_load"] * _zscore(fuel)
            + c["wind_x_dryness"] * wind_x_dry
            + c["human_proxy"] * _zscore(human)
            + c["lightning"] * lightning_n
            + c["ndvi_anom"] * _zscore(ndvi_anom)
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        fire = (rng.uniform(size=n) < p).astype(np.int8)

        panel_rows.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "date": t,
                    "tmax": tmax,
                    "rmin": rmin,
                    "wind": wind,
                    "precip": precip,
                    "vpd": vpd,
                    "erc": erc,
                    "bi": bi,
                    "pdsi": pdsi,
                    "days_since_rain": dsr.copy(),
                    "ndvi": ndvi,
                    "ndvi_anom": ndvi_anom,
                    "lightning_density": lightning,
                    "fire": fire,
                }
            )
        )

        # Build fire events for positive cells, with ignition cause + size.
        pos = np.flatnonzero(fire)
        if pos.size:
            # Cause: compare human vs lightning contribution at the positive cells.
            human_term = c["human_proxy"] * _zscore(human)[pos]
            light_term = c["lightning"] * lightning_n[pos]
            p_human = 1.0 / (1.0 + np.exp(-(human_term - light_term)))
            cause = np.where(rng.uniform(size=pos.size) < p_human, "human", "lightning")
            size_ha = np.exp(rng.normal(2.0, 1.3, pos.size) + 0.4 * _zscore(vpd)[pos])
            event_rows.append(
                pd.DataFrame(
                    {
                        "event_id": np.arange(ev, ev + pos.size),
                        "date": t,
                        "lon": lon[pos] + rng.normal(0, 0.02, pos.size),
                        "lat": lat[pos] + rng.normal(0, 0.02, pos.size),
                        "cell_id": cell_ids[pos],
                        "cause": cause,
                        "size_ha": np.clip(size_ha, 0.1, None),
                        "source": "synthetic",
                    }
                )
            )
            ev += pos.size

    panel = pd.concat(panel_rows, ignore_index=True)
    events = (
        pd.concat(event_rows, ignore_index=True)
        if event_rows
        else pd.DataFrame(
            columns=["event_id", "date", "lon", "lat", "cell_id", "cause", "size_ha", "source"]
        )
    )
    return panel, events


def generate(cfg: Config | None = None, *, quick: bool = False) -> dict:
    """Generate the full synthetic dataset.

    Returns a dict with keys ``grid`` (GeoDataFrame), ``static`` (DataFrame),
    ``weather_panel`` (DataFrame), ``fire_events`` (DataFrame).
    """
    cfg = cfg or load_config()
    spec = _spec(cfg, quick)
    rng = np.random.default_rng(spec.seed)

    grid = build_grid(cfg)
    if spec.n_cells is not None and len(grid) > spec.n_cells:
        keep = rng.choice(len(grid), size=spec.n_cells, replace=False)
        grid = grid.iloc[np.sort(keep)].reset_index(drop=True)

    static = _static_features(grid, rng)
    panel, events = _weather_and_labels(grid, static, spec.dates, rng)

    # Attach static features to the grid (grid already carries ecoregion).
    grid = grid.merge(static.drop(columns=["ecoregion"]), on="cell_id", how="left")
    return {"grid": grid, "static": static, "weather_panel": panel, "fire_events": events}
