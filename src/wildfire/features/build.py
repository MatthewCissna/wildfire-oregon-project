"""Feature engineering for the tabular (risk + count) models.

Turns the canonical ``weather_panel`` + ``grid`` static features into a modeling
matrix with engineered fire-domain features:

* **Antecedent weather windows** — rolling means over the previous 2 / 4 / 8 steps
  (~14 / 30 / 60 days at weekly resolution) for temperature, VPD, ERC, wind, etc.,
  plus antecedent precipitation **sums** (dryness memory).
* **Interactions** — wind × dryness, fuel × dryness, drought × VPD: the nonlinear
  combinations that drive fire spread.
* **Ignition-cause priors** — per-cell historical lightning/human fire densities
  (from NIFC) when available.
* **Calendar / season** — cyclical day-of-year encoding and year.
* **Autoregressive** — whether the cell burned in the previous step.

Leakage policy: same-step weather is a legitimate predictor of same-step fire risk
(this is *nowcasting*). Anything that would peek at the future (e.g. the previous
fire flag) is explicitly shifted into the past with ``groupby(cell).shift()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config

# Weather/veg columns that get antecedent rolling windows.
_ROLL_COLS = ["tmax", "vpd", "erc", "bi", "wind", "rmin", "pdsi", "ndvi"]
_ROLL_WINDOWS = [2, 4, 8]  # steps (~14/30/60 days at weekly resolution)

# Static per-cell columns carried through from the grid.
_STATIC_COLS = [
    "elevation", "slope", "aspect", "fuel_load",
    "dist_road_km", "dist_power_km", "human_proxy",
    "hist_human_fires", "hist_lightning_fires", "hist_fire_density",
]


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Antecedent rolling windows + autoregressive lag, computed per cell."""
    df = df.sort_values(["cell_id", "date"])
    g = df.groupby("cell_id", sort=False)

    for col in _ROLL_COLS:
        if col not in df:
            continue
        for w in _ROLL_WINDOWS:
            df[f"{col}_roll{w}"] = g[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )

    # Antecedent precipitation sums (dryness memory) — past windows only.
    if "precip" in df:
        for w in _ROLL_WINDOWS:
            df[f"precip_sum{w}"] = g["precip"].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum()
            )

    # Autoregressive: did this cell burn last step? (strictly past)
    if "fire" in df:
        df["fire_lag1"] = g["fire"].transform(lambda s: s.shift(1)).fillna(0).astype("int8")
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    def z(col):
        s = df[col]
        sd = s.std()
        return (s - s.mean()) / (sd if sd > 1e-9 else 1.0)

    if {"wind", "vpd"} <= set(df.columns):
        df["wind_x_dryness"] = z("wind") * z("vpd")
    if {"fuel_load", "vpd"} <= set(df.columns):
        df["fuel_x_dryness"] = z("fuel_load") * z("vpd")
    if {"pdsi", "vpd"} <= set(df.columns):
        df["drought_x_dryness"] = z("pdsi") * z("vpd") * -1  # drier drought (neg pdsi) -> higher
    return df


def build_feature_matrix(cfg: Config | None = None, data: dict | None = None) -> pd.DataFrame:
    """Assemble the (cell_id, date) modeling matrix with target ``fire``.

    If ``data`` is None, the canonical tables are loaded from disk.
    """
    cfg = cfg or load_config()
    if data is None:
        from wildfire.ingest.datasets import load_canonical

        data = load_canonical(cfg)

    panel = data["weather_panel"].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    grid = data["grid"]

    static_cols = ["cell_id", "lon", "lat", "block_id"] + (
        ["ecoregion"] if "ecoregion" in grid.columns else []
    ) + [c for c in _STATIC_COLS if c in grid.columns]
    static = pd.DataFrame(grid[static_cols].copy())

    df = panel.merge(static, on="cell_id", how="left")
    df = _add_calendar(df)
    df = _add_temporal(df)
    df = _add_interactions(df)

    # Drop rows with no static join (shouldn't happen) and reset.
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model input columns (everything except keys, geometry, and the target).

    Raw ``lon``/``lat``/``year`` are deliberately excluded: they let trees memorize
    absolute position/time, which inflates random-split scores but *hurts* spatial
    and temporal generalization (the failure mode we're explicitly avoiding). The
    model must rely on physical drivers — season is encoded cyclically instead.
    """
    exclude = {
        "cell_id", "date", "fire", "block_id", "geometry",
        "lon", "lat", "year",
        "burned_frac", "active_frac",
    }
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def save_feature_matrix(cfg: Config | None = None, df: pd.DataFrame | None = None) -> str:
    cfg = cfg or load_config()
    if df is None:
        df = build_feature_matrix(cfg)
    out = cfg.path_for("data_processed") / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return str(out)
