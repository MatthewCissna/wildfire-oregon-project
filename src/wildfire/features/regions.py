"""Aggregate the cell-level panel to regions × season for the fire-count model.

A "region" is the spatial block (``block_id`` — an H3 parent cell by default, or a
county/ecoregion if joined upstream). For each region and fire-season year we form:

    fire_count   number of cell-weeks that burned in that region-season
    exposure     number of (cell × week) observations  -> Poisson offset / log-exposure
    <features>   season-mean of the drivers (vpd, erc, drought, fuel, ...)

The Poisson/NB model predicts ``fire_count`` with ``log(exposure)`` as an offset,
so it estimates a per-exposure rate rather than confounding region size with risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config

# Drivers averaged per region-season as count-model predictors.
_AGG_FEATURES = [
    "vpd", "erc", "bi", "pdsi", "tmax", "rmin", "wind", "precip",
    "days_since_rain", "ndvi", "fuel_load", "elevation",
    "human_proxy", "lightning_density",
    "hist_fire_density",
]


def build_region_season(
    cfg: Config | None = None, features: pd.DataFrame | None = None, region_col: str = "block_id"
) -> pd.DataFrame:
    """Region × fire-season-year table for the count model."""
    cfg = cfg or load_config()
    if features is None:
        from wildfire.features.build import build_feature_matrix

        features = build_feature_matrix(cfg)

    df = features.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    agg = {c: "mean" for c in _AGG_FEATURES if c in df.columns}
    grouped = df.groupby([region_col, "year"])
    out = grouped.agg(agg)
    out["fire_count"] = grouped["fire"].sum()
    out["exposure"] = grouped.size()
    out["n_cells"] = grouped["cell_id"].nunique()
    out = out.reset_index().rename(columns={region_col: "region"})

    out["log_exposure"] = np.log(out["exposure"].clip(lower=1))
    return out


def region_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"region", "year", "fire_count", "exposure", "n_cells", "log_exposure"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
