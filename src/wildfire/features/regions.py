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

# Drivers averaged per region-season as count-model predictors. Kept deliberately
# lean: ecoregions are few, so a compact, decorrelated set keeps the GLM stable.
_AGG_FEATURES = [
    "vpd", "erc", "pdsi", "fuel_load", "lightning_density",
]


def build_region_season(
    cfg: Config | None = None, features: pd.DataFrame | None = None,
    region_col: str | None = None, region_map=None,
) -> pd.DataFrame:
    """Region × fire-season-year table for the count model.

    ``region_map`` (optional): a dict/Series keyed by ``cell_id`` mapping each cell to
    a region name (e.g. fire district). When given, cells are grouped by it — this
    lets callers use regions (like ODF districts) that aren't part of the feature build.
    """
    cfg = cfg or load_config()
    if features is None:
        from wildfire.features.build import build_feature_matrix

        features = build_feature_matrix(cfg)

    df = features.copy()
    if region_map is not None:
        df["__region__"] = df["cell_id"].map(dict(region_map))
        region_col = "__region__"
    elif region_col is None:
        # Default region unit: ecoregion when available, else the H3 spatial block.
        region_col = "ecoregion" if "ecoregion" in features.columns else "block_id"
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
