"""Tests for the grid and feature-matrix construction."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from wildfire.features.build import _add_temporal, feature_columns
from wildfire.features.grid import build_h3_grid


def test_build_h3_grid_small_bbox():
    boundary = gpd.GeoDataFrame(geometry=[box(-122.0, 44.0, -121.5, 44.5)], crs="EPSG:4326")
    grid = build_h3_grid(boundary, resolution=6)
    assert len(grid) > 0
    for col in ("cell_id", "lon", "lat", "block_id", "geometry"):
        assert col in grid.columns
    # Centroids fall within the bbox neighborhood.
    assert grid["lon"].between(-122.2, -121.3).all()
    assert grid["lat"].between(43.8, 44.7).all()


def test_feature_columns_excludes_position_and_time():
    df = pd.DataFrame(
        {
            "cell_id": ["a"], "date": pd.to_datetime(["2020-06-01"]), "fire": [0],
            "block_id": ["x"], "lon": [-121.0], "lat": [44.0], "year": [2020],
            "vpd": [1.0], "erc": [50.0],
        }
    )
    cols = feature_columns(df)
    assert "vpd" in cols and "erc" in cols
    for excluded in ("lon", "lat", "year", "cell_id", "fire", "block_id"):
        assert excluded not in cols


def test_temporal_lag_is_past_only():
    # Two steps in one cell: lag1 of the second row must equal the first row's fire.
    df = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b", "b"],
            "date": pd.to_datetime(["2020-06-01", "2020-06-08", "2020-06-01", "2020-06-08"]),
            "fire": [1, 0, 0, 1],
            "vpd": [1.0, 2.0, 1.0, 2.0],
            "precip": [0.0, 1.0, 0.0, 1.0],
        }
    )
    out = _add_temporal(df).sort_values(["cell_id", "date"]).reset_index(drop=True)
    # First step of each cell has no past -> lag filled 0.
    assert out.loc[0, "fire_lag1"] == 0
    assert out.loc[1, "fire_lag1"] == 1  # cell a, second step sees first step's fire=1
