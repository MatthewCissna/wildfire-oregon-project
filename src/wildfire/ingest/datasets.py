"""Ingest orchestration: produce the canonical tables on disk, from either source.

Canonical artifacts (written to ``data/interim``):
    grid.parquet           GeoParquet: cell geometry + static features
    weather_panel.parquet  (cell_id, date, weather/veg features, fire)
    fire_events.parquet    (event_id, date, lon, lat, cell_id, cause, size_ha, source)
    _manifest.json         provenance (source, quick flag, row counts, timestamp)

Everything downstream reads these via :func:`load_canonical`, so the modeling code
never knows or cares whether the data came from Earth Engine or the synthetic
generator.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


def _paths(cfg: Config) -> dict:
    interim = cfg.path_for("data_interim")
    return {
        "grid": interim / "grid.parquet",
        "weather_panel": interim / "weather_panel.parquet",
        "fire_events": interim / "fire_events.parquet",
        "manifest": interim / "_manifest.json",
    }


def materialize(cfg: Config | None = None, *, synthetic: bool = True, quick: bool = False) -> dict:
    """Build and persist the canonical tables. Returns the in-memory dict too."""
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    paths = _paths(cfg)

    if synthetic:
        from wildfire.ingest import synthetic as syn

        data = syn.generate(cfg, quick=quick)
        source = "synthetic"
    else:
        from wildfire.ingest import earth_engine as gee
        from wildfire.ingest import nifc

        data = gee.build_canonical_tables(cfg, quick=quick)
        # Enrich with NIFC ignition-cause static features + better point events.
        events = nifc.load_fire_events(cfg)
        if not events.empty:
            cause_feats = nifc.ignition_cause_features(data["grid"], events)
            data["grid"] = data["grid"].merge(cause_feats, on="cell_id", how="left")
            data["fire_events"] = events
        # Optionally enrich with OSM distance proxies (best-effort).
        try:
            from wildfire.ingest import osm

            osm_feats = osm.distance_features(cfg, data["grid"])
            data["grid"] = data["grid"].merge(
                osm_feats, on="cell_id", how="left", suffixes=("", "_osm")
            )
        except Exception as exc:  # pragma: no cover (network/CPU heavy)
            logger.warning("OSM enrichment skipped: %s", exc)
        source = "gee+nifc+osm"

    grid: gpd.GeoDataFrame = data["grid"]
    grid.to_parquet(paths["grid"])
    data["weather_panel"].to_parquet(paths["weather_panel"], index=False)
    data["fire_events"].to_parquet(paths["fire_events"], index=False)

    manifest = {
        "source": source,
        "quick": quick,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_cells": int(len(grid)),
        "n_panel_rows": int(len(data["weather_panel"])),
        "n_events": int(len(data["fire_events"])),
        "positive_rate": float(data["weather_panel"]["fire"].mean())
        if len(data["weather_panel"])
        else 0.0,
        "config_path": str(cfg.path),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2))
    logger.info("Materialized canonical tables: %s", manifest)
    return {**data, "manifest": manifest}


def load_canonical(cfg: Config | None = None) -> dict:
    """Read the canonical tables back from disk."""
    cfg = cfg or load_config()
    paths = _paths(cfg)
    if not paths["grid"].exists():
        raise FileNotFoundError(
            "No canonical data found. Run stage 1 first:\n"
            "    uv run python scripts/01_ingest.py --synthetic"
        )
    return {
        "grid": gpd.read_parquet(paths["grid"]),
        "weather_panel": pd.read_parquet(paths["weather_panel"]),
        "fire_events": pd.read_parquet(paths["fire_events"]),
        "manifest": json.loads(paths["manifest"].read_text()) if paths["manifest"].exists() else {},
    }
