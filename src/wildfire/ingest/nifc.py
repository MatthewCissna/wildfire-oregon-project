"""NIFC / FPA-FOD fire records — with ignition cause (lightning vs human).

Ignition cause is the signal most wildfire models leave on the table. Two sources:

1. **FPA-FOD (Short)** — the gold standard for *historical* (1992–2020) US fire
   occurrence **with cause**. It's a one-time download (Forest Service Research
   Data Archive, RDS-2013-0009.6). If present locally we use it.
2. **NIFC WFIGS (live ArcGIS)** — interagency incident locations, queried over the
   REST API for recent years; includes a coarse fire cause field.

Both are normalized to the canonical ``fire_events`` schema plus a ``cause`` column
in {``lightning``, ``human``, ``unknown``}. We also derive per-cell historical
ignition-cause densities, which become strong static features for the risk model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)

# NIFC WFIGS Locations (current/historical interagency incidents). The exact layer
# id changes over time; override via config if needed.
_WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)

_CAUSE_MAP = {
    "1": "human", "human": "human", "h": "human", "person": "human",
    "2": "lightning", "natural": "lightning", "lightning": "lightning", "l": "lightning",
}


def _norm_cause(value) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip().lower()
    for token, label in _CAUSE_MAP.items():
        if token in key:
            return label
    return "unknown"


def fetch_fpa_fod(cfg: Config | None = None, path: str | Path | None = None) -> pd.DataFrame | None:
    """Load a locally downloaded FPA-FOD file (SQLite ``.sqlite`` or CSV), Oregon only.

    Returns None if the file isn't present (so callers can fall back to NIFC live).
    Download: https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.6
    """
    cfg = cfg or load_config()
    if path is None:
        # Look for a few conventional names under data/raw/.
        raw = cfg.path_for("data_raw")
        for name in ("FPA_FOD.sqlite", "fpa_fod.sqlite", "fpa_fod.csv", "FPA_FOD.csv"):
            cand = raw / name
            if cand.exists():
                path = cand
                break
    if path is None or not Path(path).exists():
        return None

    path = Path(path)
    if path.suffix == ".sqlite":
        import sqlite3

        con = sqlite3.connect(path)
        # FPA-FOD main table is 'Fires'.
        df = pd.read_sql(
            "SELECT FIRE_YEAR, DISCOVERY_DATE, LATITUDE, LONGITUDE, "
            "NWCG_GENERAL_CAUSE AS CAUSE, FIRE_SIZE, STATE FROM Fires WHERE STATE='OR'",
            con,
        )
        con.close()
    else:
        df = pd.read_csv(path)
        df = df[df.get("STATE", "OR") == "OR"]

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df.get("DISCOVERY_DATE"), errors="coerce"),
            "lon": df["LONGITUDE"].astype(float),
            "lat": df["LATITUDE"].astype(float),
            "cause": df["CAUSE"].map(_norm_cause) if "CAUSE" in df else "unknown",
            "size_ha": df.get("FIRE_SIZE", pd.Series(index=df.index)).astype(float) * 0.404686,
            "source": "fpa_fod",
        }
    ).dropna(subset=["lon", "lat"])
    out["event_id"] = range(len(out))
    logger.info("FPA-FOD: %d Oregon fire records", len(out))
    return out


def fetch_nifc_arcgis(cfg: Config | None = None, *, max_records: int = 5000) -> pd.DataFrame:
    """Query the live NIFC WFIGS ArcGIS service for Oregon incidents with cause.

    Network-dependent; returns an empty frame (with the right columns) on failure.
    """
    cfg = cfg or load_config()
    url = cfg.get("datasets.nifc_wfigs_url", _WFIGS_URL)
    minx, miny, maxx, maxy = cfg.get("region.bbox")
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": max_records,
    }
    cols = ["event_id", "date", "lon", "lat", "cause", "size_ha", "source"]
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        gj = resp.json()
    except Exception as exc:  # pragma: no cover (network)
        logger.warning("NIFC WFIGS query failed (%s); returning empty frame.", exc)
        return pd.DataFrame(columns=cols)

    rows = []
    for i, feat in enumerate(gj.get("features", [])):
        props = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        cause_field = props.get("FireCause") or props.get("CAUSE") or props.get("FireCauseGeneral")
        size = props.get("IncidentSize") or props.get("DailyAcres") or props.get("GISAcres")
        date = props.get("FireDiscoveryDateTime") or props.get("CreatedOnDateTime")
        rows.append(
            {
                "event_id": i,
                "date": pd.to_datetime(date, unit="ms", errors="coerce")
                if isinstance(date, (int, float))
                else pd.to_datetime(date, errors="coerce"),
                "lon": coords[0],
                "lat": coords[1],
                "cause": _norm_cause(cause_field),
                "size_ha": (float(size) * 0.404686) if size else None,
                "source": "nifc_wfigs",
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    logger.info("NIFC WFIGS: %d Oregon incidents", len(df))
    return df


def load_fire_events(cfg: Config | None = None) -> pd.DataFrame:
    """Best-available historical fire events with cause (FPA-FOD, else NIFC live)."""
    cfg = cfg or load_config()
    df = fetch_fpa_fod(cfg)
    if df is None or df.empty:
        df = fetch_nifc_arcgis(cfg)
    return df


def ignition_cause_features(grid: gpd.GeoDataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-cell historical ignition densities (lightning vs human) — static features.

    Counts how many past fires of each cause fall in each grid cell; these encode
    *where* humans tend to start fires vs where lightning does — a strong prior the
    risk model can exploit.
    """
    if events.empty:
        return pd.DataFrame({"cell_id": grid["cell_id"], "hist_human_fires": 0,
                             "hist_lightning_fires": 0, "hist_fire_density": 0.0})
    ev = gpd.GeoDataFrame(
        events.copy(),
        geometry=gpd.points_from_xy(events["lon"], events["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(ev, grid[["cell_id", "geometry"]], predicate="within", how="inner")
    counts = (
        joined.groupby(["cell_id", "cause"]).size().unstack(fill_value=0).reset_index()
    )
    counts = counts.rename(
        columns={"human": "hist_human_fires", "lightning": "hist_lightning_fires"}
    )
    for col in ("hist_human_fires", "hist_lightning_fires"):
        if col not in counts:
            counts[col] = 0
    counts["hist_fire_density"] = counts["hist_human_fires"] + counts["hist_lightning_fires"]
    out = grid[["cell_id"]].merge(counts, on="cell_id", how="left").fillna(0)
    return out
