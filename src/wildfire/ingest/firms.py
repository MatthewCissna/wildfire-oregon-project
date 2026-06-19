"""Near-real-time active-fire detections (FIRMS), reduced to the H3 grid.

FIRMS (Fire Information for Resource Management System) is the standard "is something
burning right now" product: MODIS/VIIRS thermal anomalies, refreshed a few times a
day. Three backends, one output contract — a DataFrame of *active cells*:

    columns: cell_id, lat, lon, t21, confidence, frp, n_det, acq_date, source

* :func:`firms_active_cells` (default) — pulls the **Earth Engine FIRMS** collection
  for the last few days and reduces it onto the grid. Uses the project's existing EE
  service account, so no extra credential is needed.
* NASA NRT API — if a ``FIRMS_MAP_KEY`` is supplied, the lower-latency NASA endpoint is
  used instead (point detections, includes fire radiative power).
* :func:`synthetic_firms` — an offline demo that seeds plausible detections in
  high-risk cells, so the Live Fire Watch renders before EE / a key is wired up.
"""

from __future__ import annotations

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


def firms_active_cells(cfg: Config, grid, *, days: int = 2, map_key: str | None = None) -> pd.DataFrame:
    """Active-fire cells from the best available real backend (NASA key, else EE).

    Raises on failure so the caller can fall back to :func:`synthetic_firms`.
    """
    cfg = cfg or load_config()
    if map_key:
        try:
            return _firms_nasa(cfg, grid, days, map_key)
        except Exception as exc:  # network / format — fall through to EE
            logger.warning("NASA FIRMS pull failed (%s); falling back to Earth Engine.", exc)
    return _firms_ee(cfg, grid, days)


# --------------------------------------------------------------------------- #
# Earth Engine FIRMS (default — uses the existing service account)
# --------------------------------------------------------------------------- #
def _firms_ee(cfg: Config, grid, days: int) -> pd.DataFrame:
    import ee

    from wildfire.ingest.earth_engine import _reduce_over_grid
    from wildfire.ingest.ee_auth import initialize_ee

    initialize_ee(cfg)
    end = ee.Date(dt.datetime.utcnow().strftime("%Y-%m-%d")).advance(1, "day")
    start = end.advance(-int(days), "day")
    col = ee.ImageCollection(cfg["datasets"]["firms"]).filterDate(start, end)

    t21 = col.select("T21").max().unmask(0).rename("t21")
    conf = col.select("confidence").max().unmask(0).rename("confidence")
    # Per-cell detection count: stack daily "is fire" masks, sum, take the cell peak.
    n_det = col.select("T21").map(lambda im: im.gt(0)).sum().unmask(0).rename("n_det")
    img = t21.addBands(conf).addBands(n_det)

    df = _reduce_over_grid(
        img, grid, ["t21", "confidence", "n_det"], scale=1000, reducer=ee.Reducer.max(), chunk=200
    )
    df = df.merge(grid[["cell_id", "lon", "lat"]], on="cell_id", how="left")
    df = df[df["t21"].fillna(0) > 0].copy()
    df["frp"] = np.nan  # EE FIRMS carries no fire-radiative-power band
    df["acq_date"] = end.advance(-1, "day").format("YYYY-MM-dd").getInfo()
    df["source"] = "earth-engine-firms"
    return df[["cell_id", "lat", "lon", "t21", "confidence", "frp", "n_det", "acq_date", "source"]]


# --------------------------------------------------------------------------- #
# NASA FIRMS NRT API (optional — lower latency, needs a free MAP_KEY)
# --------------------------------------------------------------------------- #
def _firms_nasa(cfg: Config, grid, days: int, map_key: str) -> pd.DataFrame:
    import requests
    from scipy.spatial import cKDTree

    w, s, e, n = cfg["region"]["bbox"]
    src = cfg.get("firms.nrt_source", "VIIRS_SNPP_NRT")
    days = max(1, min(int(days), 10))  # NASA area API caps the day range at 10
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{src}/"
        f"{w},{s},{e},{n}/{days}"
    )
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    pts = pd.read_csv(io.StringIO(r.text))
    if pts.empty:
        return _empty_active()

    # Column names vary by sensor (VIIRS: bright_ti4; MODIS: brightness).
    bright_col = next((c for c in ("bright_ti4", "brightness", "bright_t31") if c in pts.columns), None)
    pts["t21"] = pts[bright_col] if bright_col else np.nan
    pts["frp"] = pts["frp"] if "frp" in pts.columns else np.nan
    pts["confidence"] = pts["confidence"].map(_nasa_conf) if "confidence" in pts.columns else np.nan

    # Snap each detection to its nearest grid cell.
    cent = grid[["cell_id", "lon", "lat"]].to_numpy()
    tree = cKDTree(cent[:, 1:3].astype(float))
    _, idx = tree.query(pts[["longitude", "latitude"]].to_numpy())
    pts["cell_id"] = cent[idx, 0]

    g = pts.groupby("cell_id").agg(
        t21=("t21", "max"), confidence=("confidence", "max"),
        frp=("frp", "max"), n_det=("t21", "size"),
        acq_date=("acq_date", "max"),
    ).reset_index()
    g = g.merge(grid[["cell_id", "lon", "lat"]], on="cell_id", how="left")
    g["source"] = f"nasa-firms-{src.lower()}"
    return g[["cell_id", "lat", "lon", "t21", "confidence", "frp", "n_det", "acq_date", "source"]]


def _nasa_conf(v):
    """NASA VIIRS confidence is l/n/h; MODIS is 0-100. Normalize to 0-100."""
    m = {"l": 30.0, "n": 65.0, "h": 90.0}
    if isinstance(v, str):
        return m.get(v.strip().lower(), np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _empty_active() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["cell_id", "lat", "lon", "t21", "confidence", "frp", "n_det", "acq_date", "source"]
    )


# --------------------------------------------------------------------------- #
# Synthetic demo (offline)
# --------------------------------------------------------------------------- #
def synthetic_firms(cfg: Config, surf: pd.DataFrame, *, seed: int | None = None,
                    max_clusters: int = 16) -> pd.DataFrame:
    """Seed plausible active-fire cells, weighted toward high modeled risk.

    Detections concentrate in the dry south and central interior the way a real early/
    mid-season day would. Clearly flagged ``source='synthetic-demo'``.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    s = surf.copy()
    risk = s["risk"].fillna(0).to_numpy()
    if risk.sum() <= 0:
        risk = np.ones_like(risk)
    # Seasonality: fewer detections outside peak summer.
    month = dt.datetime.utcnow().month
    season = {5: 0.4, 6: 0.6, 7: 1.0, 8: 1.0, 9: 0.7, 10: 0.4}.get(month, 0.15)
    n = max(2, int(round(max_clusters * season)))

    w = (risk ** 2)
    w = w / w.sum()
    pick = rng.choice(len(s), size=min(n, (w > 0).sum()), replace=False, p=w)
    chosen = s.iloc[pick]

    today = dt.datetime.utcnow().date()
    rows = []
    for _, c in chosen.iterrows():
        intensity = float(rng.uniform(0.4, 1.0)) * (0.5 + risk[pick].max())
        rows.append({
            "cell_id": c["cell_id"], "lat": round(float(c["lat"]), 3), "lon": round(float(c["lon"]), 3),
            "t21": round(320 + 60 * intensity + float(rng.uniform(0, 25)), 1),
            "confidence": round(float(rng.uniform(55, 95)), 0),
            "frp": round(float(rng.uniform(5, 120) * intensity), 1),
            "n_det": int(rng.integers(1, 6)),
            "acq_date": str(today - dt.timedelta(days=int(rng.integers(0, 2)))),
            "source": "synthetic-demo",
        })
    return pd.DataFrame(rows)
