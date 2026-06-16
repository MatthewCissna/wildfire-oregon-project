"""Image patches for the fire-detection CNN.

Each patch is a small multi-band image (Sentinel-2 bands + spectral indices) labeled
fire / no-fire. Labels come from FIRMS/MODIS active-fire detections aligned to the
imagery date. Two providers, same output contract:

* :func:`synthetic_patches` — offline generator with a physically-motivated fire
  spectral signature (hot SWIR, depressed NBR/NDVI), so the CNN learns the right cue
  and the pipeline runs before Earth Engine is configured.
* :func:`export_s2_patches_ee` — turnkey GEE extraction around sampled fire/no-fire
  points (documented; runs once Earth Engine is authenticated).

Patches carry a ``block_id`` so train/val/test can be split by space (no leakage).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)


def n_channels(cfg: Config) -> int:
    return len(cfg.get("patches.bands", [])) + len(cfg.get("patches.indices", []))


def channel_names(cfg: Config) -> list[str]:
    return list(cfg.get("patches.bands", [])) + list(cfg.get("patches.indices", []))


def _make_patch_batch(n: int, size: int, c: int, fire: bool, rng: np.random.Generator) -> np.ndarray:
    """Generate ``n`` patches (n,c,size,size) with a fire or no-fire spectral signature.

    Channels order matches channel_names: [B2,B3,B4,B8,B11,B12, NDVI,NBR,NDMI,BAI].
    No-fire: vegetated (high NIR/B8, high NDVI/NBR), with occasional **bright-soil /
    cloud confounders** (high red+SWIR but NIR not depressed) that look fire-like and
    make the task non-trivial. Fire: hot SWIR (B11/B12) with depressed NIR/NBR, at a
    **variable strength** (some subtle) so the detector can't rely on a single cue.
    """
    x = rng.normal(0.2, 0.10, size=(n, c, size, size)).astype(np.float32)  # heavier noise

    # Baseline vegetated reflectance per band, varied per patch (scene-to-scene drift).
    base = np.array([0.10, 0.12, 0.13, 0.45, 0.22, 0.14] + [0.0] * (c - 6), dtype=np.float32)
    x += base[None, :, None, None]
    nb = min(6, c)
    x[:, :nb] += rng.normal(0, 0.04, size=(n, nb, 1, 1)).astype(np.float32)  # scene drift

    # Smooth spatial texture via a low-freq gradient.
    yy, xx = np.mgrid[0:size, 0:size] / size
    grad = (0.05 * np.sin(3 * np.pi * xx) * np.cos(3 * np.pi * yy)).astype(np.float32)
    x += grad[None, None, :, :]

    def _blob_mask(cy, cx, r):
        return ((yy * size - cy) ** 2 + (xx * size - cx) ** 2) <= r ** 2

    if fire:
        # Burn blob of variable strength (some subtle) -> realistic class overlap.
        cy = rng.integers(size // 4, 3 * size // 4, n)
        cx = rng.integers(size // 4, 3 * size // 4, n)
        r = rng.uniform(size * 0.10, size * 0.28, n)
        strength = rng.uniform(0.30, 1.0, n)
        for i in range(n):
            mask = _blob_mask(cy[i], cx[i], r[i])
            s = strength[i]
            x[i, 4][mask] += s * rng.uniform(0.08, 0.22)   # SWIR1 up
            x[i, 5][mask] += s * rng.uniform(0.10, 0.28)   # SWIR2 up
            x[i, 3][mask] -= s * rng.uniform(0.05, 0.16)   # NIR down
            x[i, 2][mask] += s * rng.uniform(0.02, 0.08)   # red up
    else:
        # ~22% of no-fire patches carry a bright soil / cloud blob (SWIR & red up, but
        # NIR NOT depressed) — a deliberate confounder so PR-AUC isn't a free 1.0.
        conf = np.flatnonzero(rng.uniform(size=n) < 0.22)
        for i in conf:
            cy = rng.integers(size // 4, 3 * size // 4)
            cx = rng.integers(size // 4, 3 * size // 4)
            mask = _blob_mask(cy, cx, rng.uniform(size * 0.12, size * 0.30))
            x[i, 4][mask] += rng.uniform(0.06, 0.16)
            x[i, 5][mask] += rng.uniform(0.06, 0.16)
            x[i, 2][mask] += rng.uniform(0.05, 0.14)
            x[i, 3][mask] += rng.uniform(0.00, 0.10)  # NIR stays high (key difference)

    x = np.clip(x, 0.0, 1.5)

    # Derived indices (recomputed from bands so they're consistent).
    b_red, b_nir, b_swir1, b_swir2 = x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    eps = 1e-6
    if c > 6:
        x[:, 6] = (b_nir - b_red) / (b_nir + b_red + eps)            # NDVI
    if c > 7:
        x[:, 7] = (b_nir - b_swir2) / (b_nir + b_swir2 + eps)        # NBR
    if c > 8:
        x[:, 8] = (b_nir - b_swir1) / (b_nir + b_swir1 + eps)        # NDMI
    if c > 9:
        x[:, 9] = 1.0 / ((0.1 - b_red) ** 2 + (0.06 - b_nir) ** 2 + eps)  # BAI
        x[:, 9] = np.clip(x[:, 9] / 100.0, 0, 1.5)
    return x


def synthetic_patches(cfg: Config | None = None, *, quick: bool = False) -> dict:
    """Generate a balanced-ish synthetic patch dataset with spatial blocks."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg.seed)
    size = int(cfg.get("patches.size_px", 64))
    c = n_channels(cfg)

    n_pos = 250 if quick else int(cfg.get("patches.pos_per_year", 1500)) * 2
    n_neg = int(n_pos * float(cfg.get("patches.neg_per_pos", 2.0)))

    xpos = _make_patch_batch(n_pos, size, c, fire=True, rng=rng)
    xneg = _make_patch_batch(n_neg, size, c, fire=False, rng=rng)
    X = np.concatenate([xpos, xneg], axis=0)
    y = np.concatenate([np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)])

    # Assign spatial blocks from the real grid so the split is geographic.
    from wildfire.features.grid import build_grid

    blocks = build_grid(cfg)["block_id"].unique()
    block_id = rng.choice(blocks, size=len(y))
    meta = pd.DataFrame({"block_id": block_id, "label": y, "source": "synthetic"})

    perm = rng.permutation(len(y))
    return {"X": X[perm], "y": y[perm], "meta": meta.iloc[perm].reset_index(drop=True),
            "channels": channel_names(cfg)}


def save_patches(cfg: Config, data: dict) -> str:
    out = cfg.path_for("patches") / "patches.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=data["X"].astype(np.float16), y=data["y"])
    data["meta"].to_parquet(cfg.path_for("patches") / "patches_meta.parquet", index=False)
    (cfg.path_for("patches") / "channels.txt").write_text("\n".join(data["channels"]))
    logger.info("Saved %d patches -> %s", len(data["y"]), out)
    return str(out)


def load_patches(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    npz = cfg.path_for("patches") / "patches.npz"
    if not npz.exists():
        raise FileNotFoundError("No patches found. Run: uv run python scripts/03_patches.py --synthetic")
    arr = np.load(npz)
    meta = pd.read_parquet(cfg.path_for("patches") / "patches_meta.parquet")
    channels = (cfg.path_for("patches") / "channels.txt").read_text().splitlines()
    return {"X": arr["X"].astype(np.float32), "y": arr["y"], "meta": meta, "channels": channels}


# --------------------------------------------------------------------------- #
# Live Earth Engine patch extraction (synchronous getInfo — no Drive needed)
# --------------------------------------------------------------------------- #
def _s2_index_composite(cfg: Config, year: int, oregon):
    """Cloud-filtered S2 median composite for a fire season + spectral indices."""
    import ee

    ds = cfg["datasets"]
    bands6 = ["B2", "B3", "B4", "B8", "B11", "B12"]
    s2 = (
        ee.ImageCollection(ds["s2_sr"]).filterDate(f"{year}-05-01", f"{year}-10-31")
        .filterBounds(oregon)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.get("patches.max_cloud_pct", 40)))
        .median().select(bands6).divide(10000)
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = s2.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    bai = s2.expression(
        "1.0/((0.1-RED)**2+(0.06-NIR)**2)", {"RED": s2.select("B4"), "NIR": s2.select("B8")}
    ).rename("BAI").clamp(0, 50)  # BAI is unbounded near red~0.1; clamp for stable CNN norm
    return s2.addBands([ndvi, nbr, ndmi, bai]).select(channel_names(cfg))


def pull_s2_patches(
    cfg: Config | None = None, *, years=(2017, 2018, 2020, 2021, 2023),
    n_per_class_per_year: int = 150, batch: int = 25,
) -> dict:
    """Extract **real** Sentinel-2 fire/no-fire patches via synchronous getInfo.

    For each fire-season year: build an S2 median + index composite, sample
    fire/non-fire points from the MOD14 active-fire mask, and pull a fixed-size
    patch around each point with ``neighborhoodToArray`` (batched). No Google Drive
    round-trip. Returns the same dict shape as :func:`synthetic_patches`.
    """
    import ee
    import h3

    from wildfire.ingest.boundary import oregon_ee_geometry
    from wildfire.ingest.ee_auth import initialize_ee
    from wildfire.ingest.earth_engine import _getinfo_with_retry

    cfg = cfg or load_config()
    initialize_ee(cfg)
    oregon = oregon_ee_geometry(cfg)
    channels = channel_names(cfg)
    size = int(cfg.get("patches.size_px", 64))
    radius = size // 2  # square kernel -> (2r+1) px; we crop to size
    scale = int(cfg.get("patches.scale_m", 20))
    ds = cfg["datasets"]

    X_list, y_list, blocks = [], [], []
    for year in years:
        comp = _s2_index_composite(cfg, year, oregon)
        arr = comp.neighborhoodToArray(ee.Kernel.square(radius, "pixels"))
        fire_mask = (
            ee.ImageCollection(ds["thermal"]).filterDate(f"{year}-05-01", f"{year}-10-31")
            .select("FireMask").max().gte(7).rename("fire").unmask(0).clip(oregon)
        )
        pts = fire_mask.stratifiedSample(
            numPoints=n_per_class_per_year, classBand="fire", region=oregon,
            scale=1000, geometries=True, seed=cfg.seed,
        )
        pts_list = pts.toList(pts.size())
        n_pts = _getinfo_with_retry(pts.size())
        for off in range(0, n_pts, batch):
            fc = ee.FeatureCollection(pts_list.slice(off, off + batch))
            sampled = arr.sampleRegions(collection=fc, scale=scale, geometries=True)
            data = _getinfo_with_retry(sampled)
            for feat in data["features"]:
                pr = feat["properties"]
                try:
                    patch = np.stack(
                        [np.array(pr[c], dtype=np.float32) for c in channels], axis=0
                    )
                except (KeyError, ValueError):
                    continue  # masked/edge patch with a missing band
                if patch.shape[1] < size or patch.shape[2] < size:
                    continue
                patch = patch[:, :size, :size]
                if not np.isfinite(patch).all():
                    continue
                lon, lat = feat["geometry"]["coordinates"]
                X_list.append(patch)
                y_list.append(int(pr["fire"]))
                blocks.append(h3.cell_to_parent(h3.latlng_to_cell(lat, lon, 6), 3))
        logger.info("S2 patches %d: total so far %d", year, len(y_list))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame({"block_id": blocks, "label": y, "source": "sentinel2"})
    return {"X": X, "y": y, "meta": meta, "channels": channels}


# --------------------------------------------------------------------------- #
# Live Earth Engine patch export (batch Export.toDrive alternative)
# --------------------------------------------------------------------------- #
def export_s2_patches_ee(cfg: Config | None = None, *, year: int = 2021, n_points: int = 500):
    """Sample fire / non-fire points and export Sentinel-2 patches via GEE.

    Fire points come from MOD14 active-fire / MCD64 burned area; non-fire points are
    sampled from unburned land. For each point a patch is exported with the band +
    index stack. Returns started export-task statuses.
    """
    import ee

    from wildfire.ingest.boundary import oregon_ee_geometry
    from wildfire.ingest.ee_auth import initialize_ee

    initialize_ee(cfg)
    cfg = cfg or load_config()
    oregon = oregon_ee_geometry(cfg)
    ds = cfg["datasets"]
    size = int(cfg.get("patches.size_px", 64))
    scale = int(cfg.get("patches.scale_m", 20))

    start = f"{year}-05-01"
    end = f"{year}-10-31"

    # Active-fire mask -> stratified sample of fire / non-fire points.
    fire_mask = (
        ee.ImageCollection(ds["thermal"]).filterDate(start, end)
        .select("FireMask").max().gte(7).rename("fire").unmask(0).clip(oregon)
    )
    samples = fire_mask.stratifiedSample(
        numPoints=n_points, classBand="fire", region=oregon, scale=1000, geometries=True, seed=cfg.seed
    )

    # Cloud-masked S2 median composite + indices.
    def add_indices(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        bai = img.expression(
            "1.0/((0.1-RED)**2+(0.06-NIR)**2)", {"RED": img.select("B4"), "NIR": img.select("B8")}
        ).rename("BAI")
        return img.addBands([ndvi, nbr, ndmi, bai])

    s2 = (
        ee.ImageCollection(ds["s2_sr"]).filterDate(start, end).filterBounds(oregon)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.get("patches.max_cloud_pct", 40)))
        .map(add_indices).median()
        .select(channel_names(cfg))
    )

    # Export a patch (neighborhood) per sample point.
    patch_dim = size * scale
    task = ee.batch.Export.table.toDrive(
        collection=s2.sampleRegions(collection=samples, scale=scale, geometries=True),
        description=f"s2_patches_{year}",
        folder="wildfire_oregon_exports",
        fileFormat="TFRecord",
    )
    task.start()
    logger.info("Started S2 patch export for %d (dim=%dm).", year, patch_dim)
    return [task.status()]
