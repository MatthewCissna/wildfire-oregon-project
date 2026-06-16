# Data Sources — Comparison & Justification

All layers are clipped to Oregon and aligned to a common spatiotemporal grid
(H3 hexes, weekly steps, 2001–present where available). This document records what
we pull, the alternatives we considered, and **why** — with an emphasis on the
sources that give us an edge: ignition cause and fine-grained weather.

## Primary source: Google Earth Engine

| Role | Dataset (EE ID) | Resolution | Why this one |
|---|---|---|---|
| Burned area (label) | `MODIS/061/MCD64A1` | 500 m, monthly | Long, consistent burned-area record back to 2000; the backbone fire label |
| Active fire (label) | `MODIS/061/MOD14A1` | 1 km, daily | Daily thermal anomalies → fine-grained active-fire timing |
| Active fire (label/imagery) | `FIRMS` | 1 km, daily | Near-real-time detections; cross-checks MODIS thermal |
| Imagery (CNN) | `COPERNICUS/S2_SR_HARMONIZED` | 10–20 m | High-res multispectral for patch classification; SWIR bands key for fire |
| Imagery (alt) | `LANDSAT/LC08\|09/C02/T1_L2` | 30 m | Longer archive; fallback/cross-sensor for the detector |
| Vegetation / fuel | `MODIS/061/MOD13A1` (NDVI/EVI) | 500 m, 16-day | Greenness & fuel-state dynamics and anomalies |
| Land cover | `ESA/WorldCover/v200` | 10 m | Modern, high-res fuel-type stratification |
| Topography | `USGS/SRTMGL1_003` | 30 m | Elevation → slope & aspect (fire spread drivers) |
| Weather | `IDAHO_EPSCOR/GRIDMET` | ~4 km, daily | **Fire-relevant** bands: VPD, ERC, BI, wind, RH, precip, temp |
| Drought | `GRIDMET/DROUGHT` | ~4 km | PDSI & drought indices → fuel-dryness memory |

Spectral indices computed from imagery: **NDVI, NBR, NDMI, BAI** (vegetation,
burn, moisture, burned-area). LANDFIRE fuel layers are desirable but not reliably
available in EE; WorldCover + NDVI dynamics serve as the fuel proxy, with a
documented hook to ingest LANDFIRE rasters if obtained.

## Second source — the decision

The brief asked us to evaluate complementary sources and justify the pick. We
considered four and chose a **combination**, because each adds a distinct signal:

| Option | Adds | Cost / risk | Verdict |
|---|---|---|---|
| **GRIDMET** (weather + fire-danger) | VPD, ERC, BI — directly fire-relevant, daily, gridded | Already in EE; trivial | **Core.** This is where fine-grained weather skill comes from |
| **NIFC / FPA-FOD** (ignition cause) | **Lightning vs human** cause, discovery date, size | FPA-FOD is a one-time download; NIFC live API for recent years | **Core differentiator.** Cause is the underused signal (see literature.md) |
| **OSM** (roads / power lines) | Human-ignition proximity proxies | Heavy fetch; cached | **Yes**, as static features complementing cause |
| NOAA / RAWS stations | Ground-truth point weather | Sparse, gap-filling, station metadata churn | **Deferred.** GRIDMET already assimilates stations; marginal gain for the effort |

**Final choice:** GRIDMET (weather/fire-danger) **+** NIFC/FPA-FOD (ignition cause)
**+** OSM (infrastructure proximity). Rationale: GRIDMET delivers the fine-grained,
physically meaningful weather (VPD/ERC/BI) that separates fire days from non-fire
days; NIFC/FPA-FOD contributes the **lightning-vs-human** distinction that most
published occurrence models omit; OSM operationalizes the human-ignition geography.
RAWS/NOAA were deferred because GRIDMET already blends station data and the added
ingestion complexity wasn't justified for this scope.

## How each feeds the models

- **Risk heatmap & fire-count:** GRIDMET + drought + NDVI + topography + land cover
  on the grid, **plus** NIFC ignition-cause densities and OSM distances as static
  per-cell features.
- **Detection CNN:** Sentinel-2 (or Landsat) patches + indices; labels from
  FIRMS/MOD14 active fire aligned to the imagery date.

## Access, licensing, and reproducibility

- Earth Engine: free **noncommercial/academic** tier; project ID is a config
  variable, credentials are machine-local and git-ignored
  (see [`earth_engine_setup.md`](earth_engine_setup.md)).
- FPA-FOD: U.S. Forest Service Research Data Archive (RDS-2013-0009.6), public.
- NIFC WFIGS: public ArcGIS REST services.
- OSM: ODbL.
- Each dataset retains its own license; this repo's code is MIT.

## Offline fallback

When Earth Engine isn't configured, `wildfire.ingest.synthetic` emits the **same
canonical tables** (grid / weather panel / fire events / patches) from a latent fire
model, so the full pipeline — and every metric in [`../RESULTS.md`](../RESULTS.md) —
is reproducible end-to-end without credentials.
