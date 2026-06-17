# Oregon Wildfire ML — Interactive Atlas (website)

A self-contained static website presenting the system: an interactive risk map you
can click for per-cell detail, model results, a data explorer, and a formatted paper.

## View it

**Easiest:** double-click `index.html` (needs internet for the map library + basemap
tiles; everything else works offline).

**As a local server** (recommended — avoids any `file://` quirks):

```powershell
# from the repo root
uv run python -m http.server 8000 --directory site
# then open http://localhost:8000
```

## Tabs

- **Overview** — headline findings, key numbers, drivers, fire-by-year chart.
- **Risk Map** — interactive Leaflet map of all 7,224 Oregon hexes. Recolor by risk,
  fire rate, fuel, elevation, VPD or NDVI; filter by ecoregion; **click any hex** for
  full detail (terrain, fuel, climate normals, NDVI, fire history by year, modeled risk).
- **Models & Results** — baseline-vs-GBM tables (both CV schemes), SHAP, the CNN, and
  per-ecoregion count forecasts.
- **Data Explorer** — ecoregion summary (click a row to filter the map) + data sources.
- **Paper** — a formatted write-up of methods and findings.

## Regenerate the data

The site reads two generated files, `data/cells.js` and `data/meta.js`, built from the
pipeline outputs. After re-running the pipeline, refresh them with:

```powershell
uv run python scripts/build_site.py
```

That's the only build step — the rest is static HTML/CSS/JS (no framework, no bundler).
