"""Maps: an interactive folium risk heatmap and a static high-resolution map.

Inputs are the per-cell risk surface (a GeoDataFrame with a ``risk`` column and hex
geometry) produced by ``wildfire.models.risk.predict_surface``.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from wildfire.config import Config, load_config


def _quantile_colormap(values: np.ndarray):
    import branca.colormap as cm

    v = values[~np.isnan(values)]
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    colormap = cm.LinearColormap(
        ["#2c7bb6", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=vmin, vmax=vmax, caption="Modeled wildfire risk (mean fire probability)",
    )
    return colormap


def interactive_heatmap(
    surface: gpd.GeoDataFrame, out_html: str | Path, *, risk_col: str = "risk"
) -> str:
    """Folium choropleth of per-cell risk, saved as a standalone HTML map."""
    import folium

    gdf = surface.dropna(subset=[risk_col]).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    colormap = _quantile_colormap(gdf[risk_col].to_numpy())

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feat: {
            "fillColor": colormap(feat["properties"][risk_col])
            if feat["properties"].get(risk_col) is not None else "#00000000",
            "color": "none",
            "weight": 0,
            "fillOpacity": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in (risk_col, "cell_id") if c in gdf.columns],
            aliases=["risk", "cell"][: sum(c in gdf.columns for c in (risk_col, "cell_id"))],
        ),
        name="Wildfire risk",
    ).add_to(m)
    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return str(out_html)


def static_map(
    surface: gpd.GeoDataFrame, out_png: str | Path, *, risk_col: str = "risk", basemap: bool = True
) -> str:
    """High-resolution static risk map (matplotlib), optional contextily basemap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = surface.dropna(subset=[risk_col]).to_crs("EPSG:3857")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    gdf.plot(
        column=risk_col, cmap="inferno", linewidth=0, ax=ax, legend=True,
        legend_kwds={"label": "Modeled wildfire risk", "shrink": 0.6},
        alpha=0.85,
    )
    if basemap:
        try:
            import contextily as ctx

            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=6)
        except Exception:
            pass  # offline / tile fetch failed -> plain map
    ax.set_axis_off()
    ax.set_title("Oregon wildfire risk surface", fontsize=14)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def count_choropleth(
    region_surface: gpd.GeoDataFrame, out_html: str | Path, *, value_col: str = "pred_count"
) -> str:
    """Interactive choropleth of predicted fire counts per region."""
    import folium

    gdf = region_surface.dropna(subset=[value_col]).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    colormap = _quantile_colormap(gdf[value_col].to_numpy())
    colormap.caption = "Predicted fires per district per season"
    name_field = next((c for c in ("district", "region", "ecoregion") if c in gdf.columns), None)
    fields = ([name_field] if name_field else []) + [value_col]
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feat: {
            "fillColor": colormap(feat["properties"][value_col])
            if feat["properties"].get(value_col) is not None else "#00000000",
            "color": "#555", "weight": 0.5, "fillOpacity": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=fields),
    ).add_to(m)
    colormap.add_to(m)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return str(out_html)
