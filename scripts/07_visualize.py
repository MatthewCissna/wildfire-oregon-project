"""Stage 7 — render the deliverable maps.

  * outputs/maps/risk_heatmap.html   interactive per-cell risk heatmap
  * outputs/figures/risk_map.png     static high-resolution risk map
  * outputs/maps/fire_count_map.html interactive per-region predicted fire counts

    uv run python scripts/07_visualize.py
"""

from __future__ import annotations

import geopandas as gpd
import joblib
import pandas as pd

from wildfire.config import load_config
from wildfire.features.regions import build_region_season, region_feature_columns
from wildfire.utils import init_console
from wildfire.viz.maps import count_choropleth, interactive_heatmap, static_map


def main() -> int:
    init_console()
    cfg = load_config()

    surface = gpd.read_parquet(cfg.path_for("data_processed") / "risk_surface.parquet")
    html = interactive_heatmap(surface, cfg.path_for("maps") / "risk_heatmap.html")
    png = static_map(surface, cfg.path_for("figures") / "risk_map.png")
    print("✅ Risk maps")
    print(f"   interactive : {html}")
    print(f"   static      : {png}")

    # Per-district predicted fire counts (ODF fire-protection districts) for the
    # latest season; falls back to ecoregions if the district boundary fetch fails.
    try:
        df = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        try:
            from wildfire.ingest.districts import district_count_layer

            merged = district_count_layer(cfg, features=df)["districts"]
            unit, name_col = "district", "district"
        except Exception as exc:
            print(f"   (districts unavailable: {exc}; falling back to ecoregions)")
            from wildfire.ingest.datasets import load_canonical

            grid = load_canonical(cfg)["grid"]
            region_geom = grid.dissolve("ecoregion").reset_index()[["ecoregion", "geometry"]].rename(columns={"ecoregion": "district"})
            region = build_region_season(cfg, df)
            latest = region[region["year"] == region["year"].max()].copy()
            cm = joblib.load(cfg.path_for("models") / "count_model.joblib")
            latest["pred_count"] = cm.predict(latest)
            lo, hi = cm.predict_interval(latest)
            latest["pred_lo"], latest["pred_hi"] = lo, hi
            merged = region_geom.merge(latest.rename(columns={"region": "district"})[["district", "pred_count", "pred_lo", "pred_hi"]], on="district")
            unit, name_col = "ecoregion", "district"

        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
        cmap_html = count_choropleth(merged, cfg.path_for("maps") / "fire_count_map.html", value_col="pred_count")
        merged.drop(columns="geometry").to_csv(cfg.path_for("metrics") / "region_count_predictions.csv", index=False)
        print(f"✅ Fire-count map (by {unit})")
        print(f"   interactive : {cmap_html}")
        print(f"   predictions : {cfg.path_for('metrics') / 'region_count_predictions.csv'}")
    except Exception as exc:
        print(f"⚠️  Fire-count map skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
