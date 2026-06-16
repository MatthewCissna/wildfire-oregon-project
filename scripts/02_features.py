"""Stage 2 — build the feature matrix from the canonical tables.

    uv run python scripts/02_features.py
"""

from __future__ import annotations

from wildfire.config import load_config
from wildfire.features.build import build_feature_matrix, feature_columns, save_feature_matrix
from wildfire.features.regions import build_region_season
from wildfire.ingest.datasets import load_canonical
from wildfire.utils import init_console


def main() -> int:
    init_console()
    cfg = load_config()
    data = load_canonical(cfg)

    df = build_feature_matrix(cfg, data)
    path = save_feature_matrix(cfg, df)

    region = build_region_season(cfg, df)
    region_path = cfg.path_for("data_processed") / "region_season.parquet"
    region.to_parquet(region_path, index=False)

    feats = feature_columns(df)
    print("\n✅ Features built")
    print(f"   rows           : {len(df):,}")
    print(f"   feature columns: {len(feats)}")
    print(f"   positive rate  : {df['fire'].mean()*100:.2f}%")
    print(f"   saved          : {path}")
    print(f"   region-season  : {len(region):,} rows -> {region_path}")
    print("   top features   :", ", ".join(feats[:12]), "...")
    print("\n   Next:  uv run python scripts/04_train_tabular.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
