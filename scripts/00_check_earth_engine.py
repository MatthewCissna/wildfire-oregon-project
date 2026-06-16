"""Stage 0 — verify Earth Engine is set up correctly.

Run this after completing docs/earth_engine_setup.md:

    uv run python scripts/00_check_earth_engine.py
"""

from __future__ import annotations

import sys

from wildfire.config import load_config
from wildfire.ingest.ee_auth import EarthEngineNotConfigured, initialize_ee
from wildfire.utils import init_console


def main() -> int:
    init_console()
    cfg = load_config()
    try:
        project = initialize_ee(cfg)
    except EarthEngineNotConfigured as exc:
        print("⚠️  Earth Engine is not configured yet.")
        print(f"   {exc}")
        print("\n   The pipeline still runs on synthetic data: add --synthetic to any stage.")
        return 1
    except RuntimeError as exc:
        print("❌ Earth Engine could not initialize.")
        print(f"   {exc}")
        return 2

    print(f"✅ Earth Engine initialized with project: {project}")

    # A tiny live query to prove data access end-to-end.
    try:
        import ee

        from wildfire.ingest.boundary import oregon_ee_geometry

        oregon = oregon_ee_geometry(cfg)
        start, end = cfg.get("time.start"), cfg.get("time.end")
        col = (
            ee.ImageCollection(cfg.get("datasets.burned_area"))
            .filterDate(start, end)
            .filterBounds(oregon)
        )
        n = col.size().getInfo()
        print(f"✅ Test query OK — Oregon MODIS burned-area image count: {n}")
    except Exception as exc:  # pragma: no cover  (depends on live service)
        print(f"⚠️  Initialized, but the test query failed: {exc}")
        return 3

    print("\n🎉 Earth Engine is ready. Run:  uv run python scripts/01_ingest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
