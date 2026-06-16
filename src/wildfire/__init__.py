"""Oregon Wildfire ML System.

A reproducible pipeline for:
  1. a per-cell wildfire **risk heatmap**,
  2. a per-region **fire-count** prediction model, and
  3. a satellite-image **fire-detection** CNN.

See the README and docs/ for setup and methodology.
"""

__version__ = "0.1.0"

from wildfire.config import Config, load_config  # noqa: E402,F401
