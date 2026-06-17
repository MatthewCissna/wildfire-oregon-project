"""Human-readable labels for engineered feature codes.

Used by the SHAP figure and the website so charts read "NDVI, 8-week average"
instead of ``ndvi_roll8``.
"""

from __future__ import annotations

# Base variables.
_BASE = {
    "tmax": "Max temperature",
    "rmin": "Min relative humidity",
    "wind": "Wind speed",
    "precip": "Precipitation",
    "vpd": "Vapor pressure deficit",
    "erc": "Energy release component",
    "bi": "Burning index",
    "pdsi": "Drought (PDSI)",
    "ndvi": "NDVI (greenness)",
    "fuel_load": "Fuel load",
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "days_since_rain": "Days since rain",
    "ndvi_anom": "NDVI anomaly",
    "month": "Month",
    "doy_sin": "Seasonal cycle (sin)",
    "doy_cos": "Seasonal cycle (cos)",
    "hist_human_fires": "Historical human-caused fires",
    "hist_lightning_fires": "Historical lightning fires",
    "hist_fire_density": "Historical fire density",
    "wind_x_dryness": "Wind × dryness",
    "fuel_x_dryness": "Fuel × dryness",
    "drought_x_dryness": "Drought × dryness",
    "fire_lag1": "Burned previous week",
}

# Weeks per rolling-window step (weekly grid).
_STEP_WEEKS = {"2": "2-week", "4": "4-week", "8": "8-week"}


def label_feature(code: str) -> str:
    """Return a human-readable label for an engineered feature code."""
    if code in _BASE:
        return _BASE[code]
    for suffix, kind in (("_roll", "avg"), ("_sum", "total")):
        if suffix in code:
            base, _, win = code.partition(suffix)
            base_lbl = _BASE.get(base, base.replace("_", " ").title())
            window = _STEP_WEEKS.get(win, f"{win}-step")
            verb = "average" if kind == "avg" else "total"
            return f"{base_lbl}, {window} {verb}"
    return code.replace("_", " ").title()


def label_map(codes) -> dict:
    return {c: label_feature(c) for c in codes}
