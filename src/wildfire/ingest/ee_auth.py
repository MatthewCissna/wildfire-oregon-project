"""Earth Engine authentication & initialization.

Centralizes the one call every GEE pull needs. The project ID is read from config
(or the ``EE_PROJECT`` env var) — never hardcoded. Credentials come from the
machine-local OAuth token created by ``earthengine authenticate`` and are never
stored in the repo.
"""

from __future__ import annotations

import logging

from wildfire.config import Config, load_config

logger = logging.getLogger(__name__)

# Earth Engine's high-volume endpoint — used for many small/automated requests.
_HIGHVOLUME_URL = "https://earthengine-highvolume.googleapis.com"

_INITIALIZED = False


class EarthEngineNotConfigured(RuntimeError):
    """Raised when no EE project is configured, so callers can fall back gracefully."""


def initialize_ee(cfg: Config | None = None, *, force: bool = False) -> str:
    """Initialize the Earth Engine client and return the project ID in use.

    Raises:
        EarthEngineNotConfigured: if no project ID is set (run on synthetic data instead).
        RuntimeError: if ``earthengine-api`` isn't installed or auth is missing.
    """
    global _INITIALIZED
    cfg = cfg or load_config()
    project = cfg.ee_project
    if not project:
        raise EarthEngineNotConfigured(
            "No Earth Engine project configured. Set earth_engine.project_id in "
            "configs/config.yaml (or the EE_PROJECT env var), or run with --synthetic. "
            "See docs/earth_engine_setup.md."
        )

    try:
        import ee  # noqa: WPS433  (import here so the rest of the repo doesn't require EE)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "earthengine-api is not installed. Run `uv sync`."
        ) from exc

    if _INITIALIZED and not force:
        return project

    init_kwargs = {"project": project}
    if cfg.get("earth_engine.use_highvolume", True):
        init_kwargs["opt_url"] = _HIGHVOLUME_URL

    try:
        ee.Initialize(**init_kwargs)
    except Exception as exc:  # broad: EE raises several auth-related exception types
        # Most common cause: not yet authenticated on this machine.
        raise RuntimeError(
            "Earth Engine failed to initialize. Most likely you haven't authenticated "
            "this machine yet. Run:\n\n    uv run earthengine authenticate\n\n"
            f"Underlying error: {exc}"
        ) from exc

    _INITIALIZED = True
    logger.info("Earth Engine initialized with project %s", project)
    return project


def ee_available(cfg: Config | None = None) -> bool:
    """Return True if Earth Engine can be initialized (config + auth present)."""
    try:
        initialize_ee(cfg)
        return True
    except (EarthEngineNotConfigured, RuntimeError):
        return False
