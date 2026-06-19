"""Earth Engine authentication & initialization.

Centralizes the one call every GEE pull needs. The project ID is read from config
(or the ``EE_PROJECT`` env var) — never hardcoded. Two credential paths:

* **Local dev / interactive:** the machine-local OAuth token created by
  ``earthengine authenticate``. Default; nothing to set up beyond the project ID.
* **Headless / CI (GitHub Actions, etc.):** a Google Cloud **service account** with
  Earth Engine access. Provide its JSON key via env var:

  - ``EE_SERVICE_ACCOUNT_KEY`` — base64-encoded JSON (easy GitHub-Secret format), OR
  - ``EE_SERVICE_ACCOUNT_FILE`` — path to the JSON file on disk.

  The service account email is read from the JSON; nothing else is needed.
  See ``docs/github_deploy_setup.md`` for the one-time setup.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from pathlib import Path

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

    # Headless path: service-account credentials supplied via env var.
    creds = _service_account_credentials()
    if creds is not None:
        init_kwargs["credentials"] = creds

    try:
        ee.Initialize(**init_kwargs)
    except Exception as exc:  # broad: EE raises several auth-related exception types
        # Most common cause: not yet authenticated on this machine.
        raise RuntimeError(
            "Earth Engine failed to initialize. Most likely you haven't authenticated "
            "this machine yet. Run:\n\n    uv run earthengine authenticate\n\n"
            "Headless/CI: set EE_SERVICE_ACCOUNT_KEY (base64 JSON) or "
            "EE_SERVICE_ACCOUNT_FILE (path).\n"
            f"Underlying error: {exc}"
        ) from exc

    _INITIALIZED = True
    logger.info("Earth Engine initialized with project %s", project)
    return project


def _service_account_credentials():
    """Return ``ee.ServiceAccountCredentials`` if a service-account key env var is set.

    Accepted env vars (first wins):
        EE_SERVICE_ACCOUNT_KEY   base64-encoded service-account JSON
        EE_SERVICE_ACCOUNT_FILE  filesystem path to the service-account JSON
    """
    import ee

    key_b64 = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    key_file = os.environ.get("EE_SERVICE_ACCOUNT_FILE")
    if not (key_b64 or key_file):
        return None

    if key_b64:
        try:
            # Strip any whitespace/newlines a copy-paste into a GitHub Secret may add.
            raw = base64.b64decode("".join(key_b64.split())).decode("utf-8")
            payload = json.loads(raw)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_KEY is set but couldn't be decoded as base64 JSON."
            ) from exc
        # ee.ServiceAccountCredentials wants a file path; write the JSON to a temp file.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write(raw)
        tmp.close()
        key_file = tmp.name
    else:
        payload = json.loads(Path(key_file).read_text(encoding="utf-8"))

    email = payload.get("client_email")
    if not email:
        raise RuntimeError("Service-account JSON has no client_email field.")
    logger.info("Using Earth Engine service account: %s", email)
    return ee.ServiceAccountCredentials(email, key_file)


def ee_available(cfg: Config | None = None) -> bool:
    """Return True if Earth Engine can be initialized (config + auth present)."""
    try:
        initialize_ee(cfg)
        return True
    except (EarthEngineNotConfigured, RuntimeError):
        return False
