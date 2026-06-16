"""Configuration loading.

A thin, typed wrapper over ``configs/config.yaml`` with environment-variable
overrides. Nothing in the codebase should hardcode paths, dataset IDs, or the
Earth Engine project — everything flows through here.

Environment overrides (take precedence over the YAML):
    EE_PROJECT   -> earth_engine.project_id
    WILDFIRE_CONFIG -> path to an alternate config file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/wildfire/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@dataclass
class Config:
    """Parsed configuration with convenience accessors."""

    raw: dict[str, Any] = field(default_factory=dict)
    path: Path = DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------ #
    # Generic access
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value with a dotted path, e.g. ``cfg.get('grid.h3_resolution')``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    # ------------------------------------------------------------------ #
    # Common shortcuts
    # ------------------------------------------------------------------ #
    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))

    @property
    def ee_project(self) -> str | None:
        """Earth Engine Cloud project ID. EE_PROJECT env var overrides the YAML."""
        return os.environ.get("EE_PROJECT") or self.get("earth_engine.project_id")

    def path_for(self, key: str) -> Path:
        """Resolve a configured path (under ``paths:``) to an absolute Path."""
        rel = self.get(f"paths.{key}")
        if rel is None:
            raise KeyError(f"No path configured for paths.{key}")
        p = REPO_ROOT / rel
        return p

    def ensure_dirs(self) -> None:
        """Create all configured output/data directories if missing."""
        for key in self.raw.get("paths", {}):
            self.path_for(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML.

    Resolution order for the path:
        explicit ``path`` arg  ->  $WILDFIRE_CONFIG  ->  configs/config.yaml
    """
    cfg_path = Path(path or os.environ.get("WILDFIRE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            "Copy configs/config.yaml and adjust, or set $WILDFIRE_CONFIG."
        )
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw, path=cfg_path)
