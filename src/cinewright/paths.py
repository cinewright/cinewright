"""Data directory location and layout.

Windows: `%LOCALAPPDATA%\\Cinewright`; elsewhere `$XDG_DATA_HOME/cinewright` or `~/.local/share/cinewright`.
`CINEWRIGHT_HOME` overrides both (see `cinewright.config`).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def default_home(environ: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        base = env.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "Cinewright"
    xdg = env.get("XDG_DATA_HOME")
    base_dir = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base_dir / "cinewright"


@dataclass(frozen=True)
class DataPaths:
    """Layout below the data directory. Assets are content-addressed: `assets/<sha[:2]>/<sha>`."""

    root: Path

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    @property
    def db_file(self) -> Path:
        return self.db_dir / "cinewright.sqlite3"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def engines_dir(self) -> Path:
        return self.root / "engines"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def asset_file(self, sha256: str) -> Path:
        return self.assets_dir / sha256[:2] / sha256

    def ensure(self) -> None:
        for directory in (self.db_dir, self.assets_dir, self.models_dir, self.engines_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
