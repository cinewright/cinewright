"""Runtime configuration from the environment.

  CINEWRIGHT_HOME   data directory (default: see `cinewright.paths.default_home`)
  CINEWRIGHT_HOST   bind address (default 127.0.0.1)
  CINEWRIGHT_PORT   port (default 8737)

Read at call time, never at import time, so tests can point `CINEWRIGHT_HOME` at a temp dir.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cinewright.api.schemas import DEFAULT_PORT
from cinewright.paths import DataPaths, default_home

ENV_HOME = "CINEWRIGHT_HOME"
ENV_HOST = "CINEWRIGHT_HOST"
ENV_PORT = "CINEWRIGHT_PORT"
# The core hands its per-start enrollment token to the worker processes it spawns in this variable.
ENV_ENROLLMENT_TOKEN = "CINEWRIGHT_ENROLLMENT_TOKEN"  # noqa: S105 - variable name, not a secret

DEFAULT_HOST = "127.0.0.1"
# The browser protection accepts only these Host names (cinewright.security), so only they may be bound.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


@dataclass(frozen=True)
class Config:
    home: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @property
    def paths(self) -> DataPaths:
        return DataPaths(self.home)

    @property
    def bind(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    home = env.get(ENV_HOME)
    port_text = env.get(ENV_PORT, str(DEFAULT_PORT)).strip()
    if not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise ValueError(f"{ENV_PORT} must be a port number between 1 and 65535, not {port_text!r}.")
    return Config(
        home=Path(home).expanduser() if home else default_home(env),
        host=env.get(ENV_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=int(port_text),
    )
