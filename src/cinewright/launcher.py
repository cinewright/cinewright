"""Start Cinewright: run the core on 127.0.0.1:<port> and open the web UI in the browser.

End users start it from the Start menu (a gui-script without console), so everything goes to
`<data dir>/logs/cinewright.log`. The core provisions the WanGP engine in the background on the first
start and starts the local worker on it once it is ready. A second start only opens the browser.
`--version` exists for CI.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import uvicorn

from cinewright import __version__
from cinewright.api.schemas import API_PREFIX
from cinewright.config import Config, load_config

log = logging.getLogger("cinewright.launcher")

READY_TIMEOUT_S = 60.0
LOG_MAX_BYTES = 5 * 1024 * 1024


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logs_dir / "cinewright.log", maxBytes=LOG_MAX_BYTES, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per health poll and MCP call is noise


def _get(config: Config, path: str, timeout: float) -> httpx.Response | None:
    try:
        return httpx.get(f"{config.base_url}{API_PREFIX}{path}", timeout=timeout, trust_env=False)
    except httpx.HTTPError:
        return None


def is_running(config: Config) -> bool:
    """Whether a Cinewright core already answers on the configured port."""
    response = _get(config, "/system", timeout=2.0)
    if response is None or response.status_code != 200:
        return False
    try:
        return response.json().get("name") == "Cinewright"
    except ValueError:
        return False


def open_when_ready(config: Config, timeout: float = READY_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _get(config, "/health", timeout=1.0)
        if response is not None and response.status_code == 200:
            webbrowser.open(f"{config.base_url}/")
            return True
        time.sleep(0.25)
    log.error("The core did not answer within %.0f s, so the browser was not opened.", timeout)
    return False


def serve(config: Config) -> bool:
    """Run the core until it stops; False when it never finished starting (e.g. an invalid registry)."""
    from cinewright.app import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config=config, provision=True),
            host=config.host,
            port=config.port,
            log_config=None,  # no console: our file handler on the root logger takes uvicorn's records
            use_colors=False,
            timeout_graceful_shutdown=5,  # open event streams must not block the exit
        )
    )
    threading.Thread(target=open_when_ready, args=(config,), name="open-browser", daemon=True).start()
    server.run()
    return server.started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cinewright", description="Start Cinewright and open its web UI in the browser.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"cinewright {__version__}")
        return 0

    config = load_config()
    setup_logging(config.paths.logs_dir)
    if not config.loopback:
        log.error("CINEWRIGHT_HOST=%s is refused: binding beyond 127.0.0.1 needs an access token, which arrives in phase 1.", config.host)
        return 2
    if is_running(config):
        log.info("Cinewright already runs on %s; opening the browser.", config.bind)
        webbrowser.open(f"{config.base_url}/")
        return 0
    log.info("Starting Cinewright %s on %s with data in %s", __version__, config.bind, config.home)
    try:
        started = serve(config)
    except Exception:
        log.exception("Cinewright stopped with an error")
        return 1
    if not started:
        log.error("Cinewright did not start; the log above names the cause.")
        return 1
    return 0
