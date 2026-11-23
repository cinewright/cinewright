"""python -m cinewright_worker --engine mock|wan2gp [--wan2gp-root PATH] [--once]

The core starts worker processes itself and passes `CINEWRIGHT_ENROLLMENT_TOKEN` (required) and
`CINEWRIGHT_CORE_URL` (default http://127.0.0.1:8737) in the environment.

`--engine mock` stands in for Wan2GP: it registers as engine `wan2gp` serving the Wan2GP model types
of the bundled registry, because registry bindings name only real engines. It needs no GPU and is
meant for tests and development; its outputs record engine version `mock-1` in their provenance.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from cinewright_worker.client import WorkerClient
from cinewright_worker.loop import CORE_URL_ENV, DEFAULT_CORE_URL, ENROLLMENT_TOKEN_ENV, WorkerLoop

# Wan2GP model types bound in the bundled registry (src/cinewright/registry/models/*.yaml).
MOCK_STAND_IN_MODELS = (
    "z_image",
    "t2v_2_2",
    "i2v_2_2",
    "ltx2_22B_distilled",
    "ltx2_22B_distilled_gguf_q4_k_m",
    "ltx2_22B_distilled_gguf_q6_k",
    "ltx2_22B_distilled_gguf_q8_0",
)


def data_dir() -> str:
    """Cinewright's data directory: `CINEWRIGHT_HOME`, else the platform default (see ARCHITECTURE.md)."""
    override = os.environ.get("CINEWRIGHT_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "Cinewright")
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = xdg if os.path.isabs(xdg) else os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "cinewright")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cinewright_worker", description="Cinewright worker: pulls jobs from the core and runs them on one engine."
    )
    parser.add_argument("--engine", choices=("mock", "wan2gp"), required=True)
    parser.add_argument("--wan2gp-root", help="path of the Wan2GP checkout (required with --engine wan2gp)")
    parser.add_argument("--once", action="store_true", help="exit after one job")
    args = parser.parse_args(argv)
    if args.engine == "wan2gp" and not args.wan2gp_root:
        parser.error("--wan2gp-root is required with --engine wan2gp")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Taken out of the environment so engine code and its child processes never see it.
    token = os.environ.pop(ENROLLMENT_TOKEN_ENV, "")
    if not token:
        logging.getLogger("cinewright_worker").error("%s is not set; the core passes it to the workers it starts.", ENROLLMENT_TOKEN_ENV)
        return 2

    if args.engine == "wan2gp":
        from cinewright_worker.engines.wan2gp import Wan2GPAdapter

        adapter = Wan2GPAdapter(root=args.wan2gp_root, data_dir=data_dir())
    else:
        from cinewright_worker.engines.mock import MockAdapter

        adapter = MockAdapter(models=MOCK_STAND_IN_MODELS, engine="wan2gp")

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    client = WorkerClient(os.environ.get(CORE_URL_ENV) or DEFAULT_CORE_URL)
    return WorkerLoop(adapter, client, token, stop_event=stop, once=args.once).run()


if __name__ == "__main__":
    raise SystemExit(main())
