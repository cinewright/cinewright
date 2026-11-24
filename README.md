<!-- badges -->
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cinewright/cinewright/badge)](https://scorecard.dev/viewer/?uri=github.com/cinewright/cinewright)

# Cinewright

Cinewright is a platform for generating images, video and audio on your own computer with open-weight models. A local core decides which model, quantization and computer runs a job. Workers run the generation engines; the first engine is WanGP (Wan2GP). You use Cinewright in your browser, and AI agents can use it through its MCP server. Cinewright is free software under the MIT licence.

> **Status: scaffold.** Cinewright cannot generate real images or videos yet. This build sets up the contracts, the structure and a walking skeleton that runs jobs on a mock engine. Real generation arrives with phase 1.

## What works now

- **Walking skeleton.** A text-to-image job goes from the internal API through the router and the queue to a worker running the mock engine. It comes back as a PNG asset with a provenance record, and its events stream over server-sent events. `tests/test_e2e_mock.py` runs this end to end over real HTTP.
- **Core and internal API.** The core serves the API under `/api/v1` on `127.0.0.1:8737`. Its contract is [`openapi.json`](openapi.json). The core keeps jobs, events, asset records and settings in SQLite, and stores asset files by their SHA-256.
- **Hardware detection.** At start, and on request from the Setup view, the core reads the NVIDIA GPUs (through `nvidia-smi`), the driver, RAM and free disk space.
- **Web UI shell.** It has Setup, Generate, Queue, Library, Hosts, Models and Settings views, plus placeholders for Cast, Recipes and Presets. Fonts and icons ship with the UI, so it makes no third-party requests.
- **MCP server.** It runs over streamable HTTP at `/mcp` and offers seven tools.
- **Model registry.** It has descriptors for Z-Image Turbo 6B, LTX-2 2.3 Distilled 22B, and Wan 2.2 Image-to-Video and Text-to-Video 14B. Every model file is pinned by SHA-256, including the text encoders, VAEs, upscalers, tokenizer and config files WanGP needs next to the model.
- **Windows installer.** `scripts/build.py` builds it.
- **WanGP adapter.** The worker side exists. It has been checked against a WanGP checkout without any downloaded models.
- **Worker supervision.** The core can start the local worker in the engine's own Python environment, restart it after a crash and stop it when the core exits. The Hosts view shows the worker's state and the VRAM it uses. `tests/test_supervisor_e2e.py` runs this with the mock engine.

## What does not work yet

- **Real generation.** Model downloads, WanGP setup and calibration arrive with phase 1. The core starts its worker only once WanGP is set up. Until then, an installed Cinewright starts no worker, the Hosts view shows "No worker", and the Generate view reports that no model can run the request.
- **Later phases.** Recipes, cast and presets arrive in phase 2, remote hosts and approvals in phase 3. Their API routes return empty lists or `501 Not Implemented`.
- **Stopping Cinewright.** There is no quit command yet. The core runs until you sign out, or until you end the `pythonw.exe` process in Task Manager. A worker the core started ends with it.
- **Releases.** There is no release yet, so no installer can be downloaded.
- **Code signing.** The installer is not code-signed yet, so Windows SmartScreen warns before it runs.

## Tutorial: run the walking skeleton

This tutorial takes about ten minutes. You run a job through Cinewright on the mock engine, then open the web UI. You need Git, Python 3.12 and Node.js (`^20.19.0 || >=22.12.0`). The commands are for Git Bash or a Linux shell.

1. Get the code and install it into a virtual environment:

   ```sh
   git clone https://github.com/cinewright/cinewright.git
   cd cinewright
   python -m venv .venv
   source .venv/Scripts/activate    # Linux: source .venv/bin/activate
   python -m pip install -e ".[dev]"
   ```

2. Run the end-to-end test:

   ```sh
   pytest tests/test_e2e_mock.py -v
   ```

   The test starts a real core on a free port, with a temporary data directory. It then starts the mock worker as a separate process (`python -m cinewright_worker --engine mock`), which registers with the core's enrollment token. Next it submits a text-to-image job pinned to `z-image-turbo`. The mock worker stands in for WanGP and writes a small PNG instead of running the model. The test then checks four things: the job's events over SSE, the PNG, its provenance record (seed, model, host) and the MCP tools at `/mcp`. You should see `1 passed`.

3. Build the web UI and the wheel:

   ```sh
   python scripts/build.py --skip-installer
   ```

4. Start Cinewright:

   ```sh
   python -m cinewright
   ```

   Your browser opens `http://127.0.0.1:8737/`. The Setup view shows the hardware the core detected. The Generate view shows that no model can run a request, because this core runs no worker. The core keeps its data in the default data directory and writes its log to `logs/cinewright.log` there, so the terminal stays quiet. Stop the core with Ctrl+C.

## How-to guides

### Install Cinewright on Windows

1. Download `CinewrightSetup-<version>.exe` from the [GitHub releases](https://github.com/cinewright/cinewright/releases). No release exists yet.
2. Run it. Setup installs Cinewright for your user account, into `%LOCALAPPDATA%\Programs\Cinewright`, and needs no administrator rights. It downloads Python 3.12 and Cinewright's Python packages during installation, so it needs an internet connection.
3. Start Cinewright from the Start menu. The web UI opens in your browser. If Cinewright already runs, starting it again only opens the browser.

For a silent installation, run `CinewrightSetup-<version>.exe /S /D=<folder>`. `/D=` must come last and must not be quoted.

### Uninstall Cinewright

Open Settings > Apps > Installed apps, select Cinewright and choose Uninstall. The uninstaller removes the program, the shortcuts and the Add/Remove Programs entry. It then asks whether to delete your data in `%LOCALAPPDATA%\Cinewright`. The default is No, and a silent uninstall (`uninstall.exe /S`) always keeps the data. If Cinewright is running, the uninstaller stops with a message; end the `pythonw.exe` process first.

### Set up a development environment

You need Python 3.12 and Node.js `^20.19.0 || >=22.12.0`. To build the installer on Windows, you also need [NSIS 3](https://nsis.sourceforge.io/Download).

```sh
python -m venv .venv
source .venv/Scripts/activate    # Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Build the web UI, the wheel and the installer

```sh
python scripts/build.py                    # web UI, wheel, uv, icon, installer -> dist/
python scripts/build.py --skip-installer   # everything except makensis
python scripts/build.py --skip-web         # reuse the web UI that is already built
```

On Linux, if `makensis` is not on PATH, the script installs the `nsis` package with `apt-get` when that needs no password. Otherwise it unpacks NSIS into `build/nsis` without root. [`installer/README.md`](installer/README.md) describes the steps and what the installer does.

### Run the tests and checks

```sh
ruff check . && ruff format --check src tests
pytest -q
cd web && npm test && npm run typecheck
```

`pytest -q` also runs the web tests. The component tests need `web/node_modules`: run `npm ci` in `web/` or `python scripts/build.py` first, or pytest skips them.

### Work on the web UI

Start the core with `python -m cinewright`, then run `npm run dev` in `web/`. The Vite dev server proxies `/api` and `/mcp` to the core on `127.0.0.1:8737`. [`web/README.md`](web/README.md) lists the scripts.

### Change the API contract

The routes, `src/cinewright/api/schemas.py` and the generated `openapi.json` form the contract. After a change, regenerate the document and the web client types. A test fails while `openapi.json` does not match the code.

```sh
python -m cinewright.devtools openapi openapi.json
cd web && npm run gen:api
```

### Connect an MCP client

While Cinewright runs, point an MCP client that supports streamable HTTP at `http://127.0.0.1:8737/mcp`. The server accepts only the `Host` values `127.0.0.1` and `localhost`.

### Use another port or data directory

Set `CINEWRIGHT_PORT` or `CINEWRIGHT_HOME` before you run `python -m cinewright` (see [Environment variables](#environment-variables)). The Start menu shortcut sets neither.

## Reference

### Requirements

| Use | Needs |
| --- | --- |
| Installed app | 64-bit Windows; an internet connection during setup |
| Development | Python 3.12 or newer; Node.js `^20.19.0 \|\| >=22.12.0`; NSIS 3 to build the installer on Windows |
| Worker package | Python 3.10 or newer, standard library only |

### Environment variables

| Variable | Read by | Meaning |
| --- | --- | --- |
| `CINEWRIGHT_HOME` | core, worker | Data directory. Default: `%LOCALAPPDATA%\Cinewright` on Windows, `$XDG_DATA_HOME/cinewright` or `~/.local/share/cinewright` elsewhere |
| `CINEWRIGHT_PORT` | core | Port, default `8737` |
| `CINEWRIGHT_HOST` | core | Bind address, default `127.0.0.1`. Only `127.0.0.1` and `localhost` are accepted; any other value stops the launcher with exit code 2 |
| `CINEWRIGHT_CORE_URL` | worker | Core URL, default `http://127.0.0.1:8737` |
| `CINEWRIGHT_ENROLLMENT_TOKEN` | worker | Registration token; the core creates a new one at each start and holds it in memory only |

The core starts its worker with `CINEWRIGHT_CORE_URL`, `CINEWRIGHT_ENROLLMENT_TOKEN` and `CINEWRIGHT_HOME` set, and removes every other `CINEWRIGHT_*` variable from the worker's environment. It also sets `HTTP_PROXY` and `HTTPS_PROXY` to a closed port on `127.0.0.1` and `NO_PROXY` to `127.0.0.1,localhost` (on Linux also the lower-case names), replacing any proxy setting of your own.

### Data directory

| Path | Holds |
| --- | --- |
| `db/cinewright.sqlite3` | Jobs, events, assets, workers, settings (SQLite, WAL) |
| `assets/<sha[:2]>/<sha256>` | Inputs and outputs, stored by SHA-256 |
| `models/` | Model files (phase 1) |
| `engines/` | Engine data, e.g. `engines/wan2gp/wgp_config.json` |
| `engines/worker/cinewright_worker/` | Copy of the worker package that the engine's Python runs; the core replaces it at each start |
| `logs/cinewright.log` | Launcher and core log |
| `logs/worker.log` | Output of the worker the core started: the worker's log and the engine's messages |

### Commands

End users never need these. They exist for CI and development.

| Command | Does |
| --- | --- |
| `python -m cinewright` | Starts the core and opens the web UI in the browser |
| `python -m cinewright --version` | Prints `cinewright <version>` |
| `python -m cinewright_worker --engine mock\|wan2gp [--wan2gp-root PATH] [--once] [--stop-on-stdin-eof]` | Runs a worker. It needs `CINEWRIGHT_ENROLLMENT_TOKEN` from the core, and an installed core starts no worker until WanGP is set up (phase 1). `mock` stands in for WanGP and serves its model types. With `--stop-on-stdin-eof` the worker stops when its standard input closes; the core starts its worker this way |
| `python -m cinewright.devtools openapi [PATH]` | Writes the OpenAPI document |
| `python -m cinewright.devtools surface` | Prints the public surface: routes, schemas and MCP tools |
| `python scripts/build.py [--skip-web] [--skip-installer]` | Builds the web UI, the wheel and the installer |

### Internal API

All routes live under `/api/v1`. [`openapi.json`](openapi.json) is the contract, and [ARCHITECTURE.md](ARCHITECTURE.md) has the route table. Errors are RFC 9457 problem documents (`application/problem+json`). Operations of later phases return `501` with the problem type `urn:cinewright:problem:not-implemented`.

### MCP tools

| Tool | Does |
| --- | --- |
| `list_capabilities` | Lists capabilities, models, quantizations, limits and calibration status |
| `estimate` | Checks a job request without starting it: feasibility, route, runtime, cost |
| `submit` | Starts a job and returns its id and state |
| `job_status` | Reports state, progress, ETA, output hashes and error of a job |
| `fetch_asset` | Returns an asset's metadata, provenance and content URL |
| `list_recipes` | Lists recipes (phase 2) |
| `list_cast` | Lists cast entries (phase 2) |

The core records jobs submitted over MCP as `ui` jobs until issue #7 adds MCP client attribution.

### Models in the registry

| Registry id | WanGP model types | Capabilities | Licence |
| --- | --- | --- | --- |
| `z-image-turbo` | `z_image` | t2i | Apache-2.0 |
| `ltx-2.3-distilled-22b` | `ltx2_22B_distilled`, `ltx2_22B_distilled_gguf_*` | t2v, i2v, t2i, i2i, audio | LTX-2 Community License Agreement |
| `wan-2.2-i2v-14b` | `i2v_2_2` | i2v | Apache-2.0 |
| `wan-2.2-t2v-14b` | `t2v_2_2` | t2v, t2i | Apache-2.0 |

[`src/cinewright/registry/README.md`](src/cinewright/registry/README.md) explains the descriptor fields and where the hashes come from.

### Repository layout

| Path | Contents |
| --- | --- |
| `src/cinewright/` | Core: API, router, scheduler, asset store, database, MCP server, launcher |
| `src/cinewright_worker/` | Worker and engine adapters (`mock`, `wan2gp`) |
| `src/cinewright/registry/` | Model descriptors, shared engine files and the WanGP pin |
| `web/` | Web UI (React, TypeScript, Vite), built into `src/cinewright/web_dist/` |
| `installer/` | NSIS script, icon renderer, pins and notices of the Windows installer |
| `scripts/build.py` | Build script |
| `design/` | Design system: brand book, tokens, components, logos |
| `tests/` | pytest suite |

## Explanation

### Core and workers

The core decides and the workers compute. The core owns the API, the registry, the router, the queue and the asset store. Workers are separate processes that pull jobs from the core (`POST /worker/lease`). Each worker drives one engine through an adapter, reports progress, uploads its outputs and completes the job. The worker package uses only the Python standard library, so it can run inside an engine's own environment. WanGP, for example, needs Python 3.11, while the core needs Python 3.12.

### Worker supervision

The core keeps one long-running worker on your computer, so WanGP's models stay loaded between jobs. It runs the worker with the engine's own Python, isolated from your Python settings, and from a copy of the worker package in the data directory, never from the core's own packages.

- **Crashes.** If the worker ends unexpectedly, the core starts it again after a pause that doubles with each quick crash, from 1 second up to 1 minute. Jobs keep running on the new worker. A job that was running when the worker crashed returns to the queue when its lease expires, after two minutes.
- **Stopping.** When the core exits, it stops granting jobs, asks the worker to stop by closing the worker's standard input, and ends the worker after 5 seconds. If the core itself is killed, or ends when you sign out, Windows ends the worker too, because the worker belongs to a job object that dies with the core. On Linux the worker gets a kill signal when the core ends.
- **Network.** The worker's proxy settings point at a closed local port, so engine libraries that honour the standard proxy variables (Python's `urllib` and `requests` among them) cannot reach the internet, while the worker still reaches the core. Code that ignores these variables is not stopped: on Windows this is configuration only, because a per-user installation cannot add firewall rules.
- **Health.** The Hosts view shows the worker's state (starting, ready, busy or failed) and the VRAM it uses. The state comes from the worker's own reports and, while the worker is starting or down, from the core's view of the process.

### Routing

For each request, the router checks four things in order. Policy comes first: the data class and the licence's geographic exclusions. Feasibility comes next: capability, a pinned model, limits, a registered worker for the engine model, and calibration. Then come cost (local jobs cost €0.00) and runtime. The estimate lists every decision, so the UI and MCP clients can show why a request cannot run.

### Local only

The core listens on `127.0.0.1` only. A web page in your browser can still send requests to local addresses, so the core also accepts only the `Host` values `127.0.0.1:<port>` and `localhost:<port>`. It rejects foreign `Origin` headers, and it refuses the body types a page can send without a CORS preflight. Workers authenticate with a token per worker, which they get by registering with the core's enrollment token.

### Provenance

Every output asset has a provenance record: prompt, seed, model, quantization, engine version, host, client and job. `GET /api/v1/assets/{sha256}/provenance` exports it as sidecar JSON.

### Why WanGP is not bundled

WanGP is source-available under the WanGP Community License 2.0, and Cinewright never redistributes it. Engine provisioning (phase 1) downloads a pinned commit from WanGP's official repository.

[ARCHITECTURE.md](ARCHITECTURE.md) records the components, data flows, boundaries and current decisions.

## WanGP notice

Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright.

Licence: <https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt>

## Licence

Cinewright is licensed under the [MIT License](LICENSE). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the third-party software Cinewright uses and bundles.
