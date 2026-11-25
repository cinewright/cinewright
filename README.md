<!-- badges -->
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cinewright/cinewright/badge)](https://scorecard.dev/viewer/?uri=github.com/cinewright/cinewright)

# Cinewright

Cinewright is a platform for generating images, video and audio on your own computer with open-weight models. A local core decides which model, quantization and computer runs a job. Workers run the generation engines; the first engine is WanGP (Wan2GP). You use Cinewright in your browser, and AI agents can use it through its MCP server. Cinewright is free software under the MIT licence.

> **Status: phase 1 in progress.** Models can be installed and jobs run on WanGP, but generation has not been checked on real hardware yet. The walking skeleton still runs jobs on a mock engine for tests and the tutorial below.

## What works now

- **Walking skeleton.** A text-to-image job goes from the internal API through the router and the queue to a worker running the mock engine. It comes back as a PNG asset with a provenance record, and its events stream over server-sent events. `tests/test_e2e_mock.py` runs this end to end over real HTTP.
- **Core and internal API.** The core serves the API under `/api/v1` on `127.0.0.1:8737`. Its contract is [`openapi.json`](openapi.json). The core keeps jobs, events, asset records and settings in SQLite, and stores asset files by their SHA-256.
- **Hardware detection.** At start, and on request from the Setup view, the core reads the NVIDIA GPUs (through `nvidia-smi`), the driver, RAM and free disk space.
- **Web UI shell.** It has Setup, Generate, Queue, Library, Hosts, Models and Settings views, plus placeholders for Cast, Recipes and Presets. Fonts and icons ship with the UI, so it makes no third-party requests.
- **Generate view.** It uploads a start image for image-to-video and image-to-image jobs and offers the resolutions, frame counts and frame rates of the pinned model or of the model the router picks. After you submit, it shows the job's progress and preview frames, then opens the result in the Library with its provenance record. It has not been checked with a real model yet.
- **MCP server.** It runs over streamable HTTP at `/mcp` and offers seven tools.
- **Model registry.** It has descriptors for Z-Image Turbo 6B, LTX-2 2.3 Distilled 22B, and Wan 2.2 Image-to-Video and Text-to-Video 14B. Every model file is pinned by SHA-256, including the text encoders, VAEs, upscalers, tokenizer and config files WanGP needs next to the model.
- **Windows installer.** `scripts/build.py` builds it.
- **WanGP adapter.** The worker side exists. It runs each job in the quantization the router chose, after checking that this quantization's files are installed, makes still images with LTX-2 and Wan 2.2 Text-to-Video (see [Images from video models](#images-from-video-models)), and reports which model settings it passed on to WanGP unchanged. When a job runs out of video memory, the error names the step it reached. An earlier version was checked against a WanGP checkout without any downloaded models; the per-quantization file checks have not been run against a real checkout yet. `tests/test_wan2gp_generation.py` runs an image-to-video, a text-to-image and an image-to-image job from the request to the provenance record on a stand-in for WanGP.
- **WanGP setup.** On the first start of the installed app, the core sets up WanGP in the background: the pinned WanGP commit, ffmpeg and WanGP's own Python environment, with every file checked against its pinned SHA-256. The Setup view shows the progress. Once WanGP is ready, the core starts its worker on it. Later starts reuse the installation. Windows only.
- **Model installs.** In the Models view you install a model in one quantization, follow its download, and remove it again. The core installs it with the shared files it needs into its model store, each file checked by SHA-256. The view also shows each installed model's size, the store's total, the storage quota and the free disk space, and lets you switch your computer's text encoder variant. After every change the core restarts the local worker, so a model can run seconds after its install. The API offers the same (`POST /api/v1/models/{model_id}/install?quant=`, `DELETE /api/v1/models/{model_id}`, `PATCH /api/v1/hosts/local`); `tests/test_e2e_model_install.py` runs an install end to end with the mock engine.
- **Worker supervision.** The installed app starts the local worker in WanGP's own Python environment once the WanGP setup is ready, restarts it after a crash and stops it when the core exits. The Hosts view shows the worker's state and the VRAM it uses. `tests/test_supervisor_e2e.py` runs this with the mock engine.

## What does not work yet

- **Real generation.** Models can be installed, and each job runs the installed quantization it is routed to. But generation on WanGP has not been checked on real hardware yet. In particular, nobody has checked yet whether image-to-image on LTX-2 changes the image or returns it almost unchanged at the default strength. Calibration arrives with phase 1. Until a model is installed, the worker on WanGP has no model files to serve, so the Generate view reports that no model can run the request.
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

   The test starts a real core on a free port, with a temporary data directory. It then starts the mock worker as a separate process (`python -m cinewright_worker --engine mock`), which registers with the core's enrollment token. Next it records `z-image-turbo` in `int8` as installed, without its files, because the router routes only installed quantizations, and submits a text-to-image job pinned to that model. The mock worker stands in for WanGP and writes a small PNG instead of running the model. The test then checks four things: the job's events over SSE, the PNG, its provenance record (seed, parameters, model and its files, host) and the MCP tools at `/mcp`. You should see `1 passed`.

3. Build the web UI and the wheel:

   ```sh
   python scripts/build.py --skip-installer
   ```

4. Start Cinewright:

   ```sh
   python -m cinewright
   ```

   Your browser opens `http://127.0.0.1:8737/`. The Setup view shows the hardware the core detected. Its Engine panel says that WanGP cannot be set up here: only the installed app on Windows sets it up, with the installer's `uv.exe`. The Generate view shows that no model can run a request, because no model files are installed yet. The core keeps its data in the default data directory and writes its log to `logs/cinewright.log` there, so the terminal stays quiet. Stop the core with Ctrl+C.

## How-to guides

### Install Cinewright on Windows

1. Download `CinewrightSetup-<version>.exe` from the [GitHub releases](https://github.com/cinewright/cinewright/releases). No release exists yet.
2. Run it. Setup installs Cinewright for your user account, into `%LOCALAPPDATA%\Programs\Cinewright`, and needs no administrator rights. It downloads Python 3.12 and Cinewright's Python packages during installation, so it needs an internet connection.
3. Start Cinewright from the Start menu. The web UI opens in your browser. If Cinewright already runs, starting it again only opens the browser.
4. On the first start, Cinewright sets up its engine WanGP in the background. It downloads several gigabytes, so this takes a while; open **Setup** to watch the progress. Once the Engine panel shows **Ready**, Cinewright starts its worker on WanGP, and the **Hosts** view shows the local worker as ready. Later starts skip the setup.
5. Install a model in the **Models** view (see [Install or remove a model](#install-or-remove-a-model)).

For a silent installation, run `CinewrightSetup-<version>.exe /S /D=<folder>`. `/D=` must come last and must not be quoted.

### Retry the WanGP setup

If the WanGP setup fails, for example because the internet connection dropped, the Setup view's Engine panel says what went wrong and what that means.

1. Fix the cause the message names, for example reconnect to the internet.
2. Open **Setup** and choose **Try again** in the Engine panel. Downloads continue where they stopped, and files that were already checked are not downloaded again.

If the panel shows no **Try again**, trying again cannot help: a file that already passed its hash check does not have the content Cinewright expects, so only a Cinewright update can fix it.

Quitting Cinewright during the setup is safe too: the next start continues it. When a downloaded file does not match its pinned hash, Cinewright deletes it and downloads it again on the retry. The details of each step are in `logs/provisioning.log` in the data directory.

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

### Install or remove a model

While Cinewright runs:

1. Open **Models**. The Model store panel shows how much is installed, the storage quota and the free disk space. Each card is one model in one quantization.
2. Choose **Install** on a card. If the model does not fit on the drive, a message at the top of the view says how much space is missing, and nothing is downloaded. Otherwise the card shows the download's progress. Once every file has passed its hash check, the card shows **Installed**, and the Generate view can use the model a few seconds later, without restarting Cinewright.
3. To remove a model, choose **Remove** on its card and confirm with **Remove model**. Text encoders and VAEs that another installed model needs stay.

Jobs run only on installed quantizations. If no quantization of the model you choose in the Generate view is installed, the estimate lists the missing ones. A request through the API can pin one quantization with `quant`; if that one is not installed, the estimate names it.

If an install fails, its card says why. An install that was still running when Cinewright closed shows as failed too. **Try again** keeps the files that were already downloaded and checked; **Remove** deletes them. Removing a model that a queued or running job needs is refused with a message.

### Generate an image or a clip

While Cinewright runs, with at least one model installed:

1. Open **Generate**, enter a prompt and choose a capability, for example `i2v` for a clip that starts from an image.
2. Leave **Model** at **Router chooses**, or pin a model.
3. Choose the **Data class**, for example **Biometric** for an image that shows a face.
4. For `i2v` and `i2i`, use **Choose image** to pick a PNG, JPEG or WebP file. Cinewright uploads it right away, with the selected data class, and shows a small preview. If you choose a stricter data class afterwards, Cinewright uploads the image again, so the Library keeps it under the stricter class. A looser class never loosens the stored one. **Replace image** and **Remove image** change the image. The uploaded image also appears in the Library.
5. Set **Resolution**, **Frames** and **Frame rate**, or leave them at **Model default** (an empty **Frames** field). They offer what the model can render: the pinned model, or the model the estimate names, and stay disabled until one of them is known. A frame count off the model's frame grid moves to the nearest valid count when you leave the field. For example, LTX-2 2.3 Distilled renders 17, 25, 33 and so on, so 30 becomes 33. Still images (`t2i`, `i2i`) have no frames or frame rate. Changing the capability, or removing the pinned model before the router has picked one, sets all three back to the model defaults.
6. Choose **Generate**. While something is missing, the button says what, for example **Add a start image**.
7. The Job panel shows the progress and the latest preview frame. If the job fails, the panel shows the full cause. When the job is done, choose **Open in Library**. The Library opens with the result selected and shows its provenance: model, seed and parameters (resolution, frames, fps, steps, guidance). A still image shows no frames or fps.

If a change makes the router pick another model, for example a different data class, the fields switch to that model's choices. A resolution or frame rate it does not offer goes back to the model default, and the frame count moves to its grid. Any link of the form `http://127.0.0.1:8737/#/library/<sha256>` opens the Library with that asset selected. Selecting another asset changes the address to that asset's link, so you can copy it.

### Switch the text encoder variant

Every model on your computer uses the same text encoder variant: `int8` (WanGP's default, about half the size) or `bf16`.

1. Open **Models**. The Text encoder panel marks the variant in use.
2. Select the other variant and choose **Switch to bf16** (or **Switch to int8**). The panel shows **Switching** while Cinewright downloads the new variant's files, restarts the worker and removes the previous variant's files. **Install** and **Remove** are unavailable until the switch is done.

A switch is refused while a job runs or a model is being installed, and when the new files do not fit on the drive; a message says why. If a switch fails while it downloads, the panel shows **Switch to bf16 failed** with the reason, and the previous variant stays in use. Choose **Switch to bf16** again to retry; files that were already downloaded and checked are kept.

### Install or remove a model through the API

While Cinewright runs, with WanGP set up:

1. Pick a model id and quantization from `GET http://127.0.0.1:8737/api/v1/models`, for example `z-image-turbo` and `int8`.
2. Start the install: `curl -X POST "http://127.0.0.1:8737/api/v1/models/z-image-turbo/install?quant=int8"`. The answer is `202`, or `409` with one sentence that says why the install cannot run now (for example, how much free disk space is missing).
3. Follow it in `GET /api/v1/events` (`model` events with the downloaded bytes) or in `GET /api/v1/models`, where the quantization goes from `downloading` to `installed`, or to `failed` with the reason in `error`.
4. To remove it: `curl -X DELETE "http://127.0.0.1:8737/api/v1/models/z-image-turbo?quant=int8"`.

To switch the text encoder variant of your computer, send `curl -X PATCH http://127.0.0.1:8737/api/v1/hosts/local -H "Content-Type: application/json" -d "{\"text_encoder_variant\": \"bf16\"}"`. `GET /api/v1/hosts` shows the switch as the host's `text_encoder_switch` while it runs (or with an `error` after it failed), and the new `text_encoder_variant` once it is done.

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
| `models/` | Installed model files, where WanGP looks them up: transformer, VAE and upscaler files directly in it, each text encoder with its tokenizer and config files in its own folder (`umt5-xxl/`, `xlm-roberta-large/`, `Qwen3/`, `gemma-3-12b-it-qat-q4_0-unquantized/`). `models/.downloads/` holds downloads in progress; an interrupted one continues from its `.part` file |
| `engines/` | Engine data, e.g. `engines/wan2gp/wgp_config.json` |
| `engines/wan2gp/<commit>/` | WanGP's source at the pinned commit (`wgp.py` lies directly in it), with the pinned ffmpeg in `ffmpeg_bins/` |
| `engines/wan2gp/env/` | WanGP's Python environment (Python 3.11, PyTorch with CUDA 13.0, the locked packages) |
| `engines/wan2gp/provisioning.json` | Record of a completed setup: commit, paths and the attention mode that worked (`sage2`, `sage` or `sdpa`) |
| `engines/wan2gp/downloads/` | Checked downloads; an interrupted one continues from its `.part` file |
| `engines/python/`, `engines/uv-cache/` | The Python that uv downloaded for WanGP, and uv's package cache |
| `engines/worker/cinewright_worker/` | Copy of the worker package that the engine's Python runs; the core replaces it at each start |
| `logs/cinewright.log` | Launcher and core log |
| `logs/worker.log` | Output of the worker the core started: the worker's log and the engine's messages |
| `logs/provisioning.log` | Output of uv and of the attention check during the WanGP setup |

### Commands

End users never need these. They exist for CI and development.

| Command | Does |
| --- | --- |
| `python -m cinewright` | Starts the core and opens the web UI in the browser |
| `python -m cinewright --version` | Prints `cinewright <version>` |
| `python -m cinewright_worker --engine mock\|wan2gp [--wan2gp-root PATH] [--once] [--stop-on-stdin-eof]` | Runs a worker. It needs `CINEWRIGHT_ENROLLMENT_TOKEN` from the core; the installed app starts one on WanGP once WanGP is set up. `mock` stands in for WanGP and serves its model types; the core routes jobs only to quantizations recorded as installed, so a hand-started mock worker gets jobs only for those. With `--stop-on-stdin-eof` the worker stops when its standard input closes; the core starts its worker this way |
| `python -m cinewright.devtools openapi [PATH]` | Writes the OpenAPI document |
| `python -m cinewright.devtools surface` | Prints the public surface: routes, schemas and MCP tools |
| `python -m cinewright.devtools upstream-check [--json]` | Reports a Wan2GP commit that is eligible for a pin update. Exit code 0: none, 10: one is eligible, 1: the check failed |
| `python scripts/build.py [--skip-web] [--skip-installer]` | Builds the web UI, the wheel and the installer |

### Internal API

All routes live under `/api/v1`. [`openapi.json`](openapi.json) is the contract, and [ARCHITECTURE.md](ARCHITECTURE.md) has the route table. Errors are RFC 9457 problem documents (`application/problem+json`). Operations of later phases return `501` with the problem type `urn:cinewright:problem:not-implemented`.

`GET /api/v1/events` streams server-sent events. Each message is named after its kind, and its `data` line is one JSON object, which `openapi.json` describes: `JobEvent` for the job events `state`, `progress`, `preview` and `log` (with the fields of each kind), `SystemEvent` for the others. Job events have an `id` (`<job_id>:<seq>`) and are replayed after `Last-Event-ID`. Previews are the exception: they are sent live but not stored and have no `id`, so after its replay a stream sends only the latest preview of each running job it covers. Without `job_id`, the stream also carries `engine` events with the engine's state, progress and error, as `GET /api/v1/system` reports it, and `model` events with a model install's state, progress and error, also for a removal. They have no `id` and are not replayed. `POST /api/v1/engines/wan2gp/provision` starts the WanGP setup again after a failure; it does nothing while the setup runs or once the engine is ready.

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

| Registry id | WanGP model types | Capabilities | Clip frames and fps | Licence |
| --- | --- | --- | --- | --- |
| `z-image-turbo` | `z_image` | t2i | none (images only) | Apache-2.0 |
| `ltx-2.3-distilled-22b` | `ltx2_22B_distilled`, `ltx2_22B_distilled_gguf_*` | t2v, i2v, t2i, i2i, audio | 17 + a multiple of 8, up to 737; 24 fps | LTX-2 Community License Agreement |
| `wan-2.2-i2v-14b` | `i2v_2_2` | i2v | 5 + a multiple of 4, up to 737; 16 fps | Apache-2.0 |
| `wan-2.2-t2v-14b` | `t2v_2_2` | t2v, t2i | 5 + a multiple of 4, up to 737; 16 fps | Apache-2.0 |

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

- **Start.** The installed app starts the worker as soon as the WanGP setup is ready: right at start when an earlier start completed the setup, otherwise when the setup or a retry completes. It starts no worker while the setup runs or after it failed.
- **Crashes.** If the worker ends unexpectedly, the core starts it again after a pause that doubles with each quick crash, from 1 second up to 1 minute. Jobs keep running on the new worker. A job that was running when the worker crashed returns to the queue when its lease expires, after two minutes.
- **Stopping.** When the core exits, it stops granting jobs, asks the worker to stop by closing the worker's standard input, and ends the worker after 5 seconds. If the core itself is killed, or ends when you sign out, Windows ends the worker too, because the worker belongs to a job object that dies with the core. On Linux the worker gets a kill signal when the core ends.
- **Network.** The worker's proxy settings point at a closed local port, so engine libraries that honour the standard proxy variables (Python's `urllib` and `requests` among them) cannot reach the internet, while the worker still reaches the core. Code that ignores these variables is not stopped: on Windows this is configuration only, because a per-user installation cannot add firewall rules.
- **Health.** The Hosts view shows the worker's state (starting, ready, busy or failed) and the VRAM it uses. The state comes from the worker's own reports and, while the worker is starting or down, from the core's view of the process.

### Routing

For each request, the router checks four things in order. Policy comes first: the data class and the licence's geographic exclusions. Feasibility comes next: capability, a pinned model, limits, a quantization installed on your computer, a registered worker for the engine model, and calibration. The limits come from the model's registry entry, also in `GET /api/v1/capabilities`: resolutions, frames, fps and duration. A clip's frame count must lie on the model's frame grid, `min_frames + n * frames_step`: 17, 25, 33 and so on for LTX-2 2.3 Distilled, 5, 9, 13 and so on for Wan 2.2. The estimate names the nearest valid counts. Text-to-image and image-to-image jobs are checked only against the resolutions: they make one still image, so their frames, fps and duration do not apply (see [Images from video models](#images-from-video-models)). Then come cost (local jobs cost €0.00) and runtime. When several routes remain, the cheapest wins; among equally cheap ones, a model the registry prefers comes first (Z-Image Turbo for text-to-image, as a model made for images), then calibration, runtime and registry order. The estimate lists every decision, so the UI and MCP clients can show why a request cannot run.

### Local only

The core listens on `127.0.0.1` only. A web page in your browser can still send requests to local addresses, so the core also accepts only the `Host` values `127.0.0.1:<port>` and `localhost:<port>`. It rejects foreign `Origin` headers, and it refuses the body types a page can send without a CORS preflight. Workers authenticate with a token per worker, which they get by registering with the core's enrollment token.

### Provenance

Every output asset has a provenance record: prompt, seed, the requested parameters, model, quantization, the model files, engine version, host, client and job. `GET /api/v1/assets/{sha256}/provenance` exports it as sidecar JSON.

- **Parameters** are the request's, with the model's registry defaults filled in where the request left them open: resolution, steps, guidance, and for videos frames and fps. WanGP can still adjust them: LTX-2 2.3 Distilled always runs 8 steps at guidance 1.0. A video's frame count is one the model renders as asked: every video model in the registry has a default frame count, and the router accepts only counts on the model's frame grid. `extra` lists the model-specific settings the worker passed on to the engine unchanged, or is `null` when the worker did not report them. The WanGP worker reports them. It leaves out a setting it does not accept, one the job's own parameters replace, and one WanGP would change for this job, for example `denoising_strength` without an input image to edit, more guidance phases than the model has, or `guidance_scale` on LTX-2 2.3 Distilled. The worker knows these rules for the registry's models at the pinned WanGP commit. A listed setting reached WanGP as asked; whether the model uses it is up to WanGP.
- **Model files** are the SHA-256 of the quantization's own files plus the shared files the model loads: the text encoder in the variant your computer's worker ran, the VAE and the other engine files.
- Records stored by earlier releases have no parameters (`params` is `null`) and list only the quantization's own files.

### Images from video models

LTX-2 2.3 Distilled and Wan 2.2 Text-to-Video can also make still images. For a text-to-image or image-to-image job on them, the worker switches WanGP to its image mode. WanGP then renders the shortest clip the model can make and keeps its first frame, so the job's frames, fps and duration do not apply: the router does not check them against the model's limits, and provenance records no frames or fps. The image comes back as a JPEG, WanGP's default image format.

An image-to-image job on LTX-2 hands its image to WanGP as the control image, the input WanGP's image mode offers for this model. The model setting `denoising_strength` (in `params.extra`, default 1.0) sets how strongly the image steers the result. A text-to-image job takes no input image; an image-to-image job takes exactly one.

### WanGP setup

The installer sets up only Cinewright itself. WanGP comes on the first start, because Cinewright may not redistribute it (see below). The core sets it up in the background:

1. It downloads the source archive of the pinned WanGP commit from GitHub and checks its SHA-256 before it opens it. The commit id inside the archive proves nothing, because anyone could write it there. It then unpacks the archive into the data directory.
2. It downloads a pinned ffmpeg release and puts its checked binaries where WanGP looks for them. Otherwise WanGP would download the newest ffmpeg without any check.
3. It uses the `uv.exe` that the installer installed to download Python 3.11 and create WanGP's own environment. It then installs exactly the packages of the lock, each checked against its SHA-256. uv ignores your own uv and pip settings, and no package comes from any index the lock does not name.
4. It tries the optional SageAttention and Triton wheels, which speed up generation on recent NVIDIA cards. It installs a wheel only if the wheel matches its pinned hash, then runs one attention calculation on the GPU. WanGP uses SageAttention only if that calculation works. Otherwise it uses PyTorch's standard attention (`sdpa`), which works everywhere. If a wheel could not be downloaded, each later start tries it once more before the worker starts; a wheel that did not match its hash is not tried again.
5. Finally, it writes a record of the setup and starts the worker on WanGP. A later start that finds the record and the files downloads and installs nothing else.

During the setup, the core and uv connect to `github.com`, `codeload.github.com`, `release-assets.githubusercontent.com`, `releases.astral.sh`, `files.pythonhosted.org`, `download.pytorch.org`, `download-r2.pytorch.org` (the lock's PyTorch wheels), `aiinfra.pkgs.visualstudio.com` and `*.vsblob.vsassets.io`. WanGP itself needs no internet connection afterwards: it runs offline, and model downloads go through the core (see [Model store](#model-store)). The setup runs on Windows only so far, because the pins cover only 64-bit Windows.

### Model store

The core keeps every model file in one model store, `models/` in the data directory, and tells WanGP to look there first. Installs follow these rules:

- **Checked files.** Every file is pinned by SHA-256 and size in the registry. A download is checked before the file is used; a file that does not match is deleted, and the model is not installed. Weights are safetensors or GGUF files only. Next to them, only non-executable tokenizer and config files (`.json`, `.txt`, `.model`) are allowed.
- **Shared files.** Text encoders, VAEs and the other engine files several models need are kept once. Removing a model removes a shared file only when no other installed model and no queued or running job needs it.
- **Text encoder variant.** Each computer uses one text encoder variant, `int8` or `bf16`, for all models. Until you choose one, Cinewright installs `int8`, WanGP's default. Switching first checks the free space, then installs the other variant's files, restarts the worker and, once it has registered again, removes the previous variant's files. It is refused while a job runs or a model is being installed, and when the new files do not fit.
- **The worker learns about new models.** WanGP reports its models only when its worker starts, so the core restarts the worker after every install, removal and switch. If a job is running, the restart waits until the job ended.
- **Each job runs its own quantization.** WanGP would otherwise load its default quantization (`int8`). The worker tells WanGP for each job which quantization to load, and first checks that its files are installed, because WanGP would otherwise try to download a missing file.
- **Licence.** Models whose licence excludes your country (`country` in the settings) are not offered and cannot be installed.
- **Sizes.** The Models view shows each installed model's size with the shared files it uses, and the store's total, which counts every shared file once. So the models' sizes can add up to more than the total. On a card, **installed** marks that size; before the install, **model files** is the size of the model's own files, without the shared files it may still need. Free disk space is measured each time the view loads the model list, and every 30 seconds while a download runs.
- **Free space and quota.** Before anything is downloaded, the core checks the free space of the drive that holds the model store, counting what installs that are still running will download; if the model does not fit, the install stops with the amount that is missing. With a storage quota set (`model_store_quota_gb`), the models used least recently make room, never one that a queued or running job needs. They are removed only after the new model is installed, so a failed or cancelled download leaves every installed model in place. If the drive has room for the new model only after removing them, the install is refused with the models it would remove and their sizes, and nothing is removed until you confirm. `GET /api/v1/models` names the models the quota removed last (`store.last_eviction`) until Cinewright restarts.
- **Download hosts.** Model files come only over HTTPS from `huggingface.co` and the download host it forwards to, `us.aws.cdn.hf.co` (later also `civitai.com`).

### Why WanGP is not bundled

WanGP is source-available under the WanGP Community License 2.0, and Cinewright never redistributes it. The WanGP setup downloads a pinned commit from WanGP's official repository.

[ARCHITECTURE.md](ARCHITECTURE.md) records the components, data flows, boundaries and current decisions.

## WanGP notice

Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright.

Licence: <https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt>

## Licence

Cinewright is licensed under the [MIT License](LICENSE). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the third-party software Cinewright uses and bundles.
