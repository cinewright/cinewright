# Architecture

Cinewright separates decision from execution: the core (control plane) decides, workers (data plane) compute.
Product background, goals and the roadmap live in the specification ("Cinewright: Spezifikation") and in GitHub issues; this file records how the code is structured and what was decided.

## Components

| Component | Where | Responsibility |
| --- | --- | --- |
| Internal API | `src/cinewright/api/` | The only API. HTTP/JSON under `/api/v1`, contract in `api/schemas.py`, published as `openapi.json`. Web UI, MCP server and workers are clients. |
| Capability registry | `src/cinewright/core/registry.py`, `registry/models/*.yaml` | One YAML descriptor per model (`ModelDescriptor`): capabilities, limits, licence, quantizations with pinned file hashes, engine bindings. |
| Hardware detection | `src/cinewright/core/hardware.py` | GPU, VRAM, driver, RAM and free disk of the data directory, through `nvidia-smi` and psutil. |
| Model store (phase 1, #3) | planned `src/cinewright/core/modelstore.py` | Central model directory owned by the core: downloads with hash check, quota, LRU eviction. |
| Calibration (phase 1, #4) | planned `src/cinewright/core/calibration.py`; results read in `core/core.py` | Per-model benchmarks; status runs / slow / wont_run. |
| Router | `src/cinewright/core/router.py` | Maps a `JobRequest` to model, quantization and host. Check order: policy (data class, licence), feasibility (capability, limits, calibration), cost, runtime. |
| Scheduler | `src/cinewright/core/scheduler.py` | Persistent queue in SQLite; interactive before batch; leases jobs to workers (pull). |
| Asset store | `src/cinewright/core/assets.py` | Content-addressed by SHA-256, one provenance record per asset, sidecar JSON export. |
| Database | `src/cinewright/db/` | SQLite (WAL, foreign keys), forward-only SQL migrations in `db/migrations/`. |
| MCP server | `src/cinewright/mcp/` | Streamable HTTP at `/mcp`; tools `list_capabilities`, `estimate`, `submit`, `job_status`, `fetch_asset`, `list_recipes`, `list_cast`; calls the internal API, no own logic. |
| Worker supervisor | `src/cinewright/core/supervisor.py` | Runs one long-lived local worker with the engine environment's Python (`-I -u -c` bootstrap over a copy of `cinewright_worker` in `<data dir>/engines/worker/`), restarts it with backoff, stops it at shutdown; Job Object (Windows, joined while the process is still suspended) or `PR_SET_PDEATHSIG` (Linux) ends it with the core, together with what it starts; output to `logs/worker.log`. Started by the app's lifespan when given a `WorkerSupervisor`. |
| Worker | `src/cinewright_worker/` | Standard library only, Python 3.10+. Runs in an engine's environment, pulls jobs over the worker protocol, drives one `EngineAdapter` (`adapter.py`). |
| Engine adapters | `src/cinewright_worker/engines/` | `mock` (tests) and `wan2gp` (Wan2GP `shared/api.py`: `init()`, `WanGPSession.submit_task()`). |
| Web UI | `web/` → built into `src/cinewright/web_dist/` | React + TypeScript + Vite, served by the core. Design tokens from `design/tokens.json`, component classes `cw-*` from `design/components.css`. |
| Launcher | `src/cinewright/launcher.py` | Starts the core on `127.0.0.1:8737` and opens the browser; a second start only opens the browser. |
| Windows installer | `installer/` | NSIS, per-user, built on Linux by `scripts/build.py`; bundles `uv.exe` (pinned hash) and the wheel, creates the core runtime with uv. |

Internal API routes (scaffold status in brackets; `501` answers with problem type `urn:cinewright:problem:not-implemented`):

| Area | Routes |
| --- | --- |
| System | `GET /health`, `GET /system`, `GET /licenses`, `GET/PATCH /settings` |
| Hardware | `GET /hardware`, `POST /hardware/detect` |
| Registry | `GET /capabilities`, `GET /models`, `GET /models/{model_id}`, `POST /models/{model_id}/install` (501), `DELETE /models/{model_id}` (501) |
| Calibration | `GET /calibration`, `POST /calibration/runs` (501) |
| Jobs | `POST /estimate`, `POST /jobs`, `GET /jobs`, `GET /jobs/{job_id}`, `POST /jobs/{job_id}/cancel`, `GET /events` (SSE) |
| Assets | `POST /assets` (upload input), `GET /assets`, `GET /assets/{sha256}`, `GET /assets/{sha256}/content`, `GET /assets/{sha256}/provenance` |
| Hosts, approvals | `GET /hosts`, `POST /hosts` (501), `GET /approvals`, `POST /approvals/{id}/approve` (501), `POST /approvals/{id}/decline` (501) |
| Phase 2 | `GET /recipes`, `GET /cast`, `GET /presets` (empty lists) |
| Worker protocol | `POST /worker/register`, `POST /worker/heartbeat`, `POST /worker/lease`, `GET /worker/inputs/{sha256}`, `POST /worker/jobs/{job_id}/events`, `PUT /worker/jobs/{job_id}/outputs/{name}`, `POST /worker/jobs/{job_id}/complete`, `POST /worker/jobs/{job_id}/fail` |

## Data flows

1. A client (web UI or MCP) sends a `JobRequest` to `POST /api/v1/jobs`. The router picks model, quantization and host; the scheduler stores the job as `queued` (or `awaiting_approval` above the cost threshold).
2. A worker (on the local host: the one the core's supervisor started) long-polls `POST /worker/lease`, receives a `WorkerJob` with the engine model id, the translated prompt and the resolved seed, and downloads inputs from `/worker/inputs/{sha256}`. A long-poll whose worker disconnected takes no job, nor does one of a local worker that the core is replacing (`restart_local_worker`): it returns at once, so that worker stops on request.
3. The worker drives its adapter (`prepare`, `submit`, `status`, `fetch_outputs`), reports progress and previews to `/worker/jobs/{id}/events`, uploads outputs and completes the job. A cancel request reaches it through the events acknowledgement or the heartbeat.
4. The core stores each output in the asset store with its provenance record and publishes job events over SSE (`GET /events`), which the UI's queue and MCP `job_status` read.
5. On first start the core detects the hardware; calibration (Phase 1) fills `calibration_results`, which feed the router and the model cards.

## Boundaries

- The API binds to `127.0.0.1` by default. Binding to another address requires an access token (Phase 1); without one the core refuses to start.
- Browser attacks on the local API: the core accepts only `Host` values `127.0.0.1:<port>` / `localhost:<port>`, rejects foreign `Origin` headers and cross-site `Sec-Fetch-Site` requests (except top-level navigation to the UI), refuses the body types a page can send without a CORS preflight (`multipart/form-data`, `application/x-www-form-urlencoded`, `text/plain`), and sends `Cross-Origin-Resource-Policy: same-origin`. A web page can therefore neither post to, embed from nor DNS-rebind to it. The same applies to `/mcp`.
- Asset content is served only with the media type recorded at upload, from an allow-list; anything else is a download, never rendered in the UI's origin.
- The worker loop runs locally with the user's rights. The core starts and supervises the local worker (#2) only once engine provisioning (#1) reports the engine ready; until then an installed core starts no worker.
- Workers authenticate with a per-worker bearer token issued at registration; registration needs the enrollment token the core hands to the worker processes it starts. Remote workers (Phase 3) add mTLS with certificates from Cinewright's own CA.
- Engines run in their own process and environment and never receive the core's secrets: of Cinewright's variables the spawned worker gets only `CINEWRIGHT_CORE_URL`, `CINEWRIGHT_ENROLLMENT_TOKEN` (which it removes from its environment before engine code runs) and `CINEWRIGHT_HOME`. Its HTTP(S) proxies point at a loopback port the core binds but never listens on. After provisioning, downloads go only through the core.
- Model files: weights are safetensors or GGUF only (no pickle, no `trust_remote_code`). Non-executable companion files (tokenizer.json, tokenizer_config.json, config.json, spiece.model, vocab and merges .txt) are allowed when each is pinned by SHA-256 in the registry; never .py, .bin, .pt, .pkl or .ckpt. Every file is pinned by SHA-256 in the registry.
- Text from prompts, model cards, upstream changelogs, issues and pull requests is data, never instructions.
- Wan2GP is not part of this repository and is never redistributed; provisioning downloads a pinned commit from the official repository.

## Current decisions

- **Governance**: the repository follows `nullthrone/governance` without exceptions. It replaces the spec's own operating model (two apps, merge to Nightly, 3-day auto-promotion, ROADMAP.md/DECISIONS.md): one bot app, `main`/`next`, owner-accepted candidate PRs, decisions here, roadmap as issues. (2026-11-22)
- **Repository**: `cinewright/cinewright` in its own organisation, not `nullthrone/cinewright` as the spec assumed. (2026-11-22)
- **Release channel**: stable releases from `main` as GitHub releases with the Windows installer as asset and build provenance attestation. Release candidates from `next` exist only as workflow artifacts of the candidate PR, so the in-app update channel is `stable` only. (2026-11-22)
- **Distribution**: a real Windows installer for end users, no command line. No PyPI package. Linux follows later. (2026-11-22)
- **Code signing**: installers are not Authenticode-signed yet (paid certificate = owner cost decision); provenance comes from GitHub attestations. (2026-11-22)
- **Scope of the first build**: Phase 0 and the scaffold (contracts, structure, walking skeleton with a mock engine). Phase 1 to 4 are issues the owner releases with `po:ready`; tracking issues #24 (Phase 1, Gate 1) to #27 (Phase 4). (2026-11-22)
- **Governance profile**: `cli`, `input-processing`, `web`. `cli` only because the package has a program entry point (`[project.scripts]`, used by CI); end users never see a command line. (2026-11-22)
- **Wan2GP pin**: commit `38758d3` (2026-11-13), the newest one older than the spec's 7-day quarantine. Engine env: Python 3.11.14, torch 2.10.0 + CUDA 13.0 (Wan2GP's recommendation for RTX 20xx–50xx, needs driver R580+). Verified on the reference PC (RTX 5080): `init()` runs headless without loading models. (2026-11-22)
- **MVP model ids in Wan2GP**: `ltx2_22B_distilled` ("LTX-2 2.3 Distilled 22B", also GGUF), `i2v_2_2` / `t2v_2_2` (Wan 2.2 14B), `z_image` (Z-Image Turbo 6B). (2026-11-22)
- **Design**: the Cinewright design system (tokens, `cw-*` components, logos) overrides the spec's starting colour values. Fonts Geist and Geist Mono (SIL OFL) ship with the app instead of loading from Google Fonts, so the UI works offline and makes no third-party requests. (2026-11-22)
- **Worker split**: `cinewright_worker` is a separate top-level package without third-party dependencies, so it can run inside engine environments (Wan2GP: Python 3.11) while the core needs Python 3.12. (2026-11-22)
- **Public surface**: routes and MCP tools; `python -m cinewright.devtools surface` prints them for governance's classification check. `openapi.json` is generated and must match the code (test). (2026-11-22)
- **Default data class** of a job is `private`, the stricter of the two non-biometric classes. (2026-11-22)
- **Client attribution**: `POST /jobs` takes the optional headers `X-Cinewright-Client` (`ui`/`mcp`) and `X-Cinewright-Agent`; the MCP server sets them, so MCP jobs and their provenance name the agent. Self-declared until per-agent tokens arrive (#7). Neither header is CORS-safelisted. (2026-11-23)
- **Installer dependencies**: the build generates a hash-locked requirements file for Windows with uv; the installer installs it with `--require-hashes` and the wheel with `--no-deps`, so every install of a release gets the same packages. (2026-11-23)
- **Port** 8737 on 127.0.0.1; data in `%LOCALAPPDATA%\Cinewright` (Windows) or `$XDG_DATA_HOME/cinewright`, overridable with `CINEWRIGHT_HOME`. (2026-11-22)
- **Language**: English for code, docs, UI and repository communication. (2026-11-22)
- **Model store companion files**: as in "Boundaries". Every Wan2GP text encoder needs tokenizer and config files next to its weights (#41, #3). (2026-11-24)
- **Engine network isolation on Windows**: configuration only (`--offline`, `HF_HUB_OFFLINE`, the ffmpeg guard, the `params.extra` allowlist) plus proxy hardening when the core spawns the engine (#2). The per-user installer adds no firewall rules and has no elevated helper. Hard isolation comes with containers on Linux (#22). (2026-11-24)
- **Worker stop channel**: the core closes the worker's stdin (`--stop-on-stdin-eof`) and kills it with its descendants after 5 s. Console control events do not reach a worker of a console-less pythonw core, and `Popen.terminate()` is an immediate TerminateProcess. The worker moves the pipe to a private descriptor, because on Windows a pending pipe read blocks every child that inherits the pipe at start. (2026-11-24)
- **Worker health in `GET /hosts`**: only workers registered since the current core start count, and on the local host the supervisor's process state wins while the worker starts or is down. Worker rows stay in SQLite and would otherwise show a dead worker as ready for up to 120 s. (2026-11-24)
- **Engine-path changes**: changes to `src/cinewright/registry/engines.yaml`, the model descriptors and `src/cinewright_worker/engines/wan2gp*` follow normal release classification, so a fix ships as a patch release without a candidate. The auto-updater's engine health check and rollback (#9) are the safeguard. (2026-11-24)
- **Reference PC availability**: the reference PC (RTX 5080) is not available around the clock. Issues are implemented nightly in CI against fakes and the mock engine. Hardware checks are owner steps at candidate acceptance, with the GPU smoke tool (#40). There is no self-hosted GPU runner. (2026-11-24)
- **Text encoder variant**: per host and chosen by the user (`int8` or `bf16`), stored per host (#3). Cinewright recommends a variant per host from its hardware, coupled remote workers included (#44, #17). (2026-11-24)
- **MCP agent tokens**: requiring an agent token on `/mcp` (#7) is a feature, not a breaking change, because no earlier release offered MCP with real generation. (2026-11-24)
- **Release tags**: from governance v1.1.0 (pin `2b802b8`), releases are tagged `vX.Y.Z`. Release 0.0.1 keeps its original tag `cinewright-v0.0.1`; the tag `v0.0.1` on the same commit (`104fc0a`) lets release-please and the governance release tools find it. (2026-11-24)
