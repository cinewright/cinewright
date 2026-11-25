"""GPU smoke test: one short generation per installed smoke model on the running core, with a comparable report.

  python -m cinewright.devtools smoke [--installer CinewrightSetup-<version>.exe [--model-store DIR]]

CI has no GPU, so the owner runs this on the reference PC before releasing a build, with the installed runtime's
Python. It therefore uses only the standard library and Cinewright's runtime dependencies (no pytest, Pillow or
torch) and contacts only the local core: `http://127.0.0.1:<CINEWRIGHT_PORT, default 8737>`. With `--installer` it
first installs that installer into a temporary folder and tests the core it starts there (`cinewright.smoke_installer`).

For each smoke model (`SMOKE_JOBS`) that has an installed quantization in `GET /api/v1/models`, it submits one job
pinned to that model with a fixed seed; image-to-video jobs get a fixed start image (`render_png`), uploaded first.
Jobs run one after another. From each job's event history (`GET /api/v1/events?job_id=...`) it takes:

- seconds per step: the time between progress events of consecutive steps of one phase, divided by those steps;
- peak VRAM: the `peak_vram_mb` field of a `log` event, which the worker sends when it can measure it;
- wall time (from submission to the end state), the first output's SHA-256 and media type, and the engine version
  that output's provenance records (`mock-1` for the mock engine).

Once per run it records the Cinewright version and engine commit (`GET /api/v1/system`), GPU name and driver
(`GET /api/v1/hardware`) and the attention result of engine provisioning (its record in
`<data dir>/engines/wan2gp/`). The report (`SmokeReport`) goes to `<data dir>/logs/smoke-<version>.json`, a short
Markdown summary for the owner's review next to it; the summary replaces the folders that name the Windows account
with placeholders. A job the run leaves before its end state (an error, Ctrl+C) is cancelled.

Exit codes: 0 every selected model produced its output, 1 a generation failed, 2 the tool could not run (no core,
no smoke model installed, the installer or the setup of its temporary installation failed, the report could not be
written, the run was interrupted). The last printed line states the result and the exit code (`run_guarded`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cinewright.config import load_config
from cinewright_worker.engines.mock import render_png
from cinewright_worker.engines.wan2gp import PROVISIONING_RECORD, SAGE_ATTENTION_MODES, attention_mode, read_provisioning_record

API = "/api/v1"
EXIT_PASSED, EXIT_FAILED, EXIT_ERROR = 0, 1, 2
ENGINE_ID = "wan2gp"
TERMINAL_STATES = ("succeeded", "failed", "cancelled")
SEED = 4040
START_IMAGE_SIZE = (832, 480)  # width, height of the fixed start image of the image-to-video jobs
HTTP_TIMEOUT_S = 30.0
STREAM_READ_TIMEOUT_S = 60.0  # the event stream sends a keepalive every 15 s
ROUTE_WAIT_S = 120.0  # how long a job waits for a worker that can run it (the local worker may still be starting)
ROUTE_POLL_S = 1.0
JOB_TIMEOUT_S = 3600.0  # a job still running after this is cancelled and counts as failed
# The worker posts progress about once a second; a shorter span of timed steps says nothing about a step's duration.
MIN_TIMED_SPAN_S = 1.0
DENOISING_PHASE = "inference"  # phase of Wan2GP's (and the mock engine's) progress updates during denoising steps
PROGRESS_EVERY_S = 60.0  # at most one progress line per job and this many seconds, so a long job shows it is alive


@dataclass(frozen=True)
class SmokeJob:
    """One smoke generation: a short output at a fixed size. Frame counts lie on Wan2GP's frame grid of the model.

    `install_quant` is the quantization the `--installer` path installs (each smoke model has `int8`); the job itself
    pins only the model, and the router takes an installed quantization.
    """

    model_id: str
    capability: Literal["i2v", "t2i"]
    prompt: str
    params: Mapping[str, Any] = field(default_factory=dict)
    install_quant: str = "int8"


SMOKE_JOBS: tuple[SmokeJob, ...] = (
    SmokeJob(
        "ltx-2.3-distilled-22b",
        "i2v",
        "The camera slowly pushes in while soft light moves across the scene.",
        {"resolution": "640x384", "frames": 49},  # about 2 s at 24 fps; LTX-2 frame counts are 17 + 8n
    ),
    SmokeJob(
        "wan-2.2-i2v-14b",
        "i2v",
        "The colours drift slowly while the camera pans to the right.",
        {"resolution": "832x480", "frames": 33},  # about 2 s at 16 fps; Wan 2.2 frame counts are 5 + 4n
    ),
    SmokeJob(
        "z-image-turbo",
        "t2i",
        "A lighthouse on a rocky cliff at dusk, warm light in its windows, calm sea.",
        {"resolution": "1024x1024"},
    ),
)


# --------------------------------------------------------------------------- report format


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttentionResult(_Model):
    mode: Literal["sage2", "sage", "sdpa"] | None = Field(
        default=None, description="Wan2GP's --attention mode from the provisioning record; null without a record"
    )
    sage_attention: bool | None = Field(default=None, description="Whether Wan2GP runs with SageAttention (else it falls back to sdpa)")
    probe: Literal["passed", "failed", "not_run"] | None = Field(
        default=None,
        description="The Triton/SageAttention probe of engine provisioning (import plus a test run on the GPU); "
        "`not_run` when the optional attention wheels were not all installed",
    )
    optional_wheels: dict[str, str] | None = Field(default=None, description="Outcome per optional attention wheel, as recorded")
    reason: str | None = Field(default=None, description="Why the fields are null")


class JobResult(_Model):
    model_id: str
    capability: str
    seed: int
    params: dict[str, Any] = Field(description="The job's parameters besides the seed; the model's registry defaults fill the rest")
    job_id: str | None = Field(default=None, description="null when the job was not submitted")
    quant: str | None = Field(default=None, description="The quantization the router chose")
    state: str = Field(description="The job's last state (`succeeded` passes), or `not_submitted`")
    error: str | None = None
    wall_seconds: float | None = Field(default=None, description="From submission to the end state")
    total_steps: int | None = Field(
        default=None, description="Steps of the job's last pass, from its last progress event with a total (LTX-2 runs two passes)"
    )
    seconds_per_step: float | None = None
    seconds_per_step_reason: str | None = Field(default=None, description="Why `seconds_per_step` is null")
    peak_vram_mb: int | None = Field(default=None, description="Peak VRAM the worker measured in its own process for this job")
    peak_vram_reason: str | None = Field(default=None, description="Why `peak_vram_mb` is null")
    output_sha256: str | None = None
    output_media_type: str | None = None
    engine_version: str | None = Field(default=None, description="From the output's provenance; `mock-1` for the mock engine")

    @property
    def passed(self) -> bool:
        return self.state == "succeeded" and self.output_sha256 is not None


class SkippedModel(_Model):
    model_id: str
    reason: str


class SmokeReport(_Model):
    """The JSON report `<data dir>/logs/smoke-<version>.json`."""

    report_version: Literal[1] = 1
    result: Literal["passed", "failed"]
    exit_code: Literal[0, 1]
    cinewright_version: str
    core_url: str
    data_dir: str
    engine_state: str | None = Field(default=None, description="Engine provisioning state from GET /system")
    engine_commit: str | None = Field(default=None, description="Pinned Wan2GP commit once the engine is ready")
    gpu_name: str | None = None
    gpu_driver: str | None = None
    gpu_reason: str | None = Field(default=None, description="Why the GPU fields are null")
    attention: AttentionResult
    started_at: datetime
    finished_at: datetime
    jobs: list[JobResult]
    skipped: list[SkippedModel] = Field(default_factory=list, description="Smoke models without an installed quantization")


# --------------------------------------------------------------------------- measurements


def _created_at(event: Mapping[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(event["created_at"]))
    except (KeyError, ValueError):
        return None


def seconds_per_step(events: Iterable[Mapping[str, Any]]) -> tuple[float | None, str | None]:
    """Mean duration of a denoising step from the job's progress events, or None and the reason.

    Only pairs of consecutive progress events count that are both denoising steps (phase `inference...`, as Wan2GP
    and the mock engine name it) of the same phase and number of steps, where the step went up. So loading (Wan2GP
    counts loaded blocks as steps of another phase), decoding and saving are not part of it, a pass that counts its
    steps from 0 again (LTX-2's second pass) adds its own steps instead of a negative difference, and the time between
    two passes (Wan2GP's `inference_stage_1`, `inference_stage_2`) does not count as a step.
    """
    points: list[tuple[datetime, str, Any, int]] = []
    for event in events:
        data = event.get("data") or {}
        step = data.get("step")
        created = _created_at(event)
        if event.get("kind") != "progress" or created is None or not isinstance(step, int) or isinstance(step, bool):
            continue
        points.append((created, str(data.get("phase") or ""), data.get("total_steps"), step))
    span = 0.0
    steps = 0
    for (before, phase, total, step), (after, next_phase, next_total, next_step) in pairwise(points):
        if phase.startswith(DENOISING_PHASE) and next_phase == phase and next_total == total and next_step > step:
            span += (after - before).total_seconds()
            steps += next_step - step
    if steps == 0:
        return None, "The job's progress events show fewer than two denoising steps of one pass, so no step could be timed."
    if span < MIN_TIMED_SPAN_S:
        return None, (
            f"Its {steps} timed steps took {span:.2f} s together, less than the worker's progress interval of about "
            f"{MIN_TIMED_SPAN_S:g} s, so a step's duration cannot be told."
        )
    return span / steps, None


def peak_vram(events: Iterable[Mapping[str, Any]]) -> tuple[int | None, str | None]:
    """The peak VRAM of the job from the worker's `log` events with `peak_vram_mb`, or None and the reason."""
    values = []
    for event in events:
        value = (event.get("data") or {}).get("peak_vram_mb") if event.get("kind") == "log" else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(value)
    if not values:
        return None, (
            "The worker reported no peak VRAM; it measures it only when its engine runs on torch with CUDA, which the mock engine does not."
        )
    return round(max(values)), None


def attention_result(data_dir: Path) -> AttentionResult:
    """The attention result of engine provisioning, from its record in `<data dir>/engines/wan2gp/`."""
    record = read_provisioning_record(str(data_dir))
    if record is None:
        return AttentionResult(
            reason=f"No engine provisioning record ({ENGINE_ID}/{PROVISIONING_RECORD}) in the data directory: the engine was not "
            "provisioned here (for example with the mock engine), so the attention result is unknown."
        )
    mode = attention_mode(record)
    raw_wheels = record.get("optional_wheels")
    wheels = {str(name): str(outcome) for name, outcome in raw_wheels.items()} if isinstance(raw_wheels, dict) else {}
    if mode in SAGE_ATTENTION_MODES:
        probe = "passed"
    elif wheels and all(outcome == "installed" for outcome in wheels.values()):
        probe = "failed"
    else:
        probe = "not_run"
    return AttentionResult(mode=mode, sage_attention=mode in SAGE_ATTENTION_MODES, probe=probe, optional_wheels=wheels)


# --------------------------------------------------------------------------- core access


class CannotRun(Exception):
    """The smoke test cannot run at all (exit code 2); the message says why in one sentence."""


def json_body(response: httpx.Response) -> dict[str, Any] | None:
    """The response's JSON object, or None when its body is something else (another server, an HTML page)."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def problem_text(response: httpx.Response) -> str:
    """A one-line description of a refused request, from its problem details where there are any."""
    body = json_body(response)
    if body is None:
        return f"HTTP {response.status_code}"
    parts = [str(part).rstrip(".") for part in (body.get("title"), body.get("detail")) if part]
    text = ": ".join(parts) or f"HTTP {response.status_code}"
    reasons = body.get("reasons")
    if isinstance(reasons, list) and reasons:
        text += " (" + "; ".join(str(reason) for reason in reasons) + ")"
    return text


def iter_sse(lines: Iterable[str]) -> Iterator[dict[str, Any] | None]:
    """The `data` objects of a server-sent event stream, and None for each comment line (a keepalive), so a caller
    can look at the clock while the stream is idle. Data that is not JSON raises ValueError."""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                value = json.loads("\n".join(data))
                if isinstance(value, dict):
                    yield value
            data = []
        elif line.startswith(":"):
            yield None
        else:
            name, _, value = line.partition(":")
            if name == "data":
                data.append(value.removeprefix(" "))


def installed_models(models: dict[str, Any]) -> set[str]:
    """Ids of the models with at least one installed quantization in a `GET /models` answer."""
    return {
        item["id"]
        for item in models.get("items", [])
        if any(quant.get("install_state") == "installed" for quant in item.get("quantizations", []))
    }


def say(out: TextIO, text: str) -> None:
    """Print a line; characters the console cannot show become `?` instead of ending the run."""
    encoding = getattr(out, "encoding", None) or "utf-8"
    out.write(text.encode(encoding, "replace").decode(encoding) + "\n")
    out.flush()


def progress_line(data: Any) -> str:
    """A line about a running job from the data of one of its progress events."""
    data = data if isinstance(data, dict) else {}
    phase, step, total = data.get("phase") or "running", data.get("step"), data.get("total_steps")
    if isinstance(step, int) and isinstance(total, int):
        return f"  still running: {phase}, step {step} of {total} ..."
    return f"  still running: {phase} ..."


class SmokeRun:
    def __init__(self, http: httpx.Client, out: TextIO, *, route_wait_s: float = ROUTE_WAIT_S) -> None:
        self.http = http
        self.out = out
        self.route_wait_s = route_wait_s
        self.cancelled: set[str] = set()  # jobs this run asked the core to cancel

    def say(self, text: str) -> None:
        say(self.out, text)

    def get_json(self, path: str) -> dict[str, Any]:
        response = self.http.get(f"{API}{path}")
        if response.status_code != 200:
            raise CannotRun(f"GET {API}{path} was refused: {problem_text(response)}.")
        body = json_body(response)
        if body is None:
            raise CannotRun(f"GET {API}{path} answered with something other than a JSON object.")
        return body

    def run(self) -> int:
        code, result = self.execute()
        self.say(result)
        return code

    def execute(self) -> tuple[int, str]:
        """Run the smoke jobs and write the report; returns the exit code and the result line, which comes last."""
        started_at = datetime.now(UTC)
        url = str(self.http.base_url).rstrip("/")
        try:
            response = self.http.get(f"{API}/system")
        except httpx.TransportError as exc:
            raise CannotRun(f"No Cinewright core answers at {url} ({exc}); start Cinewright, or set CINEWRIGHT_PORT to its port.") from exc
        system = json_body(response)
        if response.status_code != 200 or system is None or system.get("name") != "Cinewright":
            raise CannotRun(
                f"The server at {url} is not a Cinewright core (GET {API}/system: {problem_text(response)}); start Cinewright, "
                "or set CINEWRIGHT_PORT to its port."
            )
        version, data_dir_text = system.get("version"), system.get("data_dir")
        if not isinstance(version, str) or not version or not isinstance(data_dir_text, str) or not data_dir_text:
            raise CannotRun(f"The Cinewright core at {url} did not report its version and data directory.")
        data_dir = Path(data_dir_text)
        installed = installed_models(self.get_json("/models"))
        selected = [job for job in SMOKE_JOBS if job.model_id in installed]
        skipped = [SkippedModel(model_id=job.model_id, reason="No quantization is installed.") for job in SMOKE_JOBS if job not in selected]
        if not selected:
            names = ", ".join(job.model_id for job in SMOKE_JOBS)
            raise CannotRun(f"None of the smoke models ({names}) is installed on this core, so there is nothing to test.")
        engines = system.get("engines") if isinstance(system.get("engines"), list) else []
        engine = next((item for item in engines if isinstance(item, dict) and item.get("id") == ENGINE_ID), {})
        gpu_name, gpu_driver, gpu_reason = self.gpu()
        start_image = self.upload_start_image() if any(job.capability == "i2v" for job in selected) else None

        self.say(f"Cinewright {version} at {url}: {len(selected)} of {len(SMOKE_JOBS)} smoke models installed.")
        jobs = [self.run_job(job, start_image) for job in selected]
        passed = all(job.passed for job in jobs)
        report = SmokeReport(
            result="passed" if passed else "failed",
            exit_code=EXIT_PASSED if passed else EXIT_FAILED,
            cinewright_version=version,
            core_url=url,
            data_dir=str(data_dir),
            engine_state=engine.get("state"),
            engine_commit=engine.get("version"),
            gpu_name=gpu_name,
            gpu_driver=gpu_driver,
            gpu_reason=gpu_reason,
            attention=attention_result(data_dir),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            jobs=jobs,
            skipped=skipped,
        )
        json_path, summary_path = write_report(report, data_dir / "logs")
        self.say(f"Report:  {json_path}")
        self.say(f"Summary: {summary_path}")
        failed = [job.model_id for job in jobs if not job.passed]
        if passed:
            return EXIT_PASSED, f"Smoke test passed: all {len(jobs)} installed smoke models produced their output. Exit code {EXIT_PASSED}."
        return EXIT_FAILED, f"Smoke test failed: no output from {', '.join(failed)}. Exit code {EXIT_FAILED}."

    def gpu(self) -> tuple[str | None, str | None, str | None]:
        response = self.http.get(f"{API}/hardware")
        if response.status_code != 200:
            return None, None, f"GET {API}/hardware was refused: {problem_text(response)}."
        gpus = (json_body(response) or {}).get("gpus")
        if not isinstance(gpus, list) or not gpus or not isinstance(gpus[0], dict):
            return None, None, "Hardware detection found no NVIDIA GPU."
        name, driver = gpus[0].get("name"), gpus[0].get("driver_version")
        return (name if isinstance(name, str) else None), (driver if isinstance(driver, str) else None), None

    def upload_start_image(self) -> str:
        image = render_png(SEED, *START_IMAGE_SIZE)
        response = self.http.post(f"{API}/assets", content=image, headers={"Content-Type": "image/png"})
        if response.status_code != 201:
            raise CannotRun(f"The start image for the image-to-video jobs was refused: {problem_text(response)}.")
        sha256 = (json_body(response) or {}).get("sha256")
        if not isinstance(sha256, str):
            raise CannotRun("The core accepted the start image for the image-to-video jobs without naming it.")
        return sha256

    def run_job(self, job: SmokeJob, start_image: str | None) -> JobResult:
        request = {
            "capability": job.capability,
            "prompt": job.prompt,
            "model": job.model_id,
            "inputs": [start_image] if job.capability == "i2v" and start_image else [],
            "params": {**job.params, "seed": SEED},
        }
        result = JobResult(model_id=job.model_id, capability=job.capability, seed=SEED, params=dict(job.params), state="not_submitted")
        self.say(f"- {job.model_id} ({job.capability}): waiting for a worker ...")
        try:
            self.follow(request, result)
        except httpx.HTTPError as exc:
            result.error = f"The core stopped answering ({exc})."
        except CannotRun as exc:
            result.error = str(exc)
        except (ValueError, LookupError) as exc:  # an answer of the core that does not have the expected form
            result.error = f"An answer of the core could not be read ({type(exc).__name__}: {exc})."
        if result.passed:
            self.say(f"  succeeded in {result.wall_seconds:.1f} s")
        else:
            self.say(f"  {result.state}: {result.error or 'no output'}")
        return result

    def wait_for_route(self, request: dict[str, Any]) -> str | None:
        """None as soon as a worker can run the request; else, after `route_wait_s`, why none can."""
        deadline = time.monotonic() + self.route_wait_s
        while True:
            response = self.http.post(f"{API}/estimate", json=request)
            if response.status_code != 200:
                return f"The estimate was refused: {problem_text(response)}."
            estimate = response.json()
            if estimate.get("feasible"):
                return None
            if time.monotonic() >= deadline:
                reasons = "; ".join(str(reason) for reason in estimate.get("reasons") or []) or "no reason given"
                return f"No worker could run the job within {self.route_wait_s:.0f} s ({reasons})."
            time.sleep(ROUTE_POLL_S)

    def follow(self, request: dict[str, Any], result: JobResult) -> None:
        """Submit the job, read its events until its end state and fill `result` in. A job left before its end state
        (an error, Ctrl+C) is cancelled, so it does not keep the GPU busy or delay the next smoke job."""
        problem = self.wait_for_route(request)
        if problem is not None:
            result.error = problem
            return
        submitted_at = time.monotonic()
        response = self.http.post(f"{API}/jobs", json=request)
        if response.status_code != 202:
            result.error = f"The job was refused: {problem_text(response)}."
            return
        job = json_body(response) or {}
        if not isinstance(job.get("id"), str):
            result.error = "The core accepted the job without naming it."
            return
        result.job_id, result.state = job["id"], str(job.get("state"))
        result.quant = (job.get("route") or {}).get("quant")
        self.say(f"  job {result.job_id} on {result.quant or 'unknown quantization'} ...")
        try:
            self.watch(result, submitted_at)
        finally:
            if result.state not in TERMINAL_STATES:
                self.cancel(result.job_id)

    def watch(self, result: JobResult, submitted_at: float) -> None:
        """Read the submitted job's events until its end state and fill in its measurements and output."""
        job_id = result.job_id or ""
        events, finished = self.events(job_id, submitted_at + JOB_TIMEOUT_S)
        if not finished:
            self.cancel(job_id)
        job = self.get_json(f"/jobs/{job_id}")
        result.wall_seconds = round(time.monotonic() - submitted_at, 3)
        result.state = str(job.get("state"))
        error = job.get("error")
        if not finished:
            result.error = f"The job did not finish within {JOB_TIMEOUT_S / 60:.0f} minutes, so the smoke test cancelled it."
        elif isinstance(error, dict):
            result.error = f"{error.get('code')}: {error.get('message')}"
        elif result.state not in TERMINAL_STATES:
            result.error = f"The job's event stream ended while the job was {result.state}, so the smoke test cancelled it."
        progress = [event["data"] for event in events if event.get("kind") == "progress" and isinstance(event.get("data"), dict)]
        totals = [data.get("total_steps") for data in progress]
        result.total_steps = next((total for total in reversed(totals) if isinstance(total, int)), None)
        result.seconds_per_step, result.seconds_per_step_reason = seconds_per_step(events)
        if result.seconds_per_step is not None:
            result.seconds_per_step = round(result.seconds_per_step, 3)
        result.peak_vram_mb, result.peak_vram_reason = peak_vram(events)
        outputs = job.get("outputs") or []
        if outputs:
            asset = self.get_json(f"/assets/{outputs[0]}")
            sha256 = asset.get("sha256")
            if not isinstance(sha256, str):
                raise CannotRun(f"The output {outputs[0]} of job {job_id} has no SHA-256.")
            result.output_sha256 = sha256
            result.output_media_type = asset.get("media_type")
            result.engine_version = ((asset.get("provenance") or {}).get("model") or {}).get("engine_version")
        elif result.state == "succeeded" and result.error is None:
            result.error = "The job succeeded without an output."

    def cancel(self, job_id: str) -> None:
        """Ask the core to cancel the job, once and best effort: an error here must not hide the reason for leaving it."""
        if job_id in self.cancelled:
            return
        self.cancelled.add(job_id)
        try:
            self.http.post(f"{API}/jobs/{job_id}/cancel")
        except httpx.HTTPError:
            pass

    def events(self, job_id: str, deadline: float) -> tuple[list[dict[str, Any]], bool]:
        """The job's events, replayed from its start; False when `deadline` passed before its end state. Prints a
        progress line at most every `PROGRESS_EVERY_S`, so a long job shows that it is still running."""
        events: list[dict[str, Any]] = []
        timeout = httpx.Timeout(HTTP_TIMEOUT_S, read=STREAM_READ_TIMEOUT_S)
        shown = time.monotonic()
        with self.http.stream("GET", f"{API}/events", params={"job_id": job_id}, timeout=timeout) as response:
            if response.status_code != 200:
                response.read()
                raise CannotRun(f"The event stream of job {job_id} was refused: {problem_text(response)}.")
            for event in iter_sse(response.iter_lines()):
                now = time.monotonic()
                if event is not None:
                    events.append(event)
                    if event.get("kind") == "progress" and now - shown >= PROGRESS_EVERY_S:
                        self.say(progress_line(event.get("data")))
                        shown = now
                if now > deadline:
                    return events, False
        return events, True


# --------------------------------------------------------------------------- output


def _file_version(version: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", version)


def _number(value: float | None, unit: str, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def _attention_text(attention: AttentionResult) -> str:
    if attention.mode is None:
        return "unknown (no engine provisioning record)"
    probe = {
        "passed": "SageAttention probe passed",
        "failed": "SageAttention probe failed, falls back to sdpa",
        "not_run": "SageAttention probe not run (attention wheels not installed), falls back to sdpa",
    }[attention.probe or "not_run"]
    return f"`{attention.mode}` ({probe})"


def private_folders(data_dir: str) -> list[tuple[str, str]]:
    """The folders of this computer whose paths name the Windows account, each with the placeholder a pasted summary
    shows instead: the data directory, the temporary folder (the worker's job folders, an `--installer` run), the
    local and roaming application data (the installed runtime) and the home folder."""
    folders = [(data_dir, "<data dir>"), (tempfile.gettempdir(), "%TEMP%")]
    folders += [(os.environ.get(name, ""), f"%{name}%") for name in ("LOCALAPPDATA", "APPDATA", "USERPROFILE")]
    try:
        folders.append((str(Path.home()), "%USERPROFILE%"))
    except RuntimeError:  # no home folder can be determined
        pass
    return folders


# A user's folder that none of the known folders covered: `C:\Users\<name>`, also with doubled backslashes as in an
# error message's repr, or `/home/<name>`. The name may hold spaces, so it ends only at a separator or a quote.
_USER_FOLDER = re.compile(r"""(?i)(\b[a-z]:(?:\\{1,2}|/)users(?:\\{1,2}|/)|(?<![\w.~-])/(?:home|users)/)[^\\/'"`|<>()\[\],;\r\n]+""")


def redact_folders(text: str, folders: Iterable[tuple[str, str]]) -> str:
    """`text` with each of `folders` replaced by its placeholder, in every spelling an error message may use (single
    and doubled backslashes, forward slashes, any letter case), longest first, so the data directory wins over the
    home folder that holds it; then any other user's folder (`C:\\Users\\<name>`) becomes `<user>`."""
    spellings: dict[str, str] = {}
    for folder, label in folders:
        if len([part for part in re.split(r"[\\/]+", folder) if part]) < 2:
            continue  # empty, or a drive or root alone: replacing it would garble every path
        for spelling in (folder, folder.replace("\\", "\\\\"), folder.replace("\\", "/")):
            spellings.setdefault(spelling.casefold(), label)
    for spelling in sorted(spellings, key=len, reverse=True):
        label = spellings[spelling]
        text = re.sub(re.escape(spelling), lambda _match, label=label: label, text, flags=re.IGNORECASE)
    return _USER_FOLDER.sub(lambda match: f"{match.group(1)}<user>", text)


def summary_markdown(report: SmokeReport, folders: Iterable[tuple[str, str]] | None = None) -> str:
    """A short summary for the owner's review, meant to be pasted publicly: the folders of this computer (`folders`,
    by default `private_folders`) become placeholders, because an engine's error message can name files in them."""
    gpu = report.gpu_name or "none"
    if report.gpu_driver:
        gpu += f", driver {report.gpu_driver}"
    engine = f"{ENGINE_ID} {report.engine_commit or 'not provisioned'} ({report.engine_state or 'unknown'})"
    lines = [
        f"### Cinewright smoke test {report.cinewright_version}: {report.result}",
        "",
        f"- Result: **{report.result}** (exit code {report.exit_code}), {report.finished_at:%Y-%m-%d %H:%M} UTC",
        f"- Engine: {engine}",
        f"- GPU: {gpu}",
        f"- Attention: {_attention_text(report.attention)}",
        "",
        "| Model | Job | Result | Peak VRAM | Seconds per step | Wall time | Output |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    by_text: dict[tuple[str, str], list[str]] = {}  # one note per text, naming every model it applies to
    for job in report.jobs:
        params = ", ".join(f"{key} {value}" for key, value in job.params.items())
        output = f"{job.output_media_type} `{job.output_sha256[:12]}`" if job.output_sha256 else "none"
        if job.engine_version:
            output += f" ({job.engine_version})"
        vram = "n/a" if job.peak_vram_mb is None else f"{job.peak_vram_mb} MB"
        quant = f" {job.quant}" if job.quant else ""
        lines.append(
            f"| {job.model_id}{quant} | {job.capability}, {params}, seed {job.seed} | {job.state} | {vram} | "
            f"{_number(job.seconds_per_step, ' s', 2)} | {_number(job.wall_seconds, ' s')} | {output} |"
        )
        for label, text in (("Error", job.error), ("Peak VRAM", job.peak_vram_reason), ("Seconds per step", job.seconds_per_step_reason)):
            if text:
                by_text.setdefault((label, text), []).append(job.model_id)
    for skipped in report.skipped:
        by_text.setdefault(("Not run", skipped.reason), []).append(skipped.model_id)
    ordered = sorted(by_text.items(), key=lambda item: item[0][0] != "Error")  # errors first
    notes = [f"- {label} ({', '.join(models)}): {text}" for (label, text), models in ordered]
    if report.gpu_reason:
        notes.append(f"- GPU: {report.gpu_reason}")
    if report.attention.reason:
        notes.append(f"- Attention: {report.attention.reason}")
    if notes:
        lines += ["", *notes]
    text = "\n".join(lines) + "\n"
    return redact_folders(text, private_folders(report.data_dir) if folders is None else folders)


def write_report(report: SmokeReport, logs_dir: Path) -> tuple[Path, Path]:
    """Write the JSON report and the Markdown summary; returns both paths."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"smoke-{_file_version(report.cinewright_version)}"
    json_path, summary_path = logs_dir / f"{stem}.json", logs_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    summary_path.write_text(summary_markdown(report), encoding="utf-8", newline="\n")
    return json_path, summary_path


def run(out: TextIO | None = None, *, installer: Path | None = None, model_store: Path | None = None) -> int:
    """Run the smoke test against the core on 127.0.0.1 and `CINEWRIGHT_PORT`, or with `installer` against a temporary
    installation of that installer (`cinewright.smoke_installer`); returns the exit code."""
    out = out if out is not None else sys.stdout
    if model_store is not None and installer is None:
        return could_not_run(out, "--model-store works only with --installer: it becomes the models folder of the temporary installation.")
    if installer is not None:
        from cinewright import smoke_installer

        return smoke_installer.run(out, installer, model_store)
    try:
        url = f"http://127.0.0.1:{load_config().port}"
    except ValueError as exc:
        return could_not_run(out, str(exc))

    def smoke_test() -> int:
        with httpx.Client(base_url=url, timeout=HTTP_TIMEOUT_S, trust_env=False) as http:
            return SmokeRun(http, out).run()

    return run_guarded(out, smoke_test, core=f"The core at {url}")


def run_guarded(out: TextIO, body: Callable[[], int], *, core: str) -> int:
    """`body()`'s exit code; every way it can end without a result line becomes exit code 2 with a last line that
    says why, so the last printed line always states the result and the exit code. `core` names the core tested."""
    try:
        return body()
    except CannotRun as exc:
        return could_not_run(out, str(exc))
    except httpx.HTTPError as exc:
        return could_not_run(out, f"{core} stopped answering ({exc}).")
    except OSError as exc:
        return could_not_run(out, f"A file or folder could not be read or written ({exc}).")
    except ValueError as exc:
        return could_not_run(out, f"An answer of {core[:1].lower()}{core[1:]} could not be read ({type(exc).__name__}: {exc}).")
    except KeyboardInterrupt:
        return could_not_run(out, "The run was interrupted. A smoke job that was still running was cancelled.")
    except Exception as exc:  # noqa: BLE001 - an answer of an unexpected form, say; the result line must still come last
        return could_not_run(out, f"The smoke test stopped on an unexpected error ({type(exc).__name__}: {exc}).")


def could_not_run(out: TextIO, problem: str) -> int:
    say(out, f"Smoke test could not run: {sentence(problem)} Exit code {EXIT_ERROR}.")
    return EXIT_ERROR


def sentence(text: str) -> str:
    """`text` on one line, ending with a full stop if it ends without one."""
    text = " ".join(str(text).split())
    return text if text.endswith((".", "!", "?")) else f"{text}."
