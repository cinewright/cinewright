"""The worker loop: register, heartbeat, long-poll leases and drive one engine adapter per job.

Per job: download the inputs, `prepare`, `submit`, poll `status` every `poll_interval` seconds and
post progress and preview events in batches, honour cancel requests (events acknowledgement or
heartbeat), upload the outputs and complete, or report a failure. A cancelled job is reported as a
failure with code `cancelled`; a job interrupted by the worker's own shutdown as `worker_stopped`
(retryable). Completing or failing a job is retried with backoff on connection errors and 5xx answers
until the core accepts it, answers 404/409 (it already closed the job), or `close_timeout` (about the
lease time) has passed; only then does the worker ask for the next lease. Tokens never appear in log
lines.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

from cinewright_worker.adapter import EngineAdapter, EngineHealth, EngineJob, EngineStatus
from cinewright_worker.client import TRANSIENT_ERRORS, ProtocolError, WorkerClient
from cinewright_worker.engines import EXTENSIONS, GENERIC_MEDIA_TYPE

log = logging.getLogger("cinewright_worker")

ENROLLMENT_TOKEN_ENV = "CINEWRIGHT_ENROLLMENT_TOKEN"  # noqa: S105 - variable name, not a secret
CORE_URL_ENV = "CINEWRIGHT_CORE_URL"
DEFAULT_CORE_URL = "http://127.0.0.1:8737"
DEFAULT_HOST_ID = "local"

UPLOAD_MEDIA_TYPES = frozenset(EXTENSIONS) | {GENERIC_MEDIA_TYPE}
HEALTH_STATES = ("starting", "ready", "busy", "failed")
TERMINAL_STATES = ("succeeded", "failed", "cancelled")
MAX_EVENTS_PER_POST = 100
MAX_PREVIEW_BYTES = 256 * 1024
MAX_MESSAGE_CHARS = 1000
CLOSE_TIMEOUT = 120.0  # matches the core's LEASE_SECONDS
MAX_CLOSE_BACKOFF = 15.0
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")

CANCELLED = ("cancelled", "The job was cancelled on request.", False)
WORKER_STOPPED = ("worker_stopped", "The worker stopped before the job finished; the job can run again.", True)


class LeaseLost(Exception):
    """The core no longer knows this lease (404/409): the job was cancelled or handed to another worker."""


class WorkerLoop:
    def __init__(
        self,
        adapter: EngineAdapter,
        client: WorkerClient,
        enrollment_token: str,
        *,
        host_id: str = DEFAULT_HOST_ID,
        stop_event: threading.Event | None = None,
        once: bool = False,
        lease_wait: float = 10.0,
        poll_interval: float = 0.5,
        event_interval: float = 1.0,
        preview_interval: float = 2.0,
        heartbeat_interval: float = 10.0,
        cancel_timeout: float = 60.0,
        register_timeout: float = 60.0,
        retry_interval: float = 2.0,
        close_timeout: float = CLOSE_TIMEOUT,
        work_dir: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.client = client
        self.stop = stop_event or threading.Event()
        self.once = once
        self.host_id = host_id
        self.lease_wait = lease_wait
        self.poll_interval = poll_interval
        self.event_interval = event_interval
        self.preview_interval = preview_interval
        self.heartbeat_interval = heartbeat_interval
        self.cancel_timeout = cancel_timeout
        self.register_timeout = register_timeout
        self.retry_interval = retry_interval
        self.close_timeout = close_timeout
        self.work_dir = work_dir
        self.poll_seconds = 1.0
        self.worker_id: str | None = None
        self._enrollment_token = enrollment_token
        self._engine_version: str | None = None
        self._cancel_ids: set[str] = set()
        self._lock = threading.Lock()
        self._busy = False
        self._finished = threading.Event()

    # ------------------------------------------------------------------ main loop

    def run(self) -> int:
        """Run until stopped (or after one job with `once`). Returns a process exit code."""
        if not self._register():
            return 1
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True, name="cinewright-heartbeat")
        heartbeat.start()
        try:
            while not self.stop.is_set():
                try:
                    job = self.client.lease(self.lease_wait)
                except ProtocolError as exc:
                    if exc.status == 401:
                        log.error("The core no longer accepts this worker (%s); exiting.", exc)
                        return 2
                    log.warning("Lease request failed: %s", exc)
                    self.stop.wait(self.retry_interval)
                    continue
                except TRANSIENT_ERRORS as exc:
                    log.warning("Core not reachable: %s", exc)
                    self.stop.wait(self.retry_interval)
                    continue
                if job is None:
                    self.stop.wait(self.poll_seconds)
                    continue
                self._run_job(job)
                if self.once:
                    break
        finally:
            self._finished.set()
            heartbeat.join(timeout=5)
        return 0

    def _register(self) -> bool:
        try:
            models = sorted({model.engine_model for model in self.adapter.capabilities() if model.available})
            health = self.adapter.health()
        except Exception:  # noqa: BLE001 - an engine that cannot start ends the worker with a log line
            log.exception("Engine %s failed to start.", self.adapter.name)
            return False
        self._engine_version = health.engine_version
        registration = {
            "host_id": self.host_id,
            "engine": self.adapter.name,
            "engine_version": health.engine_version,
            "models": models,
            "health": _health_body(health),
        }
        deadline = time.monotonic() + self.register_timeout
        while not self.stop.is_set():
            try:
                registered = self.client.register(registration, self._enrollment_token)
            except ProtocolError as exc:
                log.error("Registration refused: %s", exc)
                return False
            except TRANSIENT_ERRORS as exc:
                if time.monotonic() >= deadline:
                    log.error("Core at %s not reachable: %s", self.client.core_url, exc)
                    return False
                self.stop.wait(1.0)
                continue
            self.worker_id = registered["worker_id"]
            self.poll_seconds = float(registered.get("poll_seconds") or self.poll_seconds)
            log.info("Registered as worker %s (%s, %d models).", self.worker_id, self.adapter.name, len(models))
            return True
        return False

    def _heartbeat_loop(self) -> None:
        while not self._finished.wait(self.heartbeat_interval):
            try:
                cancel = self.client.heartbeat(self._health())
            except (ProtocolError, *TRANSIENT_ERRORS) as exc:
                log.warning("Heartbeat failed: %s", exc)
                continue
            if cancel:
                with self._lock:
                    self._cancel_ids.update(cancel)

    def _health(self) -> dict[str, Any]:
        try:
            health = self.adapter.health()
        except Exception as exc:  # noqa: BLE001 - a broken engine is reported, not raised
            return {"state": "failed", "detail": _short(f"Engine health check failed: {exc}")}
        body = _health_body(health)
        if self._busy and body["state"] == "ready":
            body["state"] = "busy"
        return body

    # ------------------------------------------------------------------ one job

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        log.info("Leased job %s (%s).", job_id, job["engine_model"])
        self._busy = True
        work = tempfile.mkdtemp(prefix="cinewright-job-", dir=self.work_dir)
        try:
            self._execute(job, work)
        except LeaseLost:
            log.warning("Lease of job %s is gone; dropped the job.", job_id)
        except Exception as exc:  # noqa: BLE001 - one broken job must not stop the worker
            log.exception("Job %s crashed in the worker.", job_id)
            self._fail(job, "engine_failed", f"The worker failed while running the job: {exc}")
        finally:
            self._busy = False
            with self._lock:
                self._cancel_ids.discard(job_id)
            shutil.rmtree(work, ignore_errors=True)

    def _execute(self, job: dict[str, Any], work: str) -> None:
        inputs_dir = os.path.join(work, "inputs")
        outputs_dir = os.path.join(work, "outputs")
        os.makedirs(inputs_dir)
        os.makedirs(outputs_dir)
        try:
            paths = [self._download(item, inputs_dir) for item in job.get("inputs", [])]
        except ProtocolError as exc:
            if exc.status in (404, 409):
                raise LeaseLost from exc
            return self._fail(job, "input_unavailable", f"An input could not be downloaded from the core: {exc}", retryable=True)
        except TRANSIENT_ERRORS as exc:
            return self._fail(job, "input_unavailable", f"An input could not be downloaded from the core: {exc}", retryable=True)
        except ValueError as exc:  # foreign URL or hash mismatch: retrying would not help
            return self._fail(job, "input_unavailable", f"An input was refused: {exc}")
        engine_job = EngineJob(
            job_id=job["job_id"],
            engine_model=job["engine_model"],
            quant=job["quant"],
            prompt=job["prompt"],
            negative_prompt=job.get("negative_prompt"),
            seed=int(job["seed"]),
            params=dict(job.get("params") or {}),
            input_paths=tuple(paths),
            output_dir=outputs_dir,
        )
        try:
            self.adapter.prepare(engine_job.engine_model, engine_job.quant)
        except Exception as exc:  # noqa: BLE001 - adapter errors become job failures
            return self._fail(job, getattr(exc, "code", "model_unavailable"), str(exc))
        try:
            engine_job_id = self.adapter.submit(engine_job)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", "engine_failed")
            # A busy engine (e.g. still aborting a dropped job) is worth another attempt.
            return self._fail(job, code, str(exc), retryable=code == "engine_busy")
        status, reason = self._follow(job, engine_job_id)
        if status.state == "succeeded":
            return self._finish(job, engine_job_id)
        if status.state == "cancelled" or reason is not None:
            code, message, retryable = reason or CANCELLED
            return self._fail(job, code, message, retryable=retryable)
        self._fail(job, status.error_code or "engine_failed", status.error_message or "The engine failed without a message.")

    def _download(self, item: dict[str, Any], inputs_dir: str) -> str:
        extension = EXTENSIONS.get(item.get("media_type", ""), ".bin")
        dest = os.path.join(inputs_dir, item["sha256"] + extension)
        return self.client.download_input(item["url"], item["sha256"], dest)

    def _follow(self, job: dict[str, Any], engine_job_id: str) -> tuple[EngineStatus, tuple[str, str, bool] | None]:
        """Poll the engine until it reaches a terminal state; returns the status and the cancel reason, if any."""
        pending: list[dict[str, Any]] = []
        last_progress: dict[str, Any] | None = None
        last_preview: str | None = None
        last_preview_at = 0.0
        last_flush = time.monotonic()
        reason: tuple[str, str, bool] | None = None
        cancel_deadline: float | None = None
        while True:
            status = self.adapter.status(engine_job_id)
            progress = {
                "phase": status.phase or status.state,
                "step": status.step,
                "total_steps": status.total_steps,
                "percent": min(100.0, max(0.0, float(status.percent))),
            }
            if progress != last_progress:
                pending.append({"kind": "progress", "data": progress})
                last_progress = progress
            if status.preview_path and status.preview_path != last_preview and time.monotonic() - last_preview_at >= self.preview_interval:
                last_preview, last_preview_at = status.preview_path, time.monotonic()  # at most one preview per interval
                preview = _preview_event(status.preview_path)
                if preview is not None:
                    pending.append(preview)
            terminal = status.state in TERMINAL_STATES
            if pending and (terminal or time.monotonic() - last_flush >= self.event_interval):
                if self._flush(job, pending, engine_job_id, terminal) and reason is None:
                    reason = CANCELLED
                last_flush = time.monotonic()
            if terminal:
                return status, reason
            if reason is None:
                reason = self._cancel_reason(job["job_id"])
            if reason is not None:
                if cancel_deadline is None:
                    log.info("Cancelling job %s (%s).", job["job_id"], reason[0])
                    self.adapter.cancel(engine_job_id)
                    cancel_deadline = time.monotonic() + self.cancel_timeout
                elif time.monotonic() > cancel_deadline:
                    log.error("Engine did not stop job %s within %.0f s.", job["job_id"], self.cancel_timeout)
                    return EngineStatus(state="cancelled", phase="cancelled"), reason
            time.sleep(self.poll_interval)

    def _cancel_reason(self, job_id: str) -> tuple[str, str, bool] | None:
        with self._lock:
            if job_id in self._cancel_ids:
                return CANCELLED
        return WORKER_STOPPED if self.stop.is_set() else None

    def _flush(self, job: dict[str, Any], pending: list[dict[str, Any]], engine_job_id: str, terminal: bool) -> bool:
        """Post pending events in batches of at most 100; returns whether the core asked to cancel."""
        cancel = False
        try:
            while pending:
                cancel = self.client.post_events(job["job_id"], job["lease_id"], pending[:MAX_EVENTS_PER_POST]) or cancel
                del pending[:MAX_EVENTS_PER_POST]
        except ProtocolError as exc:
            if exc.status in (404, 409):
                if not terminal:
                    self.adapter.cancel(engine_job_id)
                raise LeaseLost from exc
            log.warning("Posting events for job %s failed: %s", job["job_id"], exc)
            pending.clear()
        except TRANSIENT_ERRORS as exc:
            log.warning("Posting events for job %s failed: %s", job["job_id"], exc)
            del pending[:-MAX_EVENTS_PER_POST]  # keep the newest for the next attempt
        return cancel

    def _finish(self, job: dict[str, Any], engine_job_id: str) -> None:
        try:
            files = self.adapter.fetch_outputs(engine_job_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail(job, getattr(exc, "code", "engine_failed"), f"The engine's outputs could not be read: {exc}")
        if not files:
            return self._fail(job, "no_outputs", "The engine finished without producing a file.")
        completed: list[dict[str, Any]] = []
        used: set[str] = set()
        for output in files:
            name = output_name(output.name or os.path.basename(output.path), used)
            media_type = output.media_type if output.media_type in UPLOAD_MEDIA_TYPES else GENERIC_MEDIA_TYPE
            try:
                uploaded = self.client.upload_output(job["job_id"], name, output.path, media_type)
            except ProtocolError as exc:
                if exc.status in (404, 409):
                    raise LeaseLost from exc
                return self._fail(job, "upload_failed", f"The core refused output {name}: {exc}")
            except TRANSIENT_ERRORS as exc:
                return self._fail(job, "upload_failed", f"Output {name} could not be uploaded: {exc}", retryable=True)
            completed.append({"sha256": uploaded["sha256"], "media_type": media_type, "name": name})
        self._close(job, "completed", lambda: self.client.complete(job["job_id"], job["lease_id"], completed, self._engine_version))

    def _fail(self, job: dict[str, Any], code: str, message: str, *, retryable: bool = False) -> None:
        log.warning("Job %s failed: %s", job["job_id"], code)
        message = _short(message) or "The job failed."
        self._close(job, "failed", lambda: self.client.fail(job["job_id"], job["lease_id"], code, message, retryable))

    def _close(self, job: dict[str, Any], outcome: str, call: Callable[[], None]) -> None:
        """Report the outcome, retrying connection errors and 5xx answers with backoff until `close_timeout`."""
        deadline = time.monotonic() + self.close_timeout
        delay = self.retry_interval
        while True:
            try:
                call()
            except ProtocolError as exc:
                if exc.status in (404, 409):
                    log.warning("Core had already closed job %s (%s).", job["job_id"], exc)
                    return
                if exc.status < 500:
                    log.error("Reporting job %s as %s was refused: %s", job["job_id"], outcome, exc)
                    return
                error: Exception = exc
            except TRANSIENT_ERRORS as exc:
                error = exc
            else:
                log.info("Job %s %s.", job["job_id"], outcome)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.error("Reporting job %s as %s failed; giving up after %.0f s: %s", job["job_id"], outcome, self.close_timeout, error)
                return
            log.warning("Reporting job %s as %s failed, retrying: %s", job["job_id"], outcome, error)
            # Not stop.wait(): a shutdown reaches here with the stop event already set.
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, MAX_CLOSE_BACKOFF)


def output_name(raw: str, used: set[str]) -> str:
    """A unique output name matching the core's pattern `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`."""
    base = _UNSAFE_NAME_CHARS.sub("_", os.path.basename(raw)).lstrip("._-")[:128] or "output"
    stem, extension = os.path.splitext(base)
    name, counter = base, 1
    while name in used:
        suffix = f"-{counter}{extension}"
        name = (stem[: 128 - len(suffix)] or "output") + suffix
        counter += 1
    used.add(name)
    return name


def _health_body(health: EngineHealth) -> dict[str, Any]:
    return {
        "state": health.state if health.state in HEALTH_STATES else "failed",
        "vram_used_mb": health.vram_used_mb,
        "vram_total_mb": health.vram_total_mb,
        "detail": health.detail,
    }


def _preview_event(path: str) -> dict[str, Any] | None:
    try:
        if os.path.getsize(path) > MAX_PREVIEW_BYTES:
            return None
        with open(path, "rb") as preview:
            data = preview.read()
    except OSError:
        return None
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}.get(os.path.splitext(path)[1].lower(), "image/png")
    return {"kind": "preview", "data": {"media_type": media_type, "data_base64": base64.b64encode(data).decode("ascii")}}


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 1] + "…"
