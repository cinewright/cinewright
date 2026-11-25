"""`python -m cinewright.devtools smoke`: the GPU smoke test, here against the real core and the mock worker.

The core runs on uvicorn and the mock worker as its own process (`python -m cinewright_worker --engine mock`), as in
`test_e2e_mock`, whose fixtures these tests use. Nothing installs models in this build, so the tests record the
smoke models as installed in the core's database. The measurements are pure functions with their own tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import test_e2e_mock
from test_e2e_mock import LiveCore

from cinewright import __version__, devtools, smoke
from cinewright.core.registry import Registry
from cinewright_worker.engines.mock import render_png

API = "/api/v1"
SMOKE_MODELS = [job.model_id for job in smoke.SMOKE_JOBS]
# Wan2GP's frame grid (frames_minimum, frames_steps) of the video smoke models: frame counts are min + n * step.
FRAME_GRID = {"ltx-2.3-distilled-22b": (17, 8), "wan-2.2-i2v-14b": (5, 4)}

# The fixtures of the end-to-end test: the core on uvicorn and the mock worker process.
live_core = test_e2e_mock.live_core
mock_worker = test_e2e_mock.mock_worker


def mark_installed(core: LiveCore, *model_ids: str, quant: str = "bf16") -> None:
    """Record the models' quantization as installed in the core's database, as a completed install leaves it."""
    now = datetime.now(UTC).isoformat()
    with core.server.config.app.state.core.db.transaction() as conn:
        conn.executemany(
            "INSERT INTO installed_models (model_id, quant, state, installed_at) VALUES (?, ?, 'installed', ?) "
            "ON CONFLICT (model_id, quant) DO UPDATE SET state = 'installed'",
            [(model_id, quant, now) for model_id in model_ids],
        )


def use_core(monkeypatch: pytest.MonkeyPatch, core: LiveCore) -> None:
    monkeypatch.setenv("CINEWRIGHT_PORT", str(urlsplit(core.url).port))


def run_smoke(capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str]]:
    code = devtools.main(["smoke"])
    return code, capsys.readouterr().out.strip().splitlines()


def report_paths(home: Path) -> tuple[Path, Path]:
    return home / "logs" / f"smoke-{__version__}.json", home / "logs" / f"smoke-{__version__}.md"


def test_smoke_runs_every_installed_smoke_model_and_writes_the_report(live_core, mock_worker, home, monkeypatch, capsys):
    mark_installed(live_core, *SMOKE_MODELS)
    use_core(monkeypatch, live_core)

    code, lines = run_smoke(capsys)

    assert code == 0, "\n".join(lines)
    assert lines[-1].startswith("Smoke test passed") and lines[-1].endswith("Exit code 0."), lines[-1]
    json_path, summary_path = report_paths(home)
    assert f"Report:  {json_path}" in lines and f"Summary: {summary_path}" in lines
    report = smoke.SmokeReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert (report.result, report.exit_code, report.cinewright_version) == ("passed", 0, __version__)
    assert report.data_dir == str(home) and report.core_url == live_core.url
    assert report.engine_state == "not_installed" and report.engine_commit is None
    assert report.skipped == []
    # Without a provisioning record (the mock engine), the attention fields are null, with the reason.
    assert report.attention.mode is None and report.attention.sage_attention is None and report.attention.probe is None
    assert "provisioning record" in report.attention.reason
    assert (report.gpu_name is None) == (report.gpu_reason is not None)

    start_image = hashlib.sha256(render_png(smoke.SEED, *smoke.START_IMAGE_SIZE)).hexdigest()
    assert [job.model_id for job in report.jobs] == SMOKE_MODELS
    for job, spec in zip(report.jobs, smoke.SMOKE_JOBS, strict=True):
        assert job.passed and job.state == "succeeded" and job.error is None, job
        assert (job.capability, job.seed, job.params, job.quant) == (spec.capability, smoke.SEED, dict(spec.params), "bf16")
        # The mock engine writes a PNG for every job and has no torch, so no peak VRAM.
        assert (job.output_media_type, job.engine_version) == ("image/png", "mock-1")
        assert job.peak_vram_mb is None and job.peak_vram_reason
        assert (job.seconds_per_step is None) == (job.seconds_per_step_reason is not None)
        assert job.wall_seconds > 0 and job.total_steps
        content = live_core.http.get(f"{API}/assets/{job.output_sha256}/content")
        assert hashlib.sha256(content.content).hexdigest() == job.output_sha256
        # Each job pins its model and the fixed seed; image-to-video jobs start from the uploaded start image.
        request = live_core.http.get(f"{API}/jobs/{job.job_id}").json()["request"]
        assert (request["model"], request["params"]["seed"]) == (job.model_id, smoke.SEED)
        assert request["inputs"] == ([start_image] if job.capability == "i2v" else [])

    summary = summary_path.read_text(encoding="utf-8")
    assert summary.startswith(f"### Cinewright smoke test {__version__}: passed")
    assert all(model_id in summary for model_id in SMOKE_MODELS)
    assert "Peak VRAM" in summary and "Seconds per step" in summary and "Attention:" in summary
    assert str(home) not in summary  # pasted into a public review: no local paths
    assert b"\r\n" not in json_path.read_bytes() and b"\r\n" not in summary_path.read_bytes()


def test_smoke_exits_1_when_a_generation_fails(live_core, mock_worker, home, monkeypatch, capsys):
    mark_installed(live_core, *SMOKE_MODELS)
    use_core(monkeypatch, live_core)
    # The mock engine fails a job whose params.extra names an error code.
    jobs = tuple(
        dataclasses.replace(job, params={**job.params, "extra": {"mock_fail": "oom"}}) if job.model_id == "z-image-turbo" else job
        for job in smoke.SMOKE_JOBS
    )
    monkeypatch.setattr(smoke, "SMOKE_JOBS", jobs)

    code, lines = run_smoke(capsys)

    assert code == 1, "\n".join(lines)
    assert lines[-1].startswith("Smoke test failed") and "z-image-turbo" in lines[-1] and lines[-1].endswith("Exit code 1."), lines[-1]
    json_path, summary_path = report_paths(home)
    report = smoke.SmokeReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert (report.result, report.exit_code) == ("failed", 1)
    states = {job.model_id: (job.state, job.passed) for job in report.jobs}
    expected = {"ltx-2.3-distilled-22b": ("succeeded", True), "wan-2.2-i2v-14b": ("succeeded", True), "z-image-turbo": ("failed", False)}
    assert states == expected
    failed = report.jobs[-1]
    assert failed.output_sha256 is None and failed.error.startswith("oom:")
    assert "failed" in summary_path.read_text(encoding="utf-8")


def test_smoke_runs_only_the_installed_smoke_models(live_core, mock_worker, home, monkeypatch, capsys):
    mark_installed(live_core, "z-image-turbo", quant="int8")
    mark_installed(live_core, "wan-2.2-t2v-14b")  # not a smoke model
    use_core(monkeypatch, live_core)

    code, lines = run_smoke(capsys)

    assert code == 0, "\n".join(lines)
    report = smoke.SmokeReport.model_validate_json(report_paths(home)[0].read_text(encoding="utf-8"))
    assert [job.model_id for job in report.jobs] == ["z-image-turbo"]
    assert [skipped.model_id for skipped in report.skipped] == ["ltx-2.3-distilled-22b", "wan-2.2-i2v-14b"]


def test_smoke_exits_2_when_no_smoke_model_is_installed(live_core, home, monkeypatch, capsys):
    mark_installed(live_core, "wan-2.2-t2v-14b")
    use_core(monkeypatch, live_core)

    code, lines = run_smoke(capsys)

    assert code == 2
    assert lines[-1].startswith("Smoke test could not run: None of the smoke models") and lines[-1].endswith("Exit code 2."), lines[-1]
    assert not report_paths(home)[0].exists()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("port", ["free", "not-a-port"])
def test_smoke_exits_2_when_no_core_is_reachable(port, home, monkeypatch, capsys):
    monkeypatch.setenv("CINEWRIGHT_PORT", str(free_port()) if port == "free" else port)

    code, lines = run_smoke(capsys)

    assert code == 2
    assert lines[-1].startswith("Smoke test could not run:") and lines[-1].endswith("Exit code 2."), lines[-1]
    assert not (home / "logs").exists()


@pytest.fixture
def other_server(request: pytest.FixtureRequest) -> Iterator[int]:
    """Another web server on a free port (a router's admin page, say), answering every GET with `request.param`:
    (content type, body)."""
    content_type, body = request.param

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "other_server",
    [
        ("text/html", b"<html>router admin</html>"),
        ("application/json", b"[1, 2]"),
        ("application/json", b'{"status": "ok"}'),
        ("application/json", b'{"name": "Cinewright"}'),  # without its version and data directory
    ],
    indirect=True,
)
def test_smoke_exits_2_when_the_server_on_the_port_is_not_a_cinewright_core(other_server, home, monkeypatch, capsys):
    monkeypatch.setenv("CINEWRIGHT_PORT", str(other_server))

    code, lines = run_smoke(capsys)

    assert code == 2, lines
    assert lines[-1].startswith("Smoke test could not run:") and lines[-1].endswith("Exit code 2."), lines[-1]
    assert "Cinewright core" in lines[-1]
    assert not (home / "logs").exists()


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (OSError(28, "No space left on device"), "could not be read or written"),
        (ValueError("unexpected answer"), "could not be read"),
        (KeyboardInterrupt(), "The run was interrupted."),
        (KeyError("version"), "unexpected error (KeyError"),
    ],
)
def test_every_other_end_of_a_run_is_exit_2_with_a_last_line(error, text, home, monkeypatch, capsys):
    """The last printed line states the result and the exit code, also when the report cannot be written or the owner
    presses Ctrl+C."""

    def execute(self: smoke.SmokeRun) -> tuple[int, str]:
        raise error

    monkeypatch.setattr(smoke.SmokeRun, "execute", execute)

    code, lines = run_smoke(capsys)

    assert code == 2
    assert lines[-1].startswith("Smoke test could not run:") and text in lines[-1] and lines[-1].endswith("Exit code 2."), lines[-1]


def test_smoke_jobs_fit_the_registry():
    registry = Registry.load()
    assert SMOKE_MODELS == ["ltx-2.3-distilled-22b", "wan-2.2-i2v-14b", "z-image-turbo"]
    for job in smoke.SMOKE_JOBS:
        model = registry.get(job.model_id)
        assert model is not None and job.capability in model.capabilities
        assert job.params["resolution"] in model.limits.resolutions
        frames = job.params.get("frames")
        assert (frames is None) == (job.capability == "t2i")
        if frames is not None:
            minimum, step = FRAME_GRID[job.model_id]
            assert frames <= model.limits.max_frames and frames >= minimum and (frames - minimum) % step == 0


def test_smoke_tool_imports_only_runtime_dependencies():
    """It runs in the installed runtime, which has neither the dev extra (pytest, Pillow) nor torch."""
    modules = "cinewright.devtools, cinewright.smoke, cinewright.smoke_installer"
    code = f"import sys, {modules}; print(sorted({{'PIL', 'pytest', 'torch'}} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert result.stdout.strip() == "[]"


# --------------------------------------------------------------------------- measurements

T0 = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


def progress(seconds: float, step: int | None, total: int | None = 10, phase: str = "inference") -> dict:
    data = {"phase": phase, "step": step, "total_steps": total, "percent": 0.0}
    return {"kind": "progress", "data": data, "created_at": (T0 + timedelta(seconds=seconds)).isoformat()}


def log(seconds: float, **data) -> dict:
    return {"kind": "log", "data": {"message": "x", **data}, "created_at": (T0 + timedelta(seconds=seconds)).isoformat()}


def test_seconds_per_step_times_consecutive_denoising_steps_of_one_pass():
    events = [
        {"kind": "state", "data": {"state": "running"}, "created_at": T0.isoformat()},
        progress(0, 1, total=40, phase="loading"),  # Wan2GP counts loaded blocks as steps of another phase
        progress(20, 40, total=40, phase="loading"),
        progress(30, 1),  # model loading before the first step does not count
        progress(32, 2),
        progress(32.001, 3),  # posted in the same batch as step 2
        progress(36, 5),
        progress(37, 10, phase="decoding"),  # Wan2GP reports the last step of a pass as decoding
        progress(38, 0, total=3, phase="inference"),  # a second pass counts its steps from 0 again
        progress(41, 3, total=3, phase="inference"),
        progress(42, 3, total=3, phase="inference"),  # no step: no time
        progress(50, None, None, phase="saving"),
    ]
    value, reason = smoke.seconds_per_step(events)
    assert reason is None
    assert value == pytest.approx((6 + 3) / (4 + 3))


def test_seconds_per_step_does_not_time_the_gap_between_two_passes():
    # Step 2 -> 3 crosses from Wan2GP's first pass to its second; with equal totals only the phase tells them apart.
    events = [
        progress(0, 1, total=4, phase="inference_stage_1"),
        progress(5, 2, total=4, phase="inference_stage_1"),
        progress(60, 3, total=4, phase="inference_stage_2"),
        progress(65, 4, total=4, phase="inference_stage_2"),
    ]
    assert smoke.seconds_per_step(events) == (5.0, None)


@pytest.mark.parametrize(
    ("events", "text"),
    [
        ([], "fewer than two denoising steps"),
        ([progress(0, 1), progress(5, 1, total=4)], "fewer than two denoising steps"),
        ([progress(0, 1, phase="loading"), progress(5, 9, phase="loading")], "fewer than two denoising steps"),
        ([progress(0, 1), progress(0.01, 8)], "cannot be told"),
    ],
)
def test_seconds_per_step_is_null_with_a_reason(events, text):
    value, reason = smoke.seconds_per_step(events)
    assert value is None and text in reason


def failed_report(data_dir: str, *errors: str) -> smoke.SmokeReport:
    jobs = [
        smoke.JobResult(model_id=job.model_id, capability=job.capability, seed=1, params={}, state="failed", error=error)
        for job, error in zip(smoke.SMOKE_JOBS, errors, strict=False)
    ]
    return smoke.SmokeReport(
        result="failed",
        exit_code=1,
        cinewright_version="0.1.0",
        core_url="http://127.0.0.1:8737",
        data_dir=data_dir,
        attention=smoke.AttentionResult(reason="none"),
        started_at=T0,
        finished_at=T0,
        jobs=jobs,
    )


def test_summary_names_no_local_paths():
    path = r"C:\Users\Jane\AppData\Local\Cinewright"
    summary = smoke.summary_markdown(failed_report(path, "engine_failed: " + path + r"\models\x is missing"))
    assert path not in summary and "Jane" not in summary
    assert r"engine_failed: <data dir>\models\x is missing" in summary


def test_summary_replaces_the_folders_of_the_account_in_every_spelling():
    data_dir = r"C:\Users\Jane Doe\AppData\Local\Cinewright"
    folders = [
        (data_dir, "<data dir>"),
        (r"C:\Users\JANEDO~1\AppData\Local\Temp", "%TEMP%"),  # the short name the system may report for %TEMP%
        (r"C:\Users\Jane Doe\AppData\Local", "%LOCALAPPDATA%"),
        (r"C:\Users\Jane Doe", "%USERPROFILE%"),
    ]
    missing = FileNotFoundError(2, "No such file or directory", data_dir + r"\models\ltx.safetensors")
    full = OSError(28, "No space left on device", r"C:\Users\JANEDO~1\AppData\Local\Temp\cinewright-job-abc\inputs\x.png")
    report = failed_report(
        data_dir,
        f"engine_failed: The worker failed while running the job: {missing}",  # repr: doubled backslashes
        f"engine_failed: {full}; also c:/users/jane doe/appdata/local/cinewright/models/b.safetensors",
        r"engine_failed: C:\Users\Jane Doe\AppData\Local\Programs\Cinewright\runtime\python.exe and C:\Users\Other Person\x",
    )

    summary = smoke.summary_markdown(report, folders)

    assert "jane" not in summary.casefold() and "other person" not in summary.casefold(), summary
    assert r"'<data dir>\\models\\ltx.safetensors'" in summary
    assert r"'%TEMP%\\cinewright-job-abc\\inputs\\x.png'" in summary
    assert "<data dir>/models/b.safetensors" in summary
    assert r"%LOCALAPPDATA%\Programs\Cinewright\runtime\python.exe" in summary
    assert r"C:\Users\<user>\x" in summary


def test_the_summary_hides_the_home_and_temporary_folders_of_this_computer():
    folders = dict(smoke.private_folders("data"))
    assert folders["data"] == "<data dir>"
    assert str(Path.home()) in folders and tempfile.gettempdir() in folders


def test_peak_vram_comes_from_the_workers_log_event():
    assert smoke.peak_vram([log(0), log(1, peak_vram_mb=11000), log(2, peak_vram_mb=12345.4)]) == (12345, None)
    value, reason = smoke.peak_vram([log(0), progress(1, 1), log(2, peak_vram_mb=True), log(3, peak_vram_mb="9000")])
    assert value is None and "mock engine" in reason


def write_record(home: Path, record: dict) -> None:
    path = home / "engines" / "wan2gp" / "provisioning.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    ("attention", "wheels", "expected"),
    [
        ("sage2", {"triton": "installed", "sageattention": "installed"}, ("sage2", True, "passed")),
        ("sage", {"triton": "installed", "sageattention": "installed"}, ("sage", True, "passed")),
        ("sdpa", {"triton": "installed", "sageattention": "installed"}, ("sdpa", False, "failed")),
        ("sdpa", {"triton": "installed", "sageattention": "download_failed"}, ("sdpa", False, "not_run")),
        ("sdpa", {}, ("sdpa", False, "not_run")),
        ("flash", {"triton": "installed"}, ("sdpa", False, "failed")),  # an unknown mode is not passed to Wan2GP
    ],
)
def test_attention_result_reads_the_provisioning_record(home, attention, wheels, expected):
    write_record(home, {"engine": "wan2gp", "commit": "38758d3", "attention": attention, "optional_wheels": wheels})
    result = smoke.attention_result(home)
    assert (result.mode, result.sage_attention, result.probe) == expected
    assert result.optional_wheels == wheels and result.reason is None


def test_attention_result_is_null_with_a_reason_without_a_record(home):
    result = smoke.attention_result(home)
    assert result.model_dump(exclude={"reason"}) == {"mode": None, "sage_attention": None, "probe": None, "optional_wheels": None}
    assert "provisioning.json" in result.reason


def test_a_model_counts_as_installed_when_any_quantization_is():
    models = {
        "items": [
            {"id": "a", "quantizations": [{"quant": "bf16", "install_state": "absent"}, {"quant": "int8", "install_state": "installed"}]},
            {"id": "b", "quantizations": [{"quant": "bf16", "install_state": "downloading"}, {"quant": "int8", "install_state": "failed"}]},
            {"id": "c", "quantizations": []},
        ]
    }
    assert smoke.installed_models(models) == {"a"}


def test_iter_sse_yields_events_and_marks_keepalives():
    lines = [": hello", "", "id: j:1", "event: state", 'data: {"seq": 1}', "", ": keepalive", "", 'data: {"seq": 2}', ""]
    assert list(smoke.iter_sse(lines)) == [None, {"seq": 1}, None, {"seq": 2}]


# --------------------------------------------------------------------------- one job against a fake core


class FakeCore:
    """The core as `SmokeRun.run_job` uses it; `events` is the job's event stream (bytes) or an exception it raises."""

    def __init__(self, events: bytes | BaseException, final_state: str = "succeeded") -> None:
        self.events = events
        self.final_state = final_state
        self.cancels: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix(API)
        if path == "/estimate":
            return httpx.Response(200, json={"feasible": True})
        if request.method == "POST" and path == "/jobs":
            return httpx.Response(202, json={"id": "j1", "state": "queued", "route": {"quant": "int8"}})
        if path == "/events":
            if isinstance(self.events, BaseException):
                raise self.events
            return httpx.Response(200, content=self.events, headers={"Content-Type": "text/event-stream"})
        if path == "/jobs/j1/cancel":
            self.cancels.append("j1")
            return httpx.Response(200, json={"id": "j1", "state": "running"})
        if path == "/jobs/j1":
            outputs = [OUTPUT] if self.final_state == "succeeded" else []
            return httpx.Response(200, json={"id": "j1", "state": self.final_state, "outputs": outputs})
        if path == f"/assets/{OUTPUT}":
            return httpx.Response(200, json={"sha256": OUTPUT, "media_type": "video/mp4"})
        return httpx.Response(404)

    def run_job(self) -> smoke.JobResult:
        with httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(self.handler)) as http:
            return smoke.SmokeRun(http, sys.stdout).run_job(smoke.SMOKE_JOBS[2], None)


OUTPUT = "a" * 64


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


def test_a_finished_job_is_measured_and_not_cancelled_and_long_jobs_print_progress(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "PROGRESS_EVERY_S", 0.0)
    core = FakeCore(sse(progress(0, 1, total=4), progress(2, 2, total=4), {"kind": "state", "data": {"state": "succeeded"}}))

    result = core.run_job()

    assert result.passed and result.error is None and core.cancels == []
    assert (result.output_media_type, result.total_steps, result.seconds_per_step) == ("video/mp4", 4, 2.0)
    assert "  still running: inference, step 2 of 4 ..." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("events", "text"),
    [
        (httpx.ReadTimeout("timed out"), "The core stopped answering"),
        (sse(progress(0, 1)), "ended while the job was running, so the smoke test cancelled it"),
        (b"data: {not json\n\n", "could not be read"),
    ],
)
def test_a_job_left_before_its_end_state_is_cancelled(events, text, capsys):
    core = FakeCore(events, final_state="running")

    result = core.run_job()

    assert not result.passed and text in result.error, result.error
    assert core.cancels == ["j1"]


def test_ctrl_c_during_a_job_cancels_it(capsys):
    core = FakeCore(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        core.run_job()

    assert core.cancels == ["j1"]


def test_problem_text_joins_title_and_detail():
    response = httpx.Response(501, json={"title": "Not implemented yet", "detail": "Model installation arrives in phase 1."})
    assert smoke.problem_text(response) == "Not implemented yet: Model installation arrives in phase 1"
    assert smoke.problem_text(httpx.Response(502, text="<html>")) == "HTTP 502"
