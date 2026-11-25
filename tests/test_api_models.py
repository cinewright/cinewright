"""Model installs, removals and the text encoder variant over the internal API (#3).

The core runs on the fixture registry of `test_core_modelstore` (the bundled descriptors with small stand-in files)
and downloads from its fake hub; free disk space is faked. A stub stands in for the local worker's supervisor and
counts restarts, and the `model` system events the core publishes are recorded. Nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import httpx
import pytest
from conftest import API, BASE_URL, FakeClock, assert_problem, fake_detect, finish_job, mark_installed
from fastapi.testclient import TestClient
from test_core_modelstore import UMT5_INT8, WAN_I2V_INT8, WAN_T2V_INT8, Z_IMAGE_INT8_FILES, FakeHub, fixture_registry

from cinewright.api.schemas import (
    ClientKind,
    InstallRefusal,
    JobRequest,
    LeaseRequest,
    ModelInstall,
    SystemEventKind,
    WorkerRegistration,
)
from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.clock import to_iso
from cinewright.core.core import Core
from cinewright.core.modelstore import GB, STAGING_DIR, plan_files
from cinewright.core.supervisor import SupervisorStatus
from cinewright.db.database import Database

TIMEOUT_S = 10
QWEN3_INT8 = "Qwen3/qwen3_quanto_bf16_int8.safetensors"
QWEN3_BF16 = "Qwen3/qwen3_bf16.safetensors"
OTHER_KEYS = {"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "bf16", "save_path": "out"}


class StubSupervisor:
    """Stands in for the local worker's supervisor: a running process; counts restarts."""

    def __init__(self, since: datetime) -> None:
        self.since = since
        self.restarts = 0

    def status(self) -> SupervisorStatus:
        return SupervisorStatus("running", since=self.since)

    def restart(self) -> None:
        self.restarts += 1


class Store:
    """A core with the fixture registry, the fake hub, faked free space and a stub supervisor, served by a TestClient."""

    def __init__(self, config: Config, clock: FakeClock) -> None:
        self.registry, self.contents = fixture_registry()
        self.hub = FakeHub(self.contents)
        self.gate: threading.Event | None = None  # while set to an unset event, downloads wait
        self.free = 10**12
        self.clock = clock
        self.core = Core(
            config,
            registry=self.registry,
            detect=fake_detect,
            clock=clock,
            download_transport=httpx.MockTransport(self._handle),
            free_disk=lambda path: self.free,
        )
        self.events: list[ModelInstall] = []
        publish = self.core.event_log.publish_system

        def record(event: Any) -> None:
            if event.kind == SystemEventKind.MODEL:
                self.events.append(event.model)
            publish(event)

        self.core.event_log.publish_system = record
        self.supervisor = StubSupervisor(clock.now)
        self.core.supervisor = self.supervisor
        self.client = TestClient(create_app(self.core), base_url=BASE_URL)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.gate is not None:
            assert self.gate.wait(TIMEOUT_S), "the test never opened the download gate"
        return self.hub.handler(request)

    @property
    def root(self) -> Path:
        return self.core.paths.models_dir

    def placed(self) -> set[str]:
        return {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file() and STAGING_DIR not in p.parts}

    def size(self, model_id: str, quant: str, variant: str = "int8") -> int:
        return sum(f.size_bytes for f in plan_files(self.registry, model_id, quant, variant))

    def url(self, component: str, path: str) -> str:
        shared = self.registry.component(component)
        assert shared is not None
        return next(f.url for f in shared.files if f.path == path)

    def write_config(self, config: dict[str, Any]) -> None:
        self.core.models.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.core.models.config_path.write_bytes(json.dumps(config, indent=4).encode())

    def config(self) -> dict[str, Any]:
        return json.loads(self.core.models.config_path.read_text(encoding="utf-8"))

    def quant(self, model_id: str, quant: str) -> dict[str, Any]:
        response = self.client.get(f"{API}/models/{model_id}")
        assert response.status_code == 200, response.text
        return next(q for q in response.json()["quantizations"] if q["quant"] == quant)

    def idle(self) -> None:
        """Wait until no install or switch runs in the background."""
        wait_for(lambda: not self.core._background, "the background work to end")

    def install(self, model_id: str, quant: str) -> dict[str, Any]:
        response = self.client.post(f"{API}/models/{model_id}/install", params={"quant": quant})
        assert response.status_code == 202, response.text
        self.idle()
        return self.quant(model_id, quant)

    def route(self, job_id: str) -> dict[str, Any]:
        return json.loads(self.rows("SELECT route FROM jobs WHERE id = ?", (job_id,))[0]["route"])

    def local_host(self) -> dict[str, Any]:
        return next(host for host in self.client.get(f"{API}/hosts").json()["items"] if host["id"] == "local")

    def rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.core.db.query(sql, params)]

    def leased_job(self, model_id: str, quant: str | None, engine_model: str) -> tuple[str, str, str]:
        """A job for `model_id`, pinned to `quant` unless None (as the Generate view sends it), leased to a local worker
        serving `engine_model`: (worker id, job id, lease id)."""
        registration = WorkerRegistration(host_id="local", engine="wan2gp", models=[engine_model], health={"state": "ready"})
        worker_id = self.core.register_worker(registration, self.core.enrollment_token).worker_id
        pinned = {"quant": quant} if quant is not None else {}
        request = JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse", "model": model_id, **pinned})
        job_id = self.core.submit_job(request, ClientKind.UI, None).id
        granted = asyncio.run(self.core.lease(worker_id, LeaseRequest(wait_seconds=0)))
        assert granted is not None and granted.job.job_id == job_id
        return worker_id, job_id, granted.job.lease_id


def wait_for(predicate: Callable[[], object], what: str, timeout: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


@pytest.fixture
def store(config: Config, clock: FakeClock) -> Store:
    environment = Store(config, clock)
    with environment.client:
        yield environment
    environment.core.close()


# ---------------------------------------------------------------------- install


def test_installing_z_image_int8_places_its_files_merges_wgp_config_and_reports_progress(store):
    store.write_config(OTHER_KEYS)

    state = store.install("z-image-turbo", "int8")

    assert state["install_state"] == "installed" and state["error"] is None
    assert store.placed() == Z_IMAGE_INT8_FILES
    assert store.config() == {
        **OTHER_KEYS,
        "checkpoints_paths": [str(store.root.resolve()), "ckpts", "."],
        "text_encoder_quantization": "int8",
    }
    size = store.size("z-image-turbo", "int8")
    assert state["size_bytes"] == size
    models = store.client.get(f"{API}/models").json()
    assert next(m for m in models["items"] if m["id"] == "z-image-turbo")["installed_bytes"] == size
    assert models["store"] == {"total_bytes": size, "quota_bytes": None, "free_bytes": store.free, "last_eviction": None}
    assert store.local_host()["text_encoder_variant"] == "int8"  # the first install stores its variant
    # Progress as `model` events, then the result; the worker restarts once to announce the model.
    downloading = [e for e in store.events if e.install_state == "downloading"]
    assert downloading[0].downloaded_bytes == 0 and downloading[-1].downloaded_bytes == downloading[-1].total_bytes == size
    assert downloading[-1].progress.percent == 100 and downloading[-1].progress.phase == "download"
    assert (store.events[-1].model_id, store.events[-1].quant, store.events[-1].install_state) == ("z-image-turbo", "int8", "installed")
    assert store.supervisor.restarts == 1


def test_a_hash_mismatch_fails_the_install_naming_the_file_and_registers_nothing(store):
    url = store.url("z-image-shared", "Qwen3/tokenizer.json")
    store.hub.files[url] = store.hub.files[url][:-1] + b"!"

    state = store.install("z-image-turbo", "int8")

    assert state["install_state"] == "failed" and "Qwen3/tokenizer.json" in state["error"]
    assert store.rows("SELECT * FROM store_files") == [] and store.rows("SELECT * FROM installed_model_files") == []
    assert store.placed() == set()
    failed = store.events[-1]
    assert failed.install_state == "failed" and failed.error.code == "hash_mismatch" and "Qwen3/tokenizer.json" in failed.error.message
    assert store.supervisor.restarts == 0


def test_an_install_in_progress_shows_as_downloading_and_blocks_a_second_install_and_a_removal(store):
    store.gate = threading.Event()
    response = store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "int8"})
    assert response.status_code == 202, response.text
    assert store.quant("z-image-turbo", "int8")["install_state"] == "downloading"

    again = store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "int8"})
    assert assert_problem(again, 409).model_extra["code"] == "installing"
    assert (
        assert_problem(store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}), 409).model_extra["code"]
        == "installing"
    )
    assert (
        assert_problem(store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}), 409).model_extra["code"]
        == "installing"
    )

    store.gate.set()
    store.idle()
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"


def test_install_requests_must_name_a_known_quantization(store):
    assert "quant=" in assert_problem(store.client.post(f"{API}/models/z-image-turbo/install"), 422).detail
    assert_problem(store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "fp4"}), 422)
    assert_problem(store.client.post(f"{API}/models/unknown/install", params={"quant": "int8"}), 404)
    assert_problem(store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}), 404)  # not installed
    assert store.hub.requests == []


def test_an_install_that_does_not_fit_is_refused_before_any_byte_is_downloaded(store):
    store.free = 100

    response = store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "int8"})
    problem = assert_problem(response, 409)

    assert problem.model_extra["code"] == "not_enough_space"
    assert problem.detail.startswith("Installing Z-Image Turbo 6B int8 needs ") and "more free space" in problem.detail
    refusal = InstallRefusal.model_validate(response.json())  # the 409 as openapi.json declares it
    assert (refusal.code, refusal.evictions, refusal.frees_bytes) == ("not_enough_space", None, None)
    assert store.hub.requests == []
    assert store.quant("z-image-turbo", "int8")["install_state"] == "absent"


def test_a_job_lease_and_its_completion_mark_the_quantization_used(store):
    store.install("z-image-turbo", "int8")
    installed_at = store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"]

    store.clock.advance(60)
    worker_id, job_id, lease_id = store.leased_job("z-image-turbo", "int8", "z_image")
    leased_at = store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"]
    assert leased_at > installed_at

    store.clock.advance(60)
    finish_job(store.core, worker_id, job_id, lease_id)
    assert store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"] > leased_at


def test_a_job_without_a_quantization_routes_to_protects_and_marks_the_installed_one(store):
    store.install("z-image-turbo", "int8")
    installed_at = store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"]

    store.clock.advance(60)
    worker_id, job_id, lease_id = store.leased_job("z-image-turbo", None, "z_image")
    assert store.route(job_id)["quant"] == "int8"  # the installed one, although bf16 comes first in the registry
    leased_at = store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"]
    assert leased_at > installed_at

    problem = assert_problem(store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}), 409)
    assert problem.model_extra["code"] == "in_use" and store.placed() == Z_IMAGE_INT8_FILES

    store.clock.advance(60)
    finish_job(store.core, worker_id, job_id, lease_id)
    assert store.rows("SELECT last_used_at FROM installed_models")[0]["last_used_at"] > leased_at
    assert store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}).status_code == 204


def test_quota_eviction_never_removes_the_installed_quantization_a_job_without_one_routes_to(store, monkeypatch):
    store.install("z-image-turbo", "int8")
    store.clock.advance(60)
    store.leased_job("z-image-turbo", None, "z_image")
    used = store.client.get(f"{API}/models").json()["store"]["total_bytes"]
    monkeypatch.setattr(store.core.models, "_quota_bytes", lambda: used + store.size("ltx-2.3-distilled-22b", "Q4_K_M") - 1)

    problem = assert_problem(store.client.post(f"{API}/models/ltx-2.3-distilled-22b/install", params={"quant": "Q4_K_M"}), 409)

    assert problem.model_extra["code"] == "quota_exceeded"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed" and store.placed() == Z_IMAGE_INT8_FILES


def test_a_second_install_that_fits_only_without_the_first_is_refused_before_any_byte_is_downloaded(store):
    store.free = max(store.size("z-image-turbo", "int8"), store.size("wan-2.2-t2v-14b", "int8")) + 10  # room for one
    store.gate = threading.Event()
    assert store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "int8"}).status_code == 202

    problem = assert_problem(store.client.post(f"{API}/models/wan-2.2-t2v-14b/install", params={"quant": "int8"}), 409)

    assert problem.model_extra["code"] == "not_enough_space" and "installs in progress" in problem.detail
    store.gate.set()
    store.idle()
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"
    assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "absent"
    assert not any("wan2.2" in url for url, _ in store.hub.requests)


# ---------------------------------------------------------------------- remove, quota, worker restarts


def test_removing_one_of_two_models_that_share_a_text_encoder_keeps_it(store):
    store.install("wan-2.2-t2v-14b", "int8")
    store.install("wan-2.2-i2v-14b", "int8")
    restarts = store.supervisor.restarts

    response = store.client.delete(f"{API}/models/wan-2.2-t2v-14b", params={"quant": "int8"})

    assert response.status_code == 204, response.text
    placed = store.placed()
    assert UMT5_INT8 in placed and set(WAN_I2V_INT8) <= placed and not set(WAN_T2V_INT8) & placed
    assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "absent"
    assert (store.events[-1].model_id, store.events[-1].install_state) == ("wan-2.2-t2v-14b", "absent")
    assert store.supervisor.restarts == restarts + 1


def test_a_model_a_queued_job_needs_is_not_removed(store):
    store.install("z-image-turbo", "int8")
    request = JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse", "model": "z-image-turbo", "quant": "int8"})
    store.core.register_worker(
        WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"}), store.core.enrollment_token
    )
    store.core.submit_job(request, ClientKind.UI, None)

    problem = assert_problem(store.client.delete(f"{API}/models/z-image-turbo"), 409)

    assert problem.model_extra["code"] == "in_use" and store.placed() == Z_IMAGE_INT8_FILES


@pytest.mark.parametrize("protected", [False, True])
def test_quota_eviction_removes_the_least_recently_used_model_but_never_one_a_queued_job_needs(store, monkeypatch, protected):
    store.install("z-image-turbo", "int8")  # least recently used
    store.clock.advance(60)
    store.install("wan-2.2-t2v-14b", "int8")
    if protected:
        job_id = uuid4().hex
        route = json.dumps({"model_id": "z-image-turbo", "quant": "int8", "host_id": "local", "engine": "wan2gp", "reasons": []})
        store.core.db.execute(
            "INSERT INTO jobs (id, state, priority, capability, data_class, request, route, client, created_at) "
            "VALUES (?, 'queued', 'interactive', 't2i', 'private', '{}', ?, 'ui', ?)",
            (job_id, route, to_iso(store.clock())),
        )
    used = store.client.get(f"{API}/models").json()["store"]["total_bytes"]
    monkeypatch.setattr(store.core.models, "_quota_bytes", lambda: used + store.size("ltx-2.3-distilled-22b", "Q4_K_M") - 1)
    restarts = store.supervisor.restarts

    state = store.install("ltx-2.3-distilled-22b", "Q4_K_M")

    victim, kept = ("wan-2.2-t2v-14b", "z-image-turbo") if protected else ("z-image-turbo", "wan-2.2-t2v-14b")
    assert state["install_state"] == "installed"
    assert store.quant(victim, "int8")["install_state"] == "absent" and store.quant(kept, "int8")["install_state"] == "installed"
    # Ample free disk: the victim goes once the install registered, and one restart announces both changes.
    states = [(e.model_id, e.quant, e.install_state) for e in store.events if e.install_state != "downloading"]
    assert states[-2:] == [("ltx-2.3-distilled-22b", "Q4_K_M", "installed"), (victim, "int8", "absent")]
    assert store.supervisor.restarts == restarts + 1


# Z-Image int8 (least recently used) and Wan T2V int8 installed, and a quota that holds LTX-2 Q4_K_M only without
# Z-Image. Free disk is ample unless a test says otherwise, so the drive holds the download with Z-Image still there (#90).
LTX_Q4 = ("ltx-2.3-distilled-22b", "Q4_K_M")
LTX_Q4_URL_NAME = "ltx-2.3-22b-distilled-Q4_K_M_light.gguf"


def two_installed_and_a_quota_for_ltx(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    store.install("z-image-turbo", "int8")
    store.clock.advance(60)
    store.install("wan-2.2-t2v-14b", "int8")
    used = store.client.get(f"{API}/models").json()["store"]["total_bytes"]
    monkeypatch.setattr(store.core.models, "_quota_bytes", lambda: used + store.size(*LTX_Q4) - 1)


def ltx_q4_url(store: Store) -> str:
    model = store.registry.get(LTX_Q4[0])
    assert model is not None
    return next(f.url for q in model.quantizations if q.id == LTX_Q4[1] for f in q.files if f.name == LTX_Q4_URL_NAME)


def queue_job(store: Store, model_id: str, quant: str) -> None:
    route = json.dumps({"model_id": model_id, "quant": quant, "host_id": "local", "engine": "wan2gp", "reasons": []})
    store.core.db.execute(
        "INSERT INTO jobs (id, state, priority, capability, data_class, request, route, client, created_at) "
        "VALUES (?, 'queued', 'interactive', 't2i', 'private', '{}', ?, 'ui', ?)",
        (uuid4().hex, route, to_iso(store.clock())),
    )


def z_image_eviction(store: Store) -> dict[str, Any]:
    return {"model_id": "z-image-turbo", "quant": "int8", "name": "Z-Image Turbo 6B", "size_bytes": store.size("z-image-turbo", "int8")}


def model_states(store: Store, since: int = 0) -> list[tuple[str, str, str]]:
    """The `model` events since index `since`, without download progress."""
    return [(e.model_id, e.quant, e.install_state) for e in store.events[since:] if e.install_state != "downloading"]


@pytest.mark.parametrize("break_off", [True, False], ids=["broken-off", "corrupted"])
def test_a_failed_download_leaves_the_model_the_quota_would_remove_installed(store, monkeypatch, break_off):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    url = ltx_q4_url(store)
    if break_off:
        store.hub.cut[url] = 5
    else:
        store.hub.files[url] = store.hub.files[url][:-1] + b"!"
    placed, events, restarts = store.placed(), len(store.events), store.supervisor.restarts

    state = store.install(*LTX_Q4)

    assert state["install_state"] == "failed"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"
    assert Z_IMAGE_INT8_FILES <= store.placed() and store.placed() == placed
    assert model_states(store, events) == [(*LTX_Q4, "failed")]  # no `absent` event for Z-Image
    assert store.client.get(f"{API}/models").json()["store"]["last_eviction"] is None
    assert store.supervisor.restarts == restarts


def test_closing_the_core_during_an_install_leaves_the_model_the_quota_would_remove_installed(store, config, clock, monkeypatch):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.gate = threading.Event()
    assert store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params={"quant": LTX_Q4[1]}).status_code == 202

    closing = threading.Thread(target=store.core.close)
    closing.start()
    wait_for(lambda: store.core._closed, "the core to start closing")
    store.gate.set()
    closing.join(TIMEOUT_S)
    assert not closing.is_alive()

    restarted = Store(config, clock)  # the next core start
    try:
        assert restarted.quant("z-image-turbo", "int8")["install_state"] == "installed"
        assert restarted.quant(*LTX_Q4)["install_state"] == "failed"
        assert Z_IMAGE_INT8_FILES <= restarted.placed()
    finally:
        restarted.core.close()


@pytest.mark.parametrize("evict", [[], ["z-image-turbo:int8"]], ids=["no-evict", "evict-ignored"])
def test_with_room_on_the_drive_the_quota_removes_the_least_recently_used_model_after_the_install(store, monkeypatch, evict):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.gate = threading.Event()
    restarts, events, started = store.supervisor.restarts, len(store.events), store.clock()

    # Room on the drive: no confirmation is needed, and one that names the victim does not remove it before the download.
    response = store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params={"quant": LTX_Q4[1], "evict": evict})
    assert response.status_code == 202, response.text
    # While the download waits at the gate, the least recently used model is still installed with its files.
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed" and Z_IMAGE_INT8_FILES <= store.placed()
    assert model_states(store, events) == [] and store.supervisor.restarts == restarts
    store.gate.set()
    store.idle()

    assert store.quant(*LTX_Q4)["install_state"] == "installed"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "absent" and not Z_IMAGE_INT8_FILES & store.placed()
    assert model_states(store, events) == [(*LTX_Q4, "installed"), ("z-image-turbo", "int8", "absent")]
    usage = store.client.get(f"{API}/models").json()["store"]
    assert usage["total_bytes"] <= usage["quota_bytes"]
    assert store.supervisor.restarts == restarts + 1
    eviction = usage["last_eviction"]
    assert (eviction["model_id"], eviction["quant"], eviction["evicted"]) == (*LTX_Q4, [z_image_eviction(store)])
    assert started < datetime.fromisoformat(eviction["at"]) < store.clock()


@pytest.mark.parametrize("merge", ["unwritable", "invalid"])
def test_when_the_config_merge_fails_the_quota_still_removes_the_least_recently_used_model(store, monkeypatch, merge):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    if merge == "unwritable":
        store.write_config(OTHER_KEYS)  # not merged yet, so this install writes it

        def refuse(path: Path, data: object) -> None:
            raise PermissionError(13, "Permission denied", str(path))

        monkeypatch.setattr("cinewright.core.modelstore._write_json", refuse)
    else:
        store.core.models.config_path.parent.mkdir(parents=True, exist_ok=True)
        store.core.models.config_path.write_bytes(b"[]")
    restarts, events = store.supervisor.restarts, len(store.events)

    state = store.install(*LTX_Q4)

    assert state["install_state"] == "installed"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "absent"
    assert model_states(store, events) == [(*LTX_Q4, "installed"), ("z-image-turbo", "int8", "absent")]
    assert store.client.get(f"{API}/models").json()["store"]["last_eviction"]["evicted"] == [z_image_eviction(store)]
    assert store.supervisor.restarts == restarts + 1


@pytest.mark.parametrize("busy", [("z-image-turbo",), ("z-image-turbo", "wan-2.2-t2v-14b")], ids=["victim", "every-model"])
def test_a_job_queued_during_the_download_keeps_its_model_and_the_next_one_goes(store, monkeypatch, busy):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.gate = threading.Event()
    events = len(store.events)
    assert store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params={"quant": LTX_Q4[1]}).status_code == 202
    for model_id in busy:
        queue_job(store, model_id, "int8")
    store.gate.set()
    store.idle()

    assert store.quant(*LTX_Q4)["install_state"] == "installed"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"
    usage = store.client.get(f"{API}/models").json()["store"]
    if len(busy) == 1:  # the next least recently used model without a job goes instead
        assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "absent"
        assert model_states(store, events)[-1] == ("wan-2.2-t2v-14b", "int8", "absent")
        assert usage["total_bytes"] <= usage["quota_bytes"]
    else:  # none is left: the store stays above the quota
        assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "installed"
        assert model_states(store, events) == [(*LTX_Q4, "installed")]
        assert usage["total_bytes"] > usage["quota_bytes"] and usage["last_eviction"] is None


def test_an_install_that_fits_on_the_drive_only_after_removing_a_model_is_refused_naming_it(store, monkeypatch):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.free = store.size(*LTX_Q4) - 1  # the download fits only once Z-Image is gone
    requests, placed = len(store.hub.requests), store.placed()

    response = store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params={"quant": LTX_Q4[1]})
    problem = assert_problem(response, 409)

    assert problem.model_extra["code"] == "eviction_needed"
    assert problem.model_extra["evictions"] == [z_image_eviction(store)]
    assert problem.model_extra["frees_bytes"] == store.size("z-image-turbo", "int8")  # Z-Image shares no file
    refusal = InstallRefusal.model_validate(response.json())  # the 409 as openapi.json declares it
    assert refusal.code == "eviction_needed" and [e.model_dump() for e in refusal.evictions or []] == [z_image_eviction(store)]
    assert problem.detail == (
        "LTX-2 2.3 Distilled 1.0 22B Q4_K_M fits on the drive of the model store only after removing Z-Image Turbo 6B int8 (1.2 KB), "
        "so nothing was installed or removed; name it in `evict` to remove it first."
    )
    assert len(store.hub.requests) == requests and store.placed() == placed
    assert store.quant(*LTX_Q4)["install_state"] == "absent"  # not downloading
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"


def test_naming_another_quantization_than_the_planned_one_removes_nothing(store, monkeypatch):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.free = store.size(*LTX_Q4) - 1
    requests, placed = len(store.hub.requests), store.placed()

    evict = ["wan-2.2-t2v-14b:int8", "z-image-turbo:bf16"]
    problem = assert_problem(store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params={"quant": LTX_Q4[1], "evict": evict}), 409)

    assert problem.model_extra["code"] == "eviction_needed"
    assert problem.model_extra["evictions"] == [z_image_eviction(store)]  # the current list
    assert len(store.hub.requests) == requests and store.placed() == placed
    assert store.quant(*LTX_Q4)["install_state"] == "absent"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"
    assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "installed"


@pytest.mark.parametrize("value", ["z-image-turbo", "z-image-turbo/int8", ""])
def test_an_evict_value_without_a_colon_is_invalid_before_anything_is_planned(store, monkeypatch, value):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.free = store.size(*LTX_Q4) - 1
    requests, events = len(store.hub.requests), len(store.events)

    params = {"quant": LTX_Q4[1], "evict": ["wan-2.2-t2v-14b:int8", value]}
    problem = assert_problem(store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params=params), 422)

    assert [error["loc"] for error in problem.model_extra["errors"]] == [["query", "evict", 1]]
    assert len(store.hub.requests) == requests and store.events[events:] == []
    assert store.quant(*LTX_Q4)["install_state"] == "absent"
    assert store.quant("z-image-turbo", "int8")["install_state"] == "installed"


@pytest.mark.parametrize("fails", [False, True], ids=["installs", "download-fails"])
def test_a_confirmed_removal_happens_before_the_download_and_stays_when_it_fails(store, monkeypatch, fails):
    two_installed_and_a_quota_for_ltx(store, monkeypatch)
    store.free = store.size(*LTX_Q4) - 1
    if fails:
        store.hub.cut[ltx_q4_url(store)] = 5
    seen: list[set[tuple[str, str]]] = []
    handler = store.hub.handler

    def first_request(request: httpx.Request) -> httpx.Response:
        if "ltx-2.3" in request.url.path and not seen:
            seen.append(store.core.models.installed())
        return handler(request)

    monkeypatch.setattr(store.hub, "handler", first_request)
    restarts, events = store.supervisor.restarts, len(store.events)

    params = {"quant": LTX_Q4[1], "evict": ["z-image-turbo:int8", "wan-2.2-t2v-14b:int8"]}  # Wan: not in the plan, so it stays
    response = store.client.post(f"{API}/models/{LTX_Q4[0]}/install", params=params)
    assert response.status_code == 202, response.text
    store.idle()

    assert seen and ("z-image-turbo", "int8") not in seen[0]  # removed before the first request for LTX-2
    assert model_states(store, events) == [("z-image-turbo", "int8", "absent"), (*LTX_Q4, "failed" if fails else "installed")]
    assert store.quant("z-image-turbo", "int8")["install_state"] == "absent" and not Z_IMAGE_INT8_FILES & store.placed()
    assert store.quant(*LTX_Q4)["install_state"] == ("failed" if fails else "installed")
    assert store.quant("wan-2.2-t2v-14b", "int8")["install_state"] == "installed"
    eviction = store.client.get(f"{API}/models").json()["store"]["last_eviction"]
    assert (eviction["model_id"], eviction["quant"], eviction["evicted"]) == (*LTX_Q4, [z_image_eviction(store)])
    assert store.supervisor.restarts == restarts + (1 if fails else 2)  # after the removal, and after the install


def test_the_models_view_data_follows_the_quota_setting_and_measures_free_space_now(store):
    assert store.client.patch(f"{API}/settings", json={"model_store_quota_gb": 50}).status_code == 200
    first = store.client.get(f"{API}/models").json()["store"]
    store.free = 12345
    second = store.client.get(f"{API}/models").json()["store"]

    assert first == {"total_bytes": 0, "quota_bytes": 50 * GB, "free_bytes": 10**12, "last_eviction": None}
    assert second["free_bytes"] == 12345


def test_a_model_change_restarts_the_worker_only_once_its_running_job_ended(store):
    mark_installed(store.core, ("z-image-turbo", "bf16"))
    worker_id, job_id, lease_id = store.leased_job("z-image-turbo", "bf16", "z_image")

    store.install("z-image-turbo", "int8")
    assert store.supervisor.restarts == 0  # the job would be stopped

    finish_job(store.core, worker_id, job_id, lease_id)
    assert store.supervisor.restarts == 1


def test_maintenance_restarts_the_worker_for_a_model_change_once_no_job_runs(store):
    mark_installed(store.core, ("z-image-turbo", "bf16"))
    _, job_id, _ = store.leased_job("z-image-turbo", "bf16", "z_image")
    store.install("z-image-turbo", "int8")
    assert store.supervisor.restarts == 0
    store.core.db.execute("UPDATE jobs SET state = 'succeeded' WHERE id = ?", (job_id,))  # an end no job report followed

    async def one_tick() -> None:
        task = asyncio.create_task(store.core.run_maintenance(interval=0.01))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(one_tick())
    assert store.supervisor.restarts == 1


def test_a_removed_file_the_old_worker_held_open_restarts_its_replacement_once_it_is_deleted(store, monkeypatch):
    store.install("z-image-turbo", "int8")
    transformer = "ZImageTurbo_quanto_bf16_int8.safetensors"
    original = Path.unlink
    held = {"open": True}

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == transformer and held["open"]:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    restarts = store.supervisor.restarts
    assert store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}).status_code == 204
    assert store.supervisor.restarts == restarts + 1 and transformer in store.placed()

    held["open"] = False  # the old worker process ended; its replacement found the file and lists z_image
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)
    assert transformer not in store.placed() and store.supervisor.restarts == restarts + 2

    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=[], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)
    assert store.supervisor.restarts == restarts + 2


def test_a_file_left_undeleted_when_cinewright_closed_restarts_the_first_worker_of_the_next_start(store, config, clock, monkeypatch):
    store.install("z-image-turbo", "int8")
    transformer = "ZImageTurbo_quanto_bf16_int8.safetensors"
    original = Path.unlink
    held = {"open": True}

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == transformer and held["open"]:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert store.client.delete(f"{API}/models/z-image-turbo", params={"quant": "int8"}).status_code == 204
    store.core.close()  # before the replacement worker registered: the file stays registered without a reference
    assert transformer in store.placed()

    held["open"] = False  # the old worker process ended with Cinewright
    restarted = Store(config, clock)
    try:
        # The first worker of this start found the file and lists z_image; deleting the file must restart it.
        registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
        restarted.core.register_worker(registration, restarted.core.enrollment_token)
        assert transformer not in restarted.placed() and restarted.rows("SELECT * FROM store_files") == []
        assert restarted.supervisor.restarts == 1

        registration = WorkerRegistration(host_id="local", engine="wan2gp", models=[], health={"state": "ready"})
        restarted.core.register_worker(registration, restarted.core.enrollment_token)
        assert restarted.supervisor.restarts == 1
    finally:
        restarted.core.close()


def test_an_install_whose_wgp_config_cannot_be_written_still_ends_installed_and_restarts_the_worker(store, monkeypatch):
    store.write_config(OTHER_KEYS)

    def refuse(path: Path, data: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("cinewright.core.modelstore._write_json", refuse)

    state = store.install("z-image-turbo", "int8")

    assert state["install_state"] == "installed" and store.placed() == Z_IMAGE_INT8_FILES
    assert store.config() == OTHER_KEYS  # WanGP does not see the store until its config can be written
    assert (store.events[-1].model_id, store.events[-1].quant, store.events[-1].install_state) == ("z-image-turbo", "int8", "installed")
    assert not [event for event in store.events if event.install_state == "failed"]
    assert store.supervisor.restarts == 1


def test_wgp_config_written_by_wan2gp_after_an_install_is_merged_when_the_worker_registers(store):
    store.install("z-image-turbo", "int8")  # WanGP has not started yet: no config to merge into
    assert not store.core.models.config_path.exists() and store.supervisor.restarts == 1

    store.write_config(OTHER_KEYS)  # WanGP's first start wrote its defaults
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)

    assert store.config()["checkpoints_paths"][0] == str(store.root.resolve())
    assert store.config()["text_encoder_quantization"] == "int8"
    assert store.supervisor.restarts == 2  # WanGP reads its config only at import
    store.core.register_worker(registration, store.core.enrollment_token)
    assert store.supervisor.restarts == 2


# ---------------------------------------------------------------------- text encoder variant


def test_switching_the_local_text_encoder_from_int8_to_bf16(store):
    store.write_config(OTHER_KEYS)
    store.install("z-image-turbo", "int8")
    assert store.local_host()["text_encoder_variant"] == "int8" and store.supervisor.restarts == 1

    response = store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"})
    assert response.status_code == 202, response.text
    store.idle()
    assert {QWEN3_INT8, QWEN3_BF16} <= store.placed()  # the previous variant goes once the restarted worker registered
    store.supervisor.since = store.clock()
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)

    placed = store.placed()
    assert QWEN3_BF16 in placed and QWEN3_INT8 not in placed
    assert placed == (Z_IMAGE_INT8_FILES - {QWEN3_INT8}) | {QWEN3_BF16}
    assert store.config() == {
        **OTHER_KEYS,
        "checkpoints_paths": [str(store.root.resolve()), "ckpts", "."],
        "text_encoder_quantization": "bf16",
    }
    assert store.local_host()["text_encoder_variant"] == "bf16" and store.local_host()["text_encoder_switch"] is None
    assert store.supervisor.restarts == 2
    assert {r["path"] for r in store.rows("SELECT path FROM installed_model_files")} == placed
    assert store.quant("z-image-turbo", "int8")["size_bytes"] == store.size("z-image-turbo", "int8", "bf16")


def test_a_switch_deletes_the_previous_text_encoder_only_once_the_restarted_worker_registered(store, monkeypatch):
    store.write_config(OTHER_KEYS)
    store.install("z-image-turbo", "int8")
    previous = PurePosixPath(QWEN3_INT8).name
    original = Path.unlink
    held = {"open": True}  # the idle old worker holds the text encoder until its restart stopped it
    tried: list[bool] = []

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == previous:
            tried.append(held["open"])
            if held["open"]:
                raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    store.idle()
    assert store.supervisor.restarts == 2 and QWEN3_INT8 in store.placed() and tried == []

    held["open"] = False
    store.supervisor.since = store.clock()
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)  # the replacement

    assert QWEN3_INT8 not in store.placed() and tried == [False]
    assert store.supervisor.restarts == 2  # no further restart: the text encoder was never held at a collection


def test_switching_the_text_encoder_is_refused_while_a_job_runs_on_the_host(store):
    store.install("z-image-turbo", "int8")
    store.leased_job("z-image-turbo", "int8", "z_image")

    problem = assert_problem(store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}), 409)

    assert problem.model_extra["code"] == "in_use" and "job is running" in problem.detail
    assert store.local_host()["text_encoder_variant"] == "int8"
    assert QWEN3_INT8 in store.placed() and QWEN3_BF16 not in store.placed()
    assert store.hub.served(store.url("z-image-shared", QWEN3_BF16)) == 0


def test_a_job_leased_during_a_switch_keeps_the_previous_variant_until_the_worker_was_replaced(store):
    store.install("z-image-turbo", "int8")
    store.gate = threading.Event()
    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    worker_id, job_id, lease_id = store.leased_job("z-image-turbo", "int8", "z_image")  # while the bf16 files download

    store.gate.set()
    store.idle()
    assert store.local_host()["text_encoder_variant"] == "bf16"
    assert {QWEN3_INT8, QWEN3_BF16} <= store.placed()  # the running job still uses the int8 text encoder
    assert store.supervisor.restarts == 1  # the restart waits for the job

    finish_job(store.core, worker_id, job_id, lease_id)
    assert store.supervisor.restarts == 2
    store.supervisor.since = store.clock()
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["z_image"], health={"state": "ready"})
    store.core.register_worker(registration, store.core.enrollment_token)  # the replacement
    assert QWEN3_BF16 in store.placed() and QWEN3_INT8 not in store.placed()


def test_hosts_show_a_running_text_encoder_switch_and_why_it_failed(store):
    store.install("z-image-turbo", "int8")
    url = store.url("z-image-shared", QWEN3_BF16)
    store.hub.files[url] = store.contents[url][:-1] + b"!"
    store.gate = threading.Event()

    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    assert store.local_host()["text_encoder_switch"] == {"variant": "bf16", "error": None}
    store.gate.set()
    store.idle()

    host = store.local_host()
    assert host["text_encoder_variant"] == "int8"
    failed = host["text_encoder_switch"]
    assert failed["variant"] == "bf16" and failed["error"]["code"] == "hash_mismatch" and QWEN3_BF16 in failed["error"]["message"]
    assert store.client.post(f"{API}/models/wan-2.2-t2v-14b/install", params={"quant": "int8"}).status_code == 202  # no longer blocked
    store.idle()

    store.hub.files[url] = store.contents[url]
    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    store.idle()
    assert store.local_host()["text_encoder_variant"] == "bf16" and store.local_host()["text_encoder_switch"] is None


def test_after_a_switch_a_removal_keeps_the_text_encoder_a_job_leased_during_the_switch_uses(store):
    store.install("wan-2.2-t2v-14b", "int8")
    store.install("z-image-turbo", "int8")
    store.gate = threading.Event()
    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    store.leased_job("z-image-turbo", "int8", "z_image")  # while the bf16 files download
    store.gate.set()
    store.idle()

    assert store.client.delete(f"{API}/models/wan-2.2-t2v-14b", params={"quant": "int8"}).status_code == 204

    assert {QWEN3_INT8, QWEN3_BF16} <= store.placed()  # the worker running the job still uses the int8 text encoder


def test_closing_the_core_cancels_an_install_registers_no_file_and_leaves_it_failed(store):
    store.gate = threading.Event()
    assert store.client.post(f"{API}/models/z-image-turbo/install", params={"quant": "int8"}).status_code == 202

    closing = threading.Thread(target=store.core.close)
    closing.start()
    wait_for(lambda: store.core._closed, "the core to start closing")
    store.gate.set()
    closing.join(TIMEOUT_S)

    assert not closing.is_alive() and not store.core._background
    assert store.placed() == set()
    reopened = Database.open(store.core.paths.db_file)
    try:
        # Failed, not absent: its card offers Remove, which deletes the downloads it left in models/.downloads/.
        assert [row["state"] for row in reopened.query("SELECT state FROM installed_models")] == ["failed"]
        assert reopened.query("SELECT * FROM store_files") == []
    finally:
        reopened.close()


def test_a_text_encoder_switch_that_does_not_fit_is_refused(store):
    store.install("z-image-turbo", "int8")
    store.free = 10

    problem = assert_problem(store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}), 409)

    assert problem.model_extra["code"] == "not_enough_space" and "more free space" in problem.detail
    assert store.local_host()["text_encoder_variant"] == "int8"


def test_choosing_a_variant_before_any_install_stores_it_and_installs_follow_it(store):
    assert store.local_host()["text_encoder_variant"] is None

    assert store.client.patch(f"{API}/hosts/local", json={"text_encoder_variant": "bf16"}).status_code == 202
    store.idle()
    assert store.local_host()["text_encoder_variant"] == "bf16" and store.hub.requests == []

    store.install("z-image-turbo", "int8")
    assert QWEN3_BF16 in store.placed() and QWEN3_INT8 not in store.placed()


def test_an_empty_host_patch_changes_nothing_and_unknown_hosts_are_not_found(store):
    assert store.client.patch(f"{API}/hosts/local", json={}).status_code == 202
    assert store.local_host()["text_encoder_variant"] is None
    assert_problem(store.client.patch(f"{API}/hosts/elsewhere", json={"text_encoder_variant": "bf16"}), 404)


# ---------------------------------------------------------------------- licence


def test_a_model_whose_licence_excludes_the_country_is_not_offered(client):
    def offered() -> list[str]:
        return [model["id"] for model in client.get(f"{API}/models").json()["items"]]

    assert offered() == ["test-image"]  # test-video's licence excludes the EU; the default country is DE
    problem = assert_problem(client.post(f"{API}/models/test-video/install", params={"quant": "bf16"}), 409)
    assert "excludes DE" in problem.detail

    assert client.patch(f"{API}/settings", json={"country": "US"}).status_code == 200
    assert offered() == ["test-image", "test-video"]
