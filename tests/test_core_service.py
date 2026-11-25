"""Core service behaviour that needs no HTTP: settings, system, licences, assets, seeds, data classes, workers."""

from __future__ import annotations

import hashlib

import pytest
from conftest import stage

from cinewright.api.schemas import ClientKind, DataClass, JobRequest, SettingsPatch, WorkerRegistration
from cinewright.core.errors import Infeasible, Invalid, NotFound, NotImplementedFeature, Unauthorized
from cinewright.core.licenses import WANGP_NOTICE

REGISTRATION = WorkerRegistration(host_id="local", engine="wan2gp", engine_version="abc", models=["test_image"], health={"state": "ready"})


def request(**fields) -> JobRequest:
    return JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse", **fields})


def test_settings_default_update_and_validation(core):
    assert core.get_settings().country == "DE"
    updated = core.update_settings(SettingsPatch(country="FR", kwh_price_eur=0.3))
    assert (updated.country, updated.kwh_price_eur, updated.approval_threshold_eur) == ("FR", 0.3, 0.0)
    assert core.get_settings() == updated
    assert core.update_settings(SettingsPatch.model_validate({"kwh_price_eur": None})).kwh_price_eur is None
    with pytest.raises(Invalid):
        core.update_settings(SettingsPatch.model_validate({"country": None}))


def test_system_info_discloses_wangp_and_reports_engine_state(core):
    info = core.system_info()
    engine = info.engines[0]
    assert (info.name, info.bind, engine.id, engine.state, engine.version) == (
        "Cinewright",
        "127.0.0.1:8737",
        "wan2gp",
        "not_installed",
        None,
    )
    assert engine.notice == WANGP_NOTICE
    # The state comes from engine provisioning (tests/test_core_provisioning.py), not from a registered worker.
    core.register_worker(REGISTRATION, core.enrollment_token)
    engine = core.system_info().engines[0]
    assert (engine.state, engine.version) == ("not_installed", None)


def test_license_notices_name_cinewright_and_wangp(core):
    items = {item.component: item for item in core.license_notices().items}
    assert items["Cinewright"].license == "MIT"
    wangp = items["WanGP (Wan2GP)"]
    assert wangp.license == "WanGP Community License 2.0"
    assert wangp.url == "https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt"
    assert wangp.notice.startswith("Cinewright uses WanGP (Wan2GP) by deepbeepmeep")
    assert "fastapi" in {name.lower() for name in items}


def test_health_and_local_host(core):
    health = core.health()
    assert health.status == "ok"
    assert health.checks["database"] == "ok"
    hosts = core.list_hosts().items
    assert [(h.id, h.kind, h.online) for h in hosts] == [("local", "local", True)]
    assert hosts[0].allowed_data_classes == [DataClass.PUBLIC, DataClass.PRIVATE, DataClass.BIOMETRIC]


def test_identical_bytes_are_one_asset_with_the_stricter_data_class(core):
    first = core.add_asset(stage(core, b"same bytes"), "image/png", DataClass.PUBLIC)
    second = core.add_asset(stage(core, b"same bytes"), "image/png", DataClass.BIOMETRIC)
    assert first.sha256 == second.sha256 == hashlib.sha256(b"same bytes").hexdigest()
    assert second.data_class == DataClass.BIOMETRIC
    assert len(core.list_assets(None, 50).items) == 1
    path = core.asset_path(first.sha256)
    assert path.read_bytes() == b"same bytes"
    assert path.parent.name == first.sha256[:2]
    assert not list(core.paths.assets_dir.rglob(".upload-*"))


def test_asset_pagination(core):
    for index in range(5):
        core.add_asset(stage(core, f"asset {index}".encode()), "image/png", DataClass.PRIVATE)
    first = core.list_assets(None, 2)
    second = core.list_assets(first.next_cursor, 2)
    third = core.list_assets(second.next_cursor, 2)
    shas = [a.sha256 for page in (first, second, third) for a in page.items]
    assert len(set(shas)) == 5 and third.next_cursor is None
    with pytest.raises(Invalid):
        core.list_assets("not-a-cursor", 2)


def test_job_data_class_is_the_strictest_of_request_and_inputs(core):
    core.register_worker(REGISTRATION, core.enrollment_token)
    face = core.add_asset(stage(core, b"face"), "image/png", DataClass.BIOMETRIC)
    job = core.submit_job(request(capability="i2i", inputs=[face.sha256], data_class="public"), ClientKind.UI, None)
    row = core.db.query_one("SELECT data_class, seed FROM jobs WHERE id = ?", (job.id,))
    assert row["data_class"] == "biometric"
    assert 0 <= row["seed"] < 2**31
    assert job.request.params.seed is None  # the request stays as submitted; the resolved seed is recorded


def test_submit_rejects_unknown_inputs_recipes_and_infeasible_requests(core):
    with pytest.raises(Invalid):
        core.submit_job(request(inputs=["0" * 64]), ClientKind.UI, None)
    with pytest.raises(NotImplementedFeature) as recipe:
        core.submit_job(request(recipe={"id": "r", "version": 1}), ClientKind.UI, None)
    assert recipe.value.extensions == {"feature": "Recipes", "phase": "phase 2"}
    with pytest.raises(Infeasible) as infeasible:
        core.submit_job(request(), ClientKind.UI, None)
    assert infeasible.value.reasons


def test_mcp_jobs_record_their_agent(core):
    core.register_worker(REGISTRATION, core.enrollment_token)
    job = core.submit_job(request(), ClientKind.MCP, "claude-code")
    assert (job.client, job.agent) == (ClientKind.MCP, "claude-code")
    assert core.submit_job(request(), ClientKind.MCP, "claude-code").agent == "claude-code"


def test_worker_registration_needs_the_enrollment_token_and_stores_only_a_hash(core):
    with pytest.raises(Unauthorized):
        core.register_worker(REGISTRATION, "wrong")
    with pytest.raises(Invalid):
        core.register_worker(REGISTRATION.model_copy(update={"host_id": "elsewhere"}), core.enrollment_token)
    registered = core.register_worker(REGISTRATION, core.enrollment_token)
    stored = core.db.query_one("SELECT token_hash FROM workers WHERE id = ?", (registered.worker_id,))["token_hash"]
    assert stored == hashlib.sha256(registered.token.encode()).hexdigest() != registered.token
    assert core.authenticate_worker(registered.token) == registered.worker_id
    with pytest.raises(Unauthorized):
        core.authenticate_worker("nope")


def test_each_core_start_has_its_own_enrollment_token(core, config, registry_dir, clock):
    from cinewright.core.core import Core
    from cinewright.core.registry import Registry

    other = Core(config, registry=Registry.load(registry_dir), clock=clock, detect=lambda *_: core.hardware())
    try:
        assert other.enrollment_token != core.enrollment_token
    finally:
        other.close()


def test_later_phase_operations_raise_not_implemented(core):
    for call in (
        lambda: core.install_model("test-image", None),
        lambda: core.remove_model("test-image", "bf16"),
        core.start_calibration,
        core.add_host,
        lambda: core.decide_approval("a1", True),
    ):
        with pytest.raises(NotImplementedFeature):
            call()
    with pytest.raises(NotFound):
        core.install_model("unknown", None)


def test_hardware_is_detected_once_and_redetected_on_request(core):
    calls = []
    original = core._detect

    def counting(*args):
        calls.append(args)
        return original(*args)

    core._detect = counting
    core.hardware()
    core.hardware()
    assert len(calls) == 1
    core.detect_hardware()
    assert len(calls) == 2
