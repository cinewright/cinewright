"""Router: check order policy -> feasibility -> cost -> runtime, as pure functions."""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import TEST_MODELS

from cinewright.api.schemas import CalibrationResult, CalibrationStatus, DataClass, HostKind, JobRequest, Settings
from cinewright.core.router import HostInfo, WorkerInfo, effective_params, geo_excluded, plan

ALL = frozenset(DataClass)
LOCAL = HostInfo(id="local", name="Local", kind=HostKind.LOCAL, allowed_data_classes=ALL)
REMOTE = HostInfo(id="remote-1", name="Remote", kind=HostKind.REMOTE_OWN, allowed_data_classes=ALL)
REMOTE_PUBLIC = HostInfo(id="remote-2", name="Remote public", kind=HostKind.REMOTE_OWN, allowed_data_classes=frozenset({DataClass.PUBLIC}))


def worker(host: str = "local", models: tuple[str, ...] = ("test_image", "test_image_q8", "test_video")) -> WorkerInfo:
    return WorkerInfo(id=f"w-{host}", host_id=host, engine="wan2gp", models=frozenset(models))


def request(**fields) -> JobRequest:
    return JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse at dusk", **fields})


def calibration(status: CalibrationStatus, quant: str = "bf16", seconds_per_step: float | None = None, detail: str | None = None):
    return CalibrationResult(
        host_id="local",
        model_id="test-image",
        quant=quant,
        resolution="512x512",
        status=status,
        seconds_per_step=seconds_per_step,
        detail=detail,
        engine="wan2gp",
        measured_at=datetime(2026, 11, 22, tzinfo=UTC),
    )


def run(req: JobRequest, data_class: DataClass | None = None, hosts=(LOCAL,), workers=(), calib=(), settings: Settings | None = None):
    return plan(req, data_class or req.data_class, TEST_MODELS, list(hosts), list(workers), list(calib), settings or Settings())


def test_routes_to_local_with_reasons_in_check_order():
    result = run(request(), workers=[worker()])
    assert result.feasible
    assert (result.route.model_id, result.route.quant, result.route.host_id, result.route.engine) == (
        "test-image",
        "bf16",
        "local",
        "wan2gp",
    )
    assert result.cost_eur == 0.0
    assert result.eta_seconds is None
    assert not result.requires_approval
    assert [reason.split(":")[0] for reason in result.reasons] == ["policy", "policy", "feasibility", "feasibility", "cost", "runtime"]
    assert result.reasons == result.route.reasons


def test_biometric_data_stays_local_even_when_a_remote_host_allows_it():
    only_remote = run(request(data_class="biometric"), hosts=[REMOTE], workers=[worker("remote-1")])
    assert not only_remote.feasible
    assert any("biometric data stays on local hosts" in reason for reason in only_remote.reasons)

    both = run(request(data_class="biometric"), hosts=[REMOTE, LOCAL], workers=[worker("remote-1"), worker("local")])
    assert both.route.host_id == "local"


def test_host_policy_excludes_data_classes_it_does_not_accept():
    result = run(request(data_class="private"), hosts=[REMOTE_PUBLIC], workers=[worker("remote-2")])
    assert not result.feasible
    assert result.reasons[0] == "policy: host remote-2 does not accept private data"
    assert result.reasons[-1] == "policy: no host accepts private data"


def test_eu_geo_exclusion_applies_to_eu_countries_only():
    video = request(capability="t2v")
    excluded = run(video, workers=[worker()], settings=Settings(country="DE"))
    assert not excluded.feasible
    assert excluded.reasons == ["policy: the licence of test-video (Test Community License) excludes DE"]
    assert run(video, workers=[worker()], settings=Settings(country="CH")).feasible
    assert geo_excluded(["US"], "US") and not geo_excluded(["US"], "DE") and geo_excluded(["EU"], "FR")


def test_policy_is_checked_before_feasibility():
    # Both the licence (policy) and the missing worker (feasibility) fail; policy is reported.
    result = run(request(capability="t2v"), workers=[], settings=Settings(country="AT"))
    assert result.reasons == ["policy: the licence of test-video (Test Community License) excludes AT"]


def test_feasibility_reasons():
    assert run(request(), workers=[]).reasons == ["feasibility: no online worker on local serves wan2gp test_image"] + [
        "feasibility: no online worker on local serves wan2gp test_image_q8"
    ]
    assert run(request(model="nope"), workers=[worker()]).reasons == ["feasibility: model nope is not in the registry"]
    assert run(request(model="test-video", capability="t2i"), workers=[worker()], settings=Settings(country="US")).reasons == [
        "feasibility: test-video does not offer t2i"
    ]
    assert run(request(capability="audio"), workers=[worker()]).reasons == ["feasibility: no registry model offers audio"]
    assert run(request(quant="fp8"), workers=[worker()]).reasons == ["feasibility: test-image has no quantization fp8"]
    limited = run(request(params={"resolution": "640x640"}), workers=[worker()])
    assert limited.reasons == ["feasibility: test-image does not support 640x640 (supported: 512x512, 1024x1024)"]


def test_worker_must_serve_the_engine_model_of_the_pinned_quant():
    result = run(request(quant="Q8_0"), workers=[worker(models=("test_image",))])
    assert not result.feasible
    assert result.reasons == ["feasibility: no online worker on local serves wan2gp test_image_q8"]
    assert run(request(quant="Q8_0"), workers=[worker()]).route.quant == "Q8_0"


def test_calibration_wont_run_is_infeasible_and_runtime_comes_from_calibration():
    wont = run(request(quant="bf16"), workers=[worker()], calib=[calibration(CalibrationStatus.WONT_RUN, detail="needs 52 GB VRAM")])
    assert wont.reasons == ["feasibility: test-image bf16 does not run on local (needs 52 GB VRAM)"]

    timed = run(request(params={"steps": 6}), workers=[worker()], calib=[calibration(CalibrationStatus.RUNS, seconds_per_step=2.5)])
    assert timed.eta_seconds == 15.0
    assert timed.reasons[-1] == "runtime: 15 s"


def test_ranking_prefers_measured_runs_over_uncalibrated():
    result = run(request(), workers=[worker()], calib=[calibration(CalibrationStatus.RUNS, quant="Q8_0", seconds_per_step=1.0)])
    assert result.route.quant == "Q8_0"
    assert result.eta_seconds == 4.0  # registry default of 4 steps


def test_remote_hosts_need_approval_and_have_unknown_cost():
    result = run(request(data_class="public"), hosts=[REMOTE], workers=[worker("remote-1")])
    assert result.route.host_id == "remote-1"
    assert result.cost_eur is None
    assert result.requires_approval
    over_budget = run(request(data_class="public", budget_eur=1.0), hosts=[REMOTE], workers=[worker("remote-1")])
    assert over_budget.reasons == ["cost: remote-1 has no price within the budget of €1.00"]


def test_effective_params_fill_registry_defaults():
    params = effective_params(request(params={"steps": 2, "extra": {"tile": 1}}), TEST_MODELS[0], seed=7)
    assert (params.steps, params.resolution, params.seed, params.extra) == (2, "512x512", 7, {"tile": 1})
