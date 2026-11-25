"""Router: check order policy -> feasibility -> cost -> runtime, as pure functions."""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import TEST_MODELS

from cinewright.api.schemas import CalibrationResult, CalibrationStatus, DataClass, HostKind, JobRequest, Limits, Settings
from cinewright.core.registry import Registry
from cinewright.core.router import HostInfo, WorkerInfo, effective_params, geo_excluded, plan

ALL = frozenset(DataClass)
EVERY_QUANT = frozenset((m.id, q.id) for m in TEST_MODELS for q in m.quantizations)
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


def run(
    req: JobRequest,
    data_class: DataClass | None = None,
    hosts=(LOCAL,),
    workers=(),
    calib=(),
    settings: Settings | None = None,
    installed=EVERY_QUANT,
    models=TEST_MODELS,
):
    """Plan with every test quantization installed on the local host, unless `installed` says otherwise."""
    args = (req, data_class or req.data_class, models, list(hosts), list(workers), list(calib), settings or Settings())
    return plan(*args, installed={"local": installed})


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
    assert [reason.split(":")[0] for reason in result.reasons] == [
        "policy",
        "policy",
        "feasibility",
        "feasibility",
        "feasibility",
        "cost",
        "runtime",
    ]
    assert result.reasons[2] == "feasibility: test-image bf16 is installed on local"
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


GRID_VIDEO = TEST_MODELS[1].model_copy(
    update={
        "capabilities": ["t2v", "i2v", "t2i", "i2i"],
        "license": TEST_MODELS[0].license,  # no geo exclusion, so Settings() routes it
        "limits": Limits(resolutions=["832x480"], max_frames=81, min_frames=17, frames_step=8, fps=[16]),
        "defaults": {"steps": 10, "frames": 41, "fps": 16, "resolution": "832x480"},
    }
)


def run_grid(capability: str, frames: int | None):
    params = {} if frames is None else {"frames": frames}
    return run(request(capability=capability, model="test-video", params=params), workers=[worker()], models=[GRID_VIDEO])


def test_video_frame_counts_on_the_grid_route():
    for frames in (17, 25, 41, 81, None):  # None: the registry default (41) fills it
        assert run_grid("t2v", frames).feasible, frames
    assert run_grid("i2v", 73).feasible


def test_video_frame_counts_off_the_grid_are_refused_with_a_reason():
    assert run_grid("t2v", 9).reasons == ["feasibility: test-video renders at least 17 frames, 9 requested"]
    assert run_grid("i2v", 30).reasons == ["feasibility: test-video renders 17 + a multiple of 8 frames, 30 requested (nearest: 25 or 33)"]
    # When the upper neighbour exceeds max_frames, only the lower one is named.
    edge = GRID_VIDEO.model_copy(update={"limits": GRID_VIDEO.limits.model_copy(update={"max_frames": 75})})
    off = run(request(capability="t2v", model="test-video", params={"frames": 74}), workers=[worker()], models=[edge])
    assert off.reasons == ["feasibility: test-video renders 17 + a multiple of 8 frames, 74 requested (nearest: 73)"]
    # max_frames is checked first, as before.
    assert run_grid("t2v", 89).reasons == ["feasibility: test-video renders at most 81 frames, 89 requested"]


def test_still_images_are_not_checked_against_the_frame_grid():
    # Wan2GP's image mode ignores frames (it renders the shortest clip and keeps its first frame).
    for capability in ("t2i", "i2i"):
        for frames in (1, 30, None):
            assert run_grid(capability, frames).feasible, (capability, frames)


def test_still_images_skip_frame_fps_and_duration_limits_but_not_the_resolution():
    # Image mode ignores frames, fps and duration, so limits on them cannot refuse a still image; a clip keeps them.
    timed = GRID_VIDEO.model_copy(update={"limits": GRID_VIDEO.limits.model_copy(update={"max_duration_s": 5.0})})

    def limited(capability: str, params: dict):
        return run(request(capability=capability, model="test-video", params=params), workers=[worker()], models=[timed])

    ignored = {"frames": 2000, "fps": 30, "duration_s": 99.0}
    for capability in ("t2i", "i2i"):
        assert limited(capability, ignored).feasible, (capability, limited(capability, ignored).reasons)
        for key, value in ignored.items():
            assert limited(capability, {key: value}).feasible, (capability, key)
        assert limited(capability, {"resolution": "640x640"}).reasons == [
            "feasibility: test-video does not support 640x640 (supported: 832x480)"
        ]
    assert limited("t2v", {"frames": 2000}).reasons == ["feasibility: test-video renders at most 81 frames, 2000 requested"]
    assert limited("i2v", {"fps": 30}).reasons == ["feasibility: test-video does not support 30 fps"]
    assert limited("i2v", {"duration_s": 99.0}).reasons == ["feasibility: test-video renders at most 5 s, 99 s requested"]


def test_ltx_frame_grid_from_the_registry_applies_to_clips_only():
    models = Registry.load().all()
    ltx = next(m for m in models if m.id == "ltx-2.3-distilled-22b")
    installed = frozenset(("ltx-2.3-distilled-22b", q.id) for q in ltx.quantizations)
    served = tuple(b.model_type for b in ltx.engines)

    def ltx_run(capability: str, frames: int):
        req = request(capability=capability, model="ltx-2.3-distilled-22b", params={"frames": frames})
        return run(req, workers=[worker(models=served)], installed=installed, models=models)

    assert ltx_run("t2v", 121).feasible and ltx_run("i2v", 17).feasible
    assert ltx_run("t2v", 1).reasons == ["feasibility: ltx-2.3-distilled-22b renders at least 17 frames, 1 requested"]
    assert ltx_run("i2v", 120).reasons == [
        "feasibility: ltx-2.3-distilled-22b renders 17 + a multiple of 8 frames, 120 requested (nearest: 113 or 121)"
    ]
    for capability in ("t2i", "i2i"):
        assert ltx_run(capability, 1).feasible and ltx_run(capability, 120).feasible
        # Beyond max_frames (737), with an fps and a duration LTX-2.3 does not offer: image mode ignores all three.
        still = request(capability=capability, model="ltx-2.3-distilled-22b", params={"frames": 2000, "fps": 30, "duration_s": 99.0})
        assert run(still, workers=[worker(models=served)], installed=installed, models=models).feasible
    assert ltx_run("t2v", 745).reasons == ["feasibility: ltx-2.3-distilled-22b renders at most 737 frames, 745 requested"]


def test_models_without_frames_are_unaffected_by_the_frame_grid():
    assert TEST_MODELS[0].limits.min_frames is None
    assert run(request(params={"frames": 3}), workers=[worker()]).feasible
    # A video model without a grid keeps only its max_frames check.
    assert run(request(capability="t2v", params={"frames": 30}), workers=[worker()], settings=Settings(country="US")).feasible


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


def test_only_installed_quantizations_route_and_the_rest_name_model_and_quantization():
    only_q8 = run(request(), workers=[worker()], installed={("test-image", "Q8_0")})
    assert only_q8.route.quant == "Q8_0"  # although bf16 comes first in the registry

    nothing = run(request(), workers=[worker()], installed=set())
    assert not nothing.feasible
    assert nothing.reasons == [
        "feasibility: test-image bf16 is not installed on local",
        "feasibility: test-image Q8_0 is not installed on local",
    ]

    # A failed or unfinished install is not installed: the core passes only complete installs.
    pinned = run(request(quant="bf16"), workers=[worker()], installed={("test-image", "Q8_0")})
    assert pinned.reasons == ["feasibility: test-image bf16 is not installed on local"]


def test_hosts_without_a_known_model_store_are_not_checked_for_installs():
    # The core knows only the local model store; a remote host it does not name keeps routing as before (#17, #18).
    result = run(request(data_class="public"), hosts=[LOCAL, REMOTE], workers=[worker("remote-1")], installed=set())
    assert result.route.host_id == "remote-1"
    assert not any("installed" in reason for reason in result.reasons)


def test_preference_ranks_after_cost_and_before_calibration_and_registry_order():
    preferred = TEST_MODELS[1].model_copy(update={"capabilities": ["t2i"], "preference": 1, "license": TEST_MODELS[0].license})
    models = [TEST_MODELS[0], preferred]
    result = run(request(), workers=[worker()], models=models)
    assert result.route.model_id == "test-video"  # registry order alone would pick test-image

    measured = [calibration(CalibrationStatus.RUNS, seconds_per_step=1.0)]  # test-image bf16 runs, test-video is not measured
    assert run(request(), workers=[worker()], models=models, calib=measured).route.model_id == "test-video"

    remote_preferred = run(
        request(data_class="public", model=None),
        hosts=[LOCAL, REMOTE],
        workers=[worker("local", models=("test_image",)), worker("remote-1", models=("test_video",))],
        models=models,
    )
    assert (remote_preferred.route.model_id, remote_preferred.route.host_id) == ("test-image", "local")  # cost first


def test_t2i_with_every_mvp_model_installed_prefers_z_image_turbo():
    registry = Registry.load()
    models = registry.all()
    installed = frozenset((m.id, q.id) for m in models for q in m.quantizations)
    served = frozenset(b.model_type for m in models for b in m.engines)
    result = run(request(), workers=[worker(models=tuple(served))], installed=installed, models=models)
    assert result.route.model_id == "z-image-turbo"
    candidates = {m.id for m in models if "t2i" in m.capabilities}
    assert candidates == {"ltx-2.3-distilled-22b", "wan-2.2-t2v-14b", "z-image-turbo"}
