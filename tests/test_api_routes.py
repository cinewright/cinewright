"""Every route answers with its declared model or an RFC 9457 problem, with a status it declares."""

from __future__ import annotations

from typing import Any

from conftest import API, PNG, assert_problem, register_worker
from fastapi.routing import APIRoute
from pydantic import BaseModel

from cinewright.api import schemas

NOT_IMPLEMENTED = "urn:cinewright:problem:not-implemented"


class RouteChecker:
    def __init__(self, client) -> None:
        self.client = client
        self.paths = client.app.openapi()["paths"]
        self.seen: set[tuple[str, str]] = set()

    def __call__(self, method: str, template: str, expected: int, model: type[BaseModel] | None = None, path: str | None = None, **kw: Any):
        response = self.client.request(method, API + (path or template), **kw)
        assert response.status_code == expected, f"{method} {template}: {response.status_code} {response.text}"
        declared = self.paths[API + template][method.lower()]["responses"]
        assert str(expected) in declared, f"{method} {template} answered {expected}, which it does not declare"
        if expected >= 400:
            assert_problem(response, expected)
        elif model is not None:
            model.model_validate(response.json())
        self.seen.add((method, API + template))
        return response


def test_every_route_returns_its_declared_model_or_a_problem(client):
    call = RouteChecker(client)
    call("GET", "/health", 200, schemas.Health)
    call("GET", "/system", 200, schemas.SystemInfo)
    call("GET", "/licenses", 200, schemas.LicenseNotices)
    call("GET", "/settings", 200, schemas.Settings)
    call("PATCH", "/settings", 200, schemas.Settings, json={"country": "AT"})
    call("PATCH", "/settings", 422, json={"country": "Austria"})
    call("GET", "/hardware", 200, schemas.HardwareProfile)
    call("POST", "/hardware/detect", 200, schemas.HardwareProfile)
    call("GET", "/capabilities", 200, schemas.CapabilityList)
    call("GET", "/models", 200, schemas.ModelList)
    call("GET", "/models/{model_id}", 200, schemas.ModelDetail, path="/models/test-image")
    call("GET", "/models/{model_id}", 404, path="/models/unknown")
    call("POST", "/models/{model_id}/install", 501, path="/models/test-image/install")
    call("DELETE", "/models/{model_id}", 501, path="/models/test-image")
    call("GET", "/calibration", 200, schemas.CalibrationReport)
    call("POST", "/calibration/runs", 501)
    call("GET", "/hosts", 200, schemas.HostList)
    call("POST", "/hosts", 501)
    call("GET", "/approvals", 200, schemas.ApprovalList)
    call("POST", "/approvals/{approval_id}/approve", 501, path="/approvals/a1/approve")
    call("POST", "/approvals/{approval_id}/decline", 501, path="/approvals/a1/decline")
    call("GET", "/recipes", 200, schemas.RecipeList)
    call("GET", "/cast", 200, schemas.CastList)
    call("GET", "/presets", 200, schemas.PresetList)

    job_body = {"capability": "t2i", "prompt": "a lighthouse"}
    call("POST", "/estimate", 200, schemas.Estimate, json=job_body)
    call("POST", "/jobs", 409, json=job_body)  # no worker yet: infeasible
    call("POST", "/jobs", 501, json={**job_body, "recipe": {"id": "r", "version": 1}})
    call("POST", "/jobs", 422, json={"capability": "t2i"})

    upload = call("POST", "/assets", 201, schemas.Asset, content=PNG, headers={"Content-Type": "image/png"})
    sha = upload.json()["sha256"]
    call("POST", "/assets", 422, content=b"x", headers={"Content-Type": "application/zip"})
    call("GET", "/assets", 200, schemas.AssetList)
    call("GET", "/assets/{sha256}", 200, schemas.Asset, path=f"/assets/{sha}")
    call("GET", "/assets/{sha256}", 404, path=f"/assets/{'0' * 64}")
    call("GET", "/assets/{sha256}/content", 200, path=f"/assets/{sha}/content")
    call("GET", "/assets/{sha256}/provenance", 200, schemas.Provenance, path=f"/assets/{sha}/provenance")

    auth = register_worker(client)
    call(
        "POST",
        "/worker/register",
        401,
        json={"host_id": "local", "engine": "wan2gp", "models": [], "health": {"state": "ready"}},
        headers={"Authorization": "Bearer wrong"},
    )
    idle = call(
        "POST",
        "/worker/register",
        201,
        schemas.WorkerRegistered,
        json={"host_id": "local", "engine": "wan2gp", "models": ["none"], "health": {"state": "ready"}},
        headers={"Authorization": f"Bearer {client.app.state.enrollment_token}"},
    ).json()
    idle_auth = {"Authorization": f"Bearer {idle['token']}"}
    call("POST", "/worker/heartbeat", 200, schemas.HeartbeatAck, json={"health": {"state": "ready"}}, headers=auth)
    job = call("POST", "/jobs", 202, schemas.Job, json={**job_body, "capability": "i2i", "inputs": [sha]}).json()
    call("GET", "/jobs", 200, schemas.JobList)
    call("GET", "/jobs/{job_id}", 200, schemas.Job, path=f"/jobs/{job['id']}")
    call("GET", "/jobs/{job_id}", 404, path="/jobs/unknown")
    lease = call("POST", "/worker/lease", 200, schemas.Lease, json={"wait_seconds": 0}, headers=auth).json()["job"]
    # Another worker: a new lease for `auth` would release the job it still holds.
    call("POST", "/worker/lease", 204, json={"wait_seconds": 0}, headers=idle_auth)
    call("GET", "/worker/inputs/{sha256}", 200, path=f"/worker/inputs/{sha}", headers=auth)
    events = {"lease_id": lease["lease_id"], "events": [{"kind": "log", "data": {"message": "loading"}}]}
    call(
        "POST",
        "/worker/jobs/{job_id}/events",
        200,
        schemas.WorkerEventsAck,
        path=f"/worker/jobs/{job['id']}/events",
        json=events,
        headers=auth,
    )
    output = call(
        "PUT",
        "/worker/jobs/{job_id}/outputs/{name}",
        201,
        schemas.OutputUploaded,
        path=f"/worker/jobs/{job['id']}/outputs/out.png",
        content=b"output",
        headers={**auth, "Content-Type": "image/png"},
    ).json()
    completion = {"lease_id": lease["lease_id"], "outputs": [{"sha256": output["sha256"], "media_type": "image/png", "name": "out.png"}]}
    call("POST", "/worker/jobs/{job_id}/complete", 204, path=f"/worker/jobs/{job['id']}/complete", json=completion, headers=auth)
    call("POST", "/jobs/{job_id}/cancel", 409, path=f"/jobs/{job['id']}/cancel")

    second = call("POST", "/jobs", 202, schemas.Job, json=job_body).json()
    call("POST", "/jobs/{job_id}/cancel", 200, schemas.Job, path=f"/jobs/{second['id']}/cancel")
    third = call("POST", "/jobs", 202, schemas.Job, json=job_body).json()
    lease = call("POST", "/worker/lease", 200, schemas.Lease, json={}, headers=auth).json()["job"]
    failure = {"lease_id": lease["lease_id"], "error": {"code": "engine_failed", "message": "The engine stopped at step 1"}}
    call("POST", "/worker/jobs/{job_id}/fail", 204, path=f"/worker/jobs/{third['id']}/fail", json=failure, headers=auth)

    with client.stream("GET", f"{API}/events", params={"job_id": job["id"]}) as stream:
        assert stream.status_code == 200 and stream.headers["content-type"].startswith("text/event-stream")
        assert "event: state" in "".join(stream.iter_text())
    call.seen.add(("GET", f"{API}/events"))

    routes = {(method, route.path) for route in client.app.routes if isinstance(route, APIRoute) for method in route.methods}
    assert routes - call.seen == set(), "routes without a check"


def test_not_implemented_routes_name_feature_and_phase(client):
    for method, path, feature, phase in [
        ("POST", "/models/test-image/install", "Model installation", "phase 1"),
        ("DELETE", "/models/test-image", "Model removal", "phase 1"),
        ("POST", "/calibration/runs", "Calibration runs", "phase 1"),
        ("POST", "/hosts", "Adding hosts", "phase 3"),
        ("POST", "/approvals/x/approve", "Approval decisions", "phase 3"),
        ("POST", "/approvals/x/decline", "Approval decisions", "phase 3"),
    ]:
        problem = assert_problem(client.request(method, API + path), 501, NOT_IMPLEMENTED)
        assert (problem.model_extra["feature"], problem.model_extra["phase"]) == (feature, phase)
    recipe = client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "x", "recipe": {"id": "r", "version": 1}})
    assert assert_problem(recipe, 501, NOT_IMPLEMENTED).model_extra["phase"] == "phase 2"


def test_infeasible_submit_lists_the_router_reasons(client):
    problem = assert_problem(
        client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "x"}), 409, "urn:cinewright:problem:infeasible"
    )
    assert problem.model_extra["reasons"] == [
        "feasibility: no online worker on local serves wan2gp test_image",
        "feasibility: no online worker on local serves wan2gp test_image_q8",
    ]
    estimate = client.post(f"{API}/estimate", json={"capability": "t2i", "prompt": "x"}).json()
    assert estimate["feasible"] is False and estimate["reasons"] == problem.model_extra["reasons"]
