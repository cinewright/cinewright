"""Hosts, approvals and the phase 2 libraries (recipes, cast, presets)."""

from __future__ import annotations

from fastapi import APIRouter, Response

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import ApprovalList, CastList, HostList, HostPatch, PresetList, RecipeList

router = APIRouter(responses={404: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})


@router.get("/hosts", response_model=HostList, tags=["hosts"], operation_id="listHosts")
def list_hosts(core: Core) -> HostList:
    return core.list_hosts()


@router.patch("/hosts/{host_id}", status_code=202, tags=["hosts"], operation_id="updateHost", responses={409: PROBLEM_RESPONSE})
def update_host(host_id: str, patch: HostPatch, core: Core) -> Response:
    """Change a host's settings. A new `text_encoder_variant` is switched in the background.

    Checks first and refuses with 409 while a job runs on the host or a model is being installed, and when the
    variant's files do not fit on the drive of the model store. `GET /hosts` shows the new variant once WanGP's
    configuration names it.
    """
    core.update_host(host_id, patch)
    return Response(status_code=202)


@router.post("/hosts", status_code=201, tags=["hosts"], operation_id="addHost")
def add_host(core: Core) -> Response:
    core.add_host()
    return Response(status_code=201)


@router.get("/approvals", response_model=ApprovalList, tags=["approvals"], operation_id="listApprovals")
def list_approvals(core: Core) -> ApprovalList:
    return core.list_approvals()


@router.post("/approvals/{approval_id}/approve", status_code=204, tags=["approvals"], operation_id="approve")
def approve(approval_id: str, core: Core) -> Response:
    core.decide_approval(approval_id, True)
    return Response(status_code=204)


@router.post("/approvals/{approval_id}/decline", status_code=204, tags=["approvals"], operation_id="decline")
def decline(approval_id: str, core: Core) -> Response:
    core.decide_approval(approval_id, False)
    return Response(status_code=204)


@router.get("/recipes", response_model=RecipeList, tags=["library"], operation_id="listRecipes")
def list_recipes(core: Core) -> RecipeList:
    return core.list_recipes()


@router.get("/cast", response_model=CastList, tags=["library"], operation_id="listCast")
def list_cast(core: Core) -> CastList:
    return core.list_cast()


@router.get("/presets", response_model=PresetList, tags=["library"], operation_id="listPresets")
def list_presets(core: Core) -> PresetList:
    return core.list_presets()
