"""System, health, licences, settings and engine provisioning."""

from __future__ import annotations

from fastapi import APIRouter

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import EngineInfo, Health, LicenseNotices, Settings, SettingsPatch, SystemInfo

router = APIRouter(tags=["system"], responses={422: PROBLEM_RESPONSE})


@router.get("/health", response_model=Health, operation_id="getHealth")
def get_health(core: Core) -> Health:
    return core.health()


@router.get("/system", response_model=SystemInfo, operation_id="getSystem")
def get_system(core: Core) -> SystemInfo:
    return core.system_info()


@router.post(
    "/engines/{engine_id}/provision",
    response_model=EngineInfo,
    status_code=202,
    operation_id="provisionEngine",
    responses={404: PROBLEM_RESPONSE},
)
def provision_engine(engine_id: str, core: Core) -> EngineInfo:
    """Start provisioning the engine (the Setup view's retry) unless it is ready or already provisioning.

    Returns the engine's state; progress and the result follow as `engine` events on `GET /events`.
    """
    return core.provision_engine(engine_id)


@router.get("/licenses", response_model=LicenseNotices, operation_id="getLicenses")
def get_licenses(core: Core) -> LicenseNotices:
    return core.license_notices()


@router.get("/settings", response_model=Settings, operation_id="getSettings")
def get_settings(core: Core) -> Settings:
    return core.get_settings()


@router.patch("/settings", response_model=Settings, operation_id="updateSettings")
def update_settings(patch: SettingsPatch, core: Core) -> Settings:
    return core.update_settings(patch)
