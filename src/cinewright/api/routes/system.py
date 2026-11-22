"""System, health, licences and settings."""

from __future__ import annotations

from fastapi import APIRouter

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import Health, LicenseNotices, Settings, SettingsPatch, SystemInfo

router = APIRouter(tags=["system"], responses={422: PROBLEM_RESPONSE})


@router.get("/health", response_model=Health, operation_id="getHealth")
def get_health(core: Core) -> Health:
    return core.health()


@router.get("/system", response_model=SystemInfo, operation_id="getSystem")
def get_system(core: Core) -> SystemInfo:
    return core.system_info()


@router.get("/licenses", response_model=LicenseNotices, operation_id="getLicenses")
def get_licenses(core: Core) -> LicenseNotices:
    return core.license_notices()


@router.get("/settings", response_model=Settings, operation_id="getSettings")
def get_settings(core: Core) -> Settings:
    return core.get_settings()


@router.patch("/settings", response_model=Settings, operation_id="updateSettings")
def update_settings(patch: SettingsPatch, core: Core) -> Settings:
    return core.update_settings(patch)
