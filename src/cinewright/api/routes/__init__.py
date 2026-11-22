"""Internal API routers, mounted under `/api/v1` by `cinewright.app`."""

from __future__ import annotations

from fastapi import APIRouter

from cinewright.api.routes import assets, hosts, jobs, registry, system, worker
from cinewright.api.schemas import API_PREFIX


def api_router() -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)
    for module in (system, registry, jobs, assets, hosts, worker):
        router.include_router(module.router)
    return router
