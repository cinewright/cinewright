"""Map core errors and validation failures to RFC 9457 problem responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cinewright.api.schemas import PROBLEM_INVALID, PROBLEM_NOT_FOUND, Problem
from cinewright.core.errors import CoreError

PROBLEM_MEDIA_TYPE = "application/problem+json"

# Shared OpenAPI response declarations for routes.
PROBLEM_RESPONSE = {"model": Problem, "content": {PROBLEM_MEDIA_TYPE: {}}}


def problem_response(status: int, title: str, detail: str | None, problem_type: str, instance: str, **ext: object) -> JSONResponse:
    body = Problem(type=problem_type, title=title, status=status, detail=detail, instance=instance, **ext)
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, media_type=PROBLEM_MEDIA_TYPE)


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(CoreError)
    async def _core_error(request: Request, exc: CoreError) -> JSONResponse:
        return problem_response(exc.status, exc.title, exc.detail, exc.problem_type, request.url.path, **exc.extensions)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        return problem_response(
            422, "Invalid request", "The request does not match the API contract.", PROBLEM_INVALID, request.url.path, errors=errors
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        problem_type = PROBLEM_NOT_FOUND if exc.status_code == 404 else "about:blank"
        title = "Not found" if exc.status_code == 404 else str(exc.detail)
        return problem_response(exc.status_code, title, None, problem_type, request.url.path)
