"""Domain errors raised by the core and mapped to RFC 9457 problems by the API layer."""

from __future__ import annotations


class CoreError(Exception):
    """Base class. `status` and `problem_type` decide the HTTP answer."""

    status = 500
    problem_type = "about:blank"
    title = "Internal error"

    def __init__(self, detail: str, **extensions: object) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extensions = extensions


class NotFound(CoreError):
    status = 404
    problem_type = "urn:cinewright:problem:not-found"
    title = "Not found"


class Invalid(CoreError):
    status = 422
    problem_type = "urn:cinewright:problem:invalid"
    title = "Invalid request"


class Conflict(CoreError):
    status = 409
    problem_type = "urn:cinewright:problem:conflict"
    title = "Conflict"


class Infeasible(CoreError):
    """No model and host can serve the request. `reasons` lists the router's findings."""

    status = 409
    problem_type = "urn:cinewright:problem:infeasible"
    title = "No worker can run this request"

    def __init__(self, detail: str, reasons: list[str]) -> None:
        super().__init__(detail, reasons=reasons)
        self.reasons = reasons


class Unauthorized(CoreError):
    status = 401
    problem_type = "urn:cinewright:problem:unauthorized"
    title = "Unauthorized"


class NotImplementedFeature(CoreError):
    """A contract operation whose implementation belongs to a later roadmap phase."""

    status = 501
    problem_type = "urn:cinewright:problem:not-implemented"
    title = "Not implemented yet"

    def __init__(self, feature: str, phase: str) -> None:
        super().__init__(f"{feature} arrives in {phase}.", feature=feature, phase=phase)
