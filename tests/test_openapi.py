"""The committed openapi.json is generated from the code and must not drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cinewright.api.schemas import EventKind, JobEvent, SystemEvent, SystemEventKind
from cinewright.app import create_app
from cinewright.devtools import openapi_text, surface_lines

COMMITTED = Path(__file__).resolve().parent.parent / "openapi.json"


def test_openapi_json_matches_the_code():
    rendered = json.dumps(create_app().openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert rendered == openapi_text()
    assert rendered == COMMITTED.read_text(encoding="utf-8"), (
        "openapi.json is stale: run python -m cinewright.devtools openapi openapi.json"
    )


def test_web_ui_and_mcp_stay_out_of_the_schema():
    paths = create_app().openapi()["paths"]
    assert all(path.startswith("/api/v1/") for path in paths)


STREAM_ROUTE = (
    "ROUTE GET /api/v1/events params=[header:Last-Event-ID,query:job_id] body=- types=[-] "
    "-> 200:-,404:Problem,409:Problem,422:Problem,501:Problem"
)


def _refs(node: Any) -> set[str]:
    """Names of the component schemas a schema tree refers to directly."""
    if isinstance(node, dict):
        found = {node["$ref"].rsplit("/", 1)[-1]} if isinstance(node.get("$ref"), str) else set()
        return found.union(*(_refs(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_refs(value) for value in node))
    return set()


def _reachable(components: dict[str, Any], root: Any) -> set[str]:
    seen: set[str] = set()
    pending = _refs(root)
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending |= _refs(components.get(name, {}))
    return seen


def _stream() -> tuple[dict[str, Any], dict[str, Any]]:
    doc = create_app().openapi()
    return doc["paths"]["/api/v1/events"]["get"]["responses"]["200"], doc["components"]["schemas"]


def test_stream_events_response_refers_to_the_event_schemas():
    response, components = _stream()
    assert set(response["content"]) == {"text/event-stream"}  # not application/json: the route's surface line stays
    reachable = _reachable(components, response["content"]["text/event-stream"]["schema"])
    assert {"JobEvent", "EventKind", "SystemEvent", "SystemEventKind", "ModelInstall", "EngineInfo", "Progress"} <= reachable
    assert reachable <= set(components), "every $ref resolves to a component"
    assert components["EventKind"]["enum"] == [kind.value for kind in EventKind]
    assert components["SystemEventKind"]["enum"] == [kind.value for kind in SystemEventKind]
    assert STREAM_ROUTE in surface_lines()


@pytest.mark.parametrize("kind", list(EventKind), ids=str)
def test_every_job_event_kind_has_its_data_documented(kind):
    response, components = _stream()
    assert f"`{kind.value}`:" in components["JobEvent"]["properties"]["data"]["description"]
    assert f"`{kind.value}`" in response["description"]


@pytest.mark.parametrize("kind", list(SystemEventKind), ids=str)
def test_every_system_event_kind_has_a_payload_schema(kind):
    response, components = _stream()
    payload = _refs(components["SystemEvent"]["properties"][kind.value])
    assert len(payload) == 1 and payload <= set(components), payload
    assert f"`{kind.value}`:" in components["SystemEvent"]["description"]
    assert f"`{kind.value}`" in response["description"]


def test_preview_contract_is_documented():
    _, components = _stream()
    text = " ".join(components["JobEvent"]["description"].split())
    for fact in ("not stored", "never replayed", "no `id:` line", "repeats the `seq`", "latest kept preview", "once"):
        assert fact in text, fact


def test_the_stream_schemas_come_from_the_models_and_are_built_once():
    app = create_app()
    first = app.openapi()
    assert app.openapi() is first
    for model in (JobEvent, SystemEvent):
        properties = first["components"]["schemas"][model.__name__]["properties"]
        assert set(properties) == set(model.model_fields)
