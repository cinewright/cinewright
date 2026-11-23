"""The committed openapi.json is generated from the code and must not drift."""

from __future__ import annotations

import json
from pathlib import Path

from cinewright.app import create_app
from cinewright.devtools import openapi_text

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
