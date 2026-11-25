"""Maintainer tools (not for end users).

  python -m cinewright.devtools openapi [PATH]            write the OpenAPI document (default: stdout)
  python -m cinewright.devtools surface                   print the public surface, one sorted line per item
  python -m cinewright.devtools upstream-check [--json]   report a Wan2GP commit eligible for a pin update

The surface (routes, request and response schemas, MCP tools) is the public contract;
compare it between base and head: a change there is a `feat`.

upstream-check exits 0 when no upstream commit is eligible (and prints nothing), 10 when one
is, and 1 when the check failed (see cinewright.upstream_check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def openapi_document() -> dict[str, Any]:
    from cinewright.app import create_app

    return create_app().openapi()


def openapi_text() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _ref_name(schema: dict[str, Any] | None) -> str:
    if not schema:
        return "-"
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in schema:
        return "|".join(_ref_name(s) for s in schema["anyOf"])
    if schema.get("type") == "array":
        return f"list[{_ref_name(schema.get('items'))}]"
    return str(schema.get("type") or schema.get("format") or "object")


def _prop_type(schema: dict[str, Any]) -> str:
    text = _ref_name(schema)
    for key in ("enum", "const", "pattern"):
        if key in schema:
            text += f" {key}={json.dumps(schema[key], sort_keys=True)}"
    return text


def surface_lines() -> list[str]:
    doc = openapi_document()
    lines: list[str] = []
    for path, operations in doc.get("paths", {}).items():
        for method, op in operations.items():
            body = op.get("requestBody", {}).get("content", {})
            body_types = ",".join(sorted(body)) if body else "-"
            body_schema = "-"
            if "application/json" in body:
                body_schema = _ref_name(body["application/json"].get("schema"))
            params = ",".join(sorted(f"{p['in']}:{p['name']}{'!' if p.get('required') else ''}" for p in op.get("parameters", [])))
            responses = ",".join(
                f"{code}:{_ref_name(r.get('content', {}).get('application/json', {}).get('schema'))}"
                for code, r in sorted(op.get("responses", {}).items())
            )
            lines.append(f"ROUTE {method.upper()} {path} params=[{params}] body={body_schema} types=[{body_types}] -> {responses}")
    for name, schema in doc.get("components", {}).get("schemas", {}).items():
        required = set(schema.get("required", []))
        if "enum" in schema:
            lines.append(f"ENUM {name} {json.dumps(schema['enum'])}")
        for prop, prop_schema in schema.get("properties", {}).items():
            flag = " required" if prop in required else ""
            lines.append(f"SCHEMA {name}.{prop} {_prop_type(prop_schema)}{flag}")
    try:
        from cinewright.mcp.server import tool_surface
    except ImportError:  # MCP server not present in this build
        tool_surface = None
    if tool_surface is not None:
        lines.extend(f"MCP {line}" for line in tool_surface())
    return sorted(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cinewright.devtools", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("openapi")
    export.add_argument("path", nargs="?")
    sub.add_parser("surface")
    upstream = sub.add_parser("upstream-check")
    upstream.add_argument("--json", action="store_true", help="print the report as one JSON object")
    args = parser.parse_args(argv)
    if args.cmd == "upstream-check":
        from cinewright import upstream_check

        return upstream_check.run(as_json=args.json)
    if args.cmd == "openapi":
        text = openapi_text()
        if args.path:
            Path(args.path).write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        return 0
    if args.cmd == "surface":
        sys.stdout.write("\n".join(surface_lines()) + "\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
