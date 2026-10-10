#!/usr/bin/env python
"""CLI bridge for the MobilityBench sandbox tools.

Bash-only harnesses (mini-swe-agent) cannot speak MCP, so this exposes the same
9 deterministic replay tools as ordinary commands::

    sandbox_cli.py query_poi --keywords 天安门 --city 北京
    sandbox_cli.py driving_route --origin 116.4,39.9 --destination 121.5,31.2
    sandbox_cli.py list            # print the tool surface as JSON

Each call prints the raw ``{"status", "data", "error"}`` JSON payload on stdout,
identical to what the MCP server returns, so transcripts and metrics stay
comparable across harnesses.

Run with the project venv interpreter (the system python3 lacks the deps)::

    /mnt/workspace/project/MobilityBench/.venv/bin/python mcp_server/sandbox_cli.py ...
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MOBILITY_BENCH_ROOT", str(ROOT))

# stdout is the JSON channel, so nothing else may write to it. Configure logging
# *before* importing any tool module: the tools call setup_logging() at import,
# which attaches a stdout handler and prints cache-load INFO lines. Redirecting
# the root logger to stderr (and raising the level) keeps stdout pure JSON.
import logging  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.ERROR, force=True)


def _load_registry():
    from mobility_bench.tools.registry import ToolRegistry

    ToolRegistry.load_default_tools(mode="sandbox")
    return ToolRegistry


def _coerce(value: str, ptype: str | None = None):
    """Coerce a CLI string to the type the tool schema declares.

    Shell arguments arrive as strings; pydantic then rejects e.g. a float for a
    ``string`` field. Coerce strictly by the schema type so coordinates like
    ``120.15`` stay strings while ``radius`` becomes an int.
    """
    if ptype == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if ptype == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if ptype == "boolean":
        low = str(value).strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return value
    if ptype == "array":
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [p.strip() for p in str(value).split(",") if p.strip()]
    if ptype == "string":
        return value
    # Unknown type: best-effort JSON, else the raw string.
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _parse_kwargs(argv: list[str], properties: dict | None = None) -> dict:
    properties = properties or {}

    def ptype(key: str) -> str | None:
        spec = properties.get(key)
        return spec.get("type") if isinstance(spec, dict) else None

    kwargs: dict = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                key, _, val = key.partition("=")
                kwargs[key] = _coerce(val, ptype(key))
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                kwargs[key] = _coerce(argv[i + 1], ptype(key))
                i += 1
            else:
                kwargs[key] = _coerce("true", ptype(key))
        i += 1
    return kwargs


def _print_tool_list(registry) -> None:
    from mcp_server.sandbox_tools import TOOL_SPECS

    surface = [
        {"name": name, "description": doc, "parameters": schema}
        for name, (doc, schema) in TOOL_SPECS.items()
    ]
    print(json.dumps(surface, ensure_ascii=False, indent=2))


def main(argv: list[str]) -> int:
    # The MCP tool specs are the single source of truth for the tool surface.
    sys.path.insert(0, str(ROOT))
    from mcp_server.sandbox_tools import TOOL_SPECS

    if not argv or argv[0] in ("-h", "--help", "help", "list", "--list"):
        _print_tool_list(_load_registry())
        return 0

    name = argv[0]
    if name not in TOOL_SPECS:
        print(json.dumps({
            "status": 1,
            "data": {},
            "error": f"unknown tool: {name}",
            "available": list(TOOL_SPECS),
        }, ensure_ascii=False))
        return 2

    kwargs = _parse_kwargs(argv[1:], (TOOL_SPECS[name][1] or {}).get("properties"))
    tool_obj = _load_registry().get(name)
    if tool_obj is None:
        print(json.dumps({"status": 1, "data": {}, "error": f"tool load failed: {name}"}))
        return 1
    try:
        result = tool_obj.invoke(kwargs)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001 - surface any tool error as JSON
        print(json.dumps({"status": 1, "data": {}, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
