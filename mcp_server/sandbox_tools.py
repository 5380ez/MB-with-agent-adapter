"""MobilityBench sandbox tools exposed via MCP (stdio).

Wraps the 9 deterministic replay tools in ``mobility_bench.tools.sandbox`` so
that an external agent (codex CLI) can call them exactly like the built-in
LangGraph agents.

Why schemas are declared statically
-----------------------------------
Each sandbox tool module loads a multi-hundred-MB replay JSON at import time
(``driving_route_handle.json`` alone is ~1.4 GB). Importing all nine tools
takes ~20 s and delays the MCP ``tools/list`` response past the point where
codex has already sent its first model request.

codex treats a slow-starting MCP server as a *pending optional* server and
omits its tools from the request rather than waiting, so a slow startup means
the model never sees ``mcp__sandbox__*``. To avoid that we advertise the tool
surface immediately from the static schemas below and import the real
implementation lazily, only when a tool is first invoked.

Run with the project venv interpreter::

    /mnt/workspace/project/MobilityBench/.venv/bin/python mcp_server/sandbox_tools.py

The system python3 has an outdated ``mcp`` package without ``fastmcp`` — always
use the venv.

Each tool takes a single JSON-string argument ``args_json`` and returns the raw
``ToolResult.to_dict()`` payload as a JSON string, e.g.
``{"status": 0, "data": {...}, "error": null}``.
"""

import inspect
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MOBILITY_BENCH_ROOT", str(ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mobility_sandbox")

# name -> (description, input schema). Kept in sync with the ``@tool``
# definitions in ``mobility_bench/tools/sandbox/*.py``.
TOOL_SPECS: dict[str, tuple[str, dict]] = {
    "query_poi": (
        "根据模糊的位置信息，搜索得到准确的位置信息和坐标",
        {
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "要查询的关键词，例如：天安门、肯德基等"},
                "city": {"type": "string", "description": "城市名称，可选，例如：北京、上海"},
            },
            "required": ["keywords"],
        },
    ),
    "driving_route": (
        "根据起终点坐标检索符合条件的驾车路线规划方案，支持填加途径点",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点坐标，格式为 '经度,纬度'，例如 '116.481499,39.990755'"},
                "destination": {"type": "string", "description": "终点坐标，格式为 '经度,纬度'，例如 '116.465342,39.923423'"},
                "strategy": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "驾车算路策略，可选值包括：'躲避拥堵'、'高速优先'、'不走高速'、'少收费'、'大路优先'、'速度最快'。例如：['躲避拥堵', '高速优先']。最多选择2项。",
                },
                "waypoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "途经点坐标，格式为经度,纬度。例如：['116.473195,39.993253']。最多选择5个途经点。",
                },
            },
            "required": ["origin", "destination"],
        },
    ),
    "walking_route": (
        "根据起终点坐标检索符合条件的步行路线规划方案",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点坐标，格式为 '经度,纬度'，例如 '116.481499,39.990755'"},
                "destination": {"type": "string", "description": "终点坐标，格式为 '经度,纬度'，例如 '116.465342,39.923423'"},
            },
            "required": ["origin", "destination"],
        },
    ),
    "bicycling_route": (
        "根据起终点坐标检索符合条件的骑自行车路线规划方案",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点坐标，格式为 '经度,纬度'，例如 '116.481499,39.990755'"},
                "destination": {"type": "string", "description": "终点坐标，格式为 '经度,纬度'，例如 '116.465342,39.923423'"},
            },
            "required": ["origin", "destination"],
        },
    ),
    "bus_route": (
        "根据起终点坐标检索符合条件的公共交通路线规划方案，支持跨城的火车、客车、飞机等方案。",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点坐标，格式为 '经度,纬度'，例如 '116.481499,39.990755'"},
                "destination": {"type": "string", "description": "终点坐标，格式为 '经度,纬度'，例如 '116.465342,39.923423'"},
                "strategy": {
                    "type": "string",
                    "description": "路线策略，只能返回一个数字，默认为0，可选值：0-推荐模式，1-票价最低，2-换乘次数少，3-步行少，4-舒适（乘坐空调车），5-不坐地铁，7-地铁优先，8-时间短",
                },
            },
            "required": ["origin", "destination"],
        },
    ),
    "weather_query": (
        "查询指定位置的天气信息。",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如：'北京'、'上海'、'广州'等"},
                "need_forecast": {"type": "boolean", "description": "是否需要未来几天的天气预报，True表示需要预报，False表示只要当前天气"},
            },
            "required": ["city"],
        },
    ),
    "search_around_poi": (
        "根据中心点坐标，搜索周边POI信息",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "中心点坐标，格式为 '经度,纬度'，例如 '116.481499,39.990755'"},
                "keywords": {"type": "string", "description": "搜索关键词，可选，例如：加油站、餐厅"},
                "radius": {"type": "integer", "description": "搜索半径，单位米，默认10000"},
            },
            "required": ["location"],
        },
    ),
    "reverse_geocoding": (
        "根据经纬度坐标进行逆地理编码，查询该位置的详细地址信息。",
        {
            "type": "object",
            "properties": {
                "longitude": {"type": "string", "description": "经度"},
                "latitude": {"type": "string", "description": "纬度"},
                "radius": {"type": "integer", "description": "搜索半径，单位：米，默认1000"},
            },
            "required": ["longitude", "latitude"],
        },
    ),
    "traffic_status": (
        "查询指定城市某条道路的实时交通状况",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "道路名称，例如：'京港澳高速'、'外环'"},
                "city": {"type": "string", "description": "城市名称，例如：'北京市'、'上海市'"},
            },
            "required": ["name", "city"],
        },
    ),
}

TOOL_NAMES = list(TOOL_SPECS)

_impl_lock = threading.Lock()
_impls: dict[str, object] = {}


def _load_impl(name: str):
    """Import the real sandbox tool on first use (heavy JSON load, slow)."""
    with _impl_lock:
        if name in _impls:
            return _impls[name]
        from mobility_bench.tools.registry import ToolRegistry

        ToolRegistry.load_default_tools(mode="sandbox")
        tool_obj = ToolRegistry.get(name)
        _impls[name] = tool_obj
        return tool_obj


def _make_handler(name: str, schema: dict):
    """Build an MCP handler that accepts the tool's native parameters.

    The model calls the tool with its real parameters (e.g. ``city="北京"``)
    rather than a JSON blob, which is what codex expects from an MCP schema.
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = list(props)

    async def _call(**kwargs) -> str:
        # Accept the legacy ``args_json`` blob as well, in case a client sends it.
        blob = kwargs.pop("args_json", None)
        if blob:
            try:
                kwargs = {**json.loads(blob), **kwargs}
            except json.JSONDecodeError as e:
                return json.dumps({"status": 1, "data": {}, "error": f"invalid JSON args: {e}"})
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            tool_obj = _load_impl(name)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": 1, "data": {}, "error": f"tool load failed: {e}"})
        if tool_obj is None:
            return json.dumps({"status": 1, "data": {}, "error": f"unknown tool: {name}"})
        try:
            result = tool_obj.invoke(args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": 1, "data": {}, "error": str(e)})

    _call.__name__ = name
    # FastMCP derives the input schema from the signature. Give it the real
    # parameters (typed as ``object``) so the generated schema is JSON-Schema
    # compatible; the exact schema is installed below.
    _call.__annotations__ = {f: object for f in fields}
    _call.__annotations__["return"] = str
    _call.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                f,
                inspect.Parameter.KEYWORD_ONLY,
                default=(... if f in required else None),
                annotation=object,
            )
            for f in fields
        ]
    )
    return _call


def _register_tools():
    """Register all tools with static schemas so tools/list returns instantly."""
    for name, (doc, schema) in TOOL_SPECS.items():
        handler = _make_handler(name, schema)
        handler.__doc__ = doc
        mcp.tool(name=name, description=doc)(handler)

    # FastMCP derives a generic ``object``-typed schema from the signature.
    # Overwrite it with the tool's real JSON schema so the model sees the
    # correct parameter names, types and required list.
    tm = mcp._tool_manager
    for name, (_doc, schema) in TOOL_SPECS.items():
        tool = tm.get_tool(name)
        if tool is not None:
            tool.parameters = schema


_register_tools()


if __name__ == "__main__":
    mcp.run()
