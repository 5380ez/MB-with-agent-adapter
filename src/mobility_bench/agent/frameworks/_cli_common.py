"""Shared helpers for CLI-driven agent harnesses.

The harnesses in this package (``codex``, ``claude_code``, ``dsh``,
``dsh_mini``, ``mini_swe``) all work the same way: they shell out to an
external CLI, let it drive the deterministic sandbox tools over MCP, then map
the CLI's machine-readable transcript onto the runner contract::

    plan_result / planner_training_data / worker_training_data / token_usage

Everything in this module is harness-agnostic and side-effect free so the
per-harness adapters stay small.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Sandbox MCP server (9 deterministic replay tools) and the interpreter that can
# actually import it: the system python3 lacks fastmcp.
MCP_SERVER = PROJECT_ROOT / "mcp_server" / "sandbox_tools.py"
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"

PROMPT_TEMPLATE = (
    "你是出行路线规划助手，请使用 sandbox 提供的工具查询真实数据来完成任务，"
    "不要凭记忆编造坐标或路线。完成后用中文给出简洁、明确的最终答复。\n"
    "当前上下文（含用户位置、城市等）: {context}\n"
    "用户请求: {query}\n"
)


def resolve_bin(env_var: str, default: str) -> str:
    """Resolve an external CLI path, allowing an env override for testing."""
    return os.environ.get(env_var, default)


def format_context(context: Any) -> str:
    """Render the dataset context as a compact JSON string for the prompt."""
    if not context:
        return "{}"
    if isinstance(context, (dict, list)):
        return json.dumps(context, ensure_ascii=False)
    try:
        return json.dumps(json.loads(context), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(context)


def build_prompt(state: dict) -> str:
    return PROMPT_TEMPLATE.format(
        context=format_context(state.get("context")),
        query=state.get("query", ""),
    )


def strip_mcp_prefix(name: str) -> str:
    """Reduce ``mcp__<server>__<tool>`` to the bare tool name.

    Some harnesses (claude, dsh) report MCP tools fully qualified, but the tool
    metric's ground truth uses the bare name, so leaving the prefix on scores
    every tool call as a mismatch.
    """
    if name.startswith("mcp__") and name.count("__") >= 2:
        return name.split("__", 2)[2]
    return name


def parse_sandbox_args(command: str, tool: str) -> dict | None:
    """Recover a sandbox tool's own arguments from a bash command string.

    Bash-only harnesses (mini-swe-agent) reach the sandbox by running
    ``sandbox_cli.py <tool> --k v ...``; the model's ``bash`` tool call stores
    the whole shell line, so the metric sees ``{"command": ...}`` instead of the
    tool's real parameters and every call fails schema compliance. Re-parse the
    ``--k v`` flags into ``{"k": "v"}`` so the recorded call matches the schema.
    Values stay strings: the metric normalizes numbers itself and the dataset's
    ground truth stores coordinates as strings too. Returns ``None`` if the
    command is not a sandbox call to ``tool``.
    """
    tokens = command.split()
    start = -1
    for i, tok in enumerate(tokens):
        if tok.endswith("sandbox_cli.py") and i + 1 < len(tokens):
            if tokens[i + 1].strip("'\"") == tool:
                start = i + 2
                break
    if start < 0:
        return None

    args: dict[str, str] = {}
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                key, _, val = key.partition("=")
                args[key] = val.rstrip(",")
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[key] = tokens[i + 1].rstrip(",")
                i += 1
            else:
                args[key] = "true"
        i += 1
    return args


def parse_jsonl(out: str) -> list[dict]:
    """Parse newline-delimited JSON, skipping non-JSON noise lines."""
    events: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def normalize_usage(usage: dict) -> dict:
    """Normalize a provider usage block to the fields the metrics expect."""
    inp = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
    return {
        "prompt_tokens": inp,
        "completion_tokens": out,
        "total_tokens": inp + out,
        "input_tokens": inp,
        "output_tokens": out,
    }


# --- intent and planning-step inference from ACTUAL tool behavior ---
#
# External CLIs never utter the benchmark's intent vocabulary, but they DO reveal
# intent and the plan through which sandbox tools they actually call. Everything
# below is therefore driven by the tool trajectory — never by query keywords or
# the final answer text — so "I queried X" without really calling X earns nothing.

# Route tools: any of these means the agent performed route planning.
_ROUTE_TOOLS = frozenset({"driving_route", "bus_route", "walking_route", "bicycling_route"})
_ROUTE_KIND = {
    "driving_route": "驾车",
    "bus_route": "公交",
    "walking_route": "步行",
    "bicycling_route": "骑行",
}

# tool name -> intent label, in priority order (first called tool wins). Route
# tools are handled separately so 距离/出发时间 can refine 路线规划.
_INTENT_BY_TOOL: tuple[tuple[str, str], ...] = (
    ("weather_query", "天气查询"),
    ("traffic_status", "拥堵"),
    ("reverse_geocoding", "位置查询"),
    ("search_around_poi", "附近周边搜索"),
)

# Distance / departure-time projection. These two intents project only when a
# route tool fired AND the final answer actually states a distance / a concrete
# time; the query text is merely a confirmation gate and never assigns intent.
_DISTANCE_WORDS = ("多远", "多少公里", "距离", "有多远", "路程", "几公里", "里程")
_TIME_WORDS = ("几点出发", "几点走", "出发时间", "什么时候出发", "提前多久", "几点到", "应该几点")

_DISTANCE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:公里|千米|米|km)", re.IGNORECASE)
_TIME_RE = re.compile(r"\d{1,2}[:：]\d{2}|\d{1,2}点(?:\d{1,2}分)?", re.IGNORECASE)


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    return any(w in (text or "") for w in words)


def _is_meaningful(value: Any) -> bool:
    """True for a non-default strategy/waypoints value (the '有' in 有strategy)."""
    if isinstance(value, list):
        return bool(value)
    if value is None:
        return False
    return str(value).strip() not in ("", "0", "[]", "None")


def _fmt_arg(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(str(v) for v in value)
    return str(value)


def _step_for_call(name: str, args: dict) -> str | None:
    """One planning step (with its arguments) for one tool call, or None.

    The wording satisfies the PlanningMetric regexes (定位/路线规划/有偏好的路线
    规划/途径点路线规划/…), while the inlined arguments let IE/DEC check the real
    values the agent handed to the tools.
    """
    if name == "query_poi":
        return f"定位坐标：{_fmt_arg(args.get('keywords'))}（{_fmt_arg(args.get('city'))}）"
    if name == "weather_query":
        return f"查询天气，{_fmt_arg(args.get('city'))}"
    if name == "traffic_status":
        return f"查询路况，{_fmt_arg(args.get('name'))}（{_fmt_arg(args.get('city'))}）"
    if name == "reverse_geocoding":
        return f"定位坐标：{_fmt_arg(args.get('longitude'))},{_fmt_arg(args.get('latitude'))}"
    if name == "search_around_poi":
        return f"附近周边搜索：{_fmt_arg(args.get('location'))} {_fmt_arg(args.get('keywords'))}"
    if name in _ROUTE_TOOLS:
        kind = _ROUTE_KIND.get(name, "")
        origin = _fmt_arg(args.get("origin"))
        destination = _fmt_arg(args.get("destination"))
        waypoints = args.get("waypoints")
        strategy = args.get("strategy")
        if _is_meaningful(waypoints):
            return f"途径点路线规划：{origin} → {_fmt_list(waypoints)} → {destination}"
        if _is_meaningful(strategy):
            # The label alone satisfies the DEC regex; inlining the strategy value
            # (e.g. "不走高速") would leak the bare 走 into the text and fake a
            # 步行路线规划 match.
            return f"有偏好的路线规划：{origin} → {destination}"
        return f"{kind}路线规划：{origin} → {destination}"
    # Any other tool (bash, str_replace_editor, ...) does not map to a planning step.
    return None


def extract_steps_from_trajectory(
    tool_calls: list[dict], final_message: str
) -> list[list[str]]:
    """Derive planning steps from the ACTUAL tool trajectory, not the answer.

    The metrics read ``steps`` (IE uses ``thinking + steps``; DEC regex-matches
    the flattened text). Steps come from the tools the agent really called, with
    their arguments inlined. The final answer contributes only the departure-time
    back-calculation step (there is no tool for it).
    """
    steps: list[list[str]] = []
    route_completed = False
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        step = _step_for_call(name, args)
        if step:
            steps.append([step])
        if name in _ROUTE_TOOLS:
            route_completed = True
    if route_completed and _TIME_RE.search(final_message or ""):
        steps.append(["出发时间倒推计算"])
    return steps


def infer_intent(tool_calls: list[dict], final_message: str, query: str) -> str:
    """Infer intent from the agent's tool behavior, never from query keywords.

    No mapped tool -> empty intent, so an agent that only talks earns nothing on
    ID. Route tools project ``路线规划``, refined to ``距离``/``出发时间`` only
    when the final answer actually states a distance / time (query is a mere
    confirmation gate). ``query_poi`` alone -> 地点查询, but it never overrides
    the route/info tools.
    """
    names = {tc.get("name", "") for tc in tool_calls}
    if not names:
        return ""
    for tool, label in _INTENT_BY_TOOL:
        if tool in names:
            return label
    if names & _ROUTE_TOOLS:
        text = final_message or ""
        if _has_word(query, _DISTANCE_WORDS) and _DISTANCE_RE.search(text):
            return "距离"
        if _has_word(query, _TIME_WORDS) and _TIME_RE.search(text):
            return "出发时间"
        return "路线规划"
    # query_poi alone -> 地点查询. It must sit after the route/info tools because
    # it is also their shared "locate a place" precursor and must not override them.
    if "query_poi" in names:
        return "地点查询"
    return ""



def build_worker_output(tool_calls: list[dict]) -> list[dict]:
    """Map a flat list of tool calls onto the worker_training_data schema."""
    return [
        {
            "task_id": tc.get("id", ""),
            "system": "",
            "tools": [],
            "human": "",
            "tool_calls": [{
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }],
            "tool_results": {tc.get("id", ""): json.dumps(tc.get("result"), ensure_ascii=False)},
            "ai_response": "",
        }
        for tc in tool_calls
    ]


def standard_result(
    final_message: str,
    reasoning: list[str],
    tool_calls: list[dict],
    usage: dict,
    query: str,
) -> dict[str, Any]:
    """Assemble the runner contract result from a parsed CLI transcript.

    Intent and planning steps are derived from the ACTUAL tool trajectory (see
    ``infer_intent`` / ``extract_steps_from_trajectory``), never from the query
    wording or the answer text.
    """
    return {
        "plan_result": final_message,
        "planner_training_data": {
            "steps": extract_steps_from_trajectory(tool_calls, final_message),
            "thinking": "\n".join(reasoning)[:4000],
            "intent": infer_intent(tool_calls, final_message, query),
        },
        "worker_training_data": build_worker_output(tool_calls),
        "token_usage": {
            "planner": {},
            "worker": {},
            "reporter": normalize_usage(usage),
        },
    }
