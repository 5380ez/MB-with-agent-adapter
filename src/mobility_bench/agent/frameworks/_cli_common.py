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

PROMPT_TEMPLATE = """
你是出行任务助手。请先在内部识别用户请求中的地点、约束和目标，制定最短可执行计划，然后调用 sandbox 提供的工具完成任务。不要在最终答案中输出分析过程或计划。

【任务理解要求】
在调用工具前，必须识别：
1. 地点：起点、终点、途经点、查询位置及城市。
2. 约束：出行方式、时间、距离、费用、换乘、道路偏好及其他限制。
3. 目标：用户希望查询、比较、推荐或规划的具体结果。
4. 缺失信息：优先通过工具补充；只有工具无法获得时才视为缺失。

【工具使用规则】
1. 可用工具：
   query_poi、search_around_poi、driving_route、bus_route、walking_route、
   bicycling_route、weather_query、traffic_status、reverse_geocoding。
2. 必须使用工具获取坐标、距离、耗时、费用、天气和路况等事实，不得猜测或编造。
3. 只调用完成当前任务所必需的工具。
4. 工具调用总数最多为 15 次，这只是安全边界，不要因此而减少必要的工具调用。
5. 路线规划工具合计最多调用 2 次。
6. 同一工具使用完全相同的参数最多调用 1 次。
7. 每次调用前检查已有结果，避免重复查询已经获得的信息。
8. 达到调用上限仍无法完成时，应说明缺失信息，不得编造结果。
9. 工具返回多条路线时，根据用户约束选择方案；route_index 必须是该方案在工具返回数组中的零起始索引。

【任务】
当前上下文（含用户位置、城市等）：
{context}

用户请求：
{query}

【最终输出】
完成后只输出一个合法 JSON 对象，不要输出 Markdown、代码块、分析过程、计划或其他文字：

{{"intent":"","route_index":"","content":"","route_info":""}}

字段要求：
- intent：先判断该请求的意图，并从以下标签中选一个填入：地点查询、位置查询、附近周边搜索、天气查询、拥堵、距离、出发时间、路线规划。
- route_index：路线任务填写所选方案在工具返回数组中的零起始索引；没有可用路线填写 "-1"；信息查询任务填写空字符串。
- content：直接回答用户问题，或简要说明推荐方案及理由。
- route_info：填写路线详情或查询结果，其中坐标、距离、耗时、费用、天气和路况等事实必须来自工具返回。
"""


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
# Planning steps are derived from the ACTUAL tool trajectory — which sandbox tools
# were really called, and with which arguments — never from query keywords or the
# final answer text, so "I queried X" without really calling X earns nothing.
# Intent is NOT inferred here: the prompt makes the agent classify it itself (see
# ``extract_intent_from_answer``).

# Route tools: any of these means the agent performed route planning.
_ROUTE_TOOLS = frozenset({"driving_route", "bus_route", "walking_route", "bicycling_route"})
_ROUTE_KIND = {
    "driving_route": "驾车",
    "bus_route": "公交",
    "walking_route": "步行",
    "bicycling_route": "骑行",
}

# The departure-time back-calculation is the one planning step with no tool behind
# it, so ``extract_steps_from_trajectory`` detects a concrete time in the answer.
_TIME_RE = re.compile(r"\d{1,2}[:：]\d{2}|\d{1,2}点(?:\d{1,2}分)?", re.IGNORECASE)


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


def extract_intent_from_answer(answer: str) -> str:
    """Read the model's intent classification out of its final JSON answer.

    Intent is no longer inferred from the tool trajectory: the prompt asks the
    agent to classify the request itself and return the chosen label in
    ``intent``. Returns "" when the answer is empty, not JSON, or carries no
    string ``intent`` — an agent that ignores the contract earns nothing on ID.
    """
    text = (answer or "").strip()
    if not text:
        return ""
    # Tolerate a ```json fence even though the prompt forbids Markdown.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*[ \t]*\n?|\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get("intent", "")
        return value.strip() if isinstance(value, str) else ""
    # Fall back to a bare "intent": "..." scan when the JSON is malformed.
    match = re.search(r'"intent"\s*:\s*"([^"]*)"', text)
    return match.group(1).strip() if match else ""



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

    Planning steps are derived from the ACTUAL tool trajectory (see
    ``extract_steps_from_trajectory``); intent is the classification the agent
    itself produced in its final JSON answer (see ``extract_intent_from_answer``).
    """
    return {
        "plan_result": final_message,
        "planner_training_data": {
            "steps": extract_steps_from_trajectory(tool_calls, final_message),
            "thinking": "\n".join(reasoning)[:4000],
            "intent": extract_intent_from_answer(final_message),
        },
        "worker_training_data": build_worker_output(tool_calls),
        "token_usage": {
            "planner": {},
            "worker": {},
            "reporter": normalize_usage(usage),
        },
    }
