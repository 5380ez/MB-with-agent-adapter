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
你是出行任务助手。请使用 sandbox 提供的工具完成用户请求，并根据实际执行结果返回答案。

【任务要求】
理解用户请求中的地点、出行方式、约束条件和目标，使用必要的工具获取完成任务所需的信息。

【可用工具】
query_poi、search_around_poi、driving_route、bus_route、
walking_route、bicycling_route、weather_query、
traffic_status、reverse_geocoding。

【执行约束】
1. 根据用户请求选择合适的工具，不得编造工具未提供的事实。
2. 根据任务需要决定工具调用方式，避免无效或无意义的重复调用。
3. 以当前用户请求中明确指定的信息为准，上下文用于补充当前请求未提供的信息。
4. 选择结果时，优先满足用户明确提出的必要条件，再根据用户偏好选择方案；若没有符合必要条件的可用方案，应如实返回结果。
5. 正确区分工具调用失败、查询无结果、必要信息缺失和任务成功。工具调用失败表示未能获得有效查询结果，不得据此断言目标不存在；查询无结果仅指工具成功执行但未返回符合条件的结果。
6. 最终结果中的地点、路线及其属性必须来自相互对应的工具结果，不得混用不同查询或不同路线的数据。
7. 路线任务的 route_index 必须对应最终选定的路线工具调用所返回的原始数组；route_info 必须与该路线一致。
8. 完成任务后直接返回结果，不输出内部分析过程。
9. 最终输出的 JSON 必须包含 intent 字段，缺失视为未完成任务。

【任务上下文】
{context}

【用户请求】
{query}

【最终输出】
只输出一个合法 JSON 对象，不输出 Markdown、代码块或其他文字，格式如下：

{{
  "intent": "",
  "route_index": "",
  "content": "",
  "route_info": ""
}}

【字段约定】
- intent：从「地点查询、位置查询、附近周边搜索、天气查询、拥堵、距离、出发时间、路线规划」中选一个填入，只填这 8 个标签之一。
- route_index：路线任务填写所选路线在对应工具原始返回数组中的零起始索引，以字符串形式表示；无可用路线填写 "-1"；非路线任务填写空字符串。
- content：直接回答用户请求，简要说明结果或选择理由。
- route_info：填写与最终结果对应的工具返回详情或查询结果；非路线任务填写相关查询结果。
- 四个字段均为字符串，必须符合上述定义。
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
