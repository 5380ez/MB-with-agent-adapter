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


def extract_steps(messages: list[str]) -> list[list[str]]:
    """The planning metric reads ``planner_training_data['steps']``.

    CLI harnesses do not emit an explicit plan, so derive coarse steps from the
    final assistant message to give the metric something to compare.
    """
    if not messages:
        return []
    text = messages[-1]
    steps: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•0123456789. \t")
        if 0 < len(line) < 120:
            steps.append([line])
    return steps[:20]


# Classification rules keyed on the *user query* rather than the agent's own
# wording: these CLIs never utter the benchmark's intent vocabulary ("位置查询",
# "距离", ...), they just reason about the request, so matching on agent text
# collapsed most cases into "路线规划". Ordered by priority; the first match
# wins. Values mirror InstructionMetric.TASK_SCENARIO_TO_INTENT_SLOTS.
_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("出发时间", ("几点出发", "几点走", "什么时候出发", "出发时间", "几点到", "我应该几点")),
    ("位置查询", ("我在哪", "哪里", "位置", "坐标", "经度", "纬度", "什么地方", "这是哪", "这在哪")),
    ("拥堵", ("堵", "拥堵", "事故", "路况")),
    ("天气查询", ("天气", "气温", "下雨", "温度")),
    ("距离", ("多远", "多少公里", "距离", "有多远")),
    ("附近周边搜索", ("附近", "周边")),
)
# Labels mirror the instruction metric's task_scenario -> intent mapping so the
# inferred intent lands in the benchmark's vocabulary.
_INTENT_LABELS = (
    "路线规划", "地点查询", "位置查询", "附近周边搜索",
    "天气查询", "拥堵", "距离", "出发时间",
)
# Queries that are themselves a bare "where am I" question.
_SHORT_POSITION_QUERIES = ("在哪", "在哪里", "这是哪", "这在哪", "什么地方")
# Motion/route words; a query with none of these and no rule match above is
# treated as a bare place lookup.
_ROUTE_KEYWORDS = (
    "去", "到", "怎么走", "路线", "导航", "前往", "开车", "坐公交", "坐地铁",
    "乘车", "驾车", "骑车", "步行", "换乘", "不走", "优先", "高速", "低速",
)


def infer_intent(query: str, text: str) -> str:
    """The instruction metric reads ``planner_training_data['intent']``.

    These CLIs do not emit an explicit intent. Classify the request from the user
    query (reliable, mirrors the benchmark's own scenario labels), then fall back
    to the agent's wording. Never echo raw reasoning: truncating a ``<think>``
    block into ``intent`` guarantees a similarity miss and pollutes the field.
    """
    q = (query or "").strip()
    if q in _SHORT_POSITION_QUERIES:
        return "位置查询"
    for label, keywords in _INTENT_RULES:
        if any(k in q for k in keywords):
            return label
    if q and not any(k in q for k in _ROUTE_KEYWORDS):
        return "地点查询"
    for label in _INTENT_LABELS:
        if label in (text or ""):
            return label
    return "路线规划"


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
    """Assemble the runner contract result from a parsed CLI transcript."""
    joined = "\n".join(reasoning) + "\n" + final_message
    return {
        "plan_result": final_message,
        "planner_training_data": {
            "steps": extract_steps([final_message] if final_message else []),
            "thinking": "\n".join(reasoning)[:4000],
            "intent": infer_intent(query, joined),
        },
        "worker_training_data": build_worker_output(tool_calls),
        "token_usage": {
            "planner": {},
            "worker": {},
            "reporter": normalize_usage(usage),
        },
    }
