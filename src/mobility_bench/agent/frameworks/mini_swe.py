"""mini-swe-agent (``mini``) as a MobilityBench agent framework.

mini-swe-agent is a minimal SWE agent loop whose only tool is ``bash``. It
cannot speak MCP, so the sandbox is reached through the CLI bridge
(``mcp_server/sandbox_cli.py``), injected into the template context via the
child environment; see ``configs/harness/mini_swe.yaml``.

Routing goes through litellm: the endpoint is set on ``model.model_kwargs``
(``api_base``/``api_key``) and the model is addressed as ``openai/<id>``. The
run writes a trajectory JSON file, which the adapter parses into the runner
contract.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from mobility_bench.agent.frameworks._cli_common import (
    PROJECT_ROOT,
    VENV_PY,
    build_prompt,
    parse_sandbox_args,
    resolve_bin,
    standard_result,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.harness import harness_config

# mini installs into the project venv, which is not on PATH; default to that
# interpreter's script and keep the MINI_BIN override for anything else.
MINI_BIN = resolve_bin("MINI_BIN", str(PROJECT_ROOT / ".venv" / "bin" / "mini"))

# Bash CLI bridge that exposes the same 9 sandbox tools the MCP harnesses call.
SANDBOX_CLI = PROJECT_ROOT / "mcp_server" / "sandbox_cli.py"

# The 9 deterministic replay tools behind ``sandbox_cli.py``. mini's only tool
# is ``bash``, so a sandbox call arrives as a bash command that runs
# ``sandbox_cli.py <tool> ...``; recover ``<tool>`` for the tool metric, which
# matches on the bare tool name.
SANDBOX_TOOLS = frozenset({
    "query_poi",
    "driving_route",
    "walking_route",
    "bicycling_route",
    "bus_route",
    "weather_query",
    "search_around_poi",
    "reverse_geocoding",
    "traffic_status",
})


def _sandbox_call(command: str) -> tuple[str, dict] | None:
    """Resolve a bash command to the sandbox ``(tool, args)`` it invokes.

    mini's only tool is ``bash``, so a sandbox call arrives as a shell line that
    runs ``sandbox_cli.py <tool> --k v``; the model may prefix it with ``cd`` or
    ``&&``. Returns the bare tool name plus the tool's own arguments (see
    :func:`parse_sandbox_args`), or ``None`` when the line is a plain bash
    command. Surfacing the real name/args is what lets the tool metric match the
    call and score its schema.
    """
    tokens = command.split()
    for i, tok in enumerate(tokens):
        if tok.endswith("sandbox_cli.py") and i + 1 < len(tokens):
            candidate = tokens[i + 1].strip("'\"")
            if candidate in SANDBOX_TOOLS:
                return candidate, parse_sandbox_args(command, candidate) or {}
    return None

# Custom mini config: mobility framing + the sandbox tool surface. Passed with
# ``-c``, which suppresses the built-in default config, so this file is the
# whole agent definition.
MINI_CONFIG = PROJECT_ROOT / "configs" / "harness" / "mini_swe.yaml"


class _MiniGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "MiniSweFramework"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return self._fw.run_mini(initial_state)


class MiniSweFramework(BaseFramework):
    """Framework that drives the mini-swe-agent CLI as the agent under test."""

    @property
    def name(self) -> str:
        return "mini_swe"

    def build_graph(self) -> Any:
        return _MiniGraph(self)

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    def _model_name(self) -> str:
        if self.model_config and self.model_config.roles:
            return self.model_config.roles.get("planner") or self.model_config.name
        return self.model_config.name if self.model_config else "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or "http://localhost:8080/v1"
        return url.rstrip("/")

    def _api_key(self) -> str:
        key = (self.model_config.api_key if self.model_config else "") or ""
        return key or "dummy"

    def run_mini(self, state: dict) -> dict:
        hc = harness_config("mini_swe")
        prompt = build_prompt(state)

        env = {
            **os.environ,
            # First run would otherwise drop into an interactive setup wizard.
            "MSWEA_CONFIGURED": "true",
            # litellm has no price entry for the local model; without this the
            # run dies in cost accounting instead of finishing.
            "MSWEA_COST_TRACKING": "ignore_errors",
            # Resolved into the jinja context by LocalEnvironment.
            "sandbox_cli": f"{VENV_PY} {SANDBOX_CLI}",
        }

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "traj.json"
            cmd = [
                MINI_BIN,
                "-m", f"openai/{self._model_name()}",
                "-t", prompt,
                "-y",
                "-o", str(traj_path),
                "-c", str(MINI_CONFIG),
                "-c", f"model.model_kwargs.api_base={self._base_url()}",
                "-c", f"model.model_kwargs.api_key={self._api_key()}",
            ]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env,
                    cwd=str(PROJECT_ROOT), timeout=hc.timeout,
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                raw = e.stdout or ""
                stdout = raw if isinstance(raw, str) else (raw or b"").decode()
                stderr = f"mini timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started

            trajectory = None
            if traj_path.exists():
                try:
                    trajectory = json.loads(traj_path.read_text())
                except json.JSONDecodeError:
                    trajectory = None

        state["_mini"] = {
            "trajectory": trajectory,
            "stdout": stdout,
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
        }
        return state

    def extract_result(self, final_state: dict) -> dict[str, Any]:
        mini = final_state.get("_mini", {})
        traj = mini.get("trajectory") or {}
        messages = traj.get("messages", []) or []
        info = traj.get("info", {}) or {}

        assistant_text: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict] = []
        pending: dict[str, dict] = {}
        prompt_tokens = 0
        completion_tokens = 0

        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    assistant_text.append(content.strip())
                # litellm stashes the raw provider response (with usage) here.
                resp = (msg.get("extra") or {}).get("response") or {}
                usage = resp.get("usage") or {}
                prompt_tokens += usage.get("prompt_tokens", 0) or 0
                completion_tokens += usage.get("completion_tokens", 0) or 0
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function", {}) or {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    # mini's only tool is ``bash``; surface the sandbox tool the
                    # command actually runs (name *and* its own args) so the tool
                    # metric can match it and score its schema.
                    name = fn.get("name", "bash")
                    if isinstance(args, dict):
                        sandbox = _sandbox_call(str(args.get("command", "")))
                        if sandbox:
                            name, args = sandbox
                    tc = {
                        "id": call.get("id", ""),
                        "name": name,
                        "args": args,
                        "result": None,
                    }
                    tool_calls.append(tc)
                    pending[tc["id"]] = tc
            elif role == "tool":
                output = msg.get("content")
                extra = msg.get("extra") or {}
                if extra.get("raw_output") is not None:
                    output = extra["raw_output"]
                call_id = msg.get("tool_call_id") or msg.get("id", "")
                if call_id in pending:
                    pending[call_id]["result"] = output

        # mini folds thinking into the assistant content; keep it in `reasoning`
        # (truncation happens in standard_result) and use the last message as the
        # final answer.
        if assistant_text:
            reasoning = assistant_text[:-1]
        final_message = assistant_text[-1] if assistant_text else ""
        if not final_message:
            final_message = (info.get("submission") or "").strip()

        usage = _usage_from_stats(info.get("model_stats") or {}, prompt_tokens, completion_tokens)
        return standard_result(
            final_message=final_message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            query=final_state.get("query", ""),
        )


def _usage_from_stats(stats: dict, prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    """Assemble token usage.

    Tokens are summed from each assistant message's ``extra.response.usage``
    (litellm does not surface them on ``info.model_stats`` for local models, only
    cost and call count). ``api_calls`` is kept so a zero-token run is still
    distinguishable from a run that never called the model.
    """
    usage: dict = {}
    if prompt_tokens or completion_tokens:
        usage["prompt_tokens"] = prompt_tokens
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = prompt_tokens + completion_tokens
    for key in ("instance_cost", "api_calls"):
        if stats.get(key) is not None:
            usage[key] = stats[key]
    return usage
