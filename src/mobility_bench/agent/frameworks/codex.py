"""Codex CLI as a MobilityBench agent framework.

Runs the external ``codex exec`` process as the agent-under-test, pointed at a
local OpenAI-Responses-compatible endpoint (e.g. vLLM serving Qwen3-4B). The
deterministic sandbox tools are exposed to codex through the MCP server in
``mcp_server/sandbox_tools.py``.

The runner contract (see ``runner/batch.py::_run_with_graph``) requires:

* ``build_graph()`` -> object with an ``ainvoke(state)`` coroutine
* ``prepare_initial_state(query, context)`` -> state dict
* ``extract_result(state)`` -> dict with ``plan_result``,
  ``planner_training_data``, ``worker_training_data`` and ``token_usage``

Transcript parsing, intent/steps inference and token normalization live in
``_cli_common`` so every CLI harness shares one implementation.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from mobility_bench.agent.frameworks._cli_common import (
    MCP_SERVER,
    VENV_PY,
    build_prompt,
    parse_jsonl,
    resolve_bin,
    standard_result,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.harness import harness_config

CODEX_BIN = resolve_bin("CODEX_BIN", "codex")


class _CodexGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "CodexFramework"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return self._fw.run_codex(initial_state)


class CodexFramework(BaseFramework):
    """Framework that drives the codex CLI as the agent under test."""

    @property
    def name(self) -> str:
        return "codex"

    def build_graph(self) -> Any:
        return _CodexGraph(self)

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    # ------------------------------------------------------------------ codex
    def _model_name(self) -> str:
        if self.model_config and self.model_config.roles:
            return self.model_config.roles.get("planner") or self.model_config.name
        return self.model_config.name if self.model_config else "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or "http://localhost:8080/v1"
        return url.rstrip("/")

    def _codex_config(self) -> str:
        hc = harness_config("codex")
        return f"""model = "{self._model_name()}"
model_provider = "vllm"

approval_policy = "never"
sandbox_mode = "danger-full-access"
model_context_window = {hc.context_window}

# codex defers MCP tools behind a tool-search mechanism by default, so a
# weaker local model never discovers ``mcp__sandbox__*``. Disable it here.
[features]
tool_search = false
tool_search_always_defer_mcp_tools = false

[model_providers.vllm]
name = "Local vLLM"
base_url = "{self._base_url()}"
wire_api = "{hc.wire_api}"

[mcp_servers.sandbox]
command = "{VENV_PY}"
args = ["{MCP_SERVER}"]
startup_timeout_sec = 180.0
tool_timeout_sec = 120.0
default_tools_approval_mode = "approve"
"""

    def run_codex(self, state: dict) -> dict:
        hc = harness_config("codex")
        prompt = build_prompt(state)

        with tempfile.TemporaryDirectory() as home:
            (Path(home) / "config.toml").write_text(self._codex_config(), encoding="utf-8")
            env = {**os.environ, "CODEX_HOME": home}
            cmd = [
                CODEX_BIN, "exec", "--json",
                "--skip-git-repo-check",
                "--color", "never",
                prompt,
            ]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, timeout=hc.timeout
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or "" if isinstance(e.stdout, str) else (e.stdout or b"").decode()
                stderr = f"codex exec timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started

        state["_codex"] = {
            "events": parse_jsonl(stdout),
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
        }
        return state

    # ---------------------------------------------------------------- mapping
    @staticmethod
    def _merge_turn_usage(acc: dict, usage: dict) -> dict:
        """Sum ``turn.completed`` usage blocks across turns.

        Codex emits one ``turn.completed`` per turn, each carrying that turn's
        Responses-API usage (``input_tokens`` / ``output_tokens`` plus cache and
        reasoning splits). Sum the two headline fields so a multi-turn run
        counts every turn instead of silently keeping only the last one;
        ``normalize_usage`` re-derives ``total_tokens`` from the summed pair.
        Cached/reasoning splits are subsets of those two for a non-reasoning
        local model, so they are deliberately not added.
        """
        return {
            "input_tokens": (acc.get("input_tokens", 0) or 0) + (usage.get("input_tokens", 0) or 0),
            "output_tokens": (acc.get("output_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
        }

    def extract_result(self, final_state: dict) -> dict[str, Any]:
        codex = final_state.get("_codex", {})
        events = codex.get("events", [])

        agent_messages: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict] = []
        usage: dict = {}

        for e in events:
            etype = e.get("type")
            if etype == "item.completed":
                item = e.get("item", {})
                itype = item.get("type")
                if itype == "agent_message":
                    agent_messages.append(item.get("text", ""))
                elif itype == "reasoning":
                    reasoning.append(item.get("text", ""))
                elif itype == "mcp_tool_call":
                    tool_calls.append({
                        "id": item.get("id", ""),
                        "name": item.get("tool", ""),
                        "args": item.get("arguments", {}),
                        "result": item.get("result"),
                    })
            elif etype == "turn.completed":
                usage = self._merge_turn_usage(usage, e.get("usage", {}) or {})

        final_message = agent_messages[-1] if agent_messages else ""
        return standard_result(
            final_message=final_message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            query=final_state.get("query", ""),
        )

