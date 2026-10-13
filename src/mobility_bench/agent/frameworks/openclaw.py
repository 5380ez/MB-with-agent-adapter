"""OpenClaw CLI as a MobilityBench agent framework.

openclaw is a multi-channel AI gateway; its ``agent --local --json`` subcommand
runs one embedded agent turn *without* the gateway daemon — the analogue of
codex's ``exec`` and claude's ``-p``. This adapter points that turn at a local
OpenAI-completions endpoint (vLLM serving Qwen3-4B) and exposes the
deterministic sandbox tools through the same MCP server
(``mcp_server/sandbox_tools.py``) the other harnesses use.

Per run the adapter writes an isolated ``openclaw.json`` (``OPENCLAW_STATE_DIR``
+ ``OPENCLAW_CONFIG_PATH`` under a temp dir) declaring the ``vllm`` provider,
the default agent model and the sandbox MCP server. The reply text and the
aggregate token usage come from the ``--json`` stdout; the tool trajectory is
read back from the session-transcript JSONL that stdout points at.

The runner contract (see ``runner/batch.py::_run_with_graph``) requires:

* ``build_graph()`` -> object with an ``ainvoke(state)`` coroutine
* ``prepare_initial_state(query, context)`` -> state dict
* ``extract_result(state)`` -> dict with ``plan_result``,
  ``planner_training_data``, ``worker_training_data`` and ``token_usage``
"""

import json
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
    resolve_bin,
    standard_result,
    strip_mcp_prefix,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.harness import harness_config

# openclaw installs its CLI at <prefix>/bin/openclaw; override with OPENCLAW_BIN.
OPENCLAW_BIN = resolve_bin("OPENCLAW_BIN", "/opt/openclaw/bin/openclaw")

# Provider id declared in the per-run openclaw.json; the agent model is
# addressed as ``<provider>/<id>`` (e.g. ``vllm/qwen3-4b``).
_PROVIDER = "vllm"


def _bare_tool_name(name: str) -> str:
    """Reduce an openclaw MCP tool name to the bare tool name.

    openclaw reports MCP tools as ``<server>__<tool>`` (e.g.
    ``sandbox__query_poi``); some paths may already carry the ``mcp__`` form
    that :func:`strip_mcp_prefix` handles. The tool metric's ground truth uses
    the bare name, so both prefixes are stripped here.
    """
    return strip_mcp_prefix(name).split("__", 1)[-1]


def _session_file_from(stdout: str) -> str:
    """Pull the session-transcript path out of the ``--json`` stdout blob."""
    start = stdout.find("{")
    if start == -1:
        return ""
    try:
        result = json.loads(stdout[start:])
    except json.JSONDecodeError:
        return ""
    meta = result.get("meta") or {}
    agent_meta = meta.get("agentMeta") or {}
    return agent_meta.get("sessionFile") or ""


class _OpenclawGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "OpenclawFramework"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return self._fw.run_openclaw(initial_state)


class OpenclawFramework(BaseFramework):
    """Framework that drives the openclaw CLI as the agent under test."""

    @property
    def name(self) -> str:
        return "openclaw"

    def build_graph(self) -> Any:
        return _OpenclawGraph(self)

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    # --------------------------------------------------------------- openclaw
    def _model_name(self) -> str:
        if self.model_config and self.model_config.roles:
            return self.model_config.roles.get("planner") or self.model_config.name
        return self.model_config.name if self.model_config else "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or "http://localhost:8080/v1"
        return url.rstrip("/")

    def _api_key(self) -> str:
        return (self.model_config.api_key if self.model_config else "") or "dummy"

    def _config(self, home: str) -> str:
        """Build the isolated openclaw.json for one run.

        Declares the vllm provider (local endpoint), pins the default agent to
        it, and mounts the sandbox MCP server over stdio. Bootstrap/workspace
        injection is disabled so openclaw does not fold a host AGENTS.md into
        the prompt — the MobilityBench task prompt is the whole user message.
        """
        hc = harness_config("openclaw")
        model = self._model_name()
        config = {
            "models": {
                "mode": "merge",
                "providers": {
                    _PROVIDER: {
                        "api": "openai-completions",
                        "baseUrl": self._base_url(),
                        "apiKey": self._api_key(),
                        "models": [{
                            "id": model,
                            "name": model,
                            "contextWindow": hc.context_window,
                            "maxTokens": hc.max_output_tokens,
                        }],
                    }
                },
            },
            "agents": {
                "defaults": {
                    "model": f"{_PROVIDER}/{model}",
                    "timeoutSeconds": hc.timeout,
                    "workspace": str(Path(home) / "workspace"),
                    "skipBootstrap": True,
                }
            },
            "mcp": {
                "servers": {
                    "sandbox": {
                        "command": str(VENV_PY),
                        "args": [str(MCP_SERVER)],
                        "transport": "stdio",
                    }
                }
            },
        }
        return json.dumps(config, ensure_ascii=False)

    def run_openclaw(self, state: dict) -> dict:
        hc = harness_config("openclaw")
        prompt = build_prompt(state)

        with tempfile.TemporaryDirectory() as home:
            home_path = Path(home)
            config_path = home_path / "openclaw.json"
            config_path.write_text(self._config(home), encoding="utf-8")
            (home_path / "workspace").mkdir(exist_ok=True)
            prompt_path = home_path / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            env = {
                **os.environ,
                # Isolate all agent state/sessions under the temp dir.
                "OPENCLAW_STATE_DIR": home,
                "OPENCLAW_CONFIG_PATH": str(config_path),
            }
            cmd = [
                OPENCLAW_BIN, "agent",
                "--local",
                "--json",
                # `agent` needs a session selector; "main" is the default agent
                # the per-run config creates.
                "--agent", "main",
                "--message-file", str(prompt_path),
                "--timeout", str(hc.timeout),
            ]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env,
                    timeout=hc.timeout + 60,
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                raw = e.stdout or ""
                stdout = raw if isinstance(raw, str) else (raw or b"").decode()
                stderr = f"openclaw agent timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started
            # The session transcript lives under the temp state dir, so parse
            # the tool trajectory *before* the directory is cleaned up.
            tool_calls = self._read_tool_calls(_session_file_from(stdout))

        state["_openclaw"] = {
            "stdout": stdout,
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
            "tool_calls": tool_calls,
        }
        return state

    # ---------------------------------------------------------------- mapping
    def _read_tool_calls(self, session_file: str) -> list[dict]:
        """Parse the session transcript JSONL for tool calls + their results."""
        tool_calls: list[dict] = []
        path = Path(session_file)
        if not path.exists():
            return tool_calls
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "message":
                continue
            message = record.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                for block in message.get("content") or []:
                    if block.get("type") == "toolCall":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "name": _bare_tool_name(block.get("name", "")),
                            "args": block.get("arguments", {}) or {},
                            "result": None,
                        })
            elif role == "toolResult":
                call_id = message.get("toolCallId", "")
                texts = [
                    b.get("text", "")
                    for b in (message.get("content") or [])
                    if b.get("type") == "text"
                ]
                result = "\n".join(t for t in texts if t)
                for tc in tool_calls:
                    if tc["id"] == call_id:
                        tc["result"] = result
                        break
        return tool_calls

    def extract_result(self, final_state: dict) -> dict[str, Any]:
        oc = final_state.get("_openclaw", {})
        stdout = oc.get("stdout", "")

        result: dict = {}
        start = stdout.find("{")
        if start != -1:
            try:
                result = json.loads(stdout[start:])
            except json.JSONDecodeError:
                result = {}

        payloads = result.get("payloads") or []
        agent_text = "\n".join(
            (p.get("text") or "") for p in payloads if isinstance(p, dict)
        ).strip()
        meta = result.get("meta") or {}
        agent_meta = meta.get("agentMeta") or {}

        final_message = agent_text or (meta.get("finalAssistantVisibleText") or "")

        usage_block = agent_meta.get("usage") or {}
        usage = {
            "prompt_tokens": usage_block.get("input", 0) or 0,
            "completion_tokens": usage_block.get("output", 0) or 0,
        }
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        # Tool trajectory was parsed while the temp state dir still existed.
        tool_calls = oc.get("tool_calls") or []

        return standard_result(
            final_message=final_message,
            reasoning=[],
            tool_calls=tool_calls,
            usage=usage,
            query=final_state.get("query", ""),
        )
