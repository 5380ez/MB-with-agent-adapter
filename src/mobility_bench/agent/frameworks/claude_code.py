"""Claude Code CLI as a MobilityBench agent framework.

A parallel implementation to :mod:`codex`: the external ``claude`` CLI is the
agent-under-test, pointed at an Anthropic-Messages-compatible endpoint. The
deterministic sandbox tools are exposed through the same MCP server
(``mcp_server/sandbox_tools.py``) that codex uses.

Claude Code speaks the Anthropic protocol, so the model endpoint must accept
``POST {base_url}/v1/messages``. vLLM serves this route natively (alongside the
OpenAI ``/v1/chat/completions`` and ``/v1/responses`` routes), so the same
Qwen3-4B endpoint used by the other harnesses works here with no proxy. The
served model's total window also bounds claude's requested output budget, so
``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` is set from the harness config.
Everything else is declared in ``configs/harness/default.yaml``.

The runner contract matches the other harnesses:

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
    parse_jsonl,
    resolve_bin,
    standard_result,
    strip_mcp_prefix,
)
from mobility_bench.agent.frameworks.base import BaseFramework
from mobility_bench.agent.frameworks.harness import harness_config

CLAUDE_BIN = resolve_bin("CLAUDE_BIN", "claude")

# Backwards-compatible alias: claude reports MCP tools fully qualified.
_strip_mcp_prefix = strip_mcp_prefix


class _ClaudeGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "ClaudeCodeFramework"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return self._fw.run_claude(initial_state)


class ClaudeCodeFramework(BaseFramework):
    """Framework that drives the claude CLI as the agent under test."""

    @property
    def name(self) -> str:
        return "claude_code"

    def build_graph(self) -> Any:
        return _ClaudeGraph(self)

    def prepare_initial_state(self, query: str, context: str | None = None, **kwargs) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    # ----------------------------------------------------------------- claude
    def _model_name(self) -> str:
        if self.model_config and self.model_config.roles:
            return self.model_config.roles.get("planner") or self.model_config.name
        return self.model_config.name if self.model_config else "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or ""
        url = url.rstrip("/")
        # Model configs store an OpenAI-style base (e.g. `http://host:8080/v1`),
        # but claude appends its own `/v1/messages`, so a trailing `/v1` would
        # become `/v1/v1/messages`. Strip it for the Anthropic wire protocol.
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        return url

    def _api_key(self) -> str:
        return (self.model_config.api_key if self.model_config else "") or "dummy"

    def _settings(self) -> str:
        hc = harness_config("claude_code")
        model = self._model_name()
        env = {
            "ANTHROPIC_BASE_URL": self._base_url(),
            # API_KEY (not AUTH_TOKEN) is what claude accepts against a plain
            # Anthropic-compatible endpoint; AUTH_TOKEN alone yielded
            # "403 Request not allowed" against the local vLLM gateway.
            "ANTHROPIC_API_KEY": self._api_key(),
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_CONTEXT_WINDOW": str(hc.context_window),
            # Without this claude asks the endpoint for its 32000-token default
            # output budget, which the served model rejects with
            # "maximum context length is 32768 tokens ... requested 32000".
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(hc.max_output_tokens),
            "DISABLE_AUTOUPDATER": "1",
        }
        return json.dumps({"env": env, "hasCompletedOnboarding": True}, ensure_ascii=False)

    def _mcp_config(self) -> str:
        return json.dumps({
            "mcpServers": {
                "sandbox": {
                    "command": str(VENV_PY),
                    "args": [str(MCP_SERVER)],
                }
            }
        }, ensure_ascii=False)

    def run_claude(self, state: dict) -> dict:
        hc = harness_config("claude_code")
        prompt = build_prompt(state)

        with tempfile.TemporaryDirectory() as home:
            home_path = Path(home)
            (home_path / "settings.json").write_text(self._settings(), encoding="utf-8")
            mcp_path = home_path / "mcp.json"
            mcp_path.write_text(self._mcp_config(), encoding="utf-8")

            env = {**os.environ, "CLAUDE_CONFIG_DIR": home}
            cmd = [
                CLAUDE_BIN, "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--mcp-config", str(mcp_path),
            ]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, timeout=hc.timeout
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                raw = e.stdout or ""
                stdout = raw if isinstance(raw, str) else (raw or b"").decode()
                stderr = f"claude -p timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started

        state["_claude"] = {
            "events": parse_jsonl(stdout),
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
        }
        return state

    # ---------------------------------------------------------------- mapping
    def extract_result(self, final_state: dict) -> dict[str, Any]:
        claude = final_state.get("_claude", {})
        events = claude.get("events", [])

        agent_messages: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict] = []
        usage: dict = {}

        for e in events:
            etype = e.get("type")
            if etype == "assistant":
                message = e.get("message", {})
                if not usage and message.get("usage"):
                    usage = message.get("usage") or {}
                for block in message.get("content", []) or []:
                    btype = block.get("type")
                    if btype == "text":
                        agent_messages.append(block.get("text", ""))
                    elif btype == "thinking":
                        reasoning.append(block.get("thinking", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "name": _strip_mcp_prefix(block.get("name", "")),
                            "args": block.get("input", {}),
                            "result": None,
                        })
            elif etype == "user":
                # Tool results come back as user messages containing tool_result
                # blocks; attach each payload to its originating tool call.
                message = e.get("message", {})
                for block in message.get("content", []) or []:
                    if block.get("type") != "tool_result":
                        continue
                    tid = block.get("tool_use_id", "")
                    for tc in tool_calls:
                        if tc["id"] == tid:
                            tc["result"] = block.get("content")
                            break
            elif etype == "result":
                if e.get("usage"):
                    usage = e.get("usage") or usage
                if e.get("result"):
                    agent_messages.append(e["result"])

        final_message = agent_messages[-1] if agent_messages else ""
        return standard_result(
            final_message=final_message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            query=final_state.get("query", ""),
        )
