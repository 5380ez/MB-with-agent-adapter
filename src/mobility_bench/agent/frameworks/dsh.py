"""DeepSeek Harness (dsh) as a MobilityBench agent framework.

Two registrations share one implementation:

* ``dsh``       -> the one-shot ``headless`` profile with the sandbox MCP client
* ``dsh_mini``  -> the same profile with a halved model context window
                   (the paper's lightweight arm: a smaller per-run token budget)

dsh is a profile-booting launcher::

    dsh [--profile] <name> [launcher-options] [app-args...]

The one-shot app is the shipped ``headless`` profile::

    dsh --profile headless --json "<task>"

Its model route and MCP servers are configured through ``--patch`` overlays that
are applied *after* the profile layer, so nothing in this adapter hard-codes the
endpoint or the tool wiring. ``--json`` emits newline-delimited run events
(``session`` / ``status`` / ``text`` / ``thinking`` / ``tool_call`` /
``tool_result`` / ``final``); stderr carries ``dsh:`` diagnostics and reasoning.
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

DSH_BIN = resolve_bin("DSH_BIN", "dsh")

# Route name dsh's default model is repointed at via the --patch overlay.
_ROUTE = "vllm"

# Env var the overlay points dsh at for the route credential.
_API_KEY_ENV = "MOBILITYBENCH_API_KEY"


class _DshGraph:
    """Minimal async graph exposing ``ainvoke`` for the batch runner."""

    def __init__(self, framework: "_DshBase"):
        self._fw = framework

    async def ainvoke(self, initial_state: dict) -> dict:
        return self._fw.run_dsh(initial_state)


class _DshBase(BaseFramework):
    """Shared driver for the dsh CLI across profiles and MCP modes."""

    framework_name = "dsh"

    @property
    def name(self) -> str:
        return self.framework_name

    def build_graph(self) -> Any:
        return _DshGraph(self)

    def prepare_initial_state(
        self, query: str, context: str | None = None, **kwargs
    ) -> dict[str, Any]:
        state = {"query": query, "context": context}
        state.update(kwargs)
        return state

    def _model_name(self) -> str:
        if self.model_config:
            if self.model_config.roles:
                return self.model_config.roles.get("planner") or self.model_config.name
            return self.model_config.name
        return "qwen3-4b"

    def _base_url(self) -> str:
        url = (self.model_config.base_url if self.model_config else "") or "http://localhost:8080/v1"
        return url.rstrip("/")

    def _api_key(self) -> str:
        return (self.model_config.api_key if self.model_config else "") or "dummy"

    def _context_window(self) -> int:
        return harness_config(self.framework_name).context_window

    def _patch_overlay(self) -> str:
        """Build the ``--patch`` overlay routing dsh at the configured endpoint.

        Only launcher-level plugins are touched: ``llm-pi-ai`` gains an extra
        OpenAI-compatible route and ``agent-default-model`` is repointed at it.
        The model context window (and the output-token budget derived from it)
        is read from the harness config, so ``dsh`` and ``dsh_mini`` differ in
        budget here. The sandbox MCP server is inserted whenever the harness
        declares ``mcp_client: true``.
        """
        hc = harness_config(self.framework_name)
        # Leave headroom for the response inside the model's context window.
        max_tokens = min(8192, max(1024, self._context_window() // 4))

        patch: list[dict] = [
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        _ROUTE: {
                            "displayName": "MobilityBench endpoint",
                            "api": "openai-completions",
                            "baseURL": self._base_url(),
                            "apiKeyEnv": _API_KEY_ENV,
                            "models": [{
                                "id": self._model_name(),
                                "contextWindow": self._context_window(),
                                "maxTokens": max_tokens,
                            }],
                        },
                    },
                },
            },
            {
                "id": "agent-default-model",
                "config": {"provider": _ROUTE, "model": self._model_name()},
            },
        ]

        if hc.mcp_client:
            patch.append({
                "insert": [{
                    "id": "mcp-client-sandbox",
                    "name": "@deepseek-ai/dsh-mcp-client",
                    "config": {
                        "transport": "stdio",
                        "serverName": "sandbox",
                        "command": str(VENV_PY),
                        "args": [str(MCP_SERVER)],
                        "failOnStartupError": False,
                    },
                }],
            })
        return json.dumps(patch, ensure_ascii=False)

    def run_dsh(self, state: dict) -> dict:
        hc = harness_config(self.framework_name)
        prompt = build_prompt(state)

        with tempfile.TemporaryDirectory() as home:
            patch_path = Path(home) / "mobilitybench.patch.json"
            patch_path.write_text(self._patch_overlay(), encoding="utf-8")

            env = {
                **os.environ,
                "DSH_HOME": home,
                _API_KEY_ENV: self._api_key(),
            }
            cmd = [DSH_BIN, "--profile", hc.profile or "headless"]
            cmd += ["--patch", str(patch_path)]
            cmd += ["--json", prompt]

            started = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, timeout=hc.timeout
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                raw = e.stdout or ""
                stdout = raw if isinstance(raw, str) else raw.decode()
                stderr = f"dsh timed out after {hc.timeout}s"
                code = -1
            elapsed = time.time() - started

        state["_dsh"] = {
            "events": parse_jsonl(stdout),
            "stdout": stdout,
            "stderr": stderr[-4000:],
            "exit": code,
            "elapsed": elapsed,
        }
        return state

    @staticmethod
    def _merge_usage(acc: dict, usage: dict) -> dict:
        """Each ``status`` line reports that step's usage; sum them across steps."""
        return {
            "input_tokens": (acc.get("input_tokens", 0) or 0) + (usage.get("inputTokens", 0) or 0),
            "output_tokens": (acc.get("output_tokens", 0) or 0) + (usage.get("outputTokens", 0) or 0),
        }

    def extract_result(self, final_state: dict) -> dict[str, Any]:
        dsh = final_state.get("_dsh", {})
        events = dsh.get("events", [])

        agent_messages: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict] = []
        usage: dict = {}

        for e in events:
            etype = e.get("type")
            if etype == "text":
                agent_messages.append(e.get("text", ""))
            elif etype == "thinking":
                reasoning.append(e.get("text", ""))
            elif etype == "tool_call":
                tool_calls.append({
                    "id": e.get("callId", ""),
                    "name": strip_mcp_prefix(e.get("tool", "")),
                    "args": e.get("input", {}),
                    "result": None,
                })
            elif etype == "tool_result":
                call_id = e.get("callId", "")
                for tc in tool_calls:
                    if tc["id"] == call_id:
                        tc["result"] = e.get("result")
                        break
            elif etype == "status" and e.get("usage"):
                usage = self._merge_usage(usage, e["usage"])
            elif etype == "final" and not agent_messages and e.get("text"):
                # `final` is the last non-empty assistant text; use it only as a
                # fallback since it may be truncated or restate earlier turns.
                agent_messages.append(e.get("text", ""))

        # Plain-text fallback: the CLI may ignore --json (older builds).
        if not events:
            text = (dsh.get("stdout") or "").strip()
            if text:
                agent_messages.append(text)

        final_message = agent_messages[-1] if agent_messages else ""
        return standard_result(
            final_message=final_message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            query=final_state.get("query", ""),
        )


class DshFramework(_DshBase):
    """dsh with the full model context window."""

    framework_name = "dsh"


class DshMiniFramework(_DshBase):
    """dsh launched with a halved model context window (lightweight arm)."""

    framework_name = "dsh_mini"
