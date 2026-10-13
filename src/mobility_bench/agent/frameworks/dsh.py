"""DeepSeek Harness (dsh) as a MobilityBench agent framework.

Two registrations share one implementation:

* ``dsh``       -> the one-shot ``headless`` profile (the full dsh-base tree)
* ``dsh_mini``  -> the ``sdk-minimal`` profile (the paper's minimal arm: a
                   standalone tree without ``dsh-base``, driven over the SDK
                   JSON-RPC protocol rather than the one-shot ``--json`` mode)

dsh is a profile-booting launcher::

    dsh [--profile] <name> [launcher-options] [app-args...]

The full arm runs the shipped ``headless`` one-shot app — ``dsh --profile
headless --json "<task>"`` — which emits newline-delimited run events
(``session``/``status``/``text``/``thinking``/``tool_call``/``tool_result``/
``final``). The minimal arm speaks the SDK wire protocol over stdio
(``initialize``/``session/prompt``/``shutdown`` plus ``session.event`` and
``session.status`` notifications). Both are configured through ``--patch``
overlays applied *after* the profile layer, so nothing here hard-codes the
endpoint or the tool wiring.
"""

import json
import os
import select
import subprocess
import tempfile
import threading
import time
import uuid
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

# sdk-minimal is dsh's standalone minimal profile (no dsh-base). It is driven
# over the SDK JSON-RPC protocol instead of the one-shot ``--json`` mode.
_SDK_PROFILE = "sdk-minimal"


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
        """Build the ``--patch`` overlay for the full ``headless`` arm.

        ``llm-pi-ai`` gains an extra OpenAI-compatible route and
        ``agent-default-model`` is repointed at it, so dsh's default model
        selection resolves the local endpoint. The sandbox MCP client is
        inserted whenever the harness declares ``mcp_client: true``.
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

    def _is_sdk(self) -> bool:
        return harness_config(self.framework_name).profile == _SDK_PROFILE

    def _sdk_patch_overlay(self) -> str:
        """Build the ``--patch`` overlay for the ``sdk-minimal`` (minimal) tree.

        ``sdk-minimal`` is a standalone tree without ``dsh-base``, so the model
        adapter and the MCP client must be *inserted* (there is no ``id`` to
        patch), and the model route is chosen at init time through the JSON-RPC
        ``initialize`` request (there is no ``agent-default-model`` here).
        """
        max_tokens = min(8192, max(1024, self._context_window() // 4))
        patch = [
            {"insert": [{
                "id": "llm-pi-ai",
                "name": "@deepseek-ai/dsh-llm-pi-ai",
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
            }]},
            {"insert": [{
                "id": "mcp-client-sandbox",
                "name": "@deepseek-ai/dsh-mcp-client",
                "config": {
                    "transport": "stdio",
                    "serverName": "sandbox",
                    "command": str(VENV_PY),
                    "args": [str(MCP_SERVER)],
                    "failOnStartupError": False,
                },
            }]},
        ]
        return json.dumps(patch, ensure_ascii=False)

    def _run_sdk(self, state: dict) -> dict:
        """Drive the ``sdk-minimal`` profile over the JSON-RPC protocol.

        The wire is newline-delimited JSON-RPC 2.0 over stdio:
        ``initialize`` (provider/model/maxTokens), ``session/prompt`` (enqueue
        the task), then ``session.event`` notifications stream the trajectory
        until ``session.status = idle``, then ``shutdown``.
        """
        hc = harness_config(self.framework_name)
        prompt = build_prompt(state)
        max_tokens = min(8192, max(1024, self._context_window() // 4))

        with tempfile.TemporaryDirectory() as home:
            patch_path = Path(home) / "mobilitybench.patch.json"
            patch_path.write_text(self._sdk_patch_overlay(), encoding="utf-8")
            env = {**os.environ, "DSH_HOME": home, _API_KEY_ENV: self._api_key()}
            cmd = [DSH_BIN, "--profile", _SDK_PROFILE, "--patch", str(patch_path)]

            started = time.time()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
            except OSError as e:
                state["_dsh"] = {
                    "sdk": True, "frames": [], "stderr": str(e),
                    "exit": -1, "elapsed": 0.0, "timed_out": False,
                }
                return state

            # Drain stderr on a side thread so a chatty agent cannot deadlock
            # the stdout read loop (mirrors subprocess.run -> communicate()).
            stderr_chunks: list[str] = []
            def _drain() -> None:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        return
                    stderr_chunks.append(chunk)
            drain = threading.Thread(target=_drain, daemon=True)
            drain.start()

            frames: list[dict] = []
            sent_prompt = False
            idle = False
            timed_out = False

            def request(method: str, params: dict, mid: str) -> None:
                proc.stdin.write(json.dumps(
                    {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}
                ) + "\n")
                proc.stdin.flush()

            try:
                request("initialize", {
                    "cwd": os.getcwd(),
                    "provider": _ROUTE,
                    "model": self._model_name(),
                    "maxTokens": max_tokens,
                }, "mb-init")

                deadline = started + hc.timeout
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        timed_out = True
                        break
                    ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 30.0))
                    if not ready:
                        continue
                    line = proc.stdout.readline()
                    if line == "":
                        break  # EOF
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    frames.append(frame)

                    if frame.get("id") == "mb-init":
                        if frame.get("error") is not None:
                            break  # unknown provider/model -> stop early
                        if not sent_prompt:
                            request("session/prompt", {
                                "sessionId": f"mbench-{uuid.uuid4().hex[:12]}",
                                "contentBlocks": [{"type": "text", "text": prompt}],
                            }, "mb-prompt")
                            sent_prompt = True
                    elif frame.get("method") == "session.status":
                        if frame.get("params", {}).get("status") == "idle":
                            idle = True
                            break
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.write(json.dumps(
                        {"jsonrpc": "2.0", "id": "mb-shutdown", "method": "shutdown", "params": {}}
                    ) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                time.sleep(0.3)
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                drain.join(timeout=10)
                elapsed = time.time() - started
                try:
                    code = proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = -1

        state["_dsh"] = {
            "sdk": True,
            "frames": frames,
            "stderr": "".join(stderr_chunks)[-4000:],
            "exit": code,
            "elapsed": elapsed,
            "timed_out": timed_out,
            "idle": idle,
        }
        return state

    def _extract_sdk(self, frames: list[dict]) -> tuple[str, list[str], list[dict], dict]:
        """Map the ``session.event`` stream back onto the runner contract."""
        messages: list[str] = []
        tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0

        for frame in frames:
            if frame.get("method") != "session.event":
                continue
            event = frame.get("params", {}).get("event", {})
            etype = event.get("type")
            data = event.get("data", {}) or {}

            if etype == "assistant/message":
                usage = data.get("usage") or {}
                input_tokens += usage.get("inputTokens", 0) or 0
                output_tokens += usage.get("outputTokens", 0) or 0
                for block in (data.get("message", {}) or {}).get("content", []) or []:
                    if block.get("type") == "text":
                        messages.append(block.get("text", ""))
            elif etype == "tool/call":
                args = data.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({
                    "id": data.get("callId", ""),
                    "name": strip_mcp_prefix(data.get("name", "")),
                    "args": args or {},
                    "result": None,
                })
            elif etype == "tool/result":
                message = data.get("message", {}) or {}
                call_id = message.get("toolCallId", "")
                texts = [
                    b.get("text", "")
                    for b in (message.get("content", []) or [])
                    if b.get("type") == "text"
                ]
                result_raw = "\n".join(texts).strip()
                try:
                    result = json.loads(result_raw) if result_raw else None
                except json.JSONDecodeError:
                    result = result_raw or None
                for tc in tool_calls:
                    if tc["id"] == call_id:
                        tc["result"] = result
                        break

        final_message = messages[-1].strip() if messages else ""
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return final_message, [], tool_calls, usage

    def run_dsh(self, state: dict) -> dict:
        if self._is_sdk():
            return self._run_sdk(state)

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
        if dsh.get("sdk"):
            final_message, reasoning, tool_calls, usage = self._extract_sdk(dsh.get("frames", []))
            return standard_result(
                final_message=final_message,
                reasoning=reasoning,
                tool_calls=tool_calls,
                usage=usage,
                query=final_state.get("query", ""),
            )

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
    """Full dsh harness (one-shot ``headless`` profile over dsh-base)."""

    framework_name = "dsh"


class DshMiniFramework(_DshBase):
    """Minimal dsh harness (``sdk-minimal`` profile, driven over the SDK protocol)."""

    framework_name = "dsh_mini"
