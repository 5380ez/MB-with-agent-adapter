"""Harness configuration for CLI-driven agent-under-test frameworks.

Mirrors the paper's ``experiment.yaml``: every harness declares its binary,
protocol, effort/limits and timeouts in ``configs/harness/default.yaml`` instead
of hard-coding them in each adapter. Adapters call :func:`harness_config` to
read their stanza, falling back to conservative defaults when the file or the
stanza is missing (so tests and partial checkouts still work).
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Conservative defaults keyed by the framework name registered in the factory.
_DEFAULTS: dict[str, dict] = {
    "codex": {"bin": "codex", "wire_api": "responses", "context_window": 32768},
    "claude_code": {"bin": "claude", "protocol": "anthropic"},
    # dsh boots a named profile. The full arm uses the one-shot ``headless``
    # profile; the minimal arm uses the standalone ``sdk-minimal`` tree, driven
    # over the SDK JSON-RPC protocol. Both mount the sandbox MCP client.
    "dsh": {"bin": "dsh", "protocol": "openai", "profile": "headless", "mcp_client": True},
    "dsh_mini": {"bin": "dsh", "protocol": "openai", "profile": "sdk-minimal", "mcp_client": True},
    "mini_swe": {"bin": "mini", "protocol": "openai", "mcp_client": False},
    # openclaw: `agent --local --json` one-shot embedded turn, sandbox over MCP.
    "openclaw": {"bin": "openclaw", "protocol": "openai", "context_window": 32768, "mcp_client": True},
    # hermes: in-process driver over `hermes_cli.oneshot`, sandbox over MCP.
    "hermes": {"bin": "hermes", "protocol": "openai", "context_window": 32768, "mcp_client": True},
}


@dataclass
class HarnessConfig:
    """One harness's launch parameters."""

    name: str
    bin: str
    protocol: str = "openai"       # openai | anthropic
    wire_api: str = "chat"         # chat | responses (openai-protocol only)
    profile: str = ""              # dsh-style named profile
    version: str = ""              # pinned harness version, for reproducibility
    context_window: int = 32768
    max_output_tokens: int = 4096  # per-response output budget the harness requests
    timeout: int = 900
    max_steps: int = 0             # 0 = harness default
    effort: str = ""               # reasoning effort hint, when supported
    mcp_client: bool = True        # harness supports MCP tool servers
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "HarnessConfig":
        merged = {**_DEFAULTS.get(name, {}), **(data or {})}
        known = {
            "bin", "protocol", "wire_api", "profile", "version",
            "context_window", "max_output_tokens", "timeout", "max_steps",
            "effort", "mcp_client",
        }
        extra = {k: v for k, v in merged.items() if k not in known}
        return cls(
            name=name,
            bin=merged.get("bin", name),
            protocol=merged.get("protocol", "openai"),
            wire_api=merged.get("wire_api", "chat"),
            profile=merged.get("profile", ""),
            version=merged.get("version", ""),
            context_window=int(merged.get("context_window", 32768)),
            max_output_tokens=int(merged.get("max_output_tokens", 4096)),
            timeout=int(merged.get("timeout", 900)),
            max_steps=int(merged.get("max_steps", 0)),
            effort=merged.get("effort", ""),
            mcp_client=bool(merged.get("mcp_client", True)),
            extra=extra,
        )


_cache: dict[str, HarnessConfig] | None = None


def _load() -> dict[str, HarnessConfig]:
    global _cache
    if _cache is not None:
        return _cache

    path = PROJECT_ROOT / "configs" / "harness" / "default.yaml"
    data: dict = {}
    if path.exists():
        with open(path) as f:
            data = (yaml.safe_load(f) or {}).get("harnesses", {}) or {}

    _cache = {name: HarnessConfig.from_dict(name, cfg) for name, cfg in data.items()}
    return _cache


def harness_config(name: str) -> HarnessConfig:
    """Return the config for ``name``, falling back to built-in defaults."""
    cached = _load().get(name)
    if cached is not None:
        return cached
    return HarnessConfig.from_dict(name, {})
