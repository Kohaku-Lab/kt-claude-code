"""Build the ``ClaudeAgentOptions`` for one stateless model step."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code.tool_bridge import SERVER_NAME, build_mcp_server, defer_hooks

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
# Keep the CLI from adding its own memory, CLAUDE.md, attachments or connectors.
QUIET_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}


@dataclass(frozen=True)
class StepSettings:
    """Per-provider settings that stay fixed across steps."""

    model: str
    effort: str | None = None
    thinking: dict[str, Any] | None = None
    cli_path: str | None = None


def build_options(
    settings: StepSettings,
    *,
    cwd: str,
    system_prompt_path: Path | None,
    session_id: str | None,
    tools: list[ToolSchema] | None,
) -> ClaudeAgentOptions:
    """Options for one step: no built-in tools, KT tools deferred, KT prompt."""
    kwargs: dict[str, Any] = {
        "tools": [],
        "setting_sources": [],
        "strict_mcp_config": True,
        "cwd": cwd,
        "env": dict(QUIET_ENV),
        "include_partial_messages": True,
        "model": settings.model,
        "thinking": settings.thinking or DEFAULT_THINKING,
        "extra_args": {"exclude-dynamic-system-prompt-sections": None},
        "system_prompt": (
            {"type": "file", "path": str(system_prompt_path)}
            if system_prompt_path is not None
            else ""
        ),
    }
    if settings.effort in EFFORT_LEVELS:
        kwargs["effort"] = settings.effort
    if settings.cli_path:
        kwargs["cli_path"] = os.path.expanduser(settings.cli_path)
    if session_id is not None:
        kwargs["resume"] = session_id
    if tools:
        kwargs["mcp_servers"] = {SERVER_NAME: build_mcp_server(tools)}
        kwargs["hooks"] = defer_hooks()
    return ClaudeAgentOptions(**kwargs)
