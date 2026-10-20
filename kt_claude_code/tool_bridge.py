"""Expose KT tool schemas to the model without letting the CLI run them.

The schemas reach the model through an in-process MCP server; a
``PreToolUse`` hook defers every call, so the step ends with the call
returned to KohakuTerrarium.
"""

from typing import Any

from claude_agent_sdk import HookMatcher, create_sdk_mcp_server, tool

from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code.transcript import TOOL_PREFIX

SERVER_NAME = "kt"
NOT_EXECUTED = "Tool execution is handled by KohakuTerrarium, not by Claude Code."


async def _never_run(_args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": NOT_EXECUTED}], "is_error": True}


def build_mcp_server(tools: list[ToolSchema]) -> Any:
    """An SDK MCP server advertising ``tools`` with their exact schemas."""
    return create_sdk_mcp_server(
        name=SERVER_NAME,
        tools=[
            tool(schema.name, schema.description, schema.parameters)(_never_run)
            for schema in tools
        ],
    )


async def defer_tool_call(
    _input: dict[str, Any], _tool_use_id: str | None, _context: Any
) -> dict[str, Any]:
    """``PreToolUse`` callback that hands every KT tool call back to KT."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "defer",
        }
    }


def defer_hooks() -> dict[str, list[HookMatcher]]:
    """Hook config deferring every call to a KT tool."""
    return {
        "PreToolUse": [HookMatcher(matcher=f"{TOOL_PREFIX}.*", hooks=[defer_tool_call])]
    }
