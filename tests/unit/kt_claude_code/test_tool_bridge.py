"""Unit tests for ``kt_claude_code.tool_bridge``."""

from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code.options import StepSettings, build_options
from kt_claude_code.tool_bridge import NOT_EXECUTED, _never_run, defer_tool_call

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}


def test_tools_become_a_deferred_mcp_server():
    opts = build_options(
        StepSettings(model="haiku"),
        cwd="/w",
        system_prompt_path=None,
        session_id=None,
        tools=[ToolSchema("weather", "Weather.", SCHEMA)],
    )
    server = opts.mcp_servers["kt"]
    assert server["type"] == "sdk" and server["name"] == "kt"
    [matcher] = opts.hooks["PreToolUse"]
    assert matcher.matcher == "mcp__kt__.*"


async def test_the_hook_defers_and_the_handler_refuses():
    decision = await defer_tool_call({}, "toolu_1", None)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "defer"
    reply = await _never_run({"city": "x"})
    assert reply["is_error"] is True
    assert reply["content"][0]["text"] == NOT_EXECUTED
