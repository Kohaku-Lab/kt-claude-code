"""Unit tests for ``kt_claude_code.options`` and ``kt_claude_code.tool_bridge``."""

from pathlib import Path

from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code.options import (
    DEFAULT_THINKING,
    QUIET_ENV,
    StepSettings,
    build_options,
)
from kt_claude_code.tool_bridge import NOT_EXECUTED, _never_run, defer_tool_call

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}


def _build(settings=None, tools=None, session_id=None, prompt=None):
    return build_options(
        settings or StepSettings(model="haiku"),
        cwd="/w",
        system_prompt_path=prompt,
        session_id=session_id,
        tools=tools,
    )


def test_no_builtin_tools_no_settings_and_quiet_env():
    opts = _build()
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.env == QUIET_ENV
    assert "CLAUDE_CONFIG_DIR" not in opts.env
    assert opts.include_partial_messages is True
    assert opts.cwd == "/w"
    assert opts.model == "haiku"
    assert opts.thinking == DEFAULT_THINKING
    assert opts.resume is None
    assert opts.mcp_servers == {} and opts.hooks is None
    assert opts.system_prompt == ""


def test_system_prompt_file_resume_effort_and_cli_path():
    opts = _build(
        StepSettings(
            model="opus",
            effort="high",
            cli_path="~/bin/claude",
            thinking={"type": "disabled"},
        ),
        session_id="sid",
        prompt=Path("/w/system.md"),
    )
    assert opts.system_prompt == {"type": "file", "path": "/w/system.md"}
    assert opts.resume == "sid"
    assert opts.effort == "high"
    assert opts.thinking == {"type": "disabled"}
    assert not opts.cli_path.startswith("~")


def test_unknown_effort_is_not_passed():
    assert _build(StepSettings(model="m", effort="turbo")).effort is None


def test_tools_become_a_deferred_mcp_server():
    opts = _build(tools=[ToolSchema("weather", "Weather.", SCHEMA)])
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
