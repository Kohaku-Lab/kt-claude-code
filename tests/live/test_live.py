"""Contract tests against the real, signed-in Claude Code CLI.

Opt in with ``KT_CLAUDE_LIVE=1``. They spend a little of the user's own plan
usage (haiku, a few short steps) and are the alarm for CLI format changes.
"""

import base64
import os
import struct
import zlib

import pytest

import kohakuterrarium as kt
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config_types import AgentConfig, InputConfig, OutputConfig
from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code.provider import ClaudeCodeProvider
from kt_claude_code.workspace import Workspace

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("KT_CLAUDE_LIVE") != "1", reason="set KT_CLAUDE_LIVE=1"
    ),
]

MODEL = os.environ.get("KT_CLAUDE_LIVE_MODEL", "haiku")
CITY = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
WEATHER = ToolSchema("weather", "Current weather for a city.", CITY)
LOCAL_TIME = ToolSchema("local_time", "Current local time for a city.", CITY)


def _png(rgb: tuple[int, int, int]) -> str:
    raw = b"".join(b"\x00" + bytes(rgb) * 16 for _ in range(16))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture
def workspace(tmp_path):
    """A throwaway workspace whose (empty) CLI project folder is removed after."""
    ws = Workspace(tmp_path / "ws")
    yield ws
    if ws.project_dir.is_dir():
        ws.project_dir.rmdir()


@pytest.fixture
def provider(workspace):
    return ClaudeCodeProvider(model=MODEL, workspace=workspace)


async def _step(provider, messages, tools=None) -> str:
    return "".join([c async for c in provider.chat(messages, tools=tools)])


def _leftovers(provider) -> list:
    project = provider._workspace.project_dir
    files = list(project.glob("*.jsonl")) if project.exists() else []
    return files + list(provider._workspace.root.glob("system-*.md"))


async def test_plain_step_uses_kt_system_prompt_and_cleans_up(provider):
    text = await _step(
        provider,
        [
            {"role": "system", "content": "Reply with exactly: PONG-<n>."},
            {"role": "user", "content": "7"},
        ],
    )
    assert "PONG-7" in text
    assert provider.last_tool_calls == []
    assert provider.last_usage.get("completion_tokens", 0) > 0
    assert _leftovers(provider) == []


async def test_native_tool_round_trip_with_parallel_calls(provider):
    messages = [
        {"role": "system", "content": "Use tools; call independent tools together."},
        {"role": "user", "content": "Weather in Taipei and local time in Tokyo?"},
    ]
    await _step(provider, messages, tools=[WEATHER, LOCAL_TIME])
    calls = provider.last_tool_calls
    assert {c.name for c in calls} == {"weather", "local_time"}
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ],
            **provider.last_assistant_extra_fields,
        }
    )
    for c in calls:
        result = "29C, code word ZEBRA" if c.name == "weather" else "21:05"
        messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
    text = await _step(provider, messages, tools=[WEATHER, LOCAL_TIME])
    assert provider.last_tool_calls == []
    assert "21:05" in text
    assert _leftovers(provider) == []


async def test_foreign_tool_ids_and_image_results(provider):
    messages = [
        {"role": "user", "content": "Fetch the swatch image, then name its colour."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "job_7:abc",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "job_7:abc",
            "content": [
                {"type": "image_url", "image_url": {"url": _png((0, 0, 255))}},
            ],
        },
    ]
    text = await _step(provider, messages, tools=[WEATHER])
    assert "blue" in text.lower()


async def test_kt_agent_runs_its_own_tool(tmp_path, workspace):
    seen: list[str] = []

    @kt.tool
    def secret_code(city: str) -> str:
        """Return the secret code for a city."""
        seen.append(city)
        return f"code for {city} is KOHAKU-42"

    config = AgentConfig(
        name="cc-live",
        system_prompt="Answer briefly. Use tools when asked.",
        tool_format="native",
        agent_path=tmp_path,
        input=InputConfig(type="none"),
        output=OutputConfig(type="none"),
        tools=[],
    )
    agent = await Agent.build(
        config,
        llm=ClaudeCodeProvider(model=MODEL, workspace=workspace),
        io="none",
        tools=[secret_code],
    )
    await agent.start()
    try:
        result = await agent.run("Get the secret code for Kyoto and tell me it.")
    finally:
        await agent.stop()
    assert seen == ["Kyoto"]
    assert "KOHAKU-42" in result.text


async def test_not_signed_in_is_a_typed_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty-config"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    provider = ClaudeCodeProvider(model=MODEL, workspace=Workspace(tmp_path / "ws"))
    with pytest.raises(LLMNotConfiguredError, match="/login"):
        await _step(provider, [{"role": "user", "content": "hi"}])


def test_doctor_reports_sign_in():
    assert "signed in via" in ClaudeCodeProvider.doctor()
