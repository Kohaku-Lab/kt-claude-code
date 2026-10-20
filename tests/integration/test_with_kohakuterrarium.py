"""The package as a KohakuTerrarium user meets it: install, pick a preset, chat.

Only the SDK ``query`` function is scripted; KT's installer, profile
resolution, agent loop, tool execution, model switching, doctor and
uninstall all run for real.
"""

from pathlib import Path

import pytest
import yaml

import kohakuterrarium as kt
from kohakuterrarium.cli.doctor import doctor_cli
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config_types import AgentConfig, InputConfig, OutputConfig
from kohakuterrarium.llm.backends import load_backends, load_yaml_store
from kohakuterrarium.llm.profiles import resolve_controller_llm
from kohakuterrarium.packages.install import install_package, uninstall_package
from kohakuterrarium.studio.identity.api_keys import list_keys_payload

from kt_claude_code import provider as provider_mod

from tests.conftest import ScriptedQuery, message, result, text_events, tool_events

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


async def test_install_select_chat_with_tools_switch_and_uninstall(
    tmp_path, monkeypatch, capsys
):
    assert (
        install_package(str(PACKAGE_ROOT), editable=True, deps="never")
        == "kt-claude-code"
    )

    # The backend type and its presets come from the installed manifest.
    assert load_backends()["claude-code"].backend_type == "claude-code"
    profile = resolve_controller_llm({}, llm="claude-code/claude-code-haiku")
    assert (profile.backend_type, profile.model) == ("claude-code", "haiku")
    assert (profile.max_context, profile.reasoning_effort) == (1_000_000, "high")
    # Every shipped preset resolves through KT to this backend.
    for preset in yaml.safe_load((PACKAGE_ROOT / "kohaku.yaml").read_text())[
        "llm_presets"
    ]:
        resolved = resolve_controller_llm({}, llm=f"claude-code/{preset['name']}")
        assert resolved.backend_type == "claude-code"
        assert resolved.model == preset["model"]
    # KT never offers an API-key slot for it.
    assert "claude-code" not in {row["provider"] for row in list_keys_payload()}

    fake = ScriptedQuery(
        message(tool_events("toolu_1", "secret_code", {"city": "Kyoto"}, 0), "tool_use")
        + [result()],
        message(text_events("The code is KOHAKU-42.")) + [result()],
        message(text_events("Still here.")) + [result()],
        message(text_events("Deep.")) + [result()],
    )
    monkeypatch.setattr(provider_mod, "query", fake)

    seen: list[str] = []

    @kt.tool
    def secret_code(city: str) -> str:
        """Return the secret code for a city."""
        seen.append(city)
        return f"code for {city} is KOHAKU-42"

    config = AgentConfig(
        name="cc",
        system_prompt="Answer briefly.",
        tool_format="native",
        agent_path=tmp_path,
        input=InputConfig(type="none"),
        output=OutputConfig(type="none"),
        tools=[],
    )
    agent = await Agent.build(
        config, llm="claude-code/claude-code-haiku", io="none", tools=[secret_code]
    )
    await agent.start()
    try:
        assert type(agent.llm).__name__ == "ClaudeCodeProvider"
        first = await agent.run("Secret code for Kyoto?")
        # KT ran its own tool; the CLI only proposed the call.
        assert seen == ["Kyoto"]
        assert "KOHAKU-42" in first.text
        tool_step, answer_step = fake.calls[0], fake.calls[1]
        assert set(tool_step["options"].mcp_servers) == {"kt"}
        assert "Answer briefly." in tool_step["system_text"]
        assert tool_step["options"].model == "haiku"
        results = [
            block
            for entry in answer_step["transcript"]
            for block in entry["message"]["content"]
            if block.get("type") == "tool_result"
        ]
        assert [r["content"] for r in results] == ["code for Kyoto is KOHAKU-42"]

        assert agent.switch_model("claude-code/claude-code-sonnet") == (
            "claude-code/claude-code-sonnet"
        )
        second = await agent.run("Are you there?")
        assert second.text.strip() == "Still here."
        assert fake.calls[2]["options"].model == "sonnet"
        # Earlier turns now arrive as resumed history.
        assert len(fake.calls[2]["transcript"]) >= 3
        # The default effort comes from the preset.
        assert fake.calls[2]["options"].effort == "high"

        # A pinned version with variations: effort reaches the CLI, the
        # context option sets where KT compacts.
        selector = "claude-code/claude-code-opus-5.5@reasoning=max,context=200k"
        agent.switch_model(selector)
        assert (await agent.run("Think hard.")).text.strip() == "Deep."
        assert fake.calls[3]["options"].model == "claude-opus-5-5"
        assert fake.calls[3]["options"].effort == "max"
        varied = resolve_controller_llm({}, llm=selector)
        assert (varied.max_context, varied.reasoning_effort) == (200_000, "max")
    finally:
        await agent.stop()

    # Nothing package-provided leaked into the user's provider file.
    assert "claude-code" not in (load_yaml_store().get("backends") or {})

    monkeypatch.setattr(provider_mod, "login_status", lambda: "signed in via test")
    assert doctor_cli(llm="claude-code/claude-code-haiku") == 0
    assert "backend status (claude-code)" in capsys.readouterr().out

    assert uninstall_package("kt-claude-code") is True
    assert "claude-code" not in load_backends()
    with pytest.raises(ValueError, match="No LLM model configured"):
        await Agent.build(config, llm="claude-code/claude-code-haiku", io="none")
