"""The package as a KohakuTerrarium user meets it: install, pick a preset, chat.

Only the SDK client is scripted; KT's installer, profile resolution, agent
loop, tool execution, model switching, doctor and uninstall all run for real.
"""

from pathlib import Path

import pytest
import yaml

import kohakuterrarium as kt
from kohakuterrarium.cli.doctor import doctor_cli
from kohakuterrarium.core.agent import Agent
from kohakuterrarium.core.config.types import AgentConfig, InputConfig, OutputConfig
from kohakuterrarium.llm.profiles.backends import load_backends, load_yaml_store
from kohakuterrarium.llm.profiles.resolve import resolve_controller_llm
from kohakuterrarium.packages.install import install_package, uninstall_package
from kohakuterrarium.studio.identity.api_keys import list_keys_payload

from kt_claude_code import provider as provider_mod

from tests.conftest import ScriptedClients, message, result, text_events, tool_events

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

    fake = ScriptedClients(
        message(tool_events("toolu_1", "secret_code", {"city": "Kyoto"}, 0), "tool_use")
        + [result()],
        message(text_events("The code is KOHAKU-42.")) + [result()],
        message(text_events("Same process.")) + [result()],
        message(text_events("Still here.")) + [result()],
        message(text_events("Deep.")) + [result()],
    )
    monkeypatch.setattr(provider_mod, "ClaudeSDKClient", fake)

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
        # The tool step and the answer ran on one CLI process: KT's history
        # continued exactly what that process holds.
        [process] = fake.clients
        assert set(process.options.mcp_servers) == {"kt"}
        assert "Answer briefly." in process.startup["system_text"]
        assert process.options.model == "haiku"
        sent = fake.queries[1]["prompt"]
        results = [
            b for b in sent[0]["message"]["content"] if b["type"] == "tool_result"
        ]
        assert [r["content"] for r in results] == ["code for Kyoto is KOHAKU-42"]

        # The next user turn on the same model reuses it too.
        assert (await agent.run("Same one?")).text.strip() == "Same process."
        assert len(fake.clients) == 1
        assert fake.queries[2] == {"client": 0, "prompt": "Same one?"}

        assert agent.switch_model("claude-code/claude-code-sonnet") == (
            "claude-code/claude-code-sonnet"
        )
        second = await agent.run("Are you there?")
        assert second.text.strip() == "Still here."
        # Another model needs another process, started on resumed history.
        sonnet = fake.clients[1]
        assert sonnet.options.model == "sonnet"
        assert len(sonnet.startup["transcript"]) >= 3
        # The default effort comes from the preset.
        assert sonnet.options.effort == "high"

        # A pinned version with variations: effort reaches the CLI, the
        # context option sets where KT compacts.
        selector = "claude-code/claude-code-opus-5.5@reasoning=max,context=200k"
        agent.switch_model(selector)
        assert (await agent.run("Think hard.")).text.strip() == "Deep."
        assert fake.clients[2].options.model == "claude-opus-5-5"
        assert fake.clients[2].options.effort == "max"
        varied = resolve_controller_llm({}, llm=selector)
        assert (varied.max_context, varied.reasoning_effort) == (200_000, "max")
    finally:
        await agent.stop()
    # Stopping the agent stops every CLI process it left running.
    assert [c.disconnected for c in fake.clients] == [True, True, True]

    # Nothing package-provided leaked into the user's provider file.
    assert "claude-code" not in (load_yaml_store().get("backends") or {})

    monkeypatch.setattr(provider_mod, "login_status", lambda: "signed in via test")
    assert doctor_cli(llm="claude-code/claude-code-haiku") == 0
    assert "backend status (claude-code)" in capsys.readouterr().out

    assert uninstall_package("kt-claude-code") is True
    assert "claude-code" not in load_backends()
    with pytest.raises(ValueError, match="No LLM model configured"):
        await Agent.build(config, llm="claude-code/claude-code-haiku", io="none")
