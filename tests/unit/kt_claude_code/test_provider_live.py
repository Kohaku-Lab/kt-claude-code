"""Unit tests for ``kt_claude_code.provider`` keeping CLI processes alive."""

import asyncio

import pytest
from claude_agent_sdk import CLINotFoundError, SystemMessage

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.base import ToolSchema

from kt_claude_code import provider as provider_mod
from kt_claude_code.provider import ClaudeCodeProvider
from kt_claude_code.transcript import TURN_KEY

from tests.conftest import (
    ScriptedClient,
    ScriptedClients,
    message,
    rate_limited,
    result,
    text_events,
    tool_events,
)

WEATHER = ToolSchema("weather", "Weather.", {"type": "object", "properties": {}})
CLOCK = ToolSchema("clock", "Time.", {"type": "object", "properties": {}})
SYSTEM = {"role": "system", "content": "You are Kohaku."}
PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _provider(workspace, *scripts, saves=None, **kwargs):
    clients = ScriptedClients(*scripts, saves=saves, workspace=workspace)
    provider = ClaudeCodeProvider(workspace=workspace, client_factory=clients, **kwargs)
    return provider, clients


async def _step(provider, messages, tools=None):
    return "".join([c async for c in provider.chat(messages, tools=tools)])


def _text(text):
    return message(text_events(text)) + [result()]


def _calls(*ids):
    blocks = []
    for index, call_id in enumerate(ids):
        blocks += tool_events(call_id, "weather", {"city": call_id}, index)
    return message(blocks, "tool_use") + [result()]


def _assistant(provider, text=""):
    """The assistant message KT stores after a step."""
    msg = {"role": "assistant", "content": text, **provider.last_assistant_extra_fields}
    if provider.last_tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in provider.last_tool_calls
        ]
    return msg


def _results(provider, value="25C"):
    return [
        {"role": "tool", "tool_call_id": c.id, "content": value}
        for c in provider.last_tool_calls
    ]


def _leftovers(workspace):
    files = (
        list(workspace.project_dir.glob("*")) if workspace.project_dir.exists() else []
    )
    return files + list(workspace.root.glob("system-*"))


async def _settle(provider):
    pool = provider._pool(create=False)
    if pool is not None:
        await pool.wait_closed()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(provider_mod, "backoff_delay", lambda attempt, policy: 0)


class TestReuse:
    async def test_a_tool_loop_and_follow_up_run_on_one_process(self, workspace):
        provider, clients = _provider(
            workspace, _calls("toolu_1"), _text("Sunny."), _text("Bye.")
        )
        convo = [SYSTEM, {"role": "user", "content": "weather?"}]
        assert await _step(provider, convo, [WEATHER]) == ""
        convo += [_assistant(provider), *_results(provider)]
        assert await _step(provider, convo, [WEATHER]) == "Sunny."
        convo += [_assistant(provider, "Sunny."), {"role": "user", "content": "bye"}]
        assert await _step(provider, convo, [WEATHER]) == "Bye."

        [client] = clients.clients
        assert client.options.resume is None
        assert client.startup["system_text"] == "You are Kohaku."
        prompts = [q["prompt"] for q in clients.queries]
        assert prompts[0] == "weather?"
        [sent] = prompts[1]
        assert sent["message"]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "25C"}
        ]
        assert prompts[2] == "bye"
        # Parked, not closed: its files stay until it exits.
        assert not client.disconnected
        assert provider._pool().idle == 1
        await provider.close()
        assert client.disconnected
        assert _leftovers(workspace) == []

    async def test_parallel_results_go_to_the_process_in_one_message(self, workspace):
        provider, clients = _provider(
            workspace, _calls("toolu_a", "toolu_b"), _text("Both sunny.")
        )
        convo = [{"role": "user", "content": "two cities?"}]
        await _step(provider, convo, [WEATHER])
        assert len(provider.last_tool_calls) == 2
        convo += [_assistant(provider), *_results(provider)]
        assert await _step(provider, convo, [WEATHER]) == "Both sunny."
        assert len(clients.clients) == 1
        [sent] = clients.queries[1]["prompt"]
        ids = [b["tool_use_id"] for b in sent["message"]["content"]]
        assert ids == ["toolu_a", "toolu_b"]
        await provider.close()

    async def test_turns_are_recorded_from_the_live_session_file(self, workspace):
        saves = [_saved("hi", 1, "Hello"), _saved("more", 2, "Again")]
        provider, clients = _provider(
            workspace, _text("Hello"), _text("Again"), saves=saves
        )
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        first = provider.last_assistant_extra_fields[TURN_KEY]
        assert [e["uuid"] for e in first["entries"]] == ["u1", "a1", "r1"]
        convo += [_assistant(provider, "Hello"), {"role": "user", "content": "more"}]
        await _step(provider, convo)
        assert len(clients.clients) == 1
        # Only what the process saved during this step belongs to the turn.
        second = provider.last_assistant_extra_fields[TURN_KEY]
        assert second["prompt"] == "more"
        assert [e["uuid"] for e in second["entries"]] == ["u2", "a2", "r2"]
        await provider.close()
        assert _leftovers(workspace) == []


class TestFreshProcess:
    async def test_regenerating_starts_a_new_process_and_retires_the_old(
        self, workspace
    ):
        provider, clients = _provider(workspace, _text("One"), _text("Two"))
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        assert await _step(provider, convo) == "Two"
        await _settle(provider)
        old, new = clients.clients
        assert old.disconnected and not new.disconnected
        assert provider._pool().idle == 1
        await provider.close()

    async def test_an_edited_reply_resumes_a_transcript(self, workspace):
        provider, clients = _provider(workspace, _text("Hello"), _text("Fine."))
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        convo += [
            {"role": "assistant", "content": "Edited hello"},
            {"role": "user", "content": "how are you?"},
        ]
        assert await _step(provider, convo) == "Fine."
        assert len(clients.clients) == 2
        second = clients.clients[1]
        assert second.options.resume is not None
        kinds = [e["type"] for e in second.startup["transcript"]]
        assert kinds == ["user", "assistant"]
        await provider.close()
        assert _leftovers(workspace) == []

    @pytest.mark.parametrize("change", ["system", "tools", "model"])
    async def test_a_changed_setup_needs_a_new_process(self, workspace, change):
        provider, clients = _provider(workspace, _calls("toolu_1"), _text("ok"))
        convo = [SYSTEM, {"role": "user", "content": "weather?"}]
        await _step(provider, convo, [WEATHER])
        convo += [_assistant(provider), *_results(provider)]
        tools = [WEATHER]
        step_provider = provider
        if change == "system":
            convo[0] = {"role": "system", "content": "You are someone else."}
        elif change == "tools":
            tools = [WEATHER, CLOCK]
        else:
            step_provider = provider.with_model("opus")
        assert await _step(step_provider, convo, tools) == "ok"
        assert len(clients.clients) == 2
        assert clients.clients[1].options.resume is not None
        await provider.close()
        await step_provider.close()

    async def test_a_new_image_needs_a_fresh_step(self, workspace):
        provider, clients = _provider(workspace, _text("Hello"), _text("A dot."))
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        convo += [
            _assistant(provider, "Hello"),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": PIXEL}},
                ],
            },
        ]
        assert await _step(provider, convo) == "A dot."
        assert len(clients.clients) == 2
        await provider.close()

    async def test_a_dead_parked_process_falls_back_to_a_fresh_one(self, workspace):
        provider, clients = _provider(workspace, _calls("toolu_1"), _text("Sunny."))
        convo = [{"role": "user", "content": "weather?"}]
        await _step(provider, convo, [WEATHER])
        clients.clients[0].disconnected = True
        convo += [_assistant(provider), *_results(provider)]
        assert await _step(provider, convo, [WEATHER]) == "Sunny."
        fresh = clients.clients[1]
        assert fresh.options.resume is not None
        # The deferred call is answered by the fresh process's prompt.
        [sent] = clients.queries[-1]["prompt"]
        assert sent["message"]["content"][0]["tool_use_id"] == "toolu_1"
        await provider.close()

    async def test_a_parked_process_that_exited_quietly_is_replaced(self, workspace):
        provider, clients = _provider(
            workspace, _text("Hello"), ScriptedClient.END, _text("Again")
        )
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        convo += [_assistant(provider, "Hello"), {"role": "user", "content": "more"}]
        assert await _step(provider, convo) == "Again"
        await _settle(provider)
        assert [c.disconnected for c in clients.clients] == [True, False]
        assert clients.clients[1].options.resume is not None
        await provider.close()

    async def test_a_failed_start_removes_its_files(self, workspace):
        provider, clients = _provider(workspace, retry_policy={"max_retries": 0})
        clients.connect_error = OSError("spawn failed")
        convo = [
            SYSTEM,
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "go"},
        ]
        with pytest.raises(OSError, match="spawn failed"):
            await _step(provider, convo)
        await _settle(provider)
        assert clients.clients[0].disconnected
        assert _leftovers(workspace) == []

    async def test_close_before_any_step_is_a_no_op(self, workspace):
        provider, clients = _provider(workspace)
        await provider.close()
        assert clients.clients == []

    async def test_a_process_that_exits_mid_step_is_an_error(self, workspace):
        provider, clients = _provider(
            workspace, ScriptedClient.END, retry_policy={"max_retries": 0}
        )
        with pytest.raises(RuntimeError, match="before finishing"):
            await _step(provider, [{"role": "user", "content": "hi"}])
        await _settle(provider)
        assert clients.clients[0].disconnected
        assert _leftovers(workspace) == []


class TestClosing:
    async def test_idle_process_exits_after_keep_alive(self, workspace):
        provider, clients = _provider(workspace, _text("Hello"), keep_alive=0.05)
        await _step(provider, [{"role": "user", "content": "hi"}])
        assert not clients.clients[0].disconnected
        await asyncio.sleep(0.15)
        await _settle(provider)
        assert clients.clients[0].disconnected
        assert provider._pool().idle == 0
        assert _leftovers(workspace) == []

    async def test_oldest_idle_process_is_closed_over_the_limit(self, workspace):
        provider, clients = _provider(
            workspace, _text("A"), _text("B"), max_idle_sessions=1
        )
        await _step(provider, [{"role": "user", "content": "first"}])
        await _step(provider.fork(), [{"role": "user", "content": "second"}])
        await _settle(provider)
        assert [c.disconnected for c in clients.clients] == [True, False]
        await _settle(provider)
        await provider._pool().close_all()

    async def test_error_result_closes_the_process_and_retry_starts_fresh(
        self, workspace
    ):
        provider, clients = _provider(
            workspace,
            rate_limited("slow down"),
            _text("ok"),
            retry_policy={"max_retries": 1},
        )
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "ok"
        await _settle(provider)
        assert [c.disconnected for c in clients.clients] == [True, False]
        await provider.close()

    async def test_interrupt_closes_the_process_and_cleans_up(self, workspace):
        provider, clients = _provider(
            workspace,
            message(text_events("a", 0) + text_events("b", 1)) + [result()],
        )
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "go"},
        ]
        stream = provider.chat(history)
        assert await anext(stream) == "a"
        await stream.aclose()
        await _settle(provider)
        assert clients.clients[0].disconnected
        assert provider._pool().idle == 0
        assert _leftovers(workspace) == []

    async def test_no_reuse_after_the_cli_compacted(self, workspace):
        boundary = SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 900_000}},
        )
        provider, clients = _provider(
            workspace, [boundary] + _text("ok"), _text("again")
        )
        convo = [{"role": "user", "content": "hi"}]
        await _step(provider, convo)
        await _settle(provider)
        assert clients.clients[0].disconnected
        convo += [_assistant(provider, "ok"), {"role": "user", "content": "more"}]
        await _step(provider, convo)
        assert len(clients.clients) == 2
        await provider.close()

    async def test_missing_cli_is_a_configuration_error(self, workspace):
        provider, clients = _provider(workspace)
        clients.connect_error = CLINotFoundError("nope")
        with pytest.raises(LLMNotConfiguredError, match="cli_path"):
            await _step(provider, [{"role": "user", "content": "hi"}])
        await _settle(provider)
        assert _leftovers(workspace) == []

    async def test_close_leaves_other_providers_processes_running(self, workspace):
        provider, clients = _provider(workspace, _text("A"), _text("B"))
        other = provider.fork()
        await _step(provider, [{"role": "user", "content": "first"}])
        await _step(other, [{"role": "user", "content": "second"}])
        await provider.close()
        assert [c.disconnected for c in clients.clients] == [True, False]
        await other.close()
        assert clients.clients[1].disconnected

    async def test_keep_alive_zero_uses_one_shot_query(self, workspace):
        calls = []

        async def query_fn(*, prompt, options):
            calls.append(prompt)
            for item in _text("ok"):
                yield item

        clients = ScriptedClients()
        provider = ClaudeCodeProvider(
            workspace=workspace,
            query_fn=query_fn,
            client_factory=clients,
            keep_alive=0,
        )
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "ok"
        assert calls == ["hi"] and clients.clients == []
        assert provider._pool() is None


def _saved(prompt, uid, text):
    return [
        {
            "type": "user",
            "uuid": f"u{uid}",
            "message": {"role": "user", "content": prompt},
        },
        {"type": "attachment", "uuid": f"a{uid}", "attachment": {"type": "date"}},
        {"type": "api-request", "uuid": f"q{uid}"},
        {
            "type": "assistant",
            "uuid": f"r{uid}",
            "message": {
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": text}],
            },
        },
    ]
