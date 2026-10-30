"""Unit tests for ``kt_claude_code.provider`` with a scripted SDK ``query``."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from claude_agent_sdk import CLINotFoundError, ResultMessage, SystemMessage

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.base import ToolSchema
from kohakuterrarium.llm.recovery.errors import ErrorClass

from kt_claude_code import provider as provider_mod
from kt_claude_code.errors import ClaudeCodeError
from kt_claude_code.provider import ClaudeCodeProvider
from kt_claude_code.transcript import TURN_KEY

from tests.conftest import (
    ScriptedQuery,
    error_reply,
    message,
    result,
    text_events,
    thinking_events,
    tool_events,
)

WEATHER = ToolSchema("weather", "Weather.", {"type": "object", "properties": {}})


def _provider(workspace, *scripts, **kwargs):
    fake = ScriptedQuery(*scripts)
    return ClaudeCodeProvider(workspace=workspace, query_fn=fake, **kwargs), fake


async def _step(provider, messages, tools=None):
    return "".join([c async for c in provider.chat(messages, tools=tools)])


def _leftovers(workspace):
    files = (
        list(workspace.project_dir.glob("*")) if workspace.project_dir.exists() else []
    )
    return files + list(workspace.root.glob("system-*"))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(provider_mod, "backoff_delay", lambda attempt, policy: 0)


class TestOneStep:
    async def test_first_turn_has_no_transcript_and_uses_kt_system_prompt(
        self, workspace
    ):
        provider, fake = _provider(workspace, message(text_events("Hi!")) + [result()])
        text = await _step(
            provider,
            [
                {"role": "system", "content": "You are Kohaku."},
                {"role": "user", "content": "hello"},
            ],
        )
        assert text == "Hi!"
        [call] = fake.calls
        assert call["prompt"] == "hello"
        assert call["system_text"] == "You are Kohaku."
        assert call["options"].resume is None
        assert call["options"].mcp_servers == {}
        assert provider.last_tool_calls == []
        assert provider.last_usage["completion_tokens"] == 7
        assert fake.closed == 1
        assert _leftovers(workspace) == []

    async def test_tool_step_resumes_history_and_returns_calls(self, workspace):
        blocks = thinking_events("plan", 0) + tool_events(
            "toolu_9", "weather", {"city": "Kyoto"}, 1
        )
        provider, fake = _provider(workspace, message(blocks, "tool_use") + [result()])
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "weather in Kyoto?"},
        ]
        assert await _step(provider, history, tools=[WEATHER]) == ""
        [call] = fake.calls
        assert call["prompt"] == "weather in Kyoto?"
        assert [e["type"] for e in call["transcript"]] == ["user", "assistant"]
        assert call["transcript"][1]["message"]["content"] == [
            {"type": "text", "text": "ok"}
        ]
        assert set(call["options"].mcp_servers) == {"kt"}
        [tool_call] = provider.last_tool_calls
        assert (tool_call.id, tool_call.name) == ("toolu_9", "weather")
        assert json.loads(tool_call.arguments) == {"city": "Kyoto"}
        assert _leftovers(workspace) == []

    async def test_next_step_carries_tool_results_and_drops_thinking(self, workspace):
        first = message(
            thinking_events("plan", 0) + tool_events("toolu_9", "weather", {}, 1),
            "tool_use",
        )
        second = message(text_events("Sunny.")) + [result()]
        provider, fake = _provider(workspace, first, second)
        convo = [{"role": "user", "content": "weather?"}]
        await _step(provider, convo, tools=[WEATHER])
        call = provider.last_tool_calls[0]
        convo.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
                **provider.last_assistant_extra_fields,
            }
        )
        convo.append({"role": "tool", "tool_call_id": call.id, "content": "25C"})
        assert await _step(provider, convo, tools=[WEATHER]) == "Sunny."
        transcript = fake.calls[1]["transcript"]
        assistant = transcript[1]["message"]["content"]
        assert [b["type"] for b in assistant] == ["tool_use"]
        assert assistant[0]["name"] == "mcp__kt__weather"
        # The call is marked deferred and its result is the step's prompt.
        assert transcript[2]["attachment"]["type"] == "hook_deferred_tool"
        assert transcript[2]["attachment"]["toolUseID"] == "toolu_9"
        sent = [m async for m in fake.calls[1]["prompt"]]
        assert len(sent) == 1 and sent[0]["type"] == "user"
        content = sent[0]["message"]["content"]
        assert [(b["type"], b["tool_use_id"], b["content"]) for b in content] == [
            ("tool_result", "toolu_9", "25C")
        ]
        assert sent[0]["session_id"] == fake.calls[1]["options"].resume

    async def test_tool_results_without_tools_continue_from_history(self, workspace):
        first = message(tool_events("toolu_9", "weather", {}, 0), "tool_use")
        provider, fake = _provider(
            workspace, first, message(text_events("Sunny.")) + [result()]
        )
        convo = [{"role": "user", "content": "weather?"}]
        await _step(provider, convo, tools=[WEATHER])
        call = provider.last_tool_calls[0]
        convo += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "content": "25C"},
        ]
        assert await _step(provider, convo) == "Sunny."
        assert fake.calls[1]["prompt"] == "(continue)"
        kinds = [e["type"] for e in fake.calls[1]["transcript"]]
        assert kinds == ["user", "assistant", "user"]


def _saving_query(workspace, fake, saves):
    """Wrap ``fake`` so each call first saves entries like the CLI does."""
    pending = list(saves)

    def run(*, prompt, options):
        stream = fake(prompt=prompt, options=options)
        sid = options.resume or "cli-sid"
        workspace.project_dir.mkdir(parents=True, exist_ok=True)
        with open(workspace.project_dir / f"{sid}.jsonl", "a", encoding="utf-8") as f:
            for entry in pending.pop(0):
                f.write(json.dumps(entry) + "\n")

        async def relabel():
            async for item in stream:
                if isinstance(item, ResultMessage):
                    item.session_id = sid
                yield item

        return relabel()

    return run


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


class TestRecordedTurns:
    async def test_turn_is_recorded_replayed_and_cli_files_removed(self, workspace):
        fake = ScriptedQuery(
            message(text_events("Hello")) + [result()],
            message(text_events("Again")) + [result()],
        )
        saves = [_saved("hi", 1, "Hello"), _saved("more", 2, "Again")]
        provider = ClaudeCodeProvider(
            workspace=workspace, query_fn=_saving_query(workspace, fake, saves)
        )
        convo = [{"role": "user", "content": "hi"}]
        assert await _step(provider, convo) == "Hello"
        turn = provider.last_assistant_extra_fields[TURN_KEY]
        assert turn["prompt"] == "hi"
        assert [e["uuid"] for e in turn["entries"]] == ["u1", "a1", "r1"]
        # The first step had no transcript; the CLI's own file is removed too.
        assert _leftovers(workspace) == []

        convo += [
            {
                "role": "assistant",
                "content": "Hello",
                **provider.last_assistant_extra_fields,
            },
            {"role": "user", "content": "more"},
        ]
        assert await _step(provider, convo) == "Again"
        assert [e["uuid"] for e in fake.calls[1]["transcript"]] == ["u1", "a1", "r1"]
        # Only what the CLI added after KT's transcript belongs to this turn.
        second = provider.last_assistant_extra_fields[TURN_KEY]
        assert [e["uuid"] for e in second["entries"]] == ["u2", "a2", "r2"]
        assert _leftovers(workspace) == []

    async def test_no_record_after_the_cli_compacted(self, workspace):
        boundary = SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 900_000}},
        )
        fake = ScriptedQuery([boundary] + message(text_events("ok")) + [result()])
        provider = ClaudeCodeProvider(
            workspace=workspace,
            query_fn=_saving_query(workspace, fake, [_saved("hi", 1, "ok")]),
        )
        await _step(provider, [{"role": "user", "content": "hi"}])
        assert TURN_KEY not in provider.last_assistant_extra_fields
        assert _leftovers(workspace) == []


class TestDrain:
    async def test_reads_past_the_result_so_the_cli_can_finish(self, workspace):
        after: list[str] = []

        async def query_fn(*, prompt, options):
            for item in message(text_events("ok")) + [result()]:
                yield item
            after.append("cli exited")

        provider = ClaudeCodeProvider(workspace=workspace, query_fn=query_fn)
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "ok"
        assert after == ["cli exited"]

    async def test_a_cli_that_never_exits_does_not_hold_the_step(
        self, workspace, monkeypatch
    ):
        monkeypatch.setattr(provider_mod, "DRAIN_SECONDS", 0.01)

        async def query_fn(*, prompt, options):
            for item in message(text_events("ok")) + [result()]:
                yield item
            await asyncio.sleep(30)
            yield result()

        provider = ClaudeCodeProvider(workspace=workspace, query_fn=query_fn)
        text = await asyncio.wait_for(
            _step(provider, [{"role": "user", "content": "hi"}]), 5
        )
        assert text == "ok"

    async def test_an_error_while_draining_keeps_the_step_error(self, workspace):
        async def query_fn(*, prompt, options):
            for item in error_reply("authentication_failed", "Not logged in"):
                yield item
            raise RuntimeError("cli crashed on exit")

        provider = ClaudeCodeProvider(workspace=workspace, query_fn=query_fn)
        with pytest.raises(LLMNotConfiguredError):
            await _step(provider, [{"role": "user", "content": "hi"}])

    async def test_a_second_message_closes_without_draining(self, workspace):
        after: list[str] = []

        async def query_fn(*, prompt, options):
            for item in message(text_events("ok")) + message(text_events("more")):
                yield item
            after.append("drained")

        provider = ClaudeCodeProvider(workspace=workspace, query_fn=query_fn)
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "ok"
        assert after == []


class TestFailures:
    async def test_retryable_error_retries_without_repeating_text(self, workspace):
        failing = message(text_events("Hel"))[:3] + [
            ClaudeCodeError("overloaded", ErrorClass.SERVER)
        ]

        async def _fail_after_text():
            for item in failing:
                if isinstance(item, BaseException):
                    raise item
                yield item

        fake = ScriptedQuery(message(text_events("Hello")) + [result()])
        calls = {"n": 0}

        def query(*, prompt, options):
            calls["n"] += 1
            return (
                _fail_after_text()
                if calls["n"] == 1
                else fake(prompt=prompt, options=options)
            )

        provider = ClaudeCodeProvider(workspace=workspace, query_fn=query)
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "Hello"
        assert calls["n"] == 2
        assert _leftovers(workspace) == []

    async def test_overflow_asks_kt_to_compact_then_retries(self, workspace):
        provider, fake = _provider(
            workspace,
            ClaudeCodeError("Prompt is too long", ErrorClass.OVERFLOW),
            message(text_events("short now")) + [result()],
        )
        compacted = [{"role": "user", "content": "summary + latest"}]

        async def rescue():
            return compacted

        provider._overflow_rescue = rescue
        long = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert await _step(provider, long) == "short now"
        assert fake.calls[1]["prompt"] == "summary + latest"

    async def test_cli_side_compaction_still_reports_full_size_to_kt(self, workspace):
        boundary = SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 900_000}},
        )
        provider, _ = _provider(
            workspace, [boundary] + message(text_events("ok")) + [result()]
        )
        assert await _step(provider, [{"role": "user", "content": "hi"}]) == "ok"
        assert provider.last_usage["prompt_tokens"] == 900_000

    async def test_missing_cli_is_a_configuration_error(self, workspace):
        provider, _ = _provider(workspace, CLINotFoundError("nope"))
        with pytest.raises(LLMNotConfiguredError, match="cli_path"):
            await _step(provider, [{"role": "user", "content": "hi"}])
        assert _leftovers(workspace) == []

    async def test_interrupt_closes_the_cli_and_cleans_up(self, workspace):
        provider, fake = _provider(
            workspace, message(text_events("a", 0) + text_events("b", 1)) + [result()]
        )
        history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "go"},
        ]
        gen = provider.chat(history)
        assert await gen.__anext__() == "a"
        await gen.aclose()
        assert fake.closed == 1
        assert _leftovers(workspace) == []


class TestConstruction:
    def test_from_profile_reads_model_effort_and_options(self, workspace):
        profile = SimpleNamespace(
            model="opus",
            reasoning_effort="high",
            extra_body={
                "thinking": {"type": "disabled"},
                "cli_path": "/x/claude",
                "max_concurrency": 2,
            },
            retry_policy={"max_retries": 1},
        )
        provider = ClaudeCodeProvider.from_profile(profile)
        assert provider.settings.model == "opus"
        assert provider.settings.effort == "high"
        assert provider.settings.thinking == {"type": "disabled"}
        assert provider.settings.cli_path == "/x/claude"
        assert provider.max_concurrency == 2
        assert provider._retry_policy.max_retries == 1

    def test_from_profile_defaults(self):
        provider = ClaudeCodeProvider.from_profile(
            SimpleNamespace(model="", extra_body=None)
        )
        assert provider.settings.model == "sonnet"
        assert provider.settings.effort is None

    def test_with_model_shares_workspace_and_query(self, workspace):
        provider, fake = _provider(workspace, effort="low")
        other = provider.with_model("haiku")
        assert other.settings.model == "haiku" and other.settings.effort == "low"
        assert other._workspace is workspace and other._query is fake
        assert provider.with_model("sonnet") is provider


class TestConcurrencyAndComplete:
    async def test_steps_never_exceed_max_concurrency(self, workspace):
        active = {"now": 0, "peak": 0}

        async def slow(*, prompt, options):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            try:
                await asyncio.sleep(0.01)
                for item in message(text_events(prompt)) + [result()]:
                    yield item
            finally:
                active["now"] -= 1

        provider = ClaudeCodeProvider(
            workspace=workspace, query_fn=slow, max_concurrency=1
        )
        outs = await asyncio.gather(
            *(
                _step(provider.fork(), [{"role": "user", "content": f"q{i}"}])
                for i in range(3)
            )
        )
        assert sorted(outs) == ["q0", "q1", "q2"]
        assert active["peak"] == 1

    async def test_complete_chat_returns_full_text(self, workspace):
        provider, _ = _provider(workspace, message(text_events("done")) + [result()])
        response = await provider.chat_complete([{"role": "user", "content": "x"}])
        assert response.content == "done" and response.model == "sonnet"
        assert response.usage["completion_tokens"] == 7
