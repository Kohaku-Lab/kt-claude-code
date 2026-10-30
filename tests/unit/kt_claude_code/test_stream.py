"""Unit tests for ``kt_claude_code.stream``."""

import json

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    DeferredToolUse,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.recovery.errors import ErrorClass

from kt_claude_code.errors import ClaudeCodeError
from kt_claude_code.stream import StepStream

from tests.conftest import (
    error_reply,
    event,
    message,
    rate_limited,
    result,
    text_events,
    thinking_events,
    tool_events,
)


def _feed(messages):
    stream = StepStream()
    text = ""
    for msg in messages:
        text += stream.feed(msg)
        if stream.done:
            break
    return stream, text


class TestText:
    def test_text_streams_and_usage_is_merged(self):
        stream, text = _feed(message(text_events("Hello")) + [result()])
        assert text == "Hello"
        calls, usage, fields = stream.results()
        assert calls == []
        assert usage["prompt_tokens"] >= 11 and usage["completion_tokens"] == 7
        assert fields["_kt_anthropic_content"] == [{"type": "text", "text": "Hello"}]

    def test_step_waits_for_the_result_after_message_stop(self):
        # The CLI saves the turn before its result; stopping earlier loses it.
        stream = StepStream()
        for msg in message(text_events("one")):
            stream.feed(msg)
        assert stream.done is False
        stream.feed(result(session_id="cli-sid"))
        assert stream.done is True
        assert stream.session_id == "cli-sid"

    def test_second_message_is_not_consumed(self):
        stream = StepStream()
        text = "".join(stream.feed(m) for m in message(text_events("first")))
        assert not stream.done
        assert stream.feed(event({"type": "message_start", "message": {}})) == ""
        assert stream.done
        assert text == "first"

    def test_redeferral_before_the_model_call_is_skipped(self):
        call = DeferredToolUse(id="toolu_1", name="mcp__kt__w", input={})
        stream, text = _feed(
            [result(deferred_tool_use=call)] + message(text_events("ok")) + [result()]
        )
        assert text == "ok"
        assert stream.done

    def test_deferral_after_the_model_call_ends_the_step(self):
        call = DeferredToolUse(id="toolu_1", name="mcp__kt__w", input={})
        stream, _ = _feed(
            message(tool_events("toolu_1", "w", {}, 0), "tool_use")
            + [result(deferred_tool_use=call)]
        )
        assert stream.done
        assert [c.id for c in stream.results()[0]] == ["toolu_1"]

    def test_error_result_is_never_skipped(self):
        call = DeferredToolUse(id="toolu_1", name="mcp__kt__w", input={})
        with pytest.raises(ClaudeCodeError):
            _feed([result(is_error=True, result="boom", deferred_tool_use=call)])

    def test_cli_compaction_reports_the_pre_compaction_size(self):
        boundary = SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 950_000}},
        )
        stream, text = _feed([boundary] + message(text_events("ok")) + [result()])
        assert text == "ok"
        _calls, usage, _fields = stream.results()
        # KT must still see the full size, or its own compaction never fires.
        assert usage["prompt_tokens"] == 950_000
        assert usage["total_tokens"] == 950_007
        assert stream.cli_compacted_from == 950_000

    def test_usage_is_untouched_without_cli_compaction(self):
        stream, _ = _feed(message(text_events("ok")) + [result()])
        assert stream.results()[1]["prompt_tokens"] == 16
        assert stream.cli_compacted_from == 0

    def test_subagent_events_are_ignored(self):
        stream = StepStream()
        assert stream.feed(event(text_events("x")[1].event, parent="toolu_1")) == ""
        assert stream.blocks == {}


class TestToolCalls:
    def test_parallel_calls_map_back_to_kt_names(self):
        blocks = tool_events("toolu_a", "weather", {"city": "Taipei"}, 0) + tool_events(
            "toolu_b", "local_time", {"city": "Tokyo"}, 1
        )
        stream, text = _feed(message(blocks, stop_reason="tool_use"))
        calls, _usage, fields = stream.results()
        assert text == ""
        assert [(c.id, c.name) for c in calls] == [
            ("toolu_a", "weather"),
            ("toolu_b", "local_time"),
        ]
        assert json.loads(calls[1].arguments) == {"city": "Tokyo"}
        assert [b["name"] for b in fields["_kt_anthropic_content"]] == [
            "weather",
            "local_time",
        ]

    def test_reasoning_becomes_segments_but_not_text(self):
        blocks = thinking_events("pondering", 0) + text_events("done", 1)
        stream, text = _feed(message(blocks))
        _calls, _usage, fields = stream.results()
        assert text == "done"
        reasoning = [
            s for s in fields["_kt_assistant_segments"] if s["type"] == "reasoning"
        ]
        assert reasoning[0]["text"] == "pondering"


class TestFallbackAndErrors:
    def test_full_assistant_message_used_when_no_partials(self):
        msg = AssistantMessage(
            content=[
                TextBlock(text="hi"),
                ToolUseBlock(id="t1", name="mcp__kt__w", input={"a": 1}),
            ],
            model="m",
            stop_reason="tool_use",
        )
        stream, text = _feed([msg, result()])
        calls, _usage, _fields = stream.results()
        assert text == "hi"
        assert [(c.id, c.name, json.loads(c.arguments)) for c in calls] == [
            ("t1", "w", {"a": 1})
        ]

    def test_error_text_is_never_streamed(self):
        stream = StepStream()
        msgs = error_reply("server_error", "API Error: 529 overloaded")
        assert stream.feed(msgs[0]) == ""
        with pytest.raises(ClaudeCodeError) as exc:
            stream.feed(msgs[1])
        assert exc.value.error_class is ErrorClass.SERVER

    def test_not_signed_in_is_a_configuration_error(self):
        with pytest.raises(LLMNotConfiguredError, match="/login"):
            _feed(
                error_reply(
                    "authentication_failed", "Not logged in · Please run /login"
                )
            )

    def test_overflow_maps_to_overflow(self):
        with pytest.raises(ClaudeCodeError) as exc:
            _feed(
                error_reply(
                    "invalid_request",
                    "Prompt is too long",
                    terminal_reason="blocking_limit",
                )
            )
        assert exc.value.error_class is ErrorClass.OVERFLOW

    def test_usage_limit_is_rate_limit_with_reset_note(self):
        with pytest.raises(ClaudeCodeError) as exc:
            _feed(rate_limited("You've hit your limit"))
        assert exc.value.error_class is ErrorClass.RATE_LIMIT

    def test_error_result_without_assistant_error(self):
        with pytest.raises(ClaudeCodeError) as exc:
            _feed(
                [
                    result(
                        is_error=True, result="Request timed out", api_error_status=408
                    )
                ]
            )
        assert exc.value.status_code == 408
