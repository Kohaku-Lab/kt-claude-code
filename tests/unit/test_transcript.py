"""Unit tests for ``kt_claude_code.transcript``."""

from kt_claude_code.transcript import (
    CONTINUE_PROMPT,
    START_TEXT,
    cli_tool_name,
    kt_tool_name,
    plan_transcript,
    safe_tool_id,
    transcript_entries,
)

PNG = "data:image/png;base64,iVBORw0KGgo="


def _call(call_id, name="weather", args='{"city": "Taipei"}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


class TestNames:
    def test_tool_names_round_trip(self):
        assert cli_tool_name("weather") == "mcp__kt__weather"
        assert kt_tool_name("mcp__kt__weather") == "weather"
        assert kt_tool_name("plain") == "plain"

    def test_unsafe_ids_become_api_safe_and_stable(self):
        assert safe_tool_id("job_42:xyz") == "job_42_xyz"
        assert safe_tool_id("toolu_ok-1") == "toolu_ok-1"
        assert safe_tool_id("") == "toolu_kt"
        assert safe_tool_id("a.b") == safe_tool_id("a.b")


class TestPlanTranscript:
    def test_last_plain_user_text_becomes_the_prompt(self):
        plan = plan_transcript(
            [
                {"role": "system", "content": "You are Kohaku."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ]
        )
        assert plan.system == "You are Kohaku."
        assert plan.prompt == "how are you"
        assert [m["role"] for m in plan.history] == ["user", "assistant"]
        assert plan.history[1]["content"] == [{"type": "text", "text": "hello"}]

    def test_single_user_message_has_no_history(self):
        plan = plan_transcript([{"role": "user", "content": "only"}])
        assert plan.history == [] and plan.prompt == "only"

    def test_tool_results_stay_in_history_and_prompt_continues(self):
        plan = plan_transcript(
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("job:1"), _call("c2")],
                },
                {"role": "tool", "tool_call_id": "job:1", "content": "29C"},
                {"role": "tool", "tool_call_id": "c2", "content": "30C"},
            ]
        )
        assert plan.prompt == CONTINUE_PROMPT
        assistant, results = plan.history[1], plan.history[2]
        assert [b["name"] for b in assistant["content"]] == ["mcp__kt__weather"] * 2
        assert [b["id"] for b in assistant["content"]] == ["job_1", "c2"]
        assert assistant["content"][0]["input"] == {"city": "Taipei"}
        assert [b["tool_use_id"] for b in results["content"]] == ["job_1", "c2"]
        assert [b["content"] for b in results["content"]] == ["29C", "30C"]

    def test_dangling_tool_use_gets_a_result_before_reaching_the_cli(self):
        plan = plan_transcript(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
                {"role": "user", "content": "never mind"},
            ]
        )
        used = {
            b["id"]
            for m in plan.history
            for b in m["content"]
            if b["type"] == "tool_use"
        }
        answered = {
            b["tool_use_id"]
            for m in plan.history
            for b in m["content"]
            if b["type"] == "tool_result"
        }
        assert used == answered == {"c1"}

    def test_thinking_is_dropped_from_native_content(self):
        native = [
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            {"type": "text", "text": "answer"},
        ]
        plan = plan_transcript(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "_kt_anthropic_content": native,
                },
                {"role": "user", "content": "next"},
            ]
        )
        assert plan.history[1]["content"] == [{"type": "text", "text": "answer"}]

    def test_images_keep_the_turn_in_history(self):
        plan = plan_transcript(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": PNG}},
                        {"type": "text", "text": "colour?"},
                    ],
                }
            ]
        )
        assert plan.prompt == CONTINUE_PROMPT
        assert plan.history[0]["content"][0]["type"] == "image"
        assert plan.history[0]["content"][0]["source"]["type"] == "base64"

    def test_same_role_turns_merge_and_history_starts_with_user(self):
        plan = plan_transcript(
            [
                {"role": "assistant", "content": "greeting"},
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "x"},
                {"role": "user", "content": "   "},
            ]
        )
        assert plan.history[0] == {
            "role": "user",
            "content": [{"type": "text", "text": START_TEXT}],
        }
        assert [m["role"] for m in plan.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [b["text"] for b in plan.history[2]["content"]] == ["a", "b"]
        assert plan.prompt == CONTINUE_PROMPT


class TestTranscriptEntries:
    def test_entries_chain_and_describe_each_turn(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "r"}
                ],
            },
        ]
        entries = transcript_entries(
            history, session_id="sid", cwd="/w", version="9.9", model="haiku"
        )
        assert [e["type"] for e in entries] == ["user", "assistant", "user"]
        assert entries[0]["parentUuid"] is None
        assert entries[1]["parentUuid"] == entries[0]["uuid"]
        assert entries[2]["parentUuid"] == entries[1]["uuid"]
        assert {e["sessionId"] for e in entries} == {"sid"}
        assert {e["cwd"] for e in entries} == {"/w"}
        assistant = entries[1]["message"]
        assert assistant["stop_reason"] == "tool_use"
        assert assistant["model"] == "haiku" and assistant["type"] == "message"
        assert entries[0]["message"] == {
            "role": "user",
            "content": history[0]["content"],
        }
