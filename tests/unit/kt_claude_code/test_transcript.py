"""Unit tests for ``kt_claude_code.transcript``."""

from kt_claude_code.transcript import (
    CONTINUE_PROMPT,
    START_TEXT,
    TURN_KEY,
    cli_tool_name,
    kt_tool_name,
    plan_transcript,
    record_turn,
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
        # One answered call: the result travels as the prompt.
        answered = {b["tool_use_id"] for b in plan.prompt if b["type"] == "tool_result"}
        assert used == answered == {"c1"}
        assert [b["text"] for b in plan.prompt if b["type"] == "text"] == ["never mind"]

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


def _cli(kind, content=None, *, uid, model="claude-haiku-5-5", attachment=None):
    """An entry shaped like the ones the CLI saves."""
    entry = {"type": kind, "uuid": uid, "parentUuid": "p", "sessionId": "old"}
    if attachment is not None:
        entry["attachment"] = attachment
    else:
        entry["message"] = {"role": kind, "content": content}
        if kind == "assistant":
            entry["message"]["model"] = model
    return entry


def _deferred(call_id):
    return {"type": "hook_deferred_tool", "toolUseID": call_id}


def _tool_use(call_id):
    return {
        "type": "tool_use",
        "id": call_id,
        "name": "mcp__kt__weather",
        "input": {"city": "Taipei"},
        "caller": {"type": "direct"},
    }


def _entries(messages, defer_calls=True):
    plan = plan_transcript(messages, defer_calls=defer_calls)
    entries = transcript_entries(
        plan.history,
        session_id="new",
        cwd="/ws",
        version="1",
        model="haiku",
        turns=plan.turns,
        defer_calls=defer_calls,
    )
    return plan, entries


HELLO_TURN = {
    "prompt": "hi",
    "entries": [
        _cli("user", "hi", uid="u1"),
        _cli("attachment", uid="a1", attachment={"type": "environment"}),
        _cli("assistant", [{"type": "text", "text": "Hello"}], uid="r1"),
    ],
}


class TestRecordTurn:
    def test_keeps_only_entries_that_build_the_request(self):
        turn = record_turn(
            "hi",
            [
                _cli("user", "hi", uid="u1"),
                {"type": "queue-operation", "uuid": "q"},
                {"type": "api-request", "uuid": "r"},
                _cli("attachment", uid="a1", attachment={"type": "date"}),
                _cli("assistant", [{"type": "text", "text": "Hello"}], uid="r1"),
                {"type": "last-prompt", "uuid": "l"},
            ],
        )
        assert turn["prompt"] == "hi"
        assert [e["uuid"] for e in turn["entries"]] == ["u1", "a1", "r1"]

    def test_no_turn_without_an_assistant_reply(self):
        assert record_turn("hi", [_cli("user", "hi", uid="u1")]) is None
        assert record_turn("hi", []) is None

    def test_parallel_calls_drop_their_deferred_markers(self):
        # The CLI resumes only one deferred call; parallel ones go back as
        # ordinary results, so the markers must not be replayed.
        entries = [
            _cli("user", "go", uid="u1"),
            _cli("assistant", [_tool_use("c1")], uid="r1"),
            _cli("attachment", uid="d1", attachment=_deferred("c1")),
            _cli("assistant", [_tool_use("c2")], uid="r2"),
            _cli("attachment", uid="d2", attachment=_deferred("c2")),
        ]
        parallel = record_turn("go", entries)
        assert [e["uuid"] for e in parallel["entries"]] == ["u1", "r1", "r2"]
        single = record_turn("go", entries[:3])
        assert [e["uuid"] for e in single["entries"]] == ["u1", "r1", "d1"]


class TestReplay:
    def test_matching_turn_replays_the_cli_entries_verbatim(self):
        plan, entries = _entries(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello", TURN_KEY: HELLO_TURN},
                {"role": "user", "content": "next"},
            ]
        )
        assert plan.prompt == "next"
        # The recorded prompt replaces KT's own copy of it.
        assert [e["uuid"] for e in entries] == ["u1", "a1", "r1"]
        assert entries[1]["attachment"] == {"type": "environment"}
        assert [e["parentUuid"] for e in entries] == [None, "u1", "a1"]
        assert {e["sessionId"] for e in entries} == {"new"}
        # The stored record is not mutated by the rewrite.
        assert HELLO_TURN["entries"][0]["sessionId"] == "old"

    def test_edited_reply_falls_back_to_plain_entries(self):
        _plan, entries = _entries(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hi there", TURN_KEY: HELLO_TURN},
                {"role": "user", "content": "next"},
            ]
        )
        assert not {"u1", "a1", "r1"} & {e["uuid"] for e in entries}
        assert entries[1]["message"]["content"] == [
            {"type": "text", "text": "Hi there"}
        ]

    def test_edited_prompt_falls_back_to_plain_entries(self):
        _plan, entries = _entries(
            [
                {"role": "user", "content": "hello?"},
                {"role": "assistant", "content": "Hello", TURN_KEY: HELLO_TURN},
                {"role": "user", "content": "next"},
            ]
        )
        assert entries[0]["message"]["content"] == [{"type": "text", "text": "hello?"}]
        assert "r1" not in {e["uuid"] for e in entries}

    def test_malformed_record_is_ignored(self):
        for junk in ("x", {"prompt": "hi"}, {"prompt": "hi", "entries": [{}]}):
            _plan, entries = _entries(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Hello", TURN_KEY: junk},
                    {"role": "user", "content": "next"},
                ]
            )
            assert [e["type"] for e in entries] == ["user", "assistant"]

    def test_continue_turn_skips_the_cli_synthetic_reply(self):
        turn = {
            "prompt": CONTINUE_PROMPT,
            "entries": [
                _cli(
                    "assistant",
                    [{"type": "text", "text": "No response requested."}],
                    uid="s1",
                    model="<synthetic>",
                ),
                _cli("user", CONTINUE_PROMPT, uid="c1"),
                _cli("assistant", [{"type": "text", "text": "Both done."}], uid="r2"),
            ],
        }
        _plan, entries = _entries(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("c1"), _call("c2")],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "29C"},
                {"role": "tool", "tool_call_id": "c2", "content": "30C"},
                {"role": "assistant", "content": "Both done.", TURN_KEY: turn},
                {"role": "user", "content": "thanks"},
            ]
        )
        kinds = [(e["type"], e["uuid"]) for e in entries]
        # The tool results stay; the recorded continue turn follows them.
        assert kinds[2][0] == "user"
        assert entries[2]["message"]["content"][0]["type"] == "tool_result"
        assert [u for _k, u in kinds[3:]] == ["s1", "c1", "r2"]

    def test_tool_result_prompt_turn_replaces_the_result_entry(self):
        call_turn = {
            "prompt": "weather?",
            "entries": [
                _cli("user", "weather?", uid="u1"),
                _cli("assistant", [_tool_use("c1")], uid="r1"),
                _cli("attachment", uid="d1", attachment=_deferred("c1")),
            ],
        }
        results = [{"type": "tool_result", "tool_use_id": "c1", "content": "29C"}]
        answer_turn = {
            "prompt": results,
            "entries": [
                _cli("attachment", uid="d2", attachment=_deferred("c1")),
                _cli("user", results, uid="u2"),
                _cli("assistant", [{"type": "text", "text": "29C."}], uid="r2"),
            ],
        }
        _plan, entries = _entries(
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("c1")],
                    TURN_KEY: call_turn,
                },
                {"role": "tool", "tool_call_id": "c1", "content": "29C"},
                {"role": "assistant", "content": "29C.", TURN_KEY: answer_turn},
                {"role": "user", "content": "thanks"},
            ]
        )
        assert [e["uuid"] for e in entries] == ["u1", "r1", "d1", "d2", "u2", "r2"]


class TestDeferredMarkers:
    def _plan(self, calls, turn=None, defer_calls=True):
        assistant = {"role": "assistant", "content": "", "tool_calls": calls}
        if turn is not None:
            assistant[TURN_KEY] = turn
        results = [
            {"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in calls
        ]
        return _entries(
            [{"role": "user", "content": "go"}, assistant, *results],
            defer_calls=defer_calls,
        )

    def test_single_answered_call_is_marked_deferred_and_answered_by_prompt(self):
        plan, entries = self._plan([_call("c1")])
        assert [e["type"] for e in entries] == ["user", "assistant", "attachment"]
        marker = entries[-1]["attachment"]
        assert (marker["type"], marker["toolUseID"]) == ("hook_deferred_tool", "c1")
        assert marker["toolName"] == "mcp__kt__weather"
        assert [b["tool_use_id"] for b in plan.prompt] == ["c1"]

    def test_parallel_calls_get_no_marker_and_results_stay_in_history(self):
        plan, entries = self._plan([_call("c1"), _call("c2")])
        assert plan.prompt == CONTINUE_PROMPT
        assert [e["type"] for e in entries] == ["user", "assistant", "user"]

    def test_recorded_call_turn_is_not_marked_twice(self):
        turn = {
            "prompt": "go",
            "entries": [
                _cli("user", "go", uid="u1"),
                _cli("assistant", [_tool_use("c1")], uid="r1"),
                _cli("attachment", uid="d1", attachment=_deferred("c1")),
            ],
        }
        _plan, entries = self._plan([_call("c1")], turn)
        assert [e["uuid"] for e in entries] == ["u1", "r1", "d1"]

    def test_step_without_tools_never_defers(self):
        # No hook would answer the CLI re-running a deferred call.
        turn = {
            "prompt": "go",
            "entries": [
                _cli("user", "go", uid="u1"),
                _cli("assistant", [_tool_use("c1")], uid="r1"),
                _cli("attachment", uid="d1", attachment=_deferred("c1")),
            ],
        }
        plan, entries = self._plan([_call("c1")], turn, defer_calls=False)
        assert plan.prompt == CONTINUE_PROMPT
        assert [e["uuid"] for e in entries[:2]] == ["u1", "r1"]
        assert [e["type"] for e in entries[2:]] == ["user"]
        assert entries[2]["message"]["content"][0]["type"] == "tool_result"
        plain_plan, plain = self._plan([_call("c1")], defer_calls=False)
        assert plain_plan.prompt == CONTINUE_PROMPT
        assert "attachment" not in [e["type"] for e in plain]
