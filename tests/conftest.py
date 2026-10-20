"""Shared fixtures: an isolated KT config root and a scripted SDK ``query``.

The scripted query yields real SDK message objects, so everything after the
SDK boundary (stream decoding, transcript files, cleanup) runs for real.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TextBlock,
)
from claude_agent_sdk.types import RateLimitInfo

from kt_claude_code.workspace import Workspace


@pytest.fixture(autouse=True)
def isolated_homes(request, tmp_path, monkeypatch):
    """KT config and the CLI projects dir both under tmp.

    Live tests keep the real CLI config so the user's own sign-in applies.
    """
    monkeypatch.setenv("KT_CONFIG_DIR", str(tmp_path / "kt"))
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    return tmp_path


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "ws")


def event(payload: dict[str, Any], parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u", session_id="s", event=payload, parent_tool_use_id=parent
    )


def text_events(text: str, index: int = 0) -> list[StreamEvent]:
    return [
        event(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        event(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
        ),
        event({"type": "content_block_stop", "index": index}),
    ]


def thinking_events(text: str, index: int = 0) -> list[StreamEvent]:
    return [
        event(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        ),
        event(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": text},
            }
        ),
        event(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": "sig"},
            }
        ),
        event({"type": "content_block_stop", "index": index}),
    ]


def tool_events(call_id: str, name: str, args: dict, index: int) -> list[StreamEvent]:
    raw = json.dumps(args)
    return [
        event(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": f"mcp__kt__{name}",
                    "input": {},
                },
            }
        ),
        event(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": raw[:5]},
            }
        ),
        event(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": raw[5:]},
            }
        ),
        event({"type": "content_block_stop", "index": index}),
    ]


def message(blocks: list[StreamEvent], stop_reason: str = "end_turn") -> list[Any]:
    """A full streamed message: start, blocks, delta with usage, stop."""
    return [
        event(
            {
                "type": "message_start",
                "message": {
                    "usage": {"input_tokens": 11, "cache_read_input_tokens": 5}
                },
            }
        ),
        *blocks,
        event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 7},
            }
        ),
        event({"type": "message_stop"}),
    ]


def result(is_error: bool = False, **kwargs: Any) -> ResultMessage:
    base = dict(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s",
    )
    base.update(kwargs)
    return ResultMessage(**base)


def error_reply(kind: str, text: str, **result_kwargs: Any) -> list[Any]:
    return [
        AssistantMessage(content=[TextBlock(text=text)], model="m", error=kind),
        result(is_error=True, result=text, **result_kwargs),
    ]


def rate_limited(text: str) -> list[Any]:
    info = RateLimitInfo(
        status="rejected", resets_at=1700000000, rate_limit_type="five_hour"
    )
    return [
        RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s"),
        *error_reply("rate_limit", text),
    ]


class ScriptedQuery:
    """Stands in for ``claude_agent_sdk.query``; one script per call.

    Records each call's prompt and options, plus the transcript and system
    prompt files as they existed while the call ran.
    """

    def __init__(self, *scripts: list[Any] | BaseException):
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def __call__(self, *, prompt: Any, options: Any):
        script = self.scripts.pop(0)
        record = {"prompt": prompt, "options": options}
        system = options.system_prompt
        if isinstance(system, dict) and system.get("type") == "file":
            record["system_text"] = Path(system["path"]).read_text(encoding="utf-8")
        if options.resume:
            projects = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects"
            files = list(projects.rglob(f"{options.resume}.jsonl"))
            record["transcript"] = [
                json.loads(line) for line in files[0].read_text().splitlines()
            ]
        self.calls.append(record)
        return self._run(script)

    async def _run(self, script):
        try:
            if isinstance(script, BaseException):
                raise script
            for item in script:
                yield item
        finally:
            self.closed += 1
