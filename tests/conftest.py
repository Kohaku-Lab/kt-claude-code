"""Shared fixtures: an isolated KT config root and a scripted SDK ``query``.

The scripted query yields real SDK message objects, so everything after the
SDK boundary (stream decoding, transcript files, cleanup) runs for real.
"""

import asyncio
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


class ScriptedClients:
    """Stands in for ``ClaudeSDKClient``: builds clients that answer from scripts.

    Every ``query`` on any client takes the next script. When ``saves`` are
    given, each query first appends the next list of entries to the client's
    session file, as the CLI saves a step. ``queries`` records each query's
    client, prompt, options, and the transcript and system prompt the CLI
    would read at start-up.
    """

    def __init__(self, *scripts: list[Any] | BaseException, saves=None, workspace=None):
        self.scripts = list(scripts)
        self.saves = list(saves or [])
        self.workspace = workspace
        self.clients: list[ScriptedClient] = []
        self.queries: list[dict[str, Any]] = []
        self.connect_error: BaseException | None = None

    def __call__(self, options: Any) -> "ScriptedClient":
        client = ScriptedClient(self, options, len(self.clients))
        self.clients.append(client)
        return client


class ScriptedClient:
    """One fake CLI process: a queue of SDK messages fed by its queries."""

    END = object()

    def __init__(self, owner: ScriptedClients, options: Any, index: int):
        self.owner = owner
        self.options = options
        self.index = index
        self.sid = options.resume or f"cli-sid-{index}"
        self.connected = False
        self.disconnected = False
        self.startup: dict[str, Any] = {}
        self._queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        if self.owner.connect_error is not None:
            raise self.owner.connect_error
        system = self.options.system_prompt
        if isinstance(system, dict) and system.get("type") == "file":
            self.startup["system_text"] = Path(system["path"]).read_text(
                encoding="utf-8"
            )
        if self.options.resume:
            projects = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects"
            files = list(projects.rglob(f"{self.options.resume}.jsonl"))
            self.startup["transcript"] = [
                json.loads(line) for line in files[0].read_text().splitlines()
            ]
        self.connected = True

    async def query(self, prompt: Any, session_id: str = "default") -> None:
        if self.disconnected or not self.connected:
            raise RuntimeError("process is not running")
        if not isinstance(prompt, str):
            prompt = [m async for m in prompt]
        self.owner.queries.append({"client": self.index, "prompt": prompt})
        if self.owner.saves:
            folder = self.owner.workspace.project_dir
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{self.sid}.jsonl", "a", encoding="utf-8") as f:
                for entry in self.owner.saves.pop(0):
                    f.write(json.dumps(entry) + "\n")
        script = self.owner.scripts.pop(0)
        if isinstance(script, BaseException) or script is self.END:
            self._queue.put_nowait(script)
            return
        for item in script:
            if isinstance(item, ResultMessage):
                item.session_id = self.sid
            self._queue.put_nowait(item)

    async def receive_messages(self):
        while True:
            item = await self._queue.get()
            if item is self.END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def disconnect(self) -> None:
        self.disconnected = True
