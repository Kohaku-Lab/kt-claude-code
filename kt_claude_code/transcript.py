"""Turn KohakuTerrarium history into a Claude Code transcript and a prompt.

The CLI resumes the transcript as real prior turns; the prompt becomes the new
user message of this step.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kohakuterrarium.llm.anthropic.format import prepare_messages

TOOL_PREFIX = "mcp__kt__"
CONTINUE_PROMPT = "(continue)"
START_TEXT = "(conversation start)"
_KEPT_BLOCK_TYPES = frozenset({"text", "image", "document", "tool_use", "tool_result"})
# Assistant extra field holding the CLI's own transcript entries for that turn.
TURN_KEY = "_kt_claude_code_turn"
_RECORDED_ENTRY_TYPES = frozenset({"user", "assistant", "attachment"})
_DEFERRED = "hook_deferred_tool"
# The model name the CLI gives the assistant messages it inserts itself.
_SYNTHETIC_MODEL = "<synthetic>"
_UNSAFE_ID = re.compile(r"[^a-zA-Z0-9_-]")


def cli_tool_name(name: str) -> str:
    """The name the CLI shows the model for a KT tool."""
    return f"{TOOL_PREFIX}{name}"


def kt_tool_name(name: str) -> str:
    """The KT tool name behind a CLI tool name."""
    return name.removeprefix(TOOL_PREFIX)


def safe_tool_id(raw: str) -> str:
    """A tool-use id the API accepts (``[a-zA-Z0-9_-]+``), stable for one input."""
    return _UNSAFE_ID.sub("_", raw or "") or "toolu_kt"


@dataclass
class TranscriptPlan:
    """One model step: system prompt, prior turns, and the new prompt."""

    system: str
    history: list[dict[str, Any]] = field(default_factory=list)
    # Text, or the content blocks of a tool-result message.
    prompt: str | list[dict[str, Any]] = CONTINUE_PROMPT
    turns: list[dict[str, Any]] = field(default_factory=list)


def plan_transcript(
    messages: list[dict[str, Any]], *, defer_calls: bool = True
) -> TranscriptPlan:
    """Split KT messages into the system prompt, resumable history and a prompt.

    A final user message holding only text becomes the prompt, and so does
    one answering the single tool call of the assistant turn before it (the
    CLI takes it as the result of its deferred call; it resumes only one).
    That needs the step's tools, so ``defer_calls=False`` turns it off.
    Anything else stays in history and the prompt is :data:`CONTINUE_PROMPT`.
    """
    turns = [
        msg[TURN_KEY]
        for msg in messages
        if msg.get("role") == "assistant" and _valid_turn(msg.get(TURN_KEY))
    ]
    system, body = prepare_messages(messages)
    cleaned = [
        {"role": msg["role"], "content": _clean_content(msg["content"])} for msg in body
    ]
    history = _merge_same_role([msg for msg in cleaned if msg["content"]])
    if history and history[0]["role"] != "user":
        history.insert(0, {"role": "user", "content": [_text(START_TEXT)]})
    prompt: str | list[dict[str, Any]] = CONTINUE_PROMPT
    if history and history[-1]["role"] == "user":
        content = history[-1]["content"]
        text = _plain_text(content)
        if text is not None:
            history.pop()
            prompt = text
        elif defer_calls and _answers_calls(history):
            history.pop()
            prompt = content
    return TranscriptPlan(system=system, history=history, prompt=prompt, turns=turns)


def transcript_entries(
    history: list[dict[str, Any]],
    *,
    session_id: str,
    cwd: str,
    version: str,
    model: str,
    turns: list[dict[str, Any]] | None = None,
    defer_calls: bool = True,
) -> list[dict[str, Any]]:
    """CLI transcript lines for ``history``, chained by ``parentUuid``.

    An assistant turn that matches a recorded turn (see :func:`record_turn`)
    is replayed from the CLI's own entries, so the request it rebuilds is
    byte-identical to the one it sent and the prompt cache still applies.
    Without ``defer_calls`` (a step with no tools) deferred-call markers are
    left out: the CLI would try to re-run those calls. They are not part of
    the request, so the cache is unaffected.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    timestamp = now.replace("+00:00", "Z")
    pending = list(turns or [])
    deferred: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, Any] | None] = []
    for msg in history:
        turn = _take_turn(pending, msg) if msg["role"] == "assistant" else None
        if turn is not None and _replaces_prompt(turn, sources):
            if turn["prompt"] != CONTINUE_PROMPT:
                entries.pop()
                sources.pop()
            for raw in turn["entries"]:
                if not defer_calls and _is_deferred(raw):
                    continue
                entries.append(dict(raw))
                sources.append(None)
            deferred = []
            continue
        entry_uuid = str(uuid.uuid4())
        message: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
        if msg["role"] == "assistant":
            has_tool_use = any(b.get("type") == "tool_use" for b in msg["content"])
            message.update(
                model=model,
                id=f"msg_kt_{entry_uuid.replace('-', '')[:24]}",
                type="message",
                stop_reason="tool_use" if has_tool_use else "end_turn",
                stop_sequence=None,
                usage={"input_tokens": 0, "output_tokens": 0},
            )
        entries.append(
            {
                "type": msg["role"],
                "message": message,
                "uuid": entry_uuid,
                "timestamp": timestamp,
                "userType": "external",
                "entrypoint": "sdk-py",
                "cwd": cwd,
                "version": version,
            }
        )
        sources.append(msg)
        deferred = [
            {
                "type": "attachment",
                "attachment": _deferred_call(block),
                "uuid": str(uuid.uuid4()),
                "timestamp": timestamp,
                "userType": "external",
                "entrypoint": "sdk-py",
                "cwd": cwd,
                "version": version,
            }
            for block in msg["content"]
            if msg["role"] == "assistant" and block.get("type") == "tool_use"
        ]
    if defer_calls and history and history[-1]["role"] == "assistant":
        entries.extend(deferred)
    parent: str | None = None
    for entry in entries:
        entry.update(parentUuid=parent, isSidechain=False, sessionId=session_id)
        parent = entry["uuid"]
    return entries


def record_turn(
    prompt: str | list[dict[str, Any]], new_entries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The :data:`TURN_KEY` value for one step, from the entries the CLI saved."""
    kept = [e for e in new_entries if e.get("type") in _RECORDED_ENTRY_TYPES]
    calls = [
        block
        for e in kept
        if e["type"] == "assistant"
        for block in (e.get("message") or {}).get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    if len(calls) > 1:
        kept = [e for e in kept if not _is_deferred(e)]
    if not any(e["type"] == "assistant" for e in kept):
        return None
    return {"prompt": prompt, "entries": kept}


def _valid_turn(turn: Any) -> bool:
    return (
        isinstance(turn, dict)
        and isinstance(turn.get("prompt"), (str, list))
        and isinstance(turn.get("entries"), list)
        and all(isinstance(e, dict) and "uuid" in e for e in turn["entries"])
    )


def _take_turn(
    pending: list[dict[str, Any]], msg: dict[str, Any]
) -> dict[str, Any] | None:
    """Pop the first recorded turn whose assistant output equals ``msg``."""
    wanted = _signature(msg["content"])
    for index, turn in enumerate(pending):
        recorded = [
            block
            for e in turn["entries"]
            if e["type"] == "assistant"
            and (e.get("message") or {}).get("model") != _SYNTHETIC_MODEL
            for block in (e.get("message") or {}).get("content") or []
            if isinstance(block, dict)
        ]
        if _signature(recorded) == wanted:
            del pending[: index + 1]
            return turn
    return None


def _replaces_prompt(
    turn: dict[str, Any], sources: list[dict[str, Any] | None]
) -> bool:
    """Whether the entries before ``turn`` end the way its step began."""
    prompt = turn["prompt"]
    if prompt == CONTINUE_PROMPT:
        return True
    last = sources[-1] if sources else None
    if last is None or last["role"] != "user":
        return False
    if isinstance(prompt, list):
        return last["content"] == prompt
    return _plain_text(last["content"]) == prompt


def _answers_calls(history: list[dict[str, Any]]) -> bool:
    """Whether the last message answers the one tool call of the turn before."""
    if len(history) < 2 or history[-2]["role"] != "assistant":
        return False
    calls = [b.get("id") for b in history[-2]["content"] if b.get("type") == "tool_use"]
    results = [
        b.get("tool_use_id")
        for b in history[-1]["content"]
        if b.get("type") == "tool_result"
    ]
    return len(calls) == 1 and results == calls


def _is_deferred(entry: dict[str, Any]) -> bool:
    return (entry.get("attachment") or {}).get("type") == _DEFERRED


def _deferred_call(block: dict[str, Any]) -> dict[str, Any]:
    """The CLI's record of a tool call its PreToolUse hook deferred."""
    return {
        "type": _DEFERRED,
        "toolUseID": block.get("id"),
        "toolName": block.get("name"),
        "toolInput": block.get("input") or {},
        "hookName": "settings",
        "hookEvent": "PreToolUse",
        "permissionMode": "default",
    }


def _signature(blocks: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """What identifies an assistant turn: its text and its tool calls."""
    out: list[tuple[str, ...]] = []
    for block in blocks:
        if block.get("type") == "text" and str(block.get("text") or "").strip():
            out.append(("text", str(block["text"]).strip()))
        elif block.get("type") == "tool_use":
            out.append(
                (
                    "tool_use",
                    str(block.get("id") or ""),
                    str(block.get("name") or ""),
                    json.dumps(block.get("input") or {}, sort_keys=True),
                )
            )
    return out


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _clean_content(content: Any) -> list[dict[str, Any]]:
    """Content as a block list: thinking dropped, tool names and ids mapped."""
    if isinstance(content, str):
        return [_text(content)] if content.strip() else []
    blocks: list[dict[str, Any]] = []
    for block in content or []:
        if not isinstance(block, dict) or block.get("type") not in _KEPT_BLOCK_TYPES:
            continue
        btype = block["type"]
        if btype == "text":
            if str(block.get("text") or "").strip():
                blocks.append(_text(str(block["text"])))
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": safe_tool_id(str(block.get("id") or "")),
                    "name": cli_tool_name(str(block.get("name") or "")),
                    "input": block.get("input") or {},
                }
            )
        elif btype == "tool_result":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": safe_tool_id(str(block.get("tool_use_id") or "")),
                    "content": block.get("content") or "",
                    **({"is_error": True} if block.get("is_error") else {}),
                }
            )
        else:
            blocks.append({k: v for k, v in block.items() if k != "cache_control"})
    return blocks


def _merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = merged[-1]["content"] + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})
    return merged


def _plain_text(content: list[dict[str, Any]]) -> str | None:
    """The joined text when ``content`` holds only text blocks, else ``None``."""
    if not content or any(block.get("type") != "text" for block in content):
        return None
    return "\n\n".join(block["text"] for block in content)
