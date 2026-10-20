"""Turn KohakuTerrarium history into a Claude Code transcript and a prompt.

The CLI resumes the transcript as real prior turns; the prompt becomes the new
user message of this step.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kohakuterrarium.llm.anthropic_format import prepare_messages

TOOL_PREFIX = "mcp__kt__"
CONTINUE_PROMPT = "(continue)"
START_TEXT = "(conversation start)"
_KEPT_BLOCK_TYPES = frozenset({"text", "image", "document", "tool_use", "tool_result"})
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
    prompt: str = CONTINUE_PROMPT


def plan_transcript(messages: list[dict[str, Any]]) -> TranscriptPlan:
    """Split KT messages into the system prompt, resumable history and a prompt.

    A final user message holding only text becomes the prompt; anything else
    (tool results, images, an assistant turn) stays in history and the prompt
    is :data:`CONTINUE_PROMPT`.
    """
    system, body = prepare_messages(messages)
    cleaned = [
        {"role": msg["role"], "content": _clean_content(msg["content"])} for msg in body
    ]
    history = _merge_same_role([msg for msg in cleaned if msg["content"]])
    if history and history[0]["role"] != "user":
        history.insert(0, {"role": "user", "content": [_text(START_TEXT)]})
    prompt = CONTINUE_PROMPT
    if history and history[-1]["role"] == "user":
        text = _plain_text(history[-1]["content"])
        if text is not None:
            history.pop()
            prompt = text
    return TranscriptPlan(system=system, history=history, prompt=prompt)


def transcript_entries(
    history: list[dict[str, Any]],
    *,
    session_id: str,
    cwd: str,
    version: str,
    model: str,
) -> list[dict[str, Any]]:
    """CLI transcript lines for ``history``, chained by ``parentUuid``."""
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    timestamp = now.replace("+00:00", "Z")
    entries: list[dict[str, Any]] = []
    parent: str | None = None
    for msg in history:
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
                "parentUuid": parent,
                "isSidechain": False,
                "type": msg["role"],
                "message": message,
                "uuid": entry_uuid,
                "timestamp": timestamp,
                "userType": "external",
                "entrypoint": "sdk-py",
                "cwd": cwd,
                "sessionId": session_id,
                "version": version,
            }
        )
        parent = entry_uuid
    return entries


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
