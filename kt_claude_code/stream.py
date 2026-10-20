"""Decode one model step from the Claude Agent SDK message stream.

Text is taken only from partial-message stream events, so error text the CLI
reports as an assistant message never reaches the conversation.
"""

from types import SimpleNamespace
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
)

from kohakuterrarium.llm.anthropic_format import (
    KT_CONTENT_KEY,
    apply_delta,
    block_to_dict,
    merge_usage,
    normalise_started_block,
    ordered_blocks,
    tool_calls_from_blocks,
)
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.llm.openai_sanitize import strip_surrogates
from kohakuterrarium.llm.turn_segments import inject_anthropic_segments

from kt_claude_code.errors import error_from_result
from kt_claude_code.transcript import kt_tool_name


class StepStream:
    """Feed SDK messages in; read text out, then the step's results."""

    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, int] = {}
        self.stop_reason = ""
        self.done = False
        self._message_open = False
        self._message_closed = False
        self._error_kind: str | None = None
        self._error_text = ""
        self._rate_limit_note = ""

    def feed(self, msg: Any) -> str:
        """Consume one SDK message; returns visible text to stream (maybe empty).

        Raises the mapped exception when the CLI reports an error result.
        """
        if isinstance(msg, StreamEvent):
            if msg.parent_tool_use_id is None:
                return self._event(msg.event)
            return ""
        if isinstance(msg, AssistantMessage):
            return self._assistant(msg)
        if isinstance(msg, RateLimitEvent):
            info = msg.rate_limit_info
            if getattr(info, "status", "") == "rejected":
                self._rate_limit_note = (
                    f"usage limit reached ({getattr(info, 'rate_limit_type', '')}, "
                    f"resets at {getattr(info, 'resets_at', '?')})"
                )
            return ""
        if isinstance(msg, ResultMessage):
            self._result(msg)
        return ""

    def _event(self, event: dict[str, Any]) -> str:
        etype = event.get("type")
        if etype == "message_start":
            if self._message_closed:
                self.done = True
                return ""
            self._message_open = True
            self.usage = merge_usage(
                self.usage, (event.get("message") or {}).get("usage")
            )
        elif etype == "content_block_start":
            idx = int(event.get("index", len(self.blocks)))
            self.blocks[idx] = normalise_started_block(
                block_to_dict(event.get("content_block"))
            )
        elif etype == "content_block_delta":
            idx = int(event.get("index", len(self.blocks)))
            delta = SimpleNamespace(**(event.get("delta") or {}))
            piece = apply_delta(self.blocks.setdefault(idx, {}), delta)
            return strip_surrogates(piece) if piece else ""
        elif etype == "message_delta":
            delta = event.get("delta") or {}
            self.stop_reason = delta.get("stop_reason") or self.stop_reason
            if event.get("usage"):
                self.usage = merge_usage(self.usage, event["usage"])
        elif etype == "message_stop":
            self._message_closed = True
            self.done = True
        return ""

    def _assistant(self, msg: AssistantMessage) -> str:
        if msg.error:
            self._error_kind = msg.error
            self._error_text = "".join(
                getattr(block, "text", "") for block in msg.content or []
            )
            return ""
        if self._message_open or self.blocks:
            return ""
        text = ""
        for idx, block in enumerate(msg.content or []):
            data = block_to_dict(block)
            data.setdefault("type", _block_type(block))
            self.blocks[idx] = data
            if data["type"] == "text":
                text += data.get("text", "")
        self.stop_reason = msg.stop_reason or self.stop_reason
        if msg.usage:
            self.usage = merge_usage(self.usage, msg.usage)
        return strip_surrogates(text)

    def _result(self, msg: ResultMessage) -> None:
        self.done = True
        if msg.is_error or self._error_kind:
            text = self._error_text or msg.result or ""
            if self._rate_limit_note and not self._error_kind:
                self._error_kind = "rate_limit"
                text = f"{text} {self._rate_limit_note}".strip()
            raise error_from_result(
                self._error_kind,
                text,
                status_code=msg.api_error_status,
                terminal_reason=msg.terminal_reason,
            )
        if not self.usage and msg.usage:
            self.usage = merge_usage({}, msg.usage)
        self.stop_reason = self.stop_reason or (msg.stop_reason or "")

    def content_blocks(self) -> list[dict[str, Any]]:
        """Final ordered blocks with tool names mapped back to KT names."""
        blocks = ordered_blocks(self.blocks)
        for block in blocks:
            if block.get("type") == "tool_use":
                block["name"] = kt_tool_name(str(block.get("name") or ""))
        return blocks

    def results(self) -> tuple[list[NativeToolCall], dict[str, int], dict[str, Any]]:
        """``(tool_calls, usage, assistant_extra_fields)`` for the provider."""
        blocks = self.content_blocks()
        fields: dict[str, Any] = {}
        if blocks:
            fields = inject_anthropic_segments({KT_CONTENT_KEY: blocks}, blocks)
        return tool_calls_from_blocks(blocks), dict(self.usage), fields


def _block_type(block: Any) -> str:
    name = type(block).__name__
    return {
        "TextBlock": "text",
        "ThinkingBlock": "thinking",
        "ToolUseBlock": "tool_use",
    }.get(name, name)
