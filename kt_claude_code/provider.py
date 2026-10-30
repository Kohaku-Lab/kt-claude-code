"""KohakuTerrarium LLM provider that runs each step through local Claude Code.

Every ``chat()`` call is stateless: KT's history is written as a transcript,
the CLI resumes it for exactly one model step, and the transcript is removed.
KT keeps running every tool; the CLI never executes one.
"""

import asyncio
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from typing import Any

from claude_agent_sdk import CLINotFoundError, query
from claude_agent_sdk._cli_version import __cli_version__

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.base import (
    BaseLLMProvider,
    ChatResponse,
    LLMConfig,
    OverflowRecoveryState,
    ToolSchema,
)
from kohakuterrarium.llm.recovery.errors import RetryPolicy, backoff_delay
from kohakuterrarium.llm.stream.replay import ReplayFilter
from kohakuterrarium.utils.logging import get_logger

from kt_claude_code.doctor import login_status
from kt_claude_code.errors import classify
from kt_claude_code.options import StepSettings, build_options
from kt_claude_code.stream import StepStream
from kt_claude_code.transcript import (
    TURN_KEY,
    plan_transcript,
    record_turn,
    transcript_entries,
)
from kt_claude_code.workspace import Workspace

logger = get_logger(__name__)

DEFAULT_MODEL = "sonnet"
DEFAULT_CONCURRENCY = 4
# How long to let the CLI finish writing its session after the result.
DRAIN_SECONDS = 5.0

# One semaphore per (event loop, concurrency limit), shared by every provider.
_SLOTS: (
    "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]]"
) = weakref.WeakKeyDictionary()


class ClaudeCodeProvider(BaseLLMProvider):
    """Stateless KT provider backed by the Claude Agent SDK and its CLI."""

    provider_name = "claude-code"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        cli_path: str | None = None,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        retry_policy: RetryPolicy | dict[str, Any] | None = None,
        workspace: Workspace | None = None,
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
    ):
        super().__init__(LLMConfig(model=model, retry_policy=retry_policy))
        self.settings = StepSettings(
            model=model, effort=effort, thinking=thinking, cli_path=cli_path
        )
        self.max_concurrency = max(1, int(max_concurrency))
        self._retry_policy = RetryPolicy.from_value(retry_policy)
        self._workspace = workspace or Workspace()
        self._query = query_fn or query
        self._last_usage: dict[str, int] = {}
        self._last_assistant_extra_fields: dict[str, Any] = {}
        self._workspace.sweep()

    @classmethod
    def from_profile(cls, profile: Any) -> "ClaudeCodeProvider":
        """Build from a resolved KT profile (model, effort, ``extra_body`` options)."""
        extra = dict(getattr(profile, "extra_body", None) or {})
        return cls(
            model=profile.model or DEFAULT_MODEL,
            effort=getattr(profile, "reasoning_effort", "") or None,
            thinking=extra.get("thinking"),
            cli_path=extra.get("cli_path"),
            max_concurrency=extra.get("max_concurrency", DEFAULT_CONCURRENCY),
            retry_policy=getattr(profile, "retry_policy", None),
        )

    @classmethod
    def doctor(cls) -> str:
        """One-line Claude Code status for ``kt doctor``; raises when unusable."""
        return login_status()

    def with_model(self, name: str) -> "ClaudeCodeProvider":
        """A sibling provider for another model, same settings and workspace."""
        if not name or name == self.settings.model:
            return self
        return ClaudeCodeProvider(
            model=name,
            effort=self.settings.effort,
            thinking=self.settings.thinking,
            cli_path=self.settings.cli_path,
            max_concurrency=self.max_concurrency,
            retry_policy=self._retry_policy,
            workspace=self._workspace,
            query_fn=self._query,
        )

    @asynccontextmanager
    async def _slot(self):
        per_loop = _SLOTS.setdefault(asyncio.get_running_loop(), {})
        semaphore = per_loop.setdefault(
            self.max_concurrency, asyncio.Semaphore(self.max_concurrency)
        )
        async with semaphore:
            yield

    async def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolSchema] | None = None,
        provider_native_tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream one step with KT's retry and context-recovery policy."""
        current = messages
        attempt = 0
        overflow = OverflowRecoveryState()
        replay = ReplayFilter()
        while True:
            replay.begin_attempt()
            try:
                async with aclosing(self._one_step(current, tools)) as step:
                    async for chunk in step:
                        if text := replay.feed(chunk):
                            yield text
                return
            except Exception as exc:
                cls = classify(exc)
                replacement = await self._recover_context(cls, exc, current, overflow)
                if replacement is not None:
                    current = replacement
                    continue
                if (
                    cls in self._retry_policy.retry_classes
                    and attempt < self._retry_policy.max_retries
                ):
                    attempt += 1
                    delay = backoff_delay(attempt, self._retry_policy)
                    logger.warning(
                        "provider_retry",
                        attempt=attempt,
                        error_class=cls.value,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _one_step(
        self, messages: list[dict[str, Any]], tools: list[ToolSchema] | None
    ) -> AsyncIterator[str]:
        self._last_tool_calls = []
        self._last_usage = {}
        self._last_assistant_extra_fields = {}
        plan = plan_transcript(messages, defer_calls=bool(tools))
        session_id = None
        last_uuid = None
        prompt_path = None
        recorded: list[dict[str, Any]] = []
        state = StepStream()
        try:
            if plan.history:
                session_id = self._workspace.new_session_id()
                entries = transcript_entries(
                    plan.history,
                    session_id=session_id,
                    cwd=self._workspace.cwd,
                    version=__cli_version__,
                    model=self.settings.model,
                    turns=plan.turns,
                    defer_calls=bool(tools),
                )
                last_uuid = entries[-1]["uuid"]
                self._workspace.write_transcript(session_id, entries)
            if plan.system:
                prompt_path = self._workspace.write_system_prompt(plan.system)
            options = build_options(
                self.settings,
                cwd=self._workspace.cwd,
                system_prompt_path=prompt_path,
                session_id=session_id,
                tools=tools,
            )
            async with self._slot():
                prompt = plan.prompt
                if isinstance(prompt, list):
                    prompt = _user_message(prompt, session_id or "")
                stream = self._query(prompt=prompt, options=options)
                try:
                    try:
                        async for msg in stream:
                            text = state.feed(msg)
                            if text:
                                yield text
                            if state.done:
                                break
                    except Exception:
                        if state.session_id:
                            await _drain(stream)
                        raise
                    if state.session_id:
                        await _drain(stream)
                finally:
                    await stream.aclose()
            if state.session_id and not state.cli_compacted_from:
                recorded = self._workspace.saved_entries(state.session_id, last_uuid)
        except CLINotFoundError as exc:
            raise LLMNotConfiguredError(
                f"Claude Code CLI not found ({exc}). Install the Claude Agent SDK "
                "or set extra_body.cli_path."
            ) from exc
        finally:
            self._workspace.cleanup(session_id, prompt_path)
            if state.session_id and state.session_id != session_id:
                self._workspace.cleanup(state.session_id, None)
        if state.cli_compacted_from:
            logger.warning(
                "Claude Code compacted the context itself; KT should compact next",
                prompt_tokens=state.cli_compacted_from,
            )
        calls, usage, fields = state.results()
        self._last_tool_calls = calls
        self._last_usage = usage
        turn = record_turn(plan.prompt, recorded)
        if turn is not None:
            fields = {**fields, TURN_KEY: turn}
        self._last_assistant_extra_fields = fields

    async def _complete_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        """Run one step without tools and return its full text."""
        parts = [chunk async for chunk in self._stream_chat(messages)]
        return ChatResponse(
            content="".join(parts),
            finish_reason="stop",
            usage=dict(self._last_usage),
            model=self.settings.model,
        )


async def _user_message(
    content: list[dict[str, Any]], session_id: str
) -> AsyncIterator[dict[str, Any]]:
    """``content`` as the single stream-json user message of a step."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
        "session_id": session_id,
    }


async def _drain(stream: AsyncIterator[Any]) -> None:
    """Read what is left so the CLI exits after saving, within a time limit."""

    async def consume() -> None:
        async for _ in stream:
            pass

    try:
        await asyncio.wait_for(consume(), DRAIN_SECONDS)
    except asyncio.TimeoutError:
        logger.debug("Claude Code did not exit in time after its result")
    except Exception as exc:
        logger.debug("Claude Code failed while exiting", error=str(exc))
