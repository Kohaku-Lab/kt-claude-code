"""KohakuTerrarium LLM provider that runs each step through local Claude Code.

Every ``chat()`` call is stateless: KT's history is written as a transcript,
the CLI resumes it for one model step, and the transcript is removed. A CLI
process may stay running for the next step of the same conversation (see
:mod:`kt_claude_code.live`); any other step starts a new one. KT keeps
running every tool; the CLI never executes one.
"""

import asyncio
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeSDKClient, CLINotFoundError, query
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
from kt_claude_code.images import count_images
from kt_claude_code.live import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_IDLE,
    Held,
    LiveSession,
    SessionPool,
)
from kt_claude_code.options import StepSettings, build_options, tools_key
from kt_claude_code.stream import StepStream
from kt_claude_code.transcript import (
    TURN_KEY,
    TranscriptPlan,
    conversation,
    plan_transcript,
    record_turn,
    reply_signature,
    safe_tool_id,
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
# Live-session pools per event loop, keyed by (workspace, keep_alive, max_idle).
_POOLS: (
    "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple, SessionPool]]"
) = weakref.WeakKeyDictionary()


class _StaleSession(Exception):
    """A parked CLI process that failed before answering; a fresh one may work."""


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
        keep_alive: float = DEFAULT_KEEP_ALIVE,
        max_idle_sessions: int = DEFAULT_MAX_IDLE,
        workspace: Workspace | None = None,
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ):
        super().__init__(LLMConfig(model=model, retry_policy=retry_policy))
        self.settings = StepSettings(
            model=model, effort=effort, thinking=thinking, cli_path=cli_path
        )
        self.max_concurrency = max(1, int(max_concurrency))
        self.keep_alive = max(0.0, float(keep_alive))
        self.max_idle_sessions = max(0, int(max_idle_sessions))
        self._retry_policy = RetryPolicy.from_value(retry_policy)
        self._workspace = workspace or Workspace()
        self._query = query_fn or query
        self._client_factory = client_factory or ClaudeSDKClient
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
            keep_alive=extra.get("keep_alive", DEFAULT_KEEP_ALIVE),
            max_idle_sessions=extra.get("max_idle_sessions", DEFAULT_MAX_IDLE),
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
            keep_alive=self.keep_alive,
            max_idle_sessions=self.max_idle_sessions,
            workspace=self._workspace,
            query_fn=self._query,
            client_factory=self._client_factory,
        )

    async def close(self) -> None:
        """Stop the CLI processes this provider left running for a next step."""
        pool = self._pool(create=False)
        if pool is not None:
            await pool.close_owned(self)

    def _pool(self, *, create: bool = True) -> SessionPool | None:
        """This loop's pool of live sessions; ``None`` when keep-alive is off."""
        if self.keep_alive <= 0 or self.max_idle_sessions <= 0:
            return None
        per_loop = _POOLS.get(asyncio.get_running_loop())
        if per_loop is None:
            if not create:
                return None
            per_loop = _POOLS.setdefault(asyncio.get_running_loop(), {})
        key = (self._workspace.cwd, self.keep_alive, self.max_idle_sessions)
        if key not in per_loop and create:
            per_loop[key] = SessionPool(
                self._workspace.cleanup,
                keep_alive=self.keep_alive,
                max_idle=self.max_idle_sessions,
            )
        return per_loop.get(key)

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
        """One model step on a live CLI process when one can continue, else a new one."""
        self._last_tool_calls = []
        self._last_usage = {}
        self._last_assistant_extra_fields = {}
        pool = self._pool()
        if pool is None:
            async with aclosing(self._oneshot_step(messages, tools)) as step:
                async for text in step:
                    yield text
            return
        system, body = conversation(messages)
        key = (self.settings, self._workspace.cwd, system, tools_key(tools))
        leased = pool.lease(key, body)
        if leased is not None:
            session, prompt = leased
            try:
                async with aclosing(
                    self._session_step(pool, session, prompt, body, tools)
                ) as step:
                    async for text in step:
                        yield text
                return
            except _StaleSession as exc:
                logger.debug("Parked Claude Code process failed", error=str(exc))
        async with aclosing(self._fresh_step(pool, key, messages, body, tools)) as step:
            async for text in step:
                yield text

    async def _fresh_step(
        self,
        pool: SessionPool,
        key: tuple[Any, ...],
        messages: list[dict[str, Any]],
        body: list[dict[str, Any]],
        tools: list[ToolSchema] | None,
    ) -> AsyncIterator[str]:
        """Start a CLI on KT's history; it stays parked when it can continue."""
        plan = plan_transcript(messages, defer_calls=bool(tools))
        session_id, last_uuid, prompt_path = self._write_step_files(plan, tools)
        options = build_options(
            self.settings,
            cwd=self._workspace.cwd,
            system_prompt_path=prompt_path,
            session_id=session_id,
            tools=tools,
        )
        session = LiveSession(
            self._client_factory(options),
            key,
            transcript_id=session_id,
            prompt_path=prompt_path,
            owner=self,
        )
        session.tail_uuid = last_uuid
        try:
            async with self._slot():
                await session.client.connect()
        except CLINotFoundError as exc:
            pool.discard(session)
            raise LLMNotConfiguredError(
                f"Claude Code CLI not found ({exc}). Install the Claude Agent SDK "
                "or set extra_body.cli_path."
            ) from exc
        except BaseException:
            pool.discard(session)
            raise
        continues = bool(body) and body[-1]["role"] == "user"
        async with aclosing(
            self._session_step(pool, session, plan.prompt, body, tools, reuse=continues)
        ) as step:
            async for text in step:
                yield text

    async def _session_step(
        self,
        pool: SessionPool,
        session: LiveSession,
        prompt: str | list[dict[str, Any]],
        body: list[dict[str, Any]],
        tools: list[ToolSchema] | None,
        *,
        reuse: bool = True,
    ) -> AsyncIterator[str]:
        """Send ``prompt`` to a connected CLI, then park or close it."""
        state = StepStream()
        started = False
        completed = False
        try:
            async with self._slot():
                async with aclosing(session.exchange(prompt)) as stream:
                    try:
                        async for msg in stream:
                            started = True
                            text = state.feed(msg)
                            if text:
                                yield text
                            if state.done:
                                break
                    except Exception as exc:
                        if not started and session.held is not None:
                            raise _StaleSession(str(exc)) from exc
                        raise
            if not state.done:
                if not started and session.held is not None:
                    raise _StaleSession("Claude Code exited while parked")
                raise RuntimeError("Claude Code exited before finishing the step")
            completed = bool(state.session_id) and not state.cli_compacted_from
            recorded: list[dict[str, Any]] = []
            if completed:
                recorded = self._workspace.saved_entries(
                    state.session_id, session.tail_uuid
                )
            tail = next((e["uuid"] for e in reversed(recorded) if e.get("uuid")), None)
            self._finish_step(state, prompt, recorded)
            if completed and reuse:
                blocks = state.content_blocks()
                pending = frozenset(safe_tool_id(c.id) for c in self._last_tool_calls)
                session.advance(
                    Held.of(body, reply_signature(blocks), pending),
                    state.session_id,
                    tail,
                )
                pool.park(session)
            else:
                completed = False
        finally:
            if not completed:
                pool.discard(session)

    def _write_step_files(
        self, plan: TranscriptPlan, tools: list[ToolSchema] | None
    ) -> tuple[str | None, str | None, Path | None]:
        """Write the step's transcript and system prompt files.

        Returns ``(session_id, last_uuid, prompt_path)``; each is ``None``
        when the plan needs no such file.
        """
        session_id = last_uuid = prompt_path = None
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
                extra_images=count_images(plan.prompt),
            )
            last_uuid = entries[-1]["uuid"]
            self._workspace.write_transcript(session_id, entries)
        if plan.system:
            prompt_path = self._workspace.write_system_prompt(plan.system)
        return session_id, last_uuid, prompt_path

    def _finish_step(
        self,
        state: StepStream,
        prompt: str | list[dict[str, Any]],
        recorded: list[dict[str, Any]],
    ) -> None:
        """Publish the step's calls, usage and assistant fields to KT."""
        if state.cli_compacted_from:
            logger.warning(
                "Claude Code compacted the context itself; KT should compact next",
                prompt_tokens=state.cli_compacted_from,
            )
        calls, usage, fields = state.results()
        self._last_tool_calls = calls
        self._last_usage = usage
        turn = record_turn(prompt, recorded)
        if turn is not None:
            fields = {**fields, TURN_KEY: turn}
        self._last_assistant_extra_fields = fields

    async def _oneshot_step(
        self, messages: list[dict[str, Any]], tools: list[ToolSchema] | None
    ) -> AsyncIterator[str]:
        """One step on a CLI that exits after it (``keep_alive`` 0)."""
        plan = plan_transcript(messages, defer_calls=bool(tools))
        session_id = None
        last_uuid = None
        prompt_path = None
        recorded: list[dict[str, Any]] = []
        state = StepStream()
        try:
            session_id, last_uuid, prompt_path = self._write_step_files(plan, tools)
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
        self._finish_step(state, plan.prompt, recorded)

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
