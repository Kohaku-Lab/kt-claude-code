"""Claude Code processes kept running between the steps of one conversation.

A step that only adds the CLI's own last answer and one new user message to
the conversation a parked process holds is sent to that process instead of
a new one, so a tool loop runs on one process rather than one per step.
"""

import asyncio
import time
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kohakuterrarium.utils.logging import get_logger

from kt_claude_code.transcript import body_digest, continuation

logger = get_logger(__name__)

DEFAULT_KEEP_ALIVE = 300.0
DEFAULT_MAX_IDLE = 2


@dataclass(frozen=True)
class Held:
    """The conversation a live process holds, as of the end of its last step.

    The body is kept as a digest: KT's message objects may change after a
    step, while the process keeps what it was sent.
    """

    size: int
    digest: str
    opening: str
    reply: list[tuple[str, ...]]
    pending: frozenset[str]

    @classmethod
    def of(
        cls,
        body: list[dict[str, Any]],
        reply: list[tuple[str, ...]],
        pending: frozenset[str],
    ) -> "Held":
        """What a process holds after a step on ``body`` that answered ``reply``."""
        return cls(len(body), body_digest(body), body_digest(body[:1]), reply, pending)


class LiveSession:
    """One connected CLI process and the files it was started from."""

    def __init__(
        self,
        client: Any,
        key: tuple[Any, ...],
        *,
        transcript_id: str | None,
        prompt_path: Path | None,
        owner: Any,
    ):
        self.client = client
        self.key = key
        self.transcript_id = transcript_id
        self.prompt_path = prompt_path
        self.owner = weakref.ref(owner)
        self.held: Held | None = None
        self.cli_session_id = ""
        self.tail_uuid: str | None = None
        self.idle_since = 0.0
        self._inbox: AsyncIterator[Any] | None = None
        self._timer: asyncio.TimerHandle | None = None

    async def exchange(self, prompt: str | list[dict[str, Any]]) -> AsyncIterator[Any]:
        """Send ``prompt`` as one user message and yield what the CLI answers."""
        if isinstance(prompt, str):
            await self.client.query(prompt)
        else:
            await self.client.query(_one_message(prompt))
        if self._inbox is None:
            self._inbox = aiter(self.client.receive_messages())
        while True:
            try:
                msg = await anext(self._inbox)
            except StopAsyncIteration:
                return
            yield msg

    def advance(self, held: Held, cli_session_id: str, tail_uuid: str | None) -> None:
        """Record what the process holds after a completed step."""
        self.held = held
        self.cli_session_id = cli_session_id or self.cli_session_id
        self.tail_uuid = tail_uuid or self.tail_uuid

    def files(self) -> list[tuple[str | None, Path | None]]:
        """``(session_id, prompt_path)`` pairs to remove once the process exits."""
        pairs = [(self.transcript_id, self.prompt_path)]
        if self.cli_session_id and self.cli_session_id != self.transcript_id:
            pairs.append((self.cli_session_id, None))
        return pairs


class SessionPool:
    """Idle live sessions of one event loop, closed after ``keep_alive`` seconds."""

    def __init__(
        self,
        cleanup: Callable[[str | None, Path | None], None],
        *,
        keep_alive: float = DEFAULT_KEEP_ALIVE,
        max_idle: int = DEFAULT_MAX_IDLE,
    ):
        self.keep_alive = keep_alive
        self.max_idle = max(0, max_idle)
        self._cleanup = cleanup
        self._idle: list[LiveSession] = []
        self._closing: set[asyncio.Task] = set()

    @property
    def idle(self) -> int:
        return len(self._idle)

    def lease(
        self, key: tuple[Any, ...], body: list[dict[str, Any]]
    ) -> tuple[LiveSession, str | list[dict[str, Any]]] | None:
        """Take the idle session ``body`` continues, with the prompt to send it."""
        for session in reversed(self._idle):
            if session.key != key or session.held is None:
                continue
            held = session.held
            prompt = continuation(
                body, held.size, held.digest, held.reply, held.pending
            )
            if prompt is None:
                continue
            self._idle.remove(session)
            self._stop_timer(session)
            _touch(session.prompt_path)
            return session, prompt
        return None

    def park(self, session: LiveSession) -> None:
        """Keep ``session`` for its next step, or close it when it cannot stay."""
        if self.keep_alive <= 0 or self.max_idle == 0 or session.held is None:
            self.discard(session)
            return
        for old in [s for s in self._idle if _same_thread(s, session)]:
            self.discard(old)
        while len(self._idle) >= self.max_idle:
            self.discard(self._idle.pop(0))
        session.idle_since = time.monotonic()
        loop = asyncio.get_running_loop()
        session._timer = loop.call_later(self.keep_alive, self._expire, session)
        self._idle.append(session)

    def discard(self, session: LiveSession) -> None:
        """Close ``session`` in the background; safe from a cancelled step."""
        self._stop_timer(session)
        if session in self._idle:
            self._idle.remove(session)
        task = asyncio.get_running_loop().create_task(self._close(session))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def close_owned(self, owner: Any) -> None:
        """Close the idle sessions ``owner`` parked and wait for every close."""
        for session in [s for s in self._idle if s.owner() in (owner, None)]:
            self.discard(session)
        await self.wait_closed()

    async def close_all(self) -> None:
        for session in list(self._idle):
            self.discard(session)
        await self.wait_closed()

    async def wait_closed(self) -> None:
        if self._closing:
            await asyncio.gather(*list(self._closing), return_exceptions=True)

    def _expire(self, session: LiveSession) -> None:
        session._timer = None
        if session in self._idle:
            self.discard(session)

    @staticmethod
    def _stop_timer(session: LiveSession) -> None:
        if session._timer is not None:
            session._timer.cancel()
            session._timer = None

    async def _close(self, session: LiveSession) -> None:
        try:
            await session.client.disconnect()
        except Exception as exc:
            logger.debug("Claude Code failed while exiting", error=str(exc))
        for session_id, prompt_path in session.files():
            self._cleanup(session_id, prompt_path)


async def _one_message(content: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def _same_thread(old: LiveSession, new: LiveSession) -> bool:
    """Whether ``new`` replaces ``old``: same setup and same opening message.

    A conversation that was edited, regenerated or compacted starts a new
    process; the one it left behind cannot be continued any more.
    """
    if old.key != new.key or old.held is None or new.held is None:
        return False
    return old.held.size > 0 and old.held.opening == new.held.opening


def _touch(path: Path | None) -> None:
    """Keep a live session's prompt file younger than the workspace sweep age."""
    if path is not None:
        try:
            path.touch()
        except OSError:
            pass
