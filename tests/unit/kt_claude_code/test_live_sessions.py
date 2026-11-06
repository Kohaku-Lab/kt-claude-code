"""Unit tests for ``kt_claude_code.live``: the pool of parked CLI processes."""

import asyncio
import os
import time

from kt_claude_code.live import Held, LiveSession, SessionPool


class FakeClient:
    def __init__(self, fail_disconnect=False):
        self.disconnected = False
        self.fail_disconnect = fail_disconnect

    async def disconnect(self):
        self.disconnected = True
        if self.fail_disconnect:
            raise RuntimeError("already gone")


class Owner:
    """Something to hold a weak reference to, like a provider."""


def _opening(first="hi"):
    return [{"role": "user", "content": [{"type": "text", "text": first}]}]


def _held(first="hi", body=None):
    return Held.of(body or _opening(first), [("text", "Hello")], frozenset())


def _session(owner, key=("k",), first="hi", prompt_path=None, **kwargs):
    session = LiveSession(
        FakeClient(**kwargs),
        key,
        transcript_id="t-sid",
        prompt_path=prompt_path,
        owner=owner,
    )
    session.advance(_held(first), "cli-sid", "u1")
    return session


def _next_body(first="hi"):
    return _opening(first) + [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "user", "content": [{"type": "text", "text": "more"}]},
    ]


class TestLease:
    async def test_lease_finds_the_session_the_body_continues(self, tmp_path):
        owner = Owner()
        prompt = tmp_path / "system-x.md"
        prompt.write_text("sys", encoding="utf-8")
        os.utime(prompt, (0, 0))
        pool = SessionPool(lambda *_: None)
        session = _session(owner, prompt_path=prompt)
        pool.park(session)
        assert pool.lease(("other",), _next_body()) is None
        assert pool.lease(("k",), _next_body("bye")) is None
        leased, text = pool.lease(("k",), _next_body())
        assert leased is session and text == "more"
        assert pool.idle == 0
        # Leasing keeps the prompt file clear of the workspace sweep.
        assert prompt.stat().st_mtime > time.time() - 60
        assert pool.lease(("k",), _next_body()) is None

    async def test_history_changed_in_place_no_longer_matches(self):
        sent = _opening()
        session = _session(Owner())
        session.advance(_held(body=sent), "cli-sid", "u1")
        pool = SessionPool(lambda *_: None)
        pool.park(session)
        # KT edits a message object it already sent; the process holds the old text.
        sent[0]["content"][0]["text"] = "edited"
        assert pool.lease(("k",), _next_body("edited")) is None

    async def test_a_leased_session_does_not_expire(self):
        pool = SessionPool(lambda *_: None, keep_alive=0.02)
        session = _session(Owner())
        pool.park(session)
        pool.lease(("k",), _next_body())
        await asyncio.sleep(0.05)
        await pool.wait_closed()
        assert not session.client.disconnected


class TestPark:
    async def test_keep_alive_off_closes_at_once(self):
        closed = []
        pool = SessionPool(lambda sid, path: closed.append(sid), keep_alive=0)
        session = _session(Owner())
        pool.park(session)
        await pool.wait_closed()
        assert session.client.disconnected and pool.idle == 0
        assert closed == ["t-sid", "cli-sid"]

    async def test_a_session_holding_nothing_is_closed(self):
        empty = LiveSession(
            FakeClient(), ("k",), transcript_id=None, prompt_path=None, owner=Owner()
        )
        pool = SessionPool(lambda *_: None)
        pool.park(empty)
        await pool.wait_closed()
        assert empty.client.disconnected and pool.idle == 0

    async def test_a_new_session_of_the_same_conversation_retires_the_old(self):
        owner = Owner()
        pool = SessionPool(lambda *_: None, max_idle=4)
        old, other, new = _session(owner), _session(owner, first="x"), _session(owner)
        for session in (old, other, new):
            pool.park(session)
        await pool.wait_closed()
        assert old.client.disconnected
        assert not other.client.disconnected and not new.client.disconnected
        assert pool.idle == 2

    async def test_the_oldest_session_goes_over_the_limit(self):
        owner = Owner()
        pool = SessionPool(lambda *_: None, max_idle=2)
        sessions = [_session(owner, first=str(i)) for i in range(3)]
        for session in sessions:
            pool.park(session)
        await pool.wait_closed()
        assert [s.client.disconnected for s in sessions] == [True, False, False]


class TestClose:
    async def test_close_owned_spares_other_owners_and_closes_orphans(self):
        mine, theirs = Owner(), Owner()
        pool = SessionPool(lambda *_: None, max_idle=4)
        a = _session(mine, first="a")
        b = _session(theirs, first="b")
        gone = Owner()
        orphan = _session(gone, first="c")
        for session in (a, b, orphan):
            pool.park(session)
        del gone
        await pool.close_owned(mine)
        assert a.client.disconnected and orphan.client.disconnected
        assert not b.client.disconnected
        await pool.close_all()
        assert b.client.disconnected and pool.idle == 0

    async def test_a_failing_exit_still_removes_the_files(self):
        removed = []
        pool = SessionPool(lambda sid, path: removed.append(sid))
        session = _session(Owner(), fail_disconnect=True)
        pool.park(session)
        await pool.close_all()
        assert removed == ["t-sid", "cli-sid"]

    async def test_same_cli_session_id_is_removed_once(self):
        session = _session(Owner())
        session.advance(_held(), "t-sid", None)
        assert session.files() == [("t-sid", None)]
        assert session.tail_uuid == "u1"
