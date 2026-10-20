"""Unit tests for ``kt_claude_code.doctor`` using a stand-in CLI script."""

import json
import stat
import sys

import pytest

from kt_claude_code.doctor import find_cli, login_status
from kt_claude_code.provider import ClaudeCodeProvider

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")


def _fake_cli(tmp_path, stdout: str):
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{stdout}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_signed_in_status_line_names_method_and_plan(tmp_path):
    status = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "subscriptionType": "max",
        "email": "someone@example.com",
    }
    cli = _fake_cli(tmp_path, json.dumps(status))
    line = login_status(str(cli))
    assert "signed in via claude.ai (max)" in line
    assert "someone@example.com" not in line


def test_not_signed_in_raises_with_login_hint(tmp_path):
    cli = _fake_cli(tmp_path, json.dumps({"loggedIn": False}))
    with pytest.raises(RuntimeError, match="/login"):
        login_status(str(cli))


def test_unreadable_output_raises(tmp_path):
    cli = _fake_cli(tmp_path, "not json")
    with pytest.raises(RuntimeError, match="Unreadable"):
        login_status(str(cli))


def test_missing_cli(tmp_path):
    assert find_cli(str(tmp_path / "nope")) is None
    with pytest.raises(RuntimeError, match="not found"):
        login_status(str(tmp_path / "nope"))


def test_bundled_cli_is_found_by_default():
    assert find_cli() is not None


def test_provider_doctor_uses_login_status(monkeypatch):
    monkeypatch.setattr("kt_claude_code.provider.login_status", lambda: "ok line")
    assert ClaudeCodeProvider.doctor() == "ok line"
