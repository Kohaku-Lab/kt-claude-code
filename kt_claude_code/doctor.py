"""Read-only Claude Code status: which CLI runs, and whether it is signed in.

Only the CLI's own ``auth status`` output is read; no credential is touched.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import claude_agent_sdk
from claude_agent_sdk._cli_version import __cli_version__

from kt_claude_code.errors import LOGIN_HINT

STATUS_TIMEOUT = 30


def find_cli(cli_path: str | None = None) -> Path | None:
    """The CLI the SDK would run: an explicit path, the bundled one, or PATH."""
    if cli_path:
        path = Path(os.path.expanduser(cli_path))
        return path if path.exists() else None
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled"
    for name in ("claude", "claude.exe"):
        if (bundled / name).exists():
            return bundled / name
    found = shutil.which("claude")
    return Path(found) if found else None


def login_status(cli_path: str | None = None) -> str:
    """``"<cli> <version>, signed in via <method> (<plan>)"``; raises if unusable."""
    cli = find_cli(cli_path)
    if cli is None:
        raise RuntimeError("Claude Code CLI not found")
    proc = subprocess.run(
        [str(cli), "auth", "status", "--json"],
        capture_output=True,
        text=True,
        timeout=STATUS_TIMEOUT,
        check=False,
    )
    try:
        status = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unreadable `claude auth status` output: {exc}") from exc
    if not status.get("loggedIn"):
        raise RuntimeError(f"Claude Code is not signed in. {LOGIN_HINT}")
    method = status.get("authMethod") or "unknown"
    plan = status.get("subscriptionType")
    detail = f"{method} ({plan})" if plan else method
    return f"Claude Code {__cli_version__} at {cli}, signed in via {detail}"
