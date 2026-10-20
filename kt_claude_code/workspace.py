"""Per-call files the CLI reads: the resumable transcript and the system prompt.

All files live under one KT-owned working directory, so they never mix with
the user's own Claude Code projects.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from claude_agent_sdk import project_key_for_directory

from kohakuterrarium.utils.config_dir import config_dir
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

STALE_SECONDS = 3600


def claude_projects_dir() -> Path:
    """The CLI's projects directory, honouring a user-set ``CLAUDE_CONFIG_DIR``."""
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base) if base else Path.home() / ".claude") / "projects"


class Workspace:
    """Writes and removes the files of one or more concurrent calls."""

    def __init__(self, root: Path | None = None):
        self.root = (root or config_dir() / "claude-code" / "workspace").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def cwd(self) -> str:
        """The working directory passed to the CLI."""
        return str(self.root)

    @property
    def project_dir(self) -> Path:
        """Where the CLI looks for transcripts of :attr:`cwd`."""
        return claude_projects_dir() / project_key_for_directory(self.root)

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    def write_transcript(self, session_id: str, entries: list[dict[str, Any]]) -> Path:
        """Write ``entries`` as ``<session_id>.jsonl`` for ``resume``."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        path = self.project_dir / f"{session_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def write_system_prompt(self, text: str) -> Path:
        """Write the system prompt to a unique file; returns its path."""
        path = self.root / f"system-{uuid.uuid4().hex}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def cleanup(self, session_id: str | None, prompt_path: Path | None) -> None:
        """Remove one call's transcript (and any sidecar dir) and prompt file."""
        if prompt_path is not None:
            prompt_path.unlink(missing_ok=True)
        if session_id is None:
            return
        (self.project_dir / f"{session_id}.jsonl").unlink(missing_ok=True)
        sidecar = self.project_dir / session_id
        if sidecar.is_dir():
            for child in sorted(sidecar.rglob("*"), reverse=True):
                child.rmdir() if child.is_dir() else child.unlink(missing_ok=True)
            sidecar.rmdir()

    def sweep(self, max_age: float = STALE_SECONDS) -> int:
        """Remove files older than ``max_age`` seconds left by crashed calls."""
        cutoff = time.time() - max_age
        removed = 0
        for folder, pattern in (
            (self.project_dir, "*.jsonl"),
            (self.root, "system-*.md"),
        ):
            if not folder.is_dir():
                continue
            for path in folder.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError as exc:
                    logger.debug("Workspace sweep skipped a file", error=str(exc))
        return removed
