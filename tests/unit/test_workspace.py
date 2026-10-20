"""Unit tests for ``kt_claude_code.workspace``."""

import json
import os
import time

from claude_agent_sdk import project_key_for_directory

from kt_claude_code.workspace import Workspace, claude_projects_dir


def test_projects_dir_follows_claude_config_dir(isolated_homes, monkeypatch):
    assert claude_projects_dir() == isolated_homes / "claude" / "projects"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert claude_projects_dir().parts[-2:] == (".claude", "projects")


def test_default_root_is_under_the_kt_config_dir(isolated_homes):
    ws = Workspace()
    assert ws.root == (isolated_homes / "kt" / "claude-code" / "workspace").resolve()
    assert ws.root.is_dir()


def test_transcript_lands_where_the_cli_resumes_it(workspace):
    sid = workspace.new_session_id()
    path = workspace.write_transcript(sid, [{"type": "user", "n": 1}, {"type": "x"}])
    assert path.parent.name == project_key_for_directory(workspace.root)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"type": "user", "n": 1},
        {"type": "x"},
    ]


def test_cleanup_removes_transcript_sidecar_and_prompt(workspace):
    sid = workspace.new_session_id()
    transcript = workspace.write_transcript(sid, [{"type": "user"}])
    sidecar = workspace.project_dir / sid / "subagents"
    sidecar.mkdir(parents=True)
    (sidecar / "a.jsonl").write_text("{}")
    prompt = workspace.write_system_prompt("You are Kohaku.")
    assert prompt.read_text() == "You are Kohaku."
    workspace.cleanup(sid, prompt)
    assert not transcript.exists()
    assert not (workspace.project_dir / sid).exists()
    assert not prompt.exists()
    workspace.cleanup(sid, prompt)


def test_cleanup_without_session(workspace):
    prompt = workspace.write_system_prompt("x")
    workspace.cleanup(None, prompt)
    assert not prompt.exists()


def test_sweep_removes_only_stale_files(workspace):
    old = workspace.write_transcript("old", [{"type": "user"}])
    fresh = workspace.write_transcript("fresh", [{"type": "user"}])
    old_prompt = workspace.write_system_prompt("old")
    other = workspace.root / "keep.txt"
    other.write_text("not ours")
    past = time.time() - 7200
    for path in (old, old_prompt, other):
        os.utime(path, (past, past))
    assert workspace.sweep() == 2
    assert not old.exists() and not old_prompt.exists()
    assert fresh.exists() and other.exists()


def test_saved_entries_are_the_ones_after_the_given_uuid(workspace):
    sid = workspace.new_session_id()
    path = workspace.write_transcript(
        sid, [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}]
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n[1]\n")
    assert workspace.saved_entries(sid, "a") == [{"uuid": "b"}, {"uuid": "c"}]
    assert workspace.saved_entries(sid, None) == [
        {"uuid": "a"},
        {"uuid": "b"},
        {"uuid": "c"},
    ]
    # An anchor the CLI did not keep means nothing can be trusted as new.
    assert workspace.saved_entries(sid, "gone") == []
    assert workspace.saved_entries("missing", None) == []
