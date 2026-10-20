"""Turn the CLI's model list into ``llm_presets:`` entries for ``kohaku.yaml``.

The list comes from the CLI handshake (``get_server_info()["models"]``); each
entry names a model, what it resolves to, and the effort levels it accepts.
"""

import re
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

FAMILIES = ("opus", "fable", "sonnet", "haiku")
PRESET_PREFIX = "claude-code"
DEFAULT_EFFORT = "high"
CONTEXT_WINDOW = 1_000_000
# KT compacts at ``max_context * threshold``; smaller values compact earlier.
CONTEXT_GROUP: dict[str, dict[str, int]] = {
    "200k": {"max_context": 200_000},
    "400k": {"max_context": 400_000},
    "700k": {"max_context": 700_000},
    "1m": {"max_context": 1_000_000},
}
_MODEL_ID = re.compile(
    r"^claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d+))?$"
)


async def fetch_models(cwd: str) -> list[dict[str, Any]]:
    """The model list the signed-in CLI reports (starts the CLI, no model call)."""
    options = ClaudeAgentOptions(tools=[], setting_sources=[], cwd=cwd)
    async with ClaudeSDKClient(options) as client:
        info = await client.get_server_info() or {}
    return list(info.get("models") or [])


def _version(model_id: str, min_major: int) -> tuple[str, str] | None:
    """``(family, "5.5")`` for a public model id at ``min_major`` or later."""
    match = _MODEL_ID.match(model_id)
    if not match or match["family"] not in FAMILIES:
        return None
    if int(match["major"]) < min_major:
        return None
    version = match["major"] + (f".{match['minor']}" if match["minor"] else "")
    return match["family"], version


def _preset(name: str, model: str, efforts: list[str]) -> dict[str, Any]:
    preset: dict[str, Any] = {
        "name": name,
        "provider": PRESET_PREFIX,
        "model": model,
        "max_context": CONTEXT_WINDOW,
    }
    groups: dict[str, Any] = {}
    if efforts:
        preset["reasoning_effort"] = (
            DEFAULT_EFFORT if DEFAULT_EFFORT in efforts else efforts[-1]
        )
        groups["reasoning"] = {e: {"reasoning_effort": e} for e in efforts}
    groups["context"] = {k: dict(v) for k, v in CONTEXT_GROUP.items()}
    preset["variation_groups"] = groups
    return preset


def presets_from_models(
    models: list[dict[str, Any]], min_major: int = 5
) -> list[dict[str, Any]]:
    """Family aliases (newest version) first, then one preset per exact version."""
    aliases: dict[str, dict[str, Any]] = {}
    versions: dict[str, dict[str, Any]] = {}
    for entry in models:
        value = str(entry.get("value") or "")
        resolved = str(entry.get("resolvedModel") or value)
        found = _version(resolved, min_major)
        if found is None:
            continue
        family, version = found
        efforts = list(entry.get("supportedEffortLevels") or [])
        if value == family:
            aliases[family] = _preset(f"{PRESET_PREFIX}-{family}", family, efforts)
        name = f"{PRESET_PREFIX}-{family}-{version}"
        versions.setdefault(name, _preset(name, resolved, efforts))
    ordered = [aliases[f] for f in FAMILIES if f in aliases]
    return ordered + sorted(
        versions.values(), key=lambda p: _sort_key(p["name"]), reverse=True
    )


def _sort_key(name: str) -> tuple[int, tuple[int, ...]]:
    family, version = name.removeprefix(f"{PRESET_PREFIX}-").rsplit("-", 1)
    return (
        -FAMILIES.index(family),
        tuple(int(part) for part in version.split(".")),
    )
