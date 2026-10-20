"""Rewrite the ``llm_presets:`` section of ``kohaku.yaml`` from the CLI's model list.

Run with the package venv while signed in to Claude Code:

    python scripts/sync_models.py            # write kohaku.yaml
    python scripts/sync_models.py --check    # exit 1 if it would change
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import yaml

from kt_claude_code.catalog import fetch_models, presets_from_models

MANIFEST = Path(__file__).resolve().parents[1] / "kohaku.yaml"
SECTION = "llm_presets:"


def render(manifest_text: str, presets: list[dict]) -> str:
    """``manifest_text`` with its trailing ``llm_presets:`` section replaced."""
    head, sep, _old = manifest_text.partition(f"\n{SECTION}")
    if not sep:
        raise SystemExit(f"{MANIFEST.name} has no top-level {SECTION} section")
    body = yaml.safe_dump(
        {"llm_presets": presets}, sort_keys=False, default_flow_style=False
    )
    return f"{head}\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--min-major", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as cwd:
        models = asyncio.run(fetch_models(cwd))
    presets = presets_from_models(models, min_major=args.min_major)
    if not presets:
        raise SystemExit("The CLI reported no usable models; are you signed in?")
    old = MANIFEST.read_text(encoding="utf-8")
    new = render(old, presets)
    if args.check:
        return 0 if new == old else 1
    MANIFEST.write_text(new, encoding="utf-8")
    print(f"{len(presets)} presets: " + ", ".join(p["name"] for p in presets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
