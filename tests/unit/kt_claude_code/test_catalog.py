"""Unit tests for ``kt_claude_code.catalog`` and the preset sync script."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from kt_claude_code.catalog import CONTEXT_GROUP, presets_from_models

ROOT = Path(__file__).resolve().parents[3]
ALL_EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def _model(value, resolved, efforts=ALL_EFFORTS):
    entry = {"value": value, "resolvedModel": resolved, "displayName": value}
    if efforts:
        entry["supportedEffortLevels"] = efforts
    return entry


# Shape of ``get_server_info()["models"]`` as the CLI reported it (2026-10).
HANDSHAKE = [
    _model("default", "claude-opus-5-5"),
    _model("opus", "claude-opus-5-5"),
    _model("fable", "claude-fable-5-1"),
    _model("sonnet", "claude-sonnet-5-5"),
    _model("haiku", "claude-haiku-5-5"),
    _model("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", efforts=None),
    _model("claude-sonnet-5", "claude-sonnet-5"),
    _model("claude-opus-5", "claude-opus-5"),
    _model("claude-fable-5", "claude-fable-5"),
    _model("claude-opus-4-8", "claude-opus-4-8"),
    _model("claude-opus-4-6", "claude-opus-4-6", ["low", "medium", "high", "max"]),
    _model("claude-mythos-5-1", "claude-mythos-5-1"),
]


def _by_name(presets):
    return {p["name"]: p for p in presets}


class TestPresetsFromModels:
    def test_aliases_then_versions_newest_first(self):
        names = [p["name"] for p in presets_from_models(HANDSHAKE)]
        assert names == [
            "claude-code-opus",
            "claude-code-fable",
            "claude-code-sonnet",
            "claude-code-haiku",
            "claude-code-opus-5.5",
            "claude-code-opus-5",
            "claude-code-fable-5.1",
            "claude-code-fable-5",
            "claude-code-sonnet-5.5",
            "claude-code-sonnet-5",
            "claude-code-haiku-5.5",
        ]

    def test_aliases_keep_the_alias_and_versions_pin_the_id(self):
        presets = _by_name(presets_from_models(HANDSHAKE))
        assert presets["claude-code-opus"]["model"] == "opus"
        assert presets["claude-code-fable-5.1"]["model"] == "claude-fable-5-1"
        assert all(p["provider"] == "claude-code" for p in presets.values())

    def test_unlisted_families_and_older_majors_are_left_out(self):
        names = {p["name"] for p in presets_from_models(HANDSHAKE)}
        assert not any("mythos" in n or n.endswith("-4.8") for n in names)
        older = {p["name"] for p in presets_from_models(HANDSHAKE, min_major=4)}
        assert {"claude-code-opus-4.8", "claude-code-opus-4.6"} <= older
        # Dated ids are not a plain version and stay out.
        assert not any("haiku-4" in n for n in older)

    def test_effort_group_follows_each_model(self):
        presets = _by_name(presets_from_models(HANDSHAKE, min_major=4))
        opus = presets["claude-code-opus-5.5"]
        assert opus["reasoning_effort"] == "high"
        assert list(opus["variation_groups"]["reasoning"]) == ALL_EFFORTS
        assert opus["variation_groups"]["reasoning"]["max"] == {
            "reasoning_effort": "max"
        }
        old = presets["claude-code-opus-4.6"]["variation_groups"]["reasoning"]
        assert "xhigh" not in old

    def test_context_defaults_to_the_full_window_with_smaller_options(self):
        preset = presets_from_models(HANDSHAKE)[0]
        assert preset["max_context"] == 1_000_000
        assert preset["variation_groups"]["context"] == CONTEXT_GROUP
        assert preset["variation_groups"]["context"] is not CONTEXT_GROUP

    def test_model_without_effort_gets_no_reasoning_group(self):
        presets = presets_from_models(
            [_model("haiku", "claude-haiku-5-5", efforts=None)]
        )
        assert [p["name"] for p in presets] == [
            "claude-code-haiku",
            "claude-code-haiku-5.5",
        ]
        for preset in presets:
            assert "reasoning_effort" not in preset
            assert set(preset["variation_groups"]) == {"context"}


class TestSyncScript:
    @pytest.fixture
    def sync(self):
        spec = importlib.util.spec_from_file_location(
            "sync_models", ROOT / "scripts" / "sync_models.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_render_replaces_only_the_presets_section(self, sync):
        old = "name: x\nllm_backends:\n  - name: b\n\nllm_presets:\n  - name: old\n"
        new = sync.render(old, presets_from_models(HANDSHAKE[:2]))
        data = yaml.safe_load(new)
        assert data["llm_backends"] == [{"name": "b"}]
        assert [p["name"] for p in data["llm_presets"]] == [
            "claude-code-opus",
            "claude-code-opus-5.5",
        ]
        assert "&id" not in new

    def test_render_refuses_a_manifest_without_presets(self, sync):
        with pytest.raises(SystemExit):
            sync.render("name: x\n", [])

    def test_shipped_manifest_has_no_yaml_aliases(self):
        text = (ROOT / "kohaku.yaml").read_text(encoding="utf-8")
        assert "&id" not in text
        assert yaml.safe_load(text)["llm_presets"]
