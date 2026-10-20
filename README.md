# kt-claude-code

A [KohakuTerrarium](https://github.com/Kohaku-Lab/KohakuTerrarium) LLM backend
that runs each model step through **your own, already signed-in Claude Code**
(via the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)).

It is also a small, complete example of a package-provided backend
(`llm_backends:` in `kohaku.yaml`).

## No login, by design

This package has **no login method of any kind**. It never reads, copies or
stores Claude credentials, never sets `CLAUDE_CONFIG_DIR`, and never calls the
Messages API itself. You sign in to Claude Code yourself:

```bash
claude        # then type /login
```

The SDK's bundled, unmodified `claude` CLI then uses that sign-in. The only
thing the package reads is the CLI's own `claude auth status` (for `kt doctor`).

Use it for your own, personal work. Anthropic's terms apply to how you use
your plan; see [Legal and compliance](https://code.claude.com/docs/en/legal-and-compliance).

## Install

Requires a KohakuTerrarium release newer than 2.1.7 (the first with the `llm_backends:` package slot).

```bash
kt install @kt-claude-code      # from TerrariumMarket
# or: kt install https://github.com/Kohaku-Lab/kt-claude-code
kt doctor --llm claude-code/claude-code-sonnet --ping
```

`kt install` installs the `claude-agent-sdk` dependency into KohakuTerrarium's
environment. The SDK ships its own `claude` CLI.

## Use

Presets (all `claude-code/…`):

| Preset | Model |
|---|---|
| `claude-code-opus` / `-fable` / `-sonnet` / `-haiku` | Newest version of that family (CLI alias) |
| `claude-code-opus-5.5`, `-opus-5`, `-fable-5.1`, `-fable-5`, `-sonnet-5.5`, `-sonnet-5`, `-haiku-5.5` | That exact version |

Each preset has two variation groups:

| Group | Options | Default |
|---|---|---|
| `reasoning` | the effort levels the model supports (`low` … `max`) | `high` |
| `context` | `200k`, `400k`, `700k`, `1m` | `1m` |

`context` sets KT's own `max_context`, which decides when KT compacts the
conversation; the models' window is 1M either way. Example:
`claude-code/claude-code-opus-5.5@reasoning=max,context=400k`.

Pick a preset in Settings, with `/model`, or:

```yaml
# creature config.yaml
controller:
  llm: claude-code/claude-code-sonnet
  tool_format: native        # bracket / xml also work
```

Profile options:

| Field | Effect |
|---|---|
| `reasoning_effort` | Passed as the CLI `effort` (`low` … `max`). |
| `extra_body.thinking` | Thinking config; default `{type: adaptive, display: summarized}`. |
| `extra_body.cli_path` | Use another `claude` binary instead of the bundled one. |
| `extra_body.max_concurrency` | Parallel CLI processes (default 4). |

## How it works

Each `chat()` step is stateless, which matches what KohakuTerrarium expects:

1. KT's history is written as a Claude Code transcript (thinking dropped,
   tool ids made API-safe) under a KT-owned working directory.
2. The CLI resumes it with KT's system prompt, no built-in tools and no
   user settings. KT's tools are offered through an in-process MCP server.
3. The model's text streams back; any tool calls are returned to KT and
   **deferred** by a `PreToolUse` hook, so the CLI never runs a tool. KT runs
   them (plugins, sandbox and permission gates included).
4. The transcript and prompt files are deleted.

Because nothing is kept between steps, edit / regenerate / rewind / branch /
compaction / resume all work unchanged.

KT's own (non-blocking) compaction is the one that normally runs: it fires at
`max_context × threshold`, well before the CLI's auto-compact point near the
1M window. The CLI's compaction stays on only as a fallback (for example after
one very large tool result). When it happens, the package logs a warning and
reports the pre-compaction prompt size, so KT compacts on its next check.

## Limits

- About 1.2–1.5 s of CLI start-up per step, and ~1k extra prompt tokens the
  CLI adds. Prompt caching works across steps.
- `temperature`, `top_p`, stop sequences and provider-native tools are not
  supported.
- The transcript file format is internal to the CLI. The SDK pins the CLI
  version; run the live tests before upgrading `claude-agent-sdk`.

## Development

```bash
uv venv && uv pip install -e ../KohakuTerrarium -e ".[dev]"
pytest                          # unit + integration (no network)
KT_CLAUDE_LIVE=1 pytest tests/live   # real CLI, uses a little of your plan
```

The presets in `kohaku.yaml` are generated from the model list the signed-in
CLI reports. Refresh them after a new model ships (or an SDK upgrade):

```bash
python scripts/sync_models.py           # rewrite llm_presets in kohaku.yaml
python scripts/sync_models.py --check   # exit 1 if it is out of date
```

## License

Apache-2.0. See [LICENSE](LICENSE).
