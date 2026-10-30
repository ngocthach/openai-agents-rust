# Codex extension — final parity report

Date: 2026-10-30. Scope: `src/extensions/codex/**`, `tests/codex_tests.rs`,
`tests/fixtures/fake_codex.sh`, `examples/codex_tool.rs`. Python reference:
`agents.extensions.experimental.codex` (SDK 0.24).

## Closed rows

1. **Result string format** — `CodexToolResult: Display` now emits Python's `json.dumps` bytes:
   `", "` / `": "` separators, `ensure_ascii` escaping (astral chars as surrogate pairs), field
   order `thread_id, response, usage{input_tokens, cached_input_tokens, output_tokens}`. New
   module `src/extensions/codex/py_json.rs`.
   Tests: `tool::tests::serializes_the_tool_result_like_python_json_dumps` (non-ASCII + emoji),
   `py_json::tests::*`, integration
   `the_tool_result_is_byte_identical_to_pythons_json_dumps`.
2. **`_enforce_span_data_budget` full allocator** — ported verbatim into
   `src/extensions/codex/span_budget.rs` (all 8 `SPAN_TRIM_KEYS`, base-key dropping, the 1-char
   floor, the `arguments` grant, proportional weights with Python's `int()` truncation, the
   non-round-robin leftover loop, the final 1-char shrink, `_drop_empty_string_fields`).
   `span_data_max_chars` is now a **whole-payload** budget measured as compact `ensure_ascii`
   JSON, not a per-value limit. `_truncate_span_value` / `_stringify_span_value` /
   `_json_char_size` / `_merge_span_data` / `_apply_span_updates` ported too.
   Tests: `span_budget::tests::*` — every expectation is the output of the Python functions run
   on the same input (script kept out of the repo). Covers: fits, two competing keys, arguments
   kept intact, non-ASCII sizing, base-too-small key dropping, budget below the key count,
   nested-value stringification, null/empty handling.
   Also `tool::tests::the_span_budget_covers_the_whole_payload`.
3. **`TurnOptions.signal`** — `TurnOptions::cancel: Option<CancellationToken>` +
   `with_cancel()`, threaded to `CodexExecArgs::cancel`. The read loop selects (biased) on the
   token; firing it kills the child and the turn ends with the process's exit error, as Python's
   `terminate()` path does. An idle timeout now also cancels the token (Python's
   `signal.set()`), and `Thread::run_streamed` creates a token when only `idle_timeout` is set.
   Drop-cancellation (`kill_on_drop`) stays. Divergence: the select is `biased`, so once the
   token fires, lines already buffered are dropped; Python's generator keeps yielding until the
   pipe reaches EOF. Chosen for deterministic tests.
   Test: integration `a_cancellation_token_kills_the_codex_process` (fixture gained
   `FAKE_CODEX_SLEEP`, which `exec`s `sleep` so a kill reaches it).
4. **`codex_subprocess_stream_limit_bytes`** — `CodexOptions::codex_subprocess_stream_limit_bytes`
   + `CodexExec::with_subprocess_stream_limit_bytes`, resolved from the explicit value, then
   `OPENAI_AGENTS_CODEX_SUBPROCESS_STREAM_LIMIT_BYTES`, then 8 MiB; range 64 KiB..=64 MiB with
   Python's two `UserError` texts. `tokio::io::Lines` replaced by a bounded `fill_buf`/`consume`
   reader that mirrors asyncio's two overrun branches ("Separator is not found, and chunk exceed
   the limit" / "Separator is found, but chunk is longer than limit"), prefixed with
   `Codex stdout exceeded the {limit} byte subprocess stream limit: `. Trailing `\r` is kept
   (Python `rstrip("\n")`).
   Tests: `exec::tests::resolves_the_subprocess_stream_limit`,
   `the_line_reader_splits_on_newlines_and_keeps_carriage_returns`,
   `the_line_reader_rejects_lines_over_the_limit`,
   `the_line_reader_rejects_a_long_line_it_read_whole`; integration
   `a_line_over_the_subprocess_stream_limit_fails_the_turn`,
   `an_out_of_range_subprocess_stream_limit_is_rejected`.
5. **`on_stream` backpressure** — events now go through an `mpsc::unbounded_channel` drained by a
   spawned consumer task, exactly Python's unbounded `asyncio.Queue` + `_dispatch`. Order
   preserved; the sender is dropped and the consumer awaited before `consume_events` returns, so
   every event is handled. This matches `run_producer_consumer` (`agents/util/_asyncio_tasks.py:159`):
   on a producer failure it `await`s the consumer to **drain** before re-raising (no
   `fail_fast_exceptions` are passed here), which is exactly drop-sender-then-await.
   Test: `tool::tests::on_stream_callbacks_do_not_block_the_codex_stream` — the whole Codex
   stream is drained while callback #1 is blocked on a oneshot.
6. **Custom `params_json_schema` extra-field rejection** — no JSON-schema validator exists in the
   crate (`strict_schema.rs` only transforms), so a minimal one honours `additionalProperties:
   false` and `required` at the top level and on `properties.inputs.items`.
   Tests: `tool::tests::a_custom_schema_validates_its_own_fields`,
   `an_open_custom_schema_ignores_extra_fields`; existing integration
   `a_custom_parameters_schema_may_carry_extra_fields` still passes.
   Divergence, now in the strict direction: pydantic accepts a missing field that has a default
   even when the strict schema lists it in `required`; the Rust check rejects it. Strict-mode
   models always send every field, so the practical impact is nil.
9. **Vendored binary lookup** — confirmed Python's remaining order is `CODEX_PATH` →
   `shutil.which("codex")` → vendor path. Rust matches the first two; `find_codex_path` now also
   checks the executable bit on unix like `shutil.which`'s `os.access(X_OK)`, instead of only
   `is_file()`. The vendor fallback is inherent (no vendored binaries ship) and documented on
   `find_codex_path` and in the module docs. `find_codex_path` is now a thin wrapper over
   `resolve_codex_path(codex_path, search_path)` so the order is testable without touching
   process env.
   Test: `exec::tests::resolves_the_codex_path_like_shutil_which`.
11. **`tests/codex_tests.rs` `#![cfg(unix)]`** — re-verified and kept. The fixture is a POSIX
    shell script spawned as the `codex` binary; `.github/workflows/ci.yml` runs only on
    `ubuntu-latest`, and the lib unit tests (which now cover the line reader, the span allocator,
    the result format, the tracing policy and the `on_stream` queue) spawn nothing, so no
    coverage is lost.

**Per-run tracing policy (audit row 105 / old "requested core change" 2) — closed.**
`ToolContext` now carries `run_config`, and `run_internal/tools.rs` + `local_tools.rs` populate
it. `consume_events` resolves a `TracingPolicy` from it exactly like Python: `tracing_disabled`
suppresses span creation (`_handle_item_started` only), and `include_sensitive_data` is
`run_config.is_none() || run_config.trace_include_sensitive_data`. The tool-level
`include_sensitive_span_data` (which Python does not have) is kept and ANDed with the run policy,
so an existing opt-out still works and the run can only tighten it.
Tests: `tool::tests::the_run_config_decides_the_tracing_policy`,
`a_disabled_run_creates_no_command_span`, `omits_sensitive_span_data_when_disabled`.

## Remaining rows

7. **`use_run_context_thread_id` / `run_context_thread_id_key`** — still not expressible.
   Re-verified against the new `ToolContext`: `set_custom_data`/`custom_data` live on an
   `Arc<Mutex<Option<Value>>>` created fresh in `ToolContext::from_call`, so they are *per tool
   call*, not run-scoped, and they feed the tool-output item rather than the run context.
   `tool_namespace` is a tool-name namespace, not a storage key space. `RunContext<C>` still
   exposes only `&C` with no generic key/value slot. `with_thread_id_store(get, set)` is kept and
   now documents why. Python's key derivation (`codex_thread_id[_suffix]`) is meaningless without
   `setattr`/mapping access, so it was not ported.
   Test (unchanged): integration `codex_tool_stores_the_thread_id_on_the_run_context`.
8. **`_is_codex_tool` marker + duplicate-codex-tool fail-fast** — `FunctionTool` still has no
   usable slot (`is_agent_tool`, `mcp_server_name`, `namespace`, `namespace_description` are
   `pub(crate)` but carry different meaning). Requested change below. Documented in the module
   docs as a difference.
10. **`_DictLike` (`payloads.py`)** — inherent: Python-only mapping sugar over dataclasses; Rust
    uses `Serialize`/`Deserialize` + `to_value()`.
10b. **`codex_tool(...)` keyword overload, mapping inputs, `coerce_*`** — inherent: Rust uses one
    typed options struct with builder methods.

## Breaking changes (public API)

- `TurnOptions` no longer derives `PartialEq` (it holds a `CancellationToken`). Nothing in the
  crate compared it. New public field `cancel` and builder `with_cancel`.
- `CodexExecArgs` gained the public field `cancel` (struct-literal construction with `..Default::default()`
  is unaffected; exhaustive literals need the new field).
- `CodexOptions` gained the public field `codex_subprocess_stream_limit_bytes` (same note).
- `span_data_max_chars` changed meaning: a whole-payload JSON character budget instead of a
  per-value character limit. Spans built with the default 8192 now carry at most 8192 characters
  of JSON in total.
- `CodexToolResult::to_string()` output changed (spaces after `,` and `:`, `\uXXXX` escaping).
  The parsed JSON is unchanged.
- With a custom `params_json_schema`, arguments that violate its top-level/`inputs.items`
  `additionalProperties: false` or `required` are now rejected instead of ignored.
- A process killed by a signal now reports `-signal` (e.g. `Codex exec exited with code -9: …`)
  instead of `-1`, matching Python's `returncode`.
- New public items: `SUBPROCESS_STREAM_LIMIT_ENV`, `DEFAULT/MIN/MAX_SUBPROCESS_STREAM_LIMIT_BYTES`,
  `CodexExec::with_subprocess_stream_limit_bytes`,
  `CodexOptions::with_codex_subprocess_stream_limit_bytes`, re-export of
  `tokio_util::sync::CancellationToken`.

## Requested changes in files I do not own

1. `src/tool.rs` — add `pub(crate) is_codex_tool: bool` to `FunctionTool<C>` (default `false`,
   copied in `Clone`, set by `codex_tool`). Python: `tool.py:572`
   `_is_codex_tool: bool = field(default=False, kw_only=True, repr=False)`.
2. `src/agent.rs` — port `agent.py:110 _validate_codex_tool_name_collisions`, called where
   `get_all_tools` assembles the final tool list (`agent.py:317`): collect the names of tools
   with `is_codex_tool`, count every tool name, and if any Codex name occurs more than once raise
   `AgentsError::user("Duplicate Codex tool names found: " + sorted names joined with ", " +
   ". Provide a unique codex_tool(name=...) per tool instance.")`.

Nothing else is needed; the example is auto-discovered and `tests/fixtures/fake_codex.sh` keeps
its executable bit.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/misc2 \
       CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
rustfmt --edition 2024 src/extensions/codex/*.rs tests/codex_tests.rs examples/codex_tool.rs
cargo clippy --all-features --all-targets        # zero findings in src/extensions/codex/**,
                                                 # tests/codex_tests.rs, examples/codex_tool.rs
cargo test --all-features --lib extensions::codex   # 63 passed
cargo test --all-features --test codex_tests        # 18 passed
cargo test --all-features --doc extensions::codex   # 3 passed
```

Clippy findings seen in other agents' files — noted, not fixed:
`src/sandbox/events.rs:510` `match_like_matches_macro`, one `manual_is_multiple_of`, and
`tests/core_python_run_state.rs:60` dead `TRACE_ID`. `src/models/openai_chatcompletions.rs:1352`
and `:1381` also emit `unused_assignments`.

No network, no real `codex` binary: every integration test drives `tests/fixtures/fake_codex.sh`
with the dummy key `sk-test` and a fully replaced process environment, so a developer's real
`OPENAI_API_KEY` cannot leak in. The fixture keeps its executable bit (`0755`).

## Shared-file edits

None. `Cargo.toml` and `src/lib.rs` were not touched (`tokio-util 0.7` and `tokio` with `sync`
were already dependencies).
