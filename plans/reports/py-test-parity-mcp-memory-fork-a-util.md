# py-test-parity: mcp-util (fork A)

Scope: `tests/mcp/{test_mcp_util,test_caching,test_mcp_approval,test_runner_calls_mcp,test_mcp_tracing}.py`.
Counting: `def test_` per file via AST; expanded = `@pytest.mark.parametrize` list lengths multiplied
(stacked decorator on `test_structured_content_handling` counted by hand: 9 scenarios).
"Ported-ignored" = ported and compiling, `#[ignore = "cross-area: ..."]` because the fix is in a file I do not own.
No test is ignored for network/credentials.

## Per-file table

| Python file | def_test | expanded | ported (green) | ported-ignored (cross-area) | covered | python-only | cross-area unportable | unported | Rust files |
|---|---|---|---|---|---|---|---|---|---|
| test_mcp_util.py | 102 | 135 | 79 | 22 | 11 | 16 | 5 | 2 | py_mcp_util, py_mcp_util_schemas, py_mcp_util_logging |
| test_caching.py | 12 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | py_mcp_caching_approval |
| test_mcp_approval.py | 8 | 13 | 6 | 2 | 1 | 3 | 1 | 0 | py_mcp_caching_approval, py_mcp_runner_tracing |
| test_runner_calls_mcp.py | 12 | 24 | 22 | 0 | 0 | 0 | 2 | 0 | py_mcp_runner_tracing |
| test_mcp_tracing.py | 3 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | py_mcp_runner_tracing |
| **total** | 137 | 188 | 120 | 27 | 12 | 19 | 8 | 2 | |

120+27+12+19+8+2 = 188.

Rust test counts: py_mcp_util 34, py_mcp_util_logging 4, py_mcp_util_schemas 22 (17 run + 5 ignored),
py_mcp_caching_approval 12 (9 + 3 ignored), py_mcp_runner_tracing 16 (15 + 1 ignored). One Rust test often
covers several Python tests (parametrization / shared scenario). Support file `tests/py_mcpmem_support/mod.rs`
was used read-only by fork-A files only for pattern; fork A defines its own fakes in each file.

### Classification notes

- COVERED (test_mcp_util, 11): handoff-reserved names -> `py_mcp_runner_tracing::prefixed_mcp_names_do_not_collide_with_handoffs`;
  agent/server/None `failure_error_function` (3) -> `tests/core_mcp_failure_error_function.rs`; timeout handling ->
  `py_mcp_util::the_default_failure_formatter_hides_the_error_from_the_model`; 6 structured-content tests -> scenarios of
  `py_mcp_util::structured_content_handling_matches_python`. test_mcp_approval `tool_lists` -> `require_approval_per_tool_mapping`.
- PYTHON-ONLY (19): list-identity/aliasing of guardrail lists and public-constructor kwargs (Rust copies `Vec`s; policy set via
  `MCPServerConfig::with_tool_policy`); private `_get_mcp_tool_reserved_names`; `MCPError` type (2) and the `asyncio`
  cancellation group (8: `CancelledError`, `MCPToolCancellationError`, patched `asyncio.wait`; Rust cancels by dropping
  futures); legacy 3-arg `to_function_tool`; `dataclasses.replace`; invalid `require_approval` strings/dicts (3; the Rust
  `RequireApproval` enum cannot express them).
- CROSS-AREA unportable (8): `MCPTool` has no `_meta` field (3 static-meta tests; `src/mcp/mod.rs`);
  `FunctionTool` has no `mcp_annotations` accessor (3: 2 in util, 2 runner-hooks; `src/tool.rs`) ; `RequireApproval::Dynamic`
  callback gets `(agent_name, tool)`, Python also passes the run context (`src/mcp/mod.rs:215`) -> async-callable-uses-run-context test.
- Unported (2): guardrails-do-not-leak-across-servers/filtered-tools; disabled-handoff names ignored when prefixing. Both are
  small; not done for time, not for a missing seam.
- Duplicate-name-without-hint: Rust has no seam to force a post-prefix collision, so the hintless message is built by
  `duplicate_tool_names_error(.., prefixed=true)` but only the "distinct names when prefixed" half is asserted.
- Tracing: Python's `inline_snapshot` full-tree equality was replaced by field assertions on exported spans (mcp_tools
  `server`/`result`, function `mcp_data`/`output`, redaction, credential-free count == 3). Third phase of
  `test_mcp_tracing` (adding `test_tool_3`) is the same code path as phase two and is not repeated.
- Caching tests drive a real `MCPServerStreamableHttp` against an in-process JSON-RPC HTTP server (Python patches
  `ClientSession.list_tools`).

## Rust bugs fixed (src/mcp/util.rs)

1. Duplicate tool names: now collects all collisions per server, sorted, Python-`repr`-escaped (control chars, U+2028, U+202E),
   exact Python message incl. the `include_server_in_tool_names=True` hint (omitted when already prefixed).
2. Root `$schema` no longer stripped from MCP input schemas (Python passes it through).
3. Empty MCP content now returns `ToolOutput::Json([])` (Python returns `[]`) instead of empty text.
4. Schemas nested deeper than 100 levels fail with `UserError("... too deeply nested ...")` before any conversion, strict or not
   (iterative check; Python `_copy_json_schema`). Applied in `get_all_function_tools_for_agent`; the infallible public
   `to_function_tool` cannot raise.
5. SECURITY: `mcp_tools` span used the raw server name, so URL userinfo/query/fragment reached exported traces. Now
   `get_mcp_server_log_name`. (`function` span `mcp_data` was already sanitized.)
6. Bad-JSON handling: non-object JSON now says `...: expected a JSON object` (was the generic/redacted message); debug logs
   `Invalid JSON input for MCP tool` (redacted) or the full message; added the `Invoking MCP tool` debug log.
7. Redacted mode (`DONT_LOG_TOOL_DATA`) error log is now plain `Error invoking MCP tool`, not naming tool or server.

(Another session independently changed `prepare_schema` so strict conversion follows `config`; verified, tests green.)

## Cross-area deferrals (file, Python expectation, minimal fix)

- `src/strict_schema.rs` `ensure_strict_json_schema` (line ~88): no reject-open-objects mode. Python
  `ensure_strict_json_schema(schema, _reject_open_objects=not input_schema_is_empty)` makes free-form object arguments
  (`{"type":"object"}`, `properties:{}`, `required:[]`, `$schema`/`$comment` only), free-form roots, free-form items in arrays,
  anyOf/oneOf/allOf/$ref/$defs/definitions/components targets, nullable/advanced open roots, and preserved pure `$ref`
  (also `%20` encoded) fail strict conversion so the tool stays non-strict with the original schema. Fix: add the flag, fail on an
  object node with no declared `properties` (after `$ref` resolution), pass `true` unless the root schema was `{}`. Ignored tests:
  `py_mcp_util_schemas::{free_form_object_argument_falls_back_to_non_strict, free_form_root_falls_back_to_non_strict,
  free_form_object_in_array_items_and_supported_nodes, advanced_open_root_and_nullable_root_fall_back,
  preserved_pure_refs_fall_back_to_non_strict}` (22 Python tests).
- `src/mcp/mod.rs` `list_tools` (~line 1051) / `invalidate_tools_cache` (~1516): no refresh generation. Python: invalidation during an
  in-flight refresh must not be overwritten by it and `cached_tools` keeps the pre-invalidation snapshot; an older concurrent refresh must
  not overwrite a newer cache; an older refresh publishes if the newer fails. Fix: a monotonically increasing generation counter taken
  at refresh start, publish only when no newer generation published/invalidated; make invalidate mark dirty rather than clear
  `tools_cache`. Ignored: `cache_invalidation_during_refresh_is_preserved`, `older_concurrent_refresh_does_not_overwrite_newer_cache`,
  `older_refresh_publishes_when_newer_refresh_fails` (3).
- `src/run_internal/tools.rs:214`: with `pre_approval_tool_input_guardrails` the input guardrail runs while approval is already
  granted, so a resumed run executes it 3 times in total; Python runs it before the interruption and once after approval (2).
  Fix: run the pre-approval guardrail only in the `None` (undecided) arm. Ignored:
  `py_mcp_runner_tracing::pre_approval_guardrails_run_once_before_and_once_after_approval` (2 Python params; the `false`
  param of both streaming modes passes).
- `src/mcp/mod.rs`: `MCPTool` needs a `meta` (`_meta`) field forwarded by `to_function_tool` per call as a deep copy;
  `RequireApproval::Dynamic` should receive the run context.
- `src/tool.rs`: `FunctionTool` needs `mcp_annotations` (snapshot of `MCPTool.annotations`, `None` for non-MCP tools).

## Gates

- `py_mcp_util` 34, `py_mcp_util_logging` 4, `py_mcp_util_schemas` 17 (+5 ignored), `py_mcp_caching_approval` 9 (+3 ignored),
  `py_mcp_runner_tracing` 15 (+1 ignored), `py_mcp_tool_filtering` 4, `mcp_tests` 13, `--lib mcp::` 15: all green with `--all-features`.
- `cargo clippy --all-features --all-targets -- -D warnings` currently fails ONLY in other areas' files
  (`src/extensions/models/mod.rs:1016`, `src/models/openai_responses_ws.rs:553,556`), so my files could not be linted
  through clippy; rustfmt (edition 2024) applied to my files only.
- Build dir `/Users/ngocthach/.cache/oars-build/tp-mcpmem` is 2.9 GB, shared with other forks (not deleted).
- Shared-file edits: none (`src/lib.rs`, `Cargo.toml` untouched).
- Tracing processors and `DONT_LOG_TOOL_DATA` are process-global; the runner and logging test files serialize with one in-file lock.

Status: DONE_WITH_CONCERNS
Summary: 188 Python tests classified (120 green ports, 27 ported-but-ignored for cross-area fixes, 12 covered, 19 python-only, 8 cross-area unportable, 2 unported); 7 Rust bugs fixed in src/mcp/util.rs including credentials leaking into mcp_tools trace spans.
Concerns/Blockers: 27 ignored tests await fixes in src/strict_schema.rs, src/mcp/mod.rs (cache refresh generation, `_meta`, run context in approval callback), src/run_internal/tools.rs and src/tool.rs; clippy gate blocked by other areas' lint errors; 2 small tests left unported.
