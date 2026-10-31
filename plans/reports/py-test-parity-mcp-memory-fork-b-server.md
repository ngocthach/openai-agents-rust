# Py test parity: mcp-memory fork B (MCP server / transport)

Scope: `/Users/ngocthach/myGit/openai-agents-python/tests/mcp/` server, manager, transport, pagination, resume files (17 files, 361 expanded tests).
Counting: AST `def test_` with `parametrize` list lengths multiplied (stacked parametrize multiplied). Group-level classification; a Python case counts PORTED only if a Rust assertion exercises the same behavior.

## Rust test files (60 tests, 2 ignored)
| file | tests |
|---|---|
| tests/py_mcp_manager.rs | 22 |
| tests/py_mcp_pagination.rs | 9 |
| tests/py_mcp_http_transport.rs | 18 (also holds the resources/prompts/auth/session-id ports; no separate py_mcp_resources_prompts.rs) |
| tests/py_mcp_server_errors.rs | 7 |
| tests/py_mcp_resume_binding.rs | 4 (2 `#[ignore]`d, cross-area failing, see below) |

All green: `cargo test --all-features --test <each>`; clippy `-D warnings` clean for my files; rustfmt applied. Transports use local TCP mock Streamable-HTTP MCP servers (no python, no network).

## Per-file table
| python file | def_test | expanded | ported | covered | python-only | cross-area (unported) | failures found |
|---|---|---|---|---|---|---|---|
| test_mcp_server_manager.py | 51 | 76 | 38 | 0 | 38 | 0 | 7 (fixed) |
| test_mcp_server_manager_cleanup_state.py | 3 | 3 | 0 | 2 | 1 | 0 | 0 |
| test_server_errors.py | 53 | 77 | 22 | 0 | 55 | 0 | 1 (fixed) |
| test_connect_disconnect.py | 2 | 2 | 2 | 0 | 0 | 0 | 1 (fixed, shared with above) |
| test_list_page_limits.py | 8 | 32 | 14 | 8 | 10 | 0 | 2 (fixed) |
| test_mcp_pagination_integration.py | 2 | 3 | 3 | 0 | 0 | 0 | 0 |
| test_mcp_resources.py | 9 | 9 | 9 | 0 | 0 | 0 | 1 (fixed) |
| test_prompt_server.py | 10 | 11 | 3 | 0 | 8 | 0 | 0 |
| test_message_handler.py | 2 | 2 | 0 | 1 | 1 | 0 | 0 |
| test_mcp_resume_binding.py | 15 | 61 | 4 | 0 | 0 | 57 | 2 (cross-area, ignored) |
| test_streamable_http_session_id.py | 7 | 7 | 3 | 0 | 4 | 0 | 0 |
| test_mcp_auth_params.py | 7 | 7 | 3 | 0 | 4 | 0 | 0 |
| test_streamable_http_client_factory.py | 14 | 15 | 0 | 5 | 10 | 0 | 0 |
| test_mcp_v2_http.py | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| test_mcp_version_compat.py | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| test_mcp_imports.py | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| test_client_session_retries.py | 33 | 37 | 14 | 3 | 20 | 0 | 1 (fixed) |
| **total** | 239 | 361 | 115 | 19 | 170 | 57 | |

Notes on groups:
- COVERED cites: cleanup_state -> py_mcp_manager::{manager_cleanup_failure_is_recorded, manager_keeps_duplicate_servers_once_and_reports_all_servers}; list_page_limits stdio/sse transport variants share `Core` with the http-tested path (py_mcp_pagination::*); message_handler -> mcp_tests::server_notifications_reach_the_message_handler; client_factory initialized-notification cases -> mcp_tests::{streamable_http_can_keep_the_session_and_tolerate_a_rejected_initialized, transport_errors_never_carry_credentials}; retries "only failed page / shared budget / repeated cursor" -> py_mcp_pagination::{successful_page_budget_survives_failed_page_retry, repeated_cursor_is_rejected_without_naming_the_cursor}.
- PYTHON-ONLY groups: worker-task / anyio cancel-scope / task-factory / `asyncio.CancelledError` policy / `suppress_cancelled_error` / BaseException propagation (manager, server_errors, retries); lifecycle timeout type/range validation (Rust `Duration` is typed); exception-graph, traceback-locals, `ExceptionGroup`, httpx request/response attachment scrubbing (Rust errors are plain strings; the equivalent contract "no secret reaches error text" is asserted in py_mcp_server_errors); `max_list_pages` type/zero validation (Rust `Option<usize>`); non-object `arguments` (type-enforced); mocks of `ClientSession`, `sse_client`/`streamablehttp_client` kwargs, `httpx` factories/proxy/SSL (httpx stack not used; Rust takes a `reqwest::Client`); MCP v2 / httpx2 tests (Python SDK dual-version stack; Rust uses rmcp); import-path tests; `FakeMCPPromptServer` tests that exercise the Python fake itself; `_get_session_id` callback internals.
- CROSS-AREA unported (57): test_mcp_resume_binding.py restore/streaming/schema-version matrix. `RunState::from_json` takes no agent in Rust and there is no restore-time recipient binding (see below), so restore variants have no entry point. 4 in-memory variants ported.

## Rust bugs fixed (src/mcp/mod.rs)
1. `MCPServerManager` lifecycle rewritten to Python semantics: tracks a connected set; `connect_all` skips already-connected servers (idempotent) and clears prior failures; strict failure closes everything it opened plus the failed server before returning, restoring previously-connected active list; `reconnect(true)` cleans failed servers first and does not retry a server whose cleanup failed (error stays in `errors()`); `reconnect(false)` cleans all then reconnects; cleanup runs in reverse order and records failures; `errors()` is one entry per server (cleanup error replaces connect error); `active_servers()` is all servers before any attempt (Python); connect failures logged at error level.
2. New manager API (additive): `set_strict`, `is_strict`, `with_drop_failed_servers`, `drops_failed_servers`, `connect_timeout`/`cleanup_timeout` getters, `set_connect_timeout`/`set_cleanup_timeout`.
3. Not-connected error text now `Server not initialized. Make sure you call `connect()` first.` (was `Server '<name>' not initialized. Call connect() first.`).
4. Pagination errors use Python text: `... returned a repeated cursor while listing tools|prompts.` and `MCP tool|prompt listing exceeded max_list_pages. Increase max_list_pages to allow more pages, or check the server pagination.` (no cursor/payload/limit in text).
5. `list_tools` checks for a live session before serving the cache (a disconnected server no longer answers from a stale list; Python order).
6. `call_tool` validates required parameters against the cached tool schema before any request: `Failed to call tool '<t>' on MCP server '<name>': missing required parameters: a, b` (sorted; applies with `arguments=None`; skipped when tool/cache/`required` absent). New `Core::validate_required_parameters`.
(Another agent concurrently changed `list_tools` to retry per page; test results above include that change.)

## Breaking changes
- Error message strings in items 3, 4 above.
- `active_servers()` before `connect_all()` now returns all managed servers (was empty).
- `cleanup_all()` order is now reverse of registration.
- Strict `connect_all` failure now also cleans up and resets the active list; `failed_servers()` semantics follow Python.
- `list_tools` on a disconnected server with cache enabled now errors instead of returning the cache.
- `call_tool` can now fail locally with a UserError when a cached schema marks a missing required parameter.

## Cross-area deferrals (not edited)
1. Handoff/MCP name collision. py_mcp_resume_binding::mcp_approval_cannot_authorize_a_handoff_with_the_same_name (`#[ignore]`, not network-related: it fails). Rust `get_all_function_tools*` (src/mcp/util.rs ~line 159, fork A) reserves handoff names unconditionally, so an agent with a handoff named like an MCP tool fails with `Duplicate tool names found across MCP servers`. Python only reserves handoff names when `include_server_in_tool_names` is on; otherwise run resolution proceeds and raises `MCP tool call as a handoff` for an approved call / proceeds for a rejected one. Minimal fix: reserve handoff names only when prefixing; add the approved-MCP-as-handoff guard in src/run_internal tool resolution.
2. Recipient binding. py_mcp_resume_binding::approval_is_rejected_when_the_recipient_changes (`#[ignore]`, fails). Python binds an approved MCP call to the server that offered it (`ToolOrigin.mcp_server_name`) and raises `UserError: different recipient` on resume if another same-named server now provides the tool; Rust re-resolves and runs the call (result "done"). Fix location: src/run_state*/src/run_internal (store server identity on the approval item; compare at resume). Also blocks the 57 restore/streaming cases (needs `RunState::from_json(agent, ..)`-style restore).
Both ignores are for real failures (cross-area), not for network/credentials; unignore once fixed.

## Shared-file edits
None (src/lib.rs, Cargo.toml untouched; tests/py_mcpmem_support/mod.rs only read, `fake_tool` and `FakeMcpServer` used).

Status: DONE_WITH_CONCERNS
Summary: Ported/covered 134 of 361 expanded cases with 60 Rust tests (all green except 2 ignored cross-area failures), classified 170 python-only, 57 cross-area unported; fixed 6 Rust divergences in src/mcp/mod.rs (manager lifecycle, messages, cache/session order, required-parameter validation).
Concerns/Blockers: 2 real divergences live outside my ownership (handoff-vs-MCP name reservation in src/mcp/util.rs; MCP recipient binding in run state/run_internal) and are `#[ignore]`d with reasons; restore-based resume tests (57) cannot be ported until RunState restore takes an agent.
