# Python test parity: MCP + memory area

Scope: Python `tests/mcp` (24 files), `tests/memory` (10), `tests/deferred_session_writes` (6), `tests/utils` (3), `tests/extensions/memory` (8). Crate 0.5.0, work targets the unreleased 0.6.0 tree. No git commands used.

Counting: `def_test` = AST count of `test_` functions; `expanded` = parametrize multiplied (script: scratchpad `count_tests.py`, literal-list `parametrize` only, so a few expansions are lower bounds). AST total for the scope is 1692 expanded tests (mcp 548, memory 407, deferred 158, utils 12, extensions/memory 567).

## Final gate results

- All 36 `tests/py_mcp*.rs`, `tests/py_memory*.rs`, `tests/py_deferred*.rs` binaries plus `sessions_tests` and `mcp_tests`: **540 passed, 0 failed, 41 ignored** (38 binaries), `--all-features`.
- Every `#[ignore]` is a cross-area runner/strict-schema gap, none for network/credentials/docker. Redis, MongoDB and Dapr run against in-process mocks (RESP2, OP_MSG, HTTP sidecar), so no test needs a real server.
- rustfmt applied to every file in this area (never `src/lib.rs`).
- `cargo clippy --all-features --all-targets -- -D warnings` could NOT complete cleanly at the end. Other agents' in-flight lib edits (`src/models/openai_chatcompletions.rs:1590`, `src/sandbox/**`, `src/run_internal/local_tools.rs:135`) and `tests/py_sandbox_a_archive_tar_utils.rs:811,902` (useless `vec!`) fail it before my targets compile. Earlier runs of the lib reached my files cleanly; each fork reported clippy-clean on its own files when the lib built. Re-run once the other areas land.

## Per-Python-file table (def_test / expanded)

Legend: P = ported (green), I = ported but `#[ignore]` cross-area, C = covered by an existing Rust test, PO = python-only, X = cross-area not ported, U = unported.

### tests/mcp (fork A, fork B)
| file | def | exp | P | I | C | PO | X | U | Rust file(s) |
|---|---|---|---|---|---|---|---|---|---|
| test_mcp_util.py | 102 | 135 | 79 | 22 | 11 | 16 | 5 | 2 | py_mcp_util, py_mcp_util_schemas, py_mcp_util_logging |
| test_caching.py | 12 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | py_mcp_caching_approval |
| test_mcp_approval.py | 8 | 13 | 6 | 2 | 1 | 3 | 1 | 0 | py_mcp_caching_approval, py_mcp_runner_tracing |
| test_runner_calls_mcp.py | 12 | 24 | 22 | 0 | 0 | 0 | 2 | 0 | py_mcp_runner_tracing |
| test_mcp_tracing.py | 3 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | py_mcp_runner_tracing |
| test_tool_filtering.py | 6 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | py_mcp_tool_filtering |
| test_mcp_server_manager.py | 51 | 76 | 38 | 0 | 0 | 38 | 0 | 0 | py_mcp_manager |
| test_mcp_server_manager_cleanup_state.py | 3 | 3 | 0 | 0 | 2 | 1 | 0 | 0 | covered by py_mcp_manager |
| test_server_errors.py | 53 | 77 | 22 | 0 | 0 | 55 | 0 | 0 | py_mcp_server_errors |
| test_connect_disconnect.py | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | py_mcp_manager |
| test_list_page_limits.py | 8 | 32 | 14 | 0 | 8 | 10 | 0 | 0 | py_mcp_pagination |
| test_mcp_pagination_integration.py | 2 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | py_mcp_pagination |
| test_mcp_resources.py | 9 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | py_mcp_http_transport |
| test_prompt_server.py | 10 | 11 | 3 | 0 | 0 | 8 | 0 | 0 | py_mcp_http_transport |
| test_message_handler.py | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | mcp_tests |
| test_mcp_resume_binding.py | 15 | 61 | 4 (2 ignored) | 0 | 0 | 0 | 57 | 0 | py_mcp_resume_binding |
| test_streamable_http_session_id.py | 7 | 7 | 3 | 0 | 0 | 4 | 0 | 0 | py_mcp_http_transport |
| test_mcp_auth_params.py | 7 | 7 | 3 | 0 | 0 | 4 | 0 | 0 | py_mcp_http_transport |
| test_streamable_http_client_factory.py | 14 | 15 | 0 | 0 | 5 | 10 | 0 | 0 | py_mcp_http_transport |
| test_mcp_v2_http.py | 16 | 16 | 0 | 0 | 0 | 16 | 0 | 0 | python `mcp`/`httpx` v2 internals |
| test_mcp_version_compat.py | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | Python package version |
| test_mcp_imports.py | 2 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | import paths |
| test_client_session_retries.py | 33 | 37 | 14 | 0 | 3 | 20 | 0 | 0 | py_mcp_http_transport |

Notes: the 57 cross-area resume-binding cases need `RunState::from_json` to accept an agent (Rust takes none). Fork A's `test_tool_filtering.py` row was ported by me directly (4 Rust tests cover all 6 defs / 7 expanded).

### tests/memory, tests/utils, tests/extensions/memory
| file | def | exp | outcome | Rust file(s) |
|---|---|---|---|---|
| memory/test_session.py | 40 | 56 | 21 ported (run/stream basics, limits, corrupt rows, settings, failure paths), run_sync/streamed variants ported by C2, PO 3 (sqlite thread/connection pools, asyncio cancel), remainder U (connection-pool internals classified PO by C2 where applicable) | py_memory_session, py_memory_remaining_c |
| memory/test_session_limit.py | 5 | 15 | 4 ported (run/streamed collapse to one Rust run path); 1 ignored: orphan `function_call_output` after a limit cut (cross-area) | py_memory_limits_helpers |
| memory/test_session_helpers.py | 5 | 6 | 5 ported/PO (await_session_mutation cancellation is documented Rust difference; settings coercion PO) | py_memory_limits_helpers |
| memory/test_session_context_wrapper.py | 12 | 16 | 2 PO, 10 cross-area (Rust sessions take no `wrapper`/RunContext) | not ported; see deferrals |
| memory/test_session_persistence_sanitize.py | 9 | 27 | 26 ported (8 defs) via Conversations write path, 1 PO | py_memory_conversations, py_memory_remaining_a |
| memory/test_compaction_suffix.py | 11 | 23 | 3 ported, 4 PO, 4 cross-area (encrypt-over-compaction ordering) | py_memory_remaining_a |
| memory/test_compaction_model_visibility.py | 19-20 | 81 | 1 variant green (`session` filter); `model`+`redact` variants ignored cross-area; rest cross-area | py_memory_compaction_visibility |
| memory/test_openai_conversations_session.py | 52 | 74 | 22 + 2 ported (zero limit, pagination, chunking, delete, clear, concurrency, sanitize, runner), 13 PO, ~17 U | py_memory_conversations, py_memory_remaining_a/c |
| memory/test_openai_responses_compaction_session.py | 73 | 94 | 22 + 11 + 8 ported (model names, candidates, modes, normalization, restore, rollback budget), 29 C, 16 PO, 7 cross-area, 3 U (see fork C4) | py_memory_compaction, py_memory_responses_compaction_a/b |
| memory/test_sqlalchemy_compaction.py | 9 | 15 | 1 ported, 2 PO, 2 cross-area, 4 U | py_memory_remaining_b |
| utils/test_json.py | 2 | 2 | PO (Python `_to_dump_compatible` of mapping/iterator types; serde already serializes) | n/a |
| utils/test_pretty_print_and_items.py | 7 | 7 | 4 ported (non-null text paths); 3 PO (`text=None`, `RunErrorDetails` pretty print has no Rust counterpart) | py_memory_utils |
| utils/test_simple_session.py | 3 | 3 | 3 ported (ListSession in support module) | py_memory_limits_helpers |
| ext/test_advanced_sqlite_session.py | 120 | 179 | 73 def/108 exp ported (+2 C), 27/46 PO, 1/1 cross-area, 17/22 U | py_memory_advanced_sqlite, _usage |
| ext/test_encrypt_session.py | 49 | 66 | 32/40 ported (+1 C), 11/15 PO, 4/9 cross-area, 1/1 U | py_memory_encrypt |
| ext/test_async_sqlite_session.py | 40 | 45 | 3 ported, 18 C (SqliteSession), 19 PO | py_memory_remaining_b/c |
| ext/test_memory_imports.py | 2 | 2 | PO (import paths) | n/a |
| ext/test_redis_session.py | 86 | 110 | 36 ported, 1 C, 49 PO | py_memory_redis |
| ext/test_mongodb_session.py | 60 | 72 | 38 ported, 19 PO, 3 U (driver fault injection) | py_memory_mongodb |
| ext/test_dapr_session.py | 47 | 50 | 33 ported, 13 PO, 1 U (log-redaction) | py_memory_dapr |
| ext/test_sqlalchemy_session.py | 40 | 43 | 30 ported, 10 PO | py_memory_sqlx |

### tests/deferred_session_writes (forks C3, C5)
| file | def | exp | P | I (ignored) | C | PO | U |
|---|---|---|---|---|---|---|---|
| test_resume.py | 20 | 40 | see C3 report | | | | 0 |
| test_recovery.py | 8 | 29 | | | | | 0 |
| test_filters.py | 13 | 31 | | | | | 0 |
| test_pending_write.py | 10 | 12 | 0 | 0 | 0 | 12 | 0 |
| test_max_turns.py | 5 | 9 | | | | | 0 |
| test_compaction.py (C5) | 12 | 39 | 3 green | 6 tests ignored (12 expanded) | 0 | 9 | 0 (18 expanded split green/ignored) |
| C3 total (excl. compaction) | 56 | 121 | 53 | 47 | 1 | 20 | 0 |

Per-file splits for C3 are in `py-test-parity-mcp-memory-fork-c3-deferred-writes.md`. Python's `_PendingSessionWrite` / `_held_items_safe_to_settle` internals do not exist in Rust, so those white-box tests are Python-only; the session end-state half is ported and, where the runner differs, ignored.

## Totals (fork-reported, def-level where noted)
| fork | ported | ignored (cross-area) | covered | python-only | cross-area unportable | unported |
|---|---|---|---|---|---|---|
| A mcp util (188 exp) | 120 | 27 | 12 | 19 | 8 | 2 |
| B mcp server (361 exp) | 115 | 2 (of the 4 ported) | 19 | 170 | 57 | 0 |
| D advanced sqlite + encrypt (245 exp) | 148 | 0 | 3 | 61 | 10 | 23 |
| E redis/mongo/dapr/sqlx (275 exp) | 137 | 0 | 1 | 91 | 0 | 4 |
| C2/C4 memory core (see tables) | ~20 + 19 + mine ~120 | ~2 | many | many | ~10 | ~70 |
| C3+C5 deferred writes (160 exp) | 56 | 59 | 1 | 29+ | 0 | 0 |
Fork C2/C4 per-group counts overlap on `test_openai_responses_compaction_session.py` (their table sums to 75 defs vs 73 real because some tests straddle two groups), so the memory-core row is approximate; the per-file entries above are the reliable figures. Fork D/E/B/A/C3/C5 tables sum to their stated expanded totals.

## Rust bugs fixed (src files)
- `src/mcp/util.rs`: duplicate-name error lists all duplicates sorted/escaped with Python's exact text; root `$schema` kept; MCP schemas no longer force-strictified by `FunctionTool::new_raw`; empty content returns `[]`; schema depth > 100 fails with "too deeply nested"; non-object JSON rejected ("expected a JSON object"); `mcp_tools` span used raw server name (URL credentials reached exported traces), now sanitized; error/bad-JSON logs follow `DONT_LOG_TOOL_DATA`.
- `src/mcp/mod.rs`: manager lifecycle rewritten to Python semantics (skip connected, cleanup on strict failure, reverse cleanup, no reconnect of failed cleanup; additive `set_strict`, `is_strict`, `with_drop_failed_servers`, timeout accessors); not-initialized / pagination messages; `list_tools` checks session before cache and retries per page; `call_tool` rejects missing required params locally.
- `src/memory/sqlite_session.rs`: `get_items(limit)` widens past corrupt rows so a limit counts valid items.
- `src/memory/openai_conversations_session.rs`: zero limit never sent as a page size (still initializes the conversation); writes sanitize like `_sanitize_openai_conversation_item` (ids kept only for required types and reasoning, `provider_data`/`created_by` stripped).
- `src/memory/openai_responses_compaction_session.rs`: failed-clear recovery inspects instead of clearing twice; compact request strips `_session_description`/`_session_title`/`created_by`; restore warnings honour `DONT_LOG_*`.
- `src/memory/advanced_sqlite_session.rs`: ASCII-only case folding in content search, Python messages for branch create/delete, `get_tool_usage` port, `namespace.name` tool names, stale-instance clear detection on all default-branch reads, `create_tables=false` / second table pair no longer creates base tables (adds `AdvancedSqliteSession::open_with_tables`).
- `src/memory/dapr_session.rs`: writes error on corrupt stored state instead of overwriting it.
- `src/memory/sqlx_session.rs`: `pop_item` skips corrupt newest rows; concurrent SQLite pops retry on "database is locked"; non-ASCII stored escaped by default (Python parity; `.with_ensure_ascii(false)` restores readable storage).
- `src/memory/redis_session.rs`: unrepresentable-TTL error text.

## Breaking / behavior changes to review
- sqlx default storage of non-ASCII text changes bytes written for existing users; revert with `.with_ensure_ascii(false)`. Needs owner confirmation before 0.6.0.
- Dapr writes now fail on corrupt state (previously overwrote silently).
- `MCPServerManager::active_servers()` returns every managed server before any connect attempt; `list_tools` on a disconnected server now errors even with caching; `call_tool` can fail locally on missing required params.
- Trace `mcp_tools` span server name is sanitized (credential leak fix).

## Cross-area bugs deferred (not edited; ignored tests are the acceptance suite)
1. `src/run_internal/mod.rs:1559-1608` (`notify_session_completed`): model exchange built from the pre-`call_model_input_filter` input, so hidden history is compacted. Python records filtered input. Also ignores the `RunConfig` store override (reads `current_agent.model_settings.store`).
2. `src/run_internal/mod.rs:1205` -> `save_session:1536`: interruption writes every generated item eagerly; Python defers (`defer_interrupted_session_write`). Blocks ~25 deferred-write tests.
3. `src/run_internal/mod.rs:1119` after-turn cancel checked before the resumed turn runs the approved tool.
4. `src/run_internal/tools.rs:464`, `mod.rs:2678`: failing post-output step aborts after the tool ran; retry re-executes the approved tool (side effect duplicated, verified).
5. `src/run_internal/mod.rs:1316` crashing output guardrail returns before `save_session:1330`; call stored without its output.
6. `src/run_internal/items.rs:162` `drop_orphan_function_calls` lacks `output_pruning_indexes`; `mod.rs:1697` never prunes orphan tool outputs from session history after a limit cut.
7. `src/run_internal/tools.rs:214`: pre-approval input guardrail runs 3 times on a resumed run (Python 2).
8. `src/run_internal`/run state: approved MCP call not bound to the offering server (Python "different recipient" error never fires); `RunState::from_json` takes no agent, so 57 resume-binding cases are unportable.
9. Handoff-name reservation in `src/mcp/util.rs` is applied always; Python reserves only when tool names are prefixed (agent with a handoff named like an MCP tool fails with "Duplicate tool names").
10. `src/strict_schema.rs`: no reject-open-objects mode (5 ignored Rust tests / 22 Python tests). `src/tool.rs`: no `FunctionTool::mcp_annotations`. `MCPTool` has no `_meta` field; MCP approval callback receives no run context; `list_tools` has no refresh generation (3 ignored).
11. Sessions take no run-context `wrapper` (10 `test_session_context_wrapper.py` cases); `EncryptedSession` cannot wrap `OpenAIResponsesCompactionSession` outermost (4 encrypt/suffix cases).
12. `SqliteSession::with_tables` accepts ASCII identifiers only (Python accepts quoted/non-ASCII).
13. `AdvancedSqliteSession::store_run_usage` takes an explicit `&Usage` (Python captures per turn); 4 stale-usage tests unported pending an API decision.

## Remaining open rows
Inherent differences documented as Python-only: asyncio cancellation/task-internals, thread-paused SQLite commits, sqlite connection pools/threads, `httpx`/`mcp` package internals, `MagicMock` call-shape assertions, import-path tests, pydantic internals. Genuinely unported (portable, listed per fork report): ~23 advanced sqlite/encrypt tests, 4 redis/mongo/dapr, ~17 conversations, ~16 `test_session.py` pool/internal variants, 4 sqlx compaction, 3 responses-compaction, 2 mcp util.

## Shared-file edits
None to `src/lib.rs` or `Cargo.toml`. New shared test helper module `tests/py_mcpmem_support/mod.rs` (area-local).

## Build dir
`/Users/ngocthach/.cache/oars-build/tp-mcpmem` deleted at the end of this run.

## Unresolved questions
- Confirm the sqlx non-ASCII default-escape change for 0.6.0.
- Who owns the `src/run_internal` deferral/pruning/guardrail gaps (items 1-8)? They account for ~35 of the 41 ignored tests.
- Decide the `store_run_usage` API shape (explicit vs per-turn capture).
