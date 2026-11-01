# Core8 r8g9 python-only re-audit (mcp, voice, tracing, memory, deferred_session_writes, utils, model_settings, docs)

Status: DONE_WITH_CONCERNS (all 37 files examined; one real Rust gap and one library-logging limit left open, see Unresolved).

## Totals

- Before: 251 python-only cases in 37 files. Ported now: 41. Covered by an existing test: 17. Remaining python-only: 193.
- Per-file numbers: `plans/reports/py-test-parity-core8-r8g9.csv` (before - ported - covered = remaining on every row).
- Group counts inside the large manager and util files are approximate: the earlier reports assigned groups by name and some cases straddle two groups. The per-file totals are exact.

## Per-file table

| file | before | ported | covered | after | note |
|---|---:|---:|---:|---:|---|
| deferred_session_writes/test_compaction.py | 9 | 5 | 2 | 2 | ported: park records its response id x2, held and ordinary MCP approval pair defer compaction, an earlier output in the caller's input does not. covered: entry settle bookkeeping and recorded store by `py_deferred_writes_compaction.rs::check_storage_mode`. Left: calls of private `save_result_to_session` / `save_final_turn_items_after_guardrails` counting what they wrote |
| deferred_session_writes/test_max_turns.py | 5 | 5 | 0 | 0 | the held write is observable now (`RunState::pending_session_write`): a max-turns handler completion clears it for both runners; a guardrail crash or a rejected handler output after the terminal decision leaves the approved pair stored once, and a retry from the checkpoint does not write it again |
| deferred_session_writes/test_pending_write.py | 12 | 4 | 0 | 8 | ported: the park records the store setting x2, held checkpoint loads only under the label that introduced it x2. Left: private `_held_items_safe_to_settle`, `save_resumed_turn_items`, `defer_interrupted_session_write`, `extend_held_session_write` unit tests |
| deferred_session_writes/test_resume.py | 3 | 1 | 0 | 2 | held batch rides a non-streamed result into its checkpoint. Left: wrapper-aware (context-requiring) session x2, see Unresolved |
| docs/test_translate_docs.py | 9 | 0 | 0 | 9 | Python docs translation script, not the SDK |
| mcp/test_mcp_approval.py | 3 | 0 | 0 | 3 | dynamically typed invalid approval policies (typo string, typo in mapping, same tool in always and never); `RequireApproval` is an enum |
| mcp/test_mcp_auth_params.py | 1 | 0 | 0 | 1 | patches `sse_client` to check the hardened default httpx factory call |
| mcp/test_mcp_imports.py | 2 | 0 | 0 | 2 | Python import machinery |
| mcp/test_mcp_server_manager.py | 50 | 2 | 4 | 44 | ported: a failed cleanup of an interrupted connect is never retried, whether it raises or times out. covered: snapshot copies (`py_mcp_manager.rs::failed_and_error_snapshots_are_copies`), lifecycle results cannot be mutated. Left: worker tasks, lifecycle lock, cancellation policy, BaseException, task factory, one-shot iterables, dynamically typed timeouts |
| mcp/test_mcp_server_manager_cleanup_state.py | 1 | 0 | 0 | 1 | `asyncio.CancelledError` propagating out of cleanup |
| mcp/test_mcp_util.py | 16 | 2 | 1 | 13 | ported: server guardrails reach every converted tool, public stdio, SSE and HTTP constructors forward the guardrail policy. covered: tool timeout becomes the default error message (`core_mcp_failure_error_function.rs`). Left: inner and outer asyncio cancellation x9, legacy call without an agent (the Rust signature always names one), live FastMCP |
| mcp/test_streamable_http_session_id.py | 4 | 0 | 0 | 4 | inject a `_get_session_id` callback with MagicMock; the observable lifecycle is `py_mcp_http_transport.rs::session_id_lifecycle` |
| memory/test_openai_responses_compaction_session.py | 16 | 2 | 1 | 13 | ported (source fix): a pop or clear that committed and then failed invalidates the response chain. covered: default client lookup. Left: asyncio cancel x7 (cancelled pop/clear, restore under cancel), generation counter, `is_compaction_aware_session` duck typing x3, pydantic warnings kwarg, falsy client |
| memory/test_session.py | 13 | 0 | 0 | 13 | cancellation of an awaited mutation, per-thread sqlite connection tracking and `close()` (the Rust store has one connection that closes on drop), dict coercion of settings |
| memory/test_session_context_wrapper.py | 15 | 0 | 0 | 15 | signature-inspection opt-in; the feature itself is missing, see Unresolved |
| memory/test_session_helpers.py | 2 | 0 | 0 | 2 | repeated asyncio cancellation of `await_session_mutation` |
| memory/test_session_persistence_sanitize.py | 1 | 0 | 0 | 1 | non-dict item passthrough |
| memory/test_sqlalchemy_compaction.py | 7 | 7 | 0 | 0 | the in-memory engine twin runs on a single-connection `sqlx` pool over `sqlite::memory:` (`SqlxSession::with_pool`): visible suffix x2, timestamp order, parent constraint, other-instance append, pop, clear |
| model_settings/test_serialization.py | 4 | 0 | 2 | 2 | covered by `py_core5_leftover2_agent_config.rs` (reasoning extensions, unknown keys). Left: deep merge of a partially specified retry and backoff, see Unresolved (not a valid python-only) |
| tracing/test_import_side_effects.py | 7 | 0 | 2 | 5 | covered: lazily created default provider, one shared default processor (`py_tracing_core.rs`). Left: import-time side-effect checks in subprocesses, skip-bootstrap |
| tracing/test_processor_api_key.py | 1 | 0 | 0 | 1 | replaces `os.environ` with a racing object |
| tracing/test_reasoning_export.py | 1 | 0 | 0 | 1 | tuple, UserList and UserDict containers |
| tracing/test_setup.py | 5 | 2 | 2 | 1 | ported: the shutdown guard shuts the installed provider down, with the 5 s default timeout. covered: existing provider returned, default bootstrap. Left: `atexit` registration |
| tracing/test_spans_impl.py | 7 | 4 | 0 | 3 | abandoning a span-scoped future, same task and another task, tracing on and off. Left: ContextVar token reset errors, a processor raising |
| tracing/test_trace_context.py | 3 | 0 | 0 | 3 | falsy (`__bool__`) trace and response objects |
| tracing/test_traces_impl.py | 11 | 4 | 3 | 4 | ported as spans for traces. covered: lifecycle sets scope and exports, double start and finish, reattach (`py_tracing_core.rs`). Left: ContextVar token reset, `__enter__` token logging, a processor raising |
| tracing/test_tracing_env_disable.py | 1 | 0 | 0 | 1 | falsy span object |
| utils/test_json.py | 2 | 0 | 0 | 2 | `_to_dump_compatible` of iterators and `MappingProxyType` |
| utils/test_pretty_print_and_items.py | 3 | 0 | 0 | 3 | `None` text in message items, ruled out by the types |
| voice/test_input.py | 3 | 0 | 0 | 3 | numpy dtype and `queue.qsize` introspection |
| voice/test_openai_model_provider.py | 7 | 0 | 0 | 7 | constructor argument conflicts (a Rust builder overrides the client instead, tested), `shared_http_client` type, falsy client |
| voice/test_openai_stt.py | 12 | 2 | 0 | 10 | ported: interrupting setup closes the websocket, tracing on and off. Left: asyncio cancellation and task ownership, private attribute and MagicMock websocket cases, numpy dtype rejection, builtin `TimeoutError` type |
| voice/test_openai_stt_client_config.py | 2 | 1 | 0 | 1 | ported: a streaming session logs no connection or audio data at DEBUG. Left: openai `omit` sentinels |
| voice/test_openai_stt_session_config.py | 2 | 0 | 0 | 2 | dataclass field order and positional subclass fields |
| voice/test_pipeline_config.py | 3 | 0 | 0 | 3 | legacy positional constructor arguments |
| voice/test_pipeline_tracing.py | 1 | 0 | 0 | 1 | a `DefaultTraceProvider` subclass steered by a ContextVar |
| voice/test_testing.py | 7 | 0 | 0 | 7 | bare-string sugar and queue-time snapshot or freeze of mutable arguments; Rust moves the values in |

## New test files

- `tests/py_core8_r8g9_mcp.rs` (3 tests, `mcp`)
- `tests/py_core8_r8g9_sqlx_memory.rs` (4 tests, `sqlx` + `sqlite`)
- `tests/py_core8_r8g9_tracing.rs` (6 tests)
- `tests/py_core8_r8g9_voice.rs` (2 tests, `voice`)
- `tests/py_core8_r8g9_deferred.rs` (9 tests, `mcp`)
- `tests/py_core8_r8g9_memory.rs` (2 tests, `sqlite`)

All pass with `--all-features`; default-feature builds compile.

## Source bugs fixed

1. `src/memory/openai_responses_compaction_session.rs`: `pop_item` and `clear_session` dropped the stored response chain only when the underlying store reported success. A store that commits the deletion and then fails its acknowledgement left a stale `previous_response_id` and deferred marker, so the next compaction could reference history that no longer exists. Both now reset the compaction state on failure too, as Python does.

## Breaking changes

None.

## Unresolved questions

1. `ModelRetryBackoffSettings` and `ModelRetrySettings` fields are plain values, so Python's partial retry override cannot be expressed: `settings.resolve(override)` replaces `retry` wholesale, where Python deep-merges `max_retries`, `backoff` and `policy` field by field (`test_retry_resolve_deep_merges_backoff`, `test_model_settings_dictionary_override_merges_retry_settings`), and a backoff of only `jitter=false` cannot stay "unset" for the other fields. Fixing it means making those fields `Option` (public API break across about 15 call sites). Left for a decision; these are not valid python-only reasons.
2. Context-aware sessions (`test_session_context_wrapper.py`, 15 cases, and 2 in `test_resume.py`): Python passes the run's context wrapper to sessions that accept it. The Rust `Session` trait has no such parameter. This is a missing feature, not an introspection artefact; it needs a trait design decision.
3. `tungstenite` logs the whole websocket handshake request, including the query string, custom headers and the `Authorization` bearer token, at TRACE through the `log` crate. DEBUG and above are clean (ported test). An application that enables TRACE for all targets would see them; no SDK setting can silence another crate's `log::trace!`.
4. A zero lifecycle timeout is accepted by `MCPServerManager` as an immediate timeout, where Python raises `ValueError`. Kept as typed-argument validation; say if it should be rejected at connect like `max_list_pages == 0`.

Status: DONE_WITH_CONCERNS
Summary: 251 python-only cases in 37 files: 41 ported, 17 covered, 193 remain python-only; one source bug fixed in the compaction session wrapper.
Concerns: items 1 and 2 above are Rust behaviour gaps that remain classified python-only until a decision. Clippy on my files is clean for both feature sets when another agent's in-flight `needless_borrow` errors in `src/memory/advanced_sqlite_session.rs` are allowed; they are not mine.
