# Core8 r8g4 python-only re-audit (mcp, memory, voice, tracing, deferred_session_writes, utils, docs, model_settings)

Status: DONE_WITH_CONCERNS. Not every case in the 492 was ported or individually re-proved; see Coverage of the audit.

## Extra task: expired history during compaction

Decision (accepted by the coordinator): no source change.

- Python's `RuntimeError` in `test_rollback_does_not_revive_items_expired_during_compaction` is the test's own injected "replacement failed". Expiry itself never raises in Python.
- Without that injection Python would replace its pre-request snapshot and silently drop the live item. Rust re-reads the store after the request, sees the change, skips the replacement and keeps `[live]`, which matches the test's final assertion.
- `tests/py_core6_final7_compaction.rs` already pins this and needed no edit.

## Totals

- Before: 492 python-only cases in 49 files. Ported now: 70. Covered: 5. Remaining python-only: 417.
- Per-file numbers are in `plans/reports/py-test-parity-core8-r8g4.csv` (before - ported - covered = remaining on every row).

## Per-file movements

| file | before | ported | covered | after | note |
|---|---:|---:|---:|---:|---|
| memory/test_compaction_suffix.py | 24 | 22 | 0 | 2 | the `AsyncSQLiteSession` twin of 11 cases is Rust's second SQLite store, `SqlxSession` on a file database: `py_core8_r8g4_suffix_sqlx.rs` runs all 22. Left: the cancelled-suffix case x2 (asyncio cancel) |
| mcp/test_server_errors.py | 99 | 13 | 0 | 86 | prompt/resource error text x10, prompt HTTP status text, prompt request status text, paginated failure x2 |
| mcp/test_prompt_server.py | 8 | 8 | 0 | 0 | real MCP server over HTTP: list, render with arguments, missing arguments, not found, agent instructions streamed and not |
| mcp/test_client_session_retries.py | 22 | 6 | 0 | 16 | server-reported 408 earns an isolated session; backoff cap forwarded; non-finite/negative cap rejected x4 |
| mcp/test_mcp_v2_http.py | 16 | 4 | 0 | 12 | 5xx fails only that request, 5xx retried, handshake 5xx fails connect, tolerated `initialized` 5xx |
| mcp/test_streamable_http_client_factory.py | 10 | 6 | 0 | 4 | tolerant `initialized` notification: status, dropped connection, log without credentials x2, other failures preserved, flag honoured end to end |
| mcp/test_list_page_limits.py | 10 | 3 | 0 | 7 | a zero limit is rejected before connect on stdio, SSE and HTTP |
| mcp/test_message_handler.py | 3 | 3 | 0 | 0 | handler reaches stdio, SSE and HTTP servers |
| memory/test_openai_conversations_session.py | 17 | 5 | 0 | 12 | default-client lookup, create with `items: []`, constructors |
| voice/test_pipeline.py | 30 | 0 | 3 | 27 | covered by `tests/voice_tests.rs` (`a_scripted_session_close_failure_reaches_the_pipeline...`, `speech_spans_are_grouped_per_turn_and_honor_the_privacy_flags` for both sensitive-data settings) |
| voice/test_tts_voice_types.py | 1 | 0 | 1 | 0 | `tests/py_voice_misc.rs` |
| tracing/test_span_ordering.py | 1 | 0 | 1 | 0 | `tests/py_tracing_core.rs::ordered_spans_put_a_parent_before_its_child_on_a_tied_timestamp` |
| all other 38 files | 251 | 0 | 0 | 251 | not moved |

## Coverage of the audit

Files where I read the Python tests against the Rust surface and found portable behaviour: the ten above. For the other files I accepted the earlier group reasons after reading the Python test bodies of a sample, and did NOT port. The reasons I confirmed by reading:

- asyncio task, cancel and worker-task semantics: mcp manager worker/lifecycle-lock tests, voice pipeline and STT cancellation tests, compaction cancel-restore tests, `test_session.py` await_mutation, tracing generator-close/ContextVar tests.
- dynamically typed argument validation: lifecycle timeouts, dict coercion of `SessionSettings`/`ModelSettings`, non-integer page limits, `None` text in item helpers.
- pydantic/inspect/typing introspection: signature-inspection opt-ins in `test_session_context_wrapper.py`, numpy dtype spellings in voice.
- Python-only packaging and docs scripts: `docs/test_translate_docs.py`, import-side-effect and lazy-import tests, `test_mcp_imports.py`.
- private-helper tests: `_PendingSessionWrite` (`test_pending_write.py`), `_get_session_id` callbacks, httpx2 client plumbing.
- exception-graph shape (`ExceptionGroup`, `__cause__`, attached request or response, traceback locals) in `test_server_errors.py`. Rust errors are plain strings, so only their text could be ported; the text was.

Files that were not examined case by case: most of `test_server_errors.py` (cleanup log-record cases), `test_mcp_server_manager.py`, `test_mcp_util.py`, `test_session_context_wrapper.py`, the voice STT/pipeline files, tracing files other than span ordering, `deferred_session_writes/*`, `model_settings/test_serialization.py`, `memory/test_session.py`. Their remaining counts are carried over, not proved.

## Source bugs fixed (Rust diverged from Python)

1. `src/mcp/mod.rs`: failed MCP requests surfaced rmcp's raw text ("MCP error: Transport send error ... HTTP 503"). They now read as Python's: `Failed to list tools from MCP server '<name>': HTTP error 503`, `Failed to call tool 't' on MCP server '<name>': ...`, `Failed to <operation> on MCP server '<name>': ...` for prompts and resources, with "Connection lost", "Connection timeout" or "Request failed." for the other transport failures. A failed HTTP/SSE connect reads `Failed to connect to MCP server '<name>': HTTP error 503 (Service Unavailable)` or `Could not reach the server.` The error is now `AgentsError::User`.
2. `src/mcp/mod.rs`: a JSON-RPC error with code 408 now earns the isolated-session retry, as Python's `is_mcp_timeout_error` does.
3. `src/mcp/mod.rs`: `max_list_pages == Some(0)` and a negative or non-finite `retry_backoff_seconds_max` are rejected when connecting (Python rejects at construction; the Rust builders cannot fail). Previously a zero limit failed any multi-page listing and a bad cap was silently clamped.
4. `src/mcp/http.rs`: the warning for an ignored `initialized` notification failure logged the raw transport error, including the query string of the server URL. It now goes through `log_tool_action_warning` with credentials redacted and honours the tool-data logging policy.
5. `src/memory/openai_conversations_session.rs`, `src/memory/openai_responses_compaction_session.rs`: `new()` ignored the process default client set with `set_default_openai_client`; both now use it and fall back to a client built from the environment, like Python.

## Breaking changes

- MCP request and connect error messages changed (and are now `User` errors rather than `Other`). Code matching the old "MCP error: ..." text breaks; no test in the repo matched it.
- `with_max_list_pages(Some(0))` and an invalid backoff cap now make `connect` fail.
- `OpenAIConversationsSession::new()` and `OpenAIResponsesCompactionSession::new()` pick up a default client set earlier.

## New test files

- `tests/py_core8_r8g4_mcp_errors.rs` (16 tests), `tests/py_core8_r8g4_suffix_sqlx.rs` (10, covering 22 cases), `tests/py_core8_r8g4_sessions.rs` (1), `tests/py_core8_r8g4_misc.rs` (2 extra run-error-summary tests; they do not move any count because the three python-only cases of that file are `None`-text cases).
- A port of `test_runner_skips_compaction_after_interleaved_model_wait` already exists in `tests/py_core6_final7_compaction.rs`; a duplicate I wrote was deleted.

## Unresolved questions

- `test_session_context_wrapper.py` (15) is a real Python feature, not an impossibility: sessions that receive the run's context wrapper to scope storage per tenant. Rust `Session` methods take no run context. Porting needs a decision on extending the `Session` trait. Counted python-only for now.
- Python logs a sanitized warning when MCP cleanup hits a transport error; Rust's cleanup never sees such errors (rmcp swallows them), so there is nothing to log. Confirm this is acceptable.
- Python's `start_openai_conversations_session` has no Rust function; `OpenAIConversationsSession::conversation_id()` is the equivalent.

## Gates

- My binaries pass: `py_core8_r8g4_mcp_errors` (16), `py_core8_r8g4_suffix_sqlx` (10), `py_core8_r8g4_sessions` (1), `py_core8_r8g4_misc` (2). Also green after the source changes: `mcp_tests` (13), `py_mcp_server_errors` (9), `py_mcp_pagination` (7), `py_mcp_http_transport`, `py_mcp_manager`, `py_core6_mcpmisc_manager`, `py_core6_mcpmisc_mcp_retry_setup`, `py_core6_mcpmisc_compaction_session`, `py_memory_conversations`, `py_mcp_caching_approval`, `py_mcp_util_logging`.
- Default-feature clippy `-D warnings` on the lib and my four test files: clean.
- All-features clippy: my files are clean; the lib fails only in `src/memory/advanced_sqlite_session.rs` (lines 800-852, `needless_borrow`), another agent's in-flight edit.
- rustfmt applied to my four test files and to `src/mcp/mod.rs`, `src/mcp/http.rs` and the two memory session files (only my hunks differed).
