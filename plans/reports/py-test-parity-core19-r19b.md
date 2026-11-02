# Core19 r19b: python-only re-audit (memory close, mcp, realtime, voice, models, codex, config)

17 files, 173 python-only cases before (ledger counts). 24 moved to ported, 2 moved to covered,
147 remain python-only.

## Source changes (session close semantics)

Python has `close()` on `SQLiteSession` (inherited by `AdvancedSQLiteSession`),
`AsyncSQLiteSession` and `DaprSession`, and its tests check the closed state. The Rust backends
for those hold a connection or client, so they now have inherent `close()`/`is_closed()` (no
trait method), following the `MongoDBSession` precedent from 0.21.0:

- `src/memory/sqlite_session.rs`: the shared connection is `Arc<Mutex<Option<Connection>>>`.
  `SqliteSession::close(&self)` drops (closes) it; `is_closed()`. Every later operation fails
  with `SQLiteSession is closed`, including `add_items(vec![])`, which is now checked before the
  empty-batch fast path. Clones and `for_session` siblings share the connection, so they are
  closed with it. Rust has one `SqliteSession` for Python's sync and async classes, so the
  async-class message `AsyncSQLiteSession is closed` is `SQLiteSession is closed` here.
- `src/memory/advanced_sqlite_session.rs`: `close()`/`is_closed()` delegate to the base; the
  three direct connection-lock sites go through the same closed check; `add_items(vec![])`
  checks closed first (Python's inherited message `SQLiteSession is closed`).
- `src/memory/dapr_session.rs`: `DaprSession::new`/`from_env` own their `reqwest::Client`;
  `with_http_client` marks the client injected. `close().await` takes the session lock and,
  for an owned client, makes the session terminal: operations (including `add_items(vec![])`)
  fail with `DaprSession is closed`, and operations queued behind the close fail. For an
  injected client it is a no-op, as in Python. `ping()` keeps its `-> bool` signature: after
  close it answers `false` without a request (Python raises `RuntimeError`); changing it to
  `Result<bool>` would be a public break, so it was not done.
- `SqlxSession` (Python `SQLAlchemySession`) got nothing: Python's `SQLAlchemySession` has no
  `close()`.

No public API was removed or changed in signature; the additions are new inherent methods.

## New tests

- `tests/py_core19_r19b_sqlite_close.rs` (sqlite, 11 tests): closed SQLite/advanced sessions
  reject every operation and an empty append; close releases the connection (WAL file removed,
  siblings closed, history intact on reopen); close before use is terminal; repeated and
  concurrent closes; history across exiting runtimes, concurrent writers on retiring threads,
  and a live runtime on another thread; a failed advanced-structure initialization (conflicting
  `idx_structure_session_seq`) fails with "already a table" and leaves the database writable
  by a zero-timeout writer.
- `tests/py_core19_r19b_dapr_close.rs` (no feature, 6 tests): owned-client close makes every
  operation fail without a sidecar request; empty append after close; idempotent/concurrent
  close; ping after close sends nothing; close is a no-op for an injected client (session and
  client stay usable); an operation queued behind close fails (close queues behind a read the
  in-process sidecar holds open; the session lock is FIFO).
- `tests/py_core19_r19b_encrypt_backend.rs` (sqlite+encrypt, 2 tests): an encrypted pop goes
  through a backend's overridden `pop_item` exactly once; a backend's own compaction hook
  (`on_run_completed`) is forwarded by `EncryptedSession`.

## Per file

| file | python_only before | ported | covered | remaining | reason per remaining group |
|---|---|---|---|---|---|
| extensions/memory/test_advanced_sqlite_session.py | 14 | 2 | 0 | 12 | ported: closed rejects empty add, init failure releases the database. Remaining: injected `sqlite3.Connection` subclass whose rollback fails (connection invalidation), post-commit cancellation x3, auxiliary-mutation cancellation x4, switch-validation cancellation, usage-capture cancellation (asyncio cancel/settle), `_resolve_table_identifier` + `sqlite3` authorizer (private helper); one counted case could not be matched to a distinct name |
| extensions/memory/test_dapr_session.py | 14 | 11 | 0 | 3 | ported: closed operations raise, closed empty add, idempotent close, ping after close (adapted: `false`, no request), no-op for injected client, external client not closed, close-method coverage, internal client ownership, operation queued behind close; and, since Python's `__aexit__` is `close()`, `test_dapr_session_closed_after_context_manager_exit` (`py_core19_r19b_dapr_close.rs::closed_operations_on_an_owned_client_fail_without_a_request`) and `test_context_manager` (`::close_is_a_no_op_for_an_injected_client`). Remaining: failed and cancelled client cleanup retried (the Dapr SDK client's `close()` raising/being cancelled; `reqwest` has no fallible close), settings constructor from a dictionary (typed `SessionSettings`) |
| extensions/memory/test_async_sqlite_session.py | 12 | 4 | 0 | 8 | ported against `SqliteSession`: closed operations, closed empty add, close before use, idempotent close. Remaining: cancelled close, cancelled/failed initialization candidate close, quarantined connection retried by close, rollback-failure eviction, cancelled WAL retry/connect (aiosqlite worker internals, asyncio cancellation, injected connection subclasses), monkeypatched connect timeout |
| extensions/memory/test_encrypt_session.py | 10 | 2 | 1 | 7 | ported: overridden backend pop, optional compaction delegation. Covered: `test_encrypted_pop_skips_plaintext_with_inherited_sqlite_behavior` = `py_memory_encrypt.rs::pop_skips_plaintext_records` (same inputs). Remaining: cancelled compaction restore vs concurrent append, pop cancellation waits for rollback (asyncio), negative `max_scan_items`/negative limits x4 (usize), `__getattr__` delegation of arbitrary methods |
| extensions/memory/test_sqlalchemy_session.py | 9 | 0 | 0 | 9 | fake SQLAlchemy sessions/dialect flags (`delete_returning`, rowcount), cross-event-loop engines, engine registry gc, asyncio cancel waiting for transaction exit. No `close()` in Python's SQLAlchemySession |
| memory/test_session.py | 11 | 5 | 0 | 6 | ported: closed empty add, close closes connections (adapted: WAL released), reaping across exited runtimes, retired concurrent workers, live worker thread. Remaining: `_await_mutation` loop-error handler, post-commit cancellation x3 (asyncio), rollback-failure eviction and quarantined close (injected `sqlite3.Connection` subclasses) |
| memory/test_openai_conversations_session.py | 9 | 0 | 0 | 9 | asyncio cancel-settle cases, OpenAI Python SDK retry layer (`max_retries`, `with_options`; the Rust client has no SDK retry layer), falsy default client (`__bool__`) |
| mcp/test_mcp_v2_http.py | 11 | 0 | 0 | 11 | MCP Python SDK v2 (protocol `2026-07-28`, `server/discover`), httpx2 response hook/client factory/v1-auth rejection, OAuth sub-request mapping, connect/outer asyncio cancellation |
| mcp/test_mcp_util.py | 8 | 0 | 1 | 7 | covered: `test_mcp_fastmcp_behavior_verification` = `py_mcp_util.rs::fastmcp_content_shapes_convert_as_python_does`. Remaining: outer asyncio cancellation x5, legacy three-argument `to_function_tool` without an agent x2 (one Rust signature) |
| realtime/test_openai_realtime.py | 12 | 0 | 0 | 12 | monkeypatched private `_send_raw_message`/`_handle_ws_event`/normalizer, listener-callback re-entrancy, asyncio task state (cancelled send/close), private response-control state |
| realtime/test_session_tool_outputs.py | 7 | 0 | 0 | 7 | tool-end asyncio cancellation, pydantic/dataclass `model_dump`/`asdict` failure fallbacks of a private serializer |
| realtime/test_realtime_handoffs.py | 6 | 0 | 0 | 6 | sibling `is_enabled` cancellation on a raising check (Rust `is_enabled` returns bool), `inspect` arity/non-callable/callable-object/`input_type` validation, type annotations |
| voice/test_openai_stt.py | 8 | 0 | 0 | 8 | consumer cancellation, owned-task/cleanup failures with patched websockets and spans, numpy dtype, Python timeout class |
| voice/test_testing.py | 7 | 0 | 0 | 7 | bare-string coercions (typed constructors), queue-time freezing/snapshot failures of mutable arguments (moved values, `Clone` cannot fail), exception identity |
| models/test_any_llm_model.py | 7 | 0 | 0 | 7 | any-llm provider `aclose()` failure after a terminal event x4, Google SDK `http_options`/role normalization x3 |
| extensions/experiemental/codex/test_codex_exec_thread.py | 15 | 0 | 0 | 15 | mapping/kwargs/dict coercion and env stringification (dynamic typing), `json.dump`/`shutil` monkeypatch, fake processes without stdin/stdout, vendored target-triple mapping x6 and fallback (packaging) |
| test_config.py | 13 | 0 | 0 | 13 | loop-scoped WebSocket model cache, weakref/gc of closed loops, `aclose` across loops (asyncio), httpx2 client type, falsy client, string transport validation (typed enum), missing-key `OpenAIError` at `get_model` (see divergence) |

## Divergences found, not fixed

- `test_config.py::test_cc_no_default_key_errors` / `test_resp_no_default_key_errors`: Python's
  `OpenAIProvider().get_model()` raises when no key is set (the OpenAI Python SDK constructor
  raises). Rust's `get_model` succeeds and requests go out without an `Authorization` header,
  which supports keyless local/proxy endpoints. Left as is (contract of `src/models`).
- `test_encrypt_session.py::test_encrypted_session_preserves_optional_compaction_delegation`:
  Python's runner calls `run_compaction` only for turns without local tool outputs. Rust calls
  `Session::on_run_completed` for every persisted turn with `has_local_tool_outputs`, and the
  session defers. The port asserts the split through that flag. The doc of `on_run_completed`
  says "after a run finished", which undersells the per-turn call (the runner is owned by
  another agent; not changed).
- `DaprSession::ping` after close answers `false` instead of raising (signature kept).

## Gates

- New binaries: `py_core19_r19b_sqlite_close` 11/11, `py_core19_r19b_dapr_close` 6/6,
  `py_core19_r19b_encrypt_backend` 2/2 (`--all-features`).
- Related binaries, `--all-features`: 71 binaries (every memory/sqlite/dapr/encrypt/sqlx/
  session/deferred-writes/conversations test file, plus `mcp_tests`, `realtime_tests`,
  `voice_tests`, `py_mcp_util`, `py_voice_misc`, `py_realtime_session_tool_outputs`,
  `py_realtime_handoffs`, `py_voice_openai_models`): 849 passed, 0 failed. Lib unit tests
  `memory::`: 77 passed.
- clippy `-D warnings` for `--lib` plus the three new test files: clean with `--all-features`
  and with default features.
- rustfmt (edition 2024) on the three new files and the three edited memory sources.

## Read depth

- Bodies read for every python-only candidate of the memory files, `test_mcp_util.py`
  (legacy/fastmcp), the encrypt candidates, the codex candidates and `test_config.py`.
- `test_mcp_v2_http.py`, `test_openai_realtime.py`, `test_session_tool_outputs.py`,
  `test_realtime_handoffs.py`, `test_openai_stt.py`, `test_testing.py`,
  `test_any_llm_model.py`: unmatched names listed and a sample of bodies read (realtime
  send-failure retry, audio timing, listener/tools conversion; STT span-close and close-fail;
  MCP v2 negotiation and connect retry); the rest was classified by name against the groups of
  r16a/c/d/f.
- Exact case identity: for `test_advanced_sqlite_session.py` one of the 14 counted cases and
  for `test_codex_exec_thread.py` the exact 15 of the 19 valid-looking expanded cases could not
  be pinned to names; counts follow the ledger.

## Suggested changelog paragraph

> `SqliteSession`, `AdvancedSqliteSession` and `DaprSession` gain `close()` and `is_closed()`,
> ports of Python's `close()`. A closed SQLite session drops its connection (clones and
> `for_session` siblings share it) and every later operation, including an empty `add_items`,
> fails with `SQLiteSession is closed`. `DaprSession::close` makes a session that owns its HTTP
> client terminal (`DaprSession is closed`; `ping` answers `false`), waits for an operation in
> progress and fails operations queued behind it; it does nothing for a client supplied with
> `with_http_client`.

## Concerns

- `SqliteSession::check_not_closed` (used before the empty-batch fast path of `add_items`, also
  by `AdvancedSqliteSession`) takes the connection's std `Mutex` on the calling runtime thread.
  While a `spawn_blocking` statement holds it, that worker waits for the statement to finish.
  It cannot deadlock (the blocking closures are plain SQL), and Python makes the same
  synchronous check, so it was left as is.

## Unresolved questions

- Should `DaprSession::ping` become `Result<bool>` (like `MongoDBSession::ping`) so a closed
  session raises as in Python? Public break.
- Should `OpenAIProvider::get_model` fail without an API key for the official endpoint, as the
  Python SDK does?

Status: DONE_WITH_CONCERNS
