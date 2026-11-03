# py-test-parity: remote stores (redis / mongodb / dapr / sqlalchemy->sqlx), fork E

Counting: `def test_` via AST = first number; parametrize-expanded in parentheses. Parametrized
cases follow their base test's bucket. One Rust test may cover several Python tests (e.g.
`the_runner_stores_turns_and_honours_a_run_config_limit` = runner_integration + settings override).

| file | def_test (expanded) | ported | covered | python-only | unported | failures found |
|---|---|---|---|---|---|---|
| test_redis_session.py | 86 (110) | 36 | 1 | 49 | 0 | 1 msg diff fixed |
| test_mongodb_session.py | 60 (72) | 38 | 0 | 19 | 3 | 0 |
| test_dapr_session.py | 47 (50) | 33 | 0 | 13 | 1 | 1 real bug fixed |
| test_sqlalchemy_session.py | 40 (43) | 30 | 0 | 10 | 0 | 2 real bugs fixed |
| total | 233 (275) | 137 | 1 | 91 | 4 | |

Rust files: `tests/py_memory_redis.rs` (27 tests), `py_memory_mongodb.rs` (22), `py_memory_dapr.rs`
(26), `py_memory_sqlx.rs` (19). All green with `--all-features`, 5 repeat runs stable, 0 `#[ignore]`.

## Mocks (no real server needed, nothing ignored)
- Redis: in-process RESP2 TCP server (PING, LRANGE with negative indices, RPOP, DEL, EVAL of the 3
  scripts replicated by effect keyed on script text, `-NOSCRIPT` for EVALSHA, `+OK` otherwise) over
  a typed store (list/hash/string + TTL) so WRONGTYPE guards behave as Redis does.
- MongoDB: in-process OP_MSG server (driver 3.9.1 sends only OP_MSG, incl. handshake; kind-1
  document sequences parsed) with a mini engine: `$in/$ne/$lt/$exists/$or`, `$set/$setOnInsert/$inc`,
  the `$slice` pipeline, upsert, sort/limit, createIndexes counter. hello omits
  `logicalSessionTimeoutMinutes`. Converged first try; no ignore needed.
- Dapr: HTTP sidecar with ETags, first-write concurrency, seeded/persistent 409s, corruptible store.
- SQLAlchemy: temp SQLite files through `SqlxSession`.

## Rust bugs fixed
- `src/memory/dapr_session.rs`: `add_items`/`pop_item`/`replace_compaction_suffix` silently
  overwrote corrupted aggregate state (non-JSON or non-list value) with a fresh array, losing data.
  Python raises ValueError ("stored Dapr session messages ...") and leaves the value intact. Writers
  now use a strict decode (`read_messages_for_update`); `get_items` still reads corrupt state as
  empty. `decode_message_array` now returns `Option`.
- `src/memory/sqlx_session.rs`: (1) `pop_item` returned `None` after deleting a corrupt newest row
  instead of continuing to the next valid row (Python skips corrupt rows). (2) Concurrent
  `pop_item` from two pools on SQLite failed with `database is locked` (deferred tx upgrade); claim
  now retries lock contention (Python reserves the writer first) and re-claims if the row was taken.
  (3) Non-ASCII was stored unescaped; Python default is `ensure_ascii=True` (`\uXXXX`, surrogate
  pairs). Added `with_ensure_ascii(bool)`, default `true` (round-trips identically; stored bytes for
  non-ASCII text now match Python's default).
- `src/memory/redis_session.rs`: unrepresentable-TTL error text now matches Python
  ("outside Redis's supported expiration range").

## Python-only groups (reason)
- Redis (49): redis-py client ownership/close lifecycle (`close`, closed-state errors, external vs
  internal client, injected-client close no-op, operations waiting behind close); redis-py pool /
  WATCH / pipeline-reset / release-listener / cancellation internals (~30 tests: blocking pool,
  single-connection pool, post-commit reset, quarantine, ambiguous EXEC, Redis 8.1 observability);
  RedisCluster rejection and cluster reads (Rust does not support Cluster, documented);
  `decode_responses` client modes (x2); asyncio cancellation settling; subclass-overrides compaction
  test. Rust uses one multiplexed connection and no WATCH (documented in the struct docs).
  Covered (1): redis compaction with encryption wrapper ->
  `tests/sessions_tests.rs::compaction_over_an_encrypted_store_sends_plaintext_and_rewrites_envelopes`.
  Redis wrong-metadata-type error text differs by design: Rust "WRONGTYPE Redis key .. expected a
  hash" vs Python "metadata key must contain a hash"; the test asserts the shared behavior (no write).
- MongoDB (19): non-string session id ctor checks (2, typing); close/closed/cancelled-close family
  (9, pymongo client lifecycle); `driver_info` passing (2); asyncio cancellation of in-flight
  mutations (3); add serialization-before-reservation ordering and single-document-commit
  invisibility probes that patch pymongo internals (2); injected-client append metadata (1).
- Mongo unported (3), behavior exists in Rust but needs driver-level fault injection mid-operation:
  generation reads on primary after failed clear cleanup; add-reserved hidden before generation
  advance; pop re-checks generation after concurrent clear.
- Dapr (13): gRPC client ownership/close/context-manager/closed-state tests, cleanup failure and
  cancellation retry (Python Dapr SDK object lifecycle). Dapr unported (1):
  `test_metadata_write_warning_respects_data_policies` (log redaction of the metadata-failure warning;
  Rust warns with session id + error under target `LOG_TARGET`, not wired through
  `DONT_LOG_TOOL_DATA`; minor).
- SQLAlchemy (10): `delete rowcount unknown` and `RETURNING claim lost` (SQLAlchemy dialect quirks;
  Rust uses select+delete with rows_affected), mutation cancellation waits for tx exit and cancelled
  waiter / init-lock tests (asyncio), cross-event-loop tests (2), `create_tables=False` init-lock
  allocation, `engine` read-only property, SQLite configuration registry gc tests (2).
  Python's "fallback reserves writer" and "transient sqlite write lock" are ported as
  `concurrent_pops_return_each_row_once` / `first_writes_from_concurrent_sessions_are_all_kept`.

## Cross-area / shared files
- No edits to Cargo.toml, src/lib.rs, tests/py_mcpmem_support/mod.rs. No new dependencies needed
  (`mongodb::bson` and `sqlx` are reachable from integration tests under the feature).
- Clippy `--all-features --all-targets -D warnings` fails only in other areas' files
  (`src/extensions/models/mod.rs:1016`, `src/models/openai_responses_ws.rs:553,556`); none in mine.
- Intentional behavior change: `SqlxSession` now escapes non-ASCII by default (see above); call
  `.with_ensure_ascii(false)` for the previous readable storage.

## Remainder pass (fork E2)

Scope: the 4 tests left unported above (3 MongoDB generation races, 1 Dapr log redaction) and the
`test_sqlalchemy_compaction.py` cases (9 def / 15 expanded) that no earlier fork ported. Redis:
the table above has 0 unported (36 ported + 1 covered + 49 Python-only = 86 def), so nothing to do.

New files (all rustfmt'd, none touching shared helpers): `tests/py_memory_mongodb_remaining.rs`,
`tests/py_memory_dapr_remaining.rs`, `tests/py_memory_sqlx_compaction.rs`.

Verified runs (`--all-features`): mongodb_remaining 3 passed; dapr_remaining 3 passed;
sqlx_compaction 6 passed, 1 ignored; existing `py_memory_mongodb` (22) and `py_memory_dapr` (26)
still pass. Total new: 12 passed + 1 ignored (cross-area).

### Per-test classification

| Python test | Rust test | result |
|---|---|---|
| mongo `test_generation_reads_use_primary_after_failed_clear_cleanup` | `generation_reads_use_the_primary_after_a_failed_clear_cleanup` | ported, found a bug (fixed) |
| mongo `test_clear_session_hides_add_reserved_before_generation_advance` | `clear_hides_a_batch_reserved_before_the_generation_advance` | ported, green |
| mongo `test_pop_rechecks_generation_after_concurrent_clear` | `pop_rechecks_the_generation_after_a_concurrent_clear` | ported, green |
| dapr `test_metadata_write_warning_respects_data_policies` (3 ids) | `the_metadata_warning_*` (3 tests) | ported, found a bug (fixed) |
| sqlalchemy_compaction `test_limited_sqlalchemy_compacts_visible_suffix` (2 exp) | `a_limited_store_compacts_only_the_visible_suffix` (both streamed variants in one test) | ported, green |
| `test_other_instance_mutation_invalidates_suffix` (3 exp) | already `py_memory_remaining_b::sqlx_compaction_is_abandoned_when_another_instance_mutates_the_suffix` | covered |
| `test_encrypted_prefix_cleanup_is_atomic` (4 exp: outer x reject_insert) | `encrypted_prefix_cleanup_commits_with_the_replacement`, `a_rejected_summary_insert_rolls_back_the_prefix_cleanup` (inner); `encrypted_prefix_cleanup_with_the_encryption_outermost` (outer) | inner 2 ported green; outer ported but `#[ignore]` cross-area (below), 1 expanded case |
| `test_replacement_uses_timestamp_order_not_id_order` | `replacement_follows_timestamp_order_not_id_order` | ported, green |
| `test_encrypted_unknown_row_is_retained` | `a_row_the_key_cannot_verify_survives_prefix_cleanup` | ported, green |
| `test_parent_constraint_rejection_propagates` | `a_rejected_session_row_propagates_and_leaves_the_store_usable` | ported, green |
| `test_cancelled_replacement_settles_before_other_instance_append` | none | Python-only: pauses a SQLAlchemy `AsyncSession.execute` mid-transaction and cancels the asyncio task; no equivalent seam in `sqlx`, and Rust futures cannot be cancel-then-awaited |
| `test_compaction_with_engine_managed_sqlite_begin` | none | Python-only: SQLAlchemy `begin` event / `exec_driver_sql("BEGIN")` transaction-control config |
| `test_failed_replacement_preserves_history_during_other_instance_access` | none | Python-only: same mid-transaction `AsyncSession.execute` hook; the rollback half is covered by `a_rejected_summary_insert_rolls_back_the_prefix_cleanup` |

Totals for `test_sqlalchemy_compaction.py` (9 def / 15 expanded), counted by expanded case:
ported green 7 (limited 2, encrypted-prefix inner 2, timestamp order 1, unknown row 1, parent
constraint 1); covered 3 (other-instance mutation, `py_memory_remaining_b`); cross-area 2
(encrypted-prefix `outer=True` x reject-insert on/off; one representative ignored test, the second
has the same root cause and is not duplicated); Python-only 3 (cancelled replacement, engine-managed
BEGIN, failed replacement during other-instance access). 7 + 3 + 2 + 3 = 15. The `file`/`memory`
engine fixture parameter is not an expanded case in the AST count; the in-memory variant is not
ported because an in-memory SQLite database is per-connection and a pooled `sqlx` session cannot
share one.

Combined with the 4 tests from the earlier E table: Mongo 3 and Dapr 1 are now ported, so fork E's
unported count drops from 4 to 0 and its expanded ported count rises by those 4 (137 -> 141).

### Rust bugs fixed
- `src/memory/dapr_session.rs` (`update_metadata`): the give-up warning for a failed metadata write
  wrote the session id and the provider error unconditionally. It now goes through
  `logger::log_model_and_tool_data_warning`, so it is redacted when either `DONT_LOG_MODEL_DATA` or
  `DONT_LOG_TOOL_DATA` is on and carries the session id and error only when both are off.
- `src/memory/mongodb_session.rs` (`generation`): the session-document read that decides which
  history generation is visible used the client's default read preference. With a
  secondary-preferring client a lagging secondary can still report the pre-`clear_session`
  generation and expose cleared history. It now pins the read to `ReadPreference::Primary`, as
  Python's `_get_generation` does (verified through a mongos-mode mock that shows the driver's
  `$readPreference` on the wire).

### Cross-area deferral
- `src/memory/encrypted_session.rs:119`: `EncryptedSession` does not forward
  `Session::on_run_completed` (or the model exchange) to a wrapped `OpenAIResponsesCompactionSession`,
  so Python's `outer=True` ordering (`EncryptedSession(CompactionSession(store))`) never compacts under
  the runner. Expected (Python): one compact request, expired envelopes reclaimed, one stored summary.
  Minimal fix: forward `on_run_completed`, `wants_model_exchange` and the compaction snapshot hooks to the
  underlying session. Ignored test: `encrypted_prefix_cleanup_with_the_encryption_outermost`. Owner:
  fork D's file.

### Notes
- No edits to `src/lib.rs`, `Cargo.toml`, `tests/py_mcpmem_support/mod.rs`, `src/run*.rs` or
  `src/run_internal/**`; no new dependencies (`rusqlite` and `futures` are existing deps).
- The Mongo file carries its own copy of the OP_MSG mock (about 450 lines) because integration
  tests cannot import from another test binary; it adds a one-shot command gate, a command-failure
  list and a mongos mode.
- Clippy: `cargo clippy --all-features --lib --test <my three files> -- -D warnings` stops in the lib on
  the core agent's in-flight files (`src/run_internal/agent_identity.rs:106`, `src/run_internal/mod.rs:2329,2694`,
  `src/stream_events.rs:48`) before reaching my targets, so my files were not linted this round; rustfmt applied.
