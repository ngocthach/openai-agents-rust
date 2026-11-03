# Python test parity, core round 8, r8g3: extensions (codex, memory, sandbox providers, trimmer)

Re-audit of the python-only groups of 20 files. Totals: 441 python-only before, 34 moved to ported, 8 moved to covered, 399 remain.
CSV: `py-test-parity-core8-r8g3.csv` (before - ported - covered = remaining on every row).

## Per file

| file | before | ported | covered | after | reason for the remaining groups |
|---|--:|--:|--:|--:|---|
| codex/test_codex_exec_thread.py | 22 | 0 | 0 | 22 | monkeypatched rmtree/json.dump, dict/kwargs/`inspect` option coercion, dynamically typed env, FakeProcess without stdin/stdout, asyncio pipe drain and cancel, `asyncio.Event` watcher, vendored-wheel binary lookup and target triples |
| codex/test_codex_tool.py | 27 | 0 | 0 | 27 | dict/kwarg option coercion, `inspect`, pydantic/frozen/slots/mapping run contexts and key derivation (Rust takes reader/writer closures), BaseException and asyncio dispatcher cancellation, private resolver caches |
| codex/test_payloads.py | 2 | 0 | 0 | 2 | `_DictLike` mapping protocol and KeyError |
| hosted_multi_agent/test_model.py | 2 | 0 | 0 | 2 | `-1` and `True` concurrency: unsigned typed field |
| memory/test_advanced_sqlite_session.py | 46 | 7 | 3 | 36 | logger injection and `caplog` record fields, subclass fault injection, rollback-failure connection invalidation, `close()`, asyncio cancellation and gated private workers, queued/async usage-capture interleavings, event-loop reuse |
| memory/test_async_sqlite_session.py | 24 | 1 | 2 | 21 | no `close()` (RAII), aiosqlite internals, cancellation, unserializable item |
| memory/test_dapr_session.py | 14 | 0 | 0 | 14 | client ownership, `close`, context manager, cancelled cleanup |
| memory/test_encrypt_session.py | 15 | 0 | 0 | 15 | asyncio lock/cancel interleavings of compaction restore, overridden-pop subclass, negative scan budgets (unsigned), Python wrapper delegation |
| memory/test_memory_imports.py | 13 | 0 | 0 | 13 | optional-import machinery (cargo features) |
| memory/test_mongodb_session.py | 28 | 2 | 3 | 23 | non-string ids, `close`, driver_info/append_metadata (vendor client), unserializable item, asyncio cancellation |
| memory/test_redis_session.py | 60 | 8 | 0 | 52 | redis-py pool/WATCH/cancellation/cluster/decode_responses internals, client ownership and `close`, `_get_next_id` and counter private API |
| memory/test_sqlalchemy_session.py | 13 | 3 | 0 | 10 | SQLAlchemy engine registry, dialect fallbacks, cross-event-loop engines, cancellation |
| sandbox/test_blaxel.py | 40 | 0 | 0 | 40 | import guards, SDK duck typing, pydantic, asyncio shielding, private PTY entry fakes, test doubles, Docker driver hook absent |
| sandbox/test_cloudflare.py | 9 | 4 | 0 | 5 | cancellation settling (3), private hooks (2) |
| sandbox/test_daytona.py | 6 | 0 | 0 | 6 | vendor SDK handle and exception-class internals, re-exports |
| sandbox/test_e2b.py | 23 | 4 | 0 | 19 | re-exports, pydantic defaults, `workspace_root_ready` bookkeeping (6), SDK handle internals (7), cancellation |
| sandbox/test_modal.py | 41 | 0 | 0 | 41 | re-exports, pydantic/from_state API, asyncio, SDK kwargs/env, examples, Python-only path shapes |
| sandbox/test_vercel.py | 47 | 5 | 0 | 42 | re-exports, pydantic/from_state, asyncio locks and cancellation, lazily recreated sandbox, SDK env |
| sandbox/test_vercel_provider_contract.py | 5 | 0 | 0 | 5 | monkeypatches the installed vendor SDK |
| test_tool_output_trimmer.py | 4 | 0 | 0 | 4 | negative `recent_turns`/`preview_chars` (unsigned), bytes allowlist (dynamic) |

## New Rust tests (all pass, `--all-features`; default-feature builds compile)

- `tests/py_core8_r8g3_redis_compaction.rs` (3): compaction over `RedisSession` never exposes an empty history (4 Python ids), whole-history replacement behaves as clear-then-add (2), encryption outside and inside compaction (2).
- `tests/py_core8_r8g3_sqlite_concurrency.rs` (4): destructive read claimed by exactly one connection, branch allocation serialized (4 ids), concurrent structure-table claims, partial structure write rolls back.
- `tests/py_core8_r8g3_sqlite_open.rs` (1): concurrent first opens of one file.
- `tests/py_core8_r8g3_sqlx_first_access.rs` (3): concurrent first writes, write waits for a transient lock.
- `tests/py_core8_r8g3_mongodb_failures.rs` (2): pending batch invisible, failed cleanup keeps a known claim.
- `tests/py_core8_r8g3_trimmer_replay.rs` (1): SDK-produced structured output replays through the Chat Completions converter (extra assertion for already-counted cases).
- `tests/py_core8_r8g3_e2b_shutdown.rs` (1), `py_core8_r8g3_cloudflare_shutdown.rs` (1), `py_core8_r8g3_vercel_live_manifest.rs` (5).

Covered (verified by reading): advanced sqlite structure-failure rollback/retry/preserve history -> `py_memory_advanced_sqlite.rs::a_failed_batch_does_not_hide_or_roll_back_earlier_history`; async sqlite failed clear / failed pop -> `py_memory_session.rs::a_failed_clear_session_leaves_the_write_lock_free`, `a_failed_pop_item_leaves_the_write_lock_free`; mongodb index-once (2) -> `indexes_are_created_once_per_session_and_again_for_a_new_client`, partial cleanup -> `a_clear_hides_history_even_when_cleanup_leaves_documents_behind`.

## Source bugs fixed

1. `src/memory/openai_responses_compaction_session.rs`, `src/memory/mod.rs`, `src/memory/redis_session.rs`: manual compaction replaced history with clear then add, leaving an empty window on Redis. New defaulted trait method `Session::replace_history_atomically` (port of Python's `_atomic_replace_items`); `RedisSession` implements it with one Lua script; the compaction session tries it first. Lua is untested against a real Redis (none available); the mock replicates it by script text.
2. `src/memory/advanced_sqlite_session.rs`, `src/memory/sqlite_session.rs`: write transactions that read before writing (pop, delete_branch, clear, suffix replace, insert) were deferred and failed immediately with "database is locked" under a second connection; now `BEGIN IMMEDIATE`.
3. Same files: two connections claiming one fresh file with different table pairs left the loser's tables behind; base and structure tables are now created in one immediate transaction that rolls back on rejection.
4. `src/memory/sqlite_session.rs`: concurrent first opens failed the WAL switch with "database is locked"; retried for up to 5 s.
5. `src/extensions/sandbox/e2b/mod.rs`: `shutdown` returned kill failures and logged the pause failure unredacted; now logs through the tool-data policy and never raises (as Python).
6. `src/extensions/sandbox/cloudflare/sandbox.rs`: failed-delete debug log carried no detail; now includes the HTTP status and worker message only when tool data may be logged and no credentialed mount is configured.
7. `src/sandbox/session.rs`, `events.rs`, `runtime.rs`, `src/extensions/sandbox/vercel/mod.rs`: new defaulted hook `SandboxSession::validate_manifest_application` (Python `_validate_manifest_application`), called by `apply_live_session_manifest`; Vercel refuses changed S3 topology and non-mount deltas on a stopped session. Previously a stopped injected Vercel session silently skipped the delta.

## Breaking changes

- `EncryptedSession`/others unaffected. `SandboxSession::shutdown` for E2B no longer returns release errors. Two additive defaulted trait methods (`Session::replace_history_atomically`, `SandboxSession::validate_manifest_application`); external implementors are unaffected.
- `apply_live_session_manifest` now calls `running()` once on every changed manifest before the early return for a not-yet-started session (it did already) and runs the hook on unchanged manifests.

## Unresolved questions

- Encrypt compaction-restore concurrency (`test_encrypted_compaction_restores_original_tokens_before_concurrent_append`, 2) and E2B `workspace_root_ready` (6) look portable with extra harness work; left python-only.
- Python swallows `store_run_usage` failures; Rust returns them (kept; explicit API).
- Redis Lua replace script unverified on a real server.

Status: DONE_WITH_CONCERNS
Summary: 441 python-only before, 34 ported, 8 covered, 399 remain across 20 files; seven source fixes; all py_core8_r8g3_* and 476 related binaries pass; clippy and rustfmt clean for my files in both feature sets.
Concerns: other agents' files (py_core8_r8g1_docker, r8g4_sessions, r8g6_docstrings) currently fail clippy/compile and are not mine; the Redis Lua change is untested on a real Redis.
