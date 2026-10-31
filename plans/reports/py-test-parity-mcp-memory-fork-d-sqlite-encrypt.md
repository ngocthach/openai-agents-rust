# Py test parity: advanced_sqlite + encrypt (fork D)

Counting: AST over the Python files (`def test_*`, parametrize expanded by literal list length). `test_tool` in advanced_sqlite is a `@function_tool` helper that AST counts as a test (python-only).

| file | def_test | expanded | ported | covered | python-only | cross-area | unported | failures found |
|---|---|---|---|---|---|---|---|---|
| test_advanced_sqlite_session.py | 120 | 179 | 73 def / 108 exp | 2 / 2 | 27 / 46 | 1 / 1 | 17 / 22 | 11 (all fixed, below) |
| test_encrypt_session.py | 49 | 66 | 32 / 40 | 1 / 1 | 11 / 15 | 4 / 9 | 1 / 1 | 1 (cross-area gap) |

Totals: 169 def / 245 expanded. Rust tests: `tests/py_memory_advanced_sqlite.rs` (38), `tests/py_memory_advanced_sqlite_usage.rs` (30), `tests/py_memory_encrypt.rs` (32). All green with `--all-features`. `sessions_tests` (37) and `memory::` lib tests (75) still green. rustfmt applied to my files. Clippy: no findings in my files; the full gate currently fails only in other areas' files (`extensions/models/mod.rs:1016`, `models/openai_responses_ws.rs:553,556`, and other agents' `py_*` tests).

## Rust bugs fixed (src/memory/advanced_sqlite_session.rs)

1. `find_turns_by_content` folded Unicode case (`CAFÉ` matched `é`); Python's SQLite `LIKE` folds ASCII only. Now `to_ascii_lowercase`.
2. `create_branch_from_content` matched the 100-char preview, was Unicode-case-insensitive and raised `No user turn contains the text "x"`; now uses the full decoded match and Python's `No user turns found containing 'x'`.
3. Blank branch name accepted; now `Branch name cannot be empty` (from_turn and from_content).
4. `delete_branch`: added `Branch ID cannot be empty`; current-branch message now `Cannot delete current branch '<id>'. Use force=True or switch branches first`; unknown branch now `Branch '<id>' does not exist` and leaves the current branch untouched on force.
5. `get_tool_usage` ported Python's query: legacy NULL names for shell/apply_patch calls, `tool_search_output` counted only when its turn has no `tool_search_call`, SUM semantics.
6. `extract_tool_name` ignored `namespace`; now `namespace.name`, collapsing a namespace equal to the name (reserved deferred shape).
7. A stale instance did not notice an external `clear_session` on `add_items`, `pop_item`, `get_conversation_turns`, `get_turn_structure`, `find_turns_by_content`, `get_tool_usage`, `get_turn_usage` or `list_branches` (only `get_items` did). All now resolve the branch through the generation check.
8. Opening a file with `create_tables = false`, or with a second base-table pair, created base tables before ownership was validated. Added a pre-flight claim check and `AdvancedSqliteSession::open_with_tables(id, path, sessions, messages, create_tables)`. `open_with` now pre-flights too; error text for an unclaimed file now contains `create_tables=True`.

Also fixed earlier in this fork run (shared with other forks, files outside my ownership, mentioned for completeness): none.

## Cross-area / remaining gaps (not fixed)

- **Outer encrypted wrapper over a compaction session.** Python's `EncryptedSession(OpenAIResponsesCompactionSession(...))` forwards `run_compaction`/deferred compaction and feeds it decrypted reads. Rust `EncryptedSession` does not implement `on_run_completed`/`wants_model_exchange`, and the compaction session reads its own (ciphertext) store, so only the reverse order `Compaction(Encrypted(store))` compacts correctly (covered by `sessions_tests::compaction_over_an_encrypted_store_sends_plaintext_and_rewrites_envelopes`). Affects python tests 280, 362, 393 (+ the outer halves of 192/249). Fix needs a reader hook from the wrapper into `src/memory/openai_responses_compaction_session.rs` (not my file); my encrypt ports use the supported order.
- **Pending session write recovery** (`test_runner_recovers_encrypted_resumed_append`): `_PendingSessionWrite`/`defer_interrupted_session_write` do not exist in `src/run*.rs`.
- **Non-ASCII custom table names** (`test_structure_tables_reject_distinct_non_ascii_identifiers`): `SqliteSession::with_tables` (src/memory/sqlite_session.rs `valid_identifier`) accepts ASCII only. Minimal fix: allow quoted identifiers; the same limit blocks `test_structure_tables_accept_quoted_custom_session_table`.
- Unported advanced tests (17): 945 (shell/patch via Runner by branch), 1414/1486/1526/1567 (branch reservation rollback + legacy backfill), 3063 (corrupt-row limit on advanced get_items), 3172 (concurrent add on file db), 3748 (store usage after reopen), 4246/4278/4317/4357 (stale `store_run_usage` after pop; Rust `store_run_usage` has no turn-capture, so these may expose a real gap), 4577 (default read must not create generation table), 4682, 4725, 4774 (rollback after earlier delete), 4985 (quoted sessions table). Encrypt unported: 1075 (SQLAlchemy backend; `SqlxSession` port pending).

## Python-only groups

- asyncio cancellation/gating/event-loop tests (`_await_mutation`, `to_thread` gates, queued-add ordering, cross-loop turn ordering, switch-validation cancellation): no Rust equivalent; tokio drop semantics already documented on `await_session_mutation`.
- connection-internal tests (`_connections`, quarantine, rollback-failure eviction, closed-session errors, recording cursor): Rust owns one `Mutex<Connection>`; no such state.
- logging-policy tests (`DONT_LOG_MODEL_DATA` with Mock logger/caplog) and multiprocessing tests (`multiprocessing` spawn): not portable as-is.
- encrypt: kwarg `wrapper=` forwarding, `__getattr__` custom-method delegation, negative `limit` (Rust `limit` is `usize`), monkeypatched `cryptography.fernet.time` (Rust tests use real sleeps with ttl=2).

## Behavior differences documented, not bugs

- Python `get_conversation_by_turns` (dict of structure rows) corresponds to Rust `get_turn_structure`; Rust `get_turn_usage` always returns a `Vec`; `store_run_usage` takes `&Usage`; `ConversationTurn.full_content` is the joined text, not the raw list.
- Python swallows usage-write failures; Rust returns them (lock-release behavior is identical and tested).
- Zero-limit `get_items` on an encrypted/bounded session never reaches the store in Rust (Python records a read of 0).

## Shared-file edits
None (no Cargo.toml, no src/lib.rs). New public API: `AdvancedSqliteSession::open_with_tables`.

Status: DONE_WITH_CONCERNS
Summary: 102 Python tests ported or covered across both files (3 Rust files, 100 tests green), 8 source fixes in advanced_sqlite_session.rs; remaining items are classified above.
Concerns/Blockers: outer-encrypt-over-compaction wrapper unsupported (needs compaction-session hook); 18 advanced/encrypt tests still unported (listed); build dir /Users/ngocthach/.cache/oars-build/tp-mcpmem left for the parent (shared with other forks, ~2 GB).

## Remainder pass (fork D2)

Scope: the 17 unported advanced-SQLite defs (22 expanded) and the 1 unported encrypt def listed above. New files: `tests/py_memory_advanced_sqlite_remaining.rs` (21 tests: 20 pass, 1 ignored), `tests/py_memory_encrypt_remaining.rs` (1 test, passes). Counts are from actual runs with `--all-features`; the existing `py_memory_advanced_sqlite` (38), `py_memory_advanced_sqlite_usage` (30), `py_memory_encrypt` (32) and `sessions_tests` (37) still pass.

Coordinator decisions applied: `store_run_usage` keeps its explicit `&Usage` argument (documented divergence below); the quoted-identifier gap stays recorded against `sqlite_session.rs`.

### Per-test classification
| Python test (line) | exp. | outcome | Rust test |
|---|---|---|---|
| tool usage tracks runner shell and patch calls by branch (945) | 1 | PORTED | `tool_usage_tracks_runner_shell_and_patch_calls_by_branch` |
| failed branch reservation rolls back and allows retry (1414) | 1 | PORTED (fault injected with a SQLite trigger, not a monkeypatch) | `a_failed_branch_reservation_rolls_back_and_allows_retry` |
| branch reservations migrate existing populated branches (1486) | 1 | PORTED | `branch_reservations_migrate_existing_populated_branches` |
| legacy branch ids are backfilled before destructive ops (1526; clear/delete/pop) | 3 | PORTED; delete and pop failed first, fixed (bug 2) | `legacy_branch_ids_are_backfilled_before_{clear,delete,pop}` |
| legacy destructive no-ops leave database unlocked (1567; missing_delete/empty_pop) | 2 | PORTED | `a_legacy_missing_delete_leaves_the_database_unlocked`, `a_legacy_empty_pop_leaves_the_database_unlocked` |
| get_items limit skips corrupt newest rows (3063) | 1 | PORTED; failed first, fixed (bug 1) | `get_items_limit_skips_corrupt_newest_rows` |
| concurrent add_items preserves structure for file db (3172) | 1 | PORTED | `concurrent_add_items_keep_messages_and_structure_aligned_for_a_file_db` |
| store_run_usage after reopening cleared session (3748; create_tables False/True) | 2 | PORTED | `store_run_usage_after_reopening_a_cleared_session_{without_creating,creating}_tables` |
| stale usage skipped when turn removed by pop (4246) | 1 | PORTED sequential half; gated race half is a DOCUMENTED DIVERGENCE | `usage_is_not_attributed_to_a_turn_removed_by_pop` |
| stale usage not recorded against reused turn number (4278) | 1 | PORTED observable half (usage row removed with its turn, reused number starts empty); gated race half DOCUMENTED DIVERGENCE | `a_reused_turn_number_does_not_inherit_usage_recorded_for_the_removed_turn` |
| usage survives unrelated branch deletion (4317) | 1 | PORTED sequentially | `store_run_usage_survives_unrelated_branch_deletion` |
| usage skips when current branch has no turn (4357) | 1 | PORTED | `store_run_usage_skips_a_branch_without_a_turn` |
| default read does not initialize generation table (4577) | 1 | PORTED | `a_default_read_does_not_initialize_the_clear_generation_table` |
| post-clear switch synchronizes generation (4682) | 1 | PORTED | `a_post_clear_switch_synchronizes_the_generation_before_the_next_write` |
| pop rolls back on failure after earlier delete (4725) | 1 | PORTED (trigger on message delete) | `pop_item_rolls_back_when_a_later_delete_fails` |
| clear rolls back on failure after earlier delete (4774) | 1 | PORTED (trigger on turn_usage delete) | `clear_session_rolls_back_when_a_later_delete_fails` |
| quoted custom session table (4985; create_tables False/True) | 2 | PORTED but `#[ignore = "cross-area: ..."]` | `structure_tables_accept_a_quoted_custom_session_table` |
| encrypt: pop preserves SQLAlchemy history (1075) | 1 | PORTED against `SqlxSession` | `encrypted_pop_preserves_sqlx_history` |

Totals: 22 advanced expanded + 1 encrypt = 23 expanded. 21 ported and green; 2 ignored cross-area (the quoted-table test covers 2 expanded cases); 0 unported. The two gated stale-usage race halves are documented divergences inside tests that are otherwise ported and green, not failures.

### Rust bugs fixed (`src/memory/advanced_sqlite_session.rs`)
1. `get_items` / `get_items_for_branch` with a limit counted stored rows, not readable items, so corrupt newest rows shrank the window (`["valid 2"]` instead of `["valid 1","valid 2"]`). The fetch window now widens past corrupt rows, as in Python.
2. `delete_branch` and `pop_item` did not back-fill branch reservations before deleting the rows that are their only evidence in a database written before reservations existed, so a deleted or emptied legacy branch id could be reused. Both now call `ensure_branch_reservations` first inside their transaction (`clear_session` already did). An empty pop now commits the back-fill instead of dropping it.

### Documented divergence
- `store_run_usage` takes an explicit `&Usage` and reads the current turn and writes the row in one blocking call, so Python's `_gate_worker("_update_sync")` race tests (4246, 4278) have no Rust equivalent; usage rows are removed with their turn (`pop_item`), which is the observable invariant, and that is what the ported tests pin.

### Cross-area / not done
- **Quoted custom session table (4985):** `SqliteSession::with_tables` (`src/memory/sqlite_session.rs`, `valid_identifier`) accepts ASCII letters, digits and underscores only, so `"quoted_sessions"` is rejected before `AdvancedSqliteSession` can use it. Minimal fix: accept SQL-quoted identifiers there. Test is ignored with that reason.
- **EncryptedSession outermost over a compaction session** (requested by the E2 fork; blocks `encrypted_prefix_cleanup_with_the_encryption_outermost` in `tests/py_memory_sqlx_compaction.rs` and the 4 cross-area encrypt tests above): NOT fixed, not feasible inside `encrypted_session.rs`. `EncryptedSession(CompactionSession(store))` stores envelopes in `store`; the compaction session reads and rewrites that store directly (`openai_responses_compaction_session.rs:471, 632, 843, 846`). Merely forwarding `on_run_completed`/`wants_model_exchange` from the wrapper would make it send ciphertext envelopes to the compaction API and write the plaintext summary into an encrypted store, which is worse than not compacting. Minimal correct fix needs a reader/writer hook in the compaction session: a `with_item_codec(decrypt_on_read, encrypt_on_write)`-style seam that `EncryptedSession` installs when its underlying is a compaction session, plus forwarding `on_run_completed`, `wants_model_exchange` and `run_compaction` in `encrypted_session.rs`. Until then only `Compaction(Encrypted(store))` is supported, and that order is documented in the type docs and covered by `sessions_tests::compaction_over_an_encrypted_store_sends_plaintext_and_rewrites_envelopes`.

### Gates
- rustfmt applied to `src/memory/advanced_sqlite_session.rs` and both new test files.
- `cargo clippy --all-features --lib --test py_memory_advanced_sqlite_remaining --test py_memory_encrypt_remaining -- -D warnings` could not reach my files: it stops on other areas' lint errors (`src/run_internal/agent_identity.rs:106` complex type, `src/run_internal/mod.rs:2329,2384,2749` needless borrow, `src/stream_events.rs:48` large enum variant). No finding is in files I own.
