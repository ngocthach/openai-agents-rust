# Sessions / memory final parity report

Scope: `src/memory/**`, `tests/sessions_tests.rs`, `examples/sessions_encrypted.rs`, plus two
additive `Cargo.toml` lines. Python reference: `openai-agents` 0.24.

## Closed rows

| # | Row | How | Test |
| --- | --- | --- | --- |
| 1 | `EncryptedSession::pop_item` leaves an unauthenticatable envelope in place | New defaulted `Session::pop_item_validated(PopValidator)`; `SqliteSession` claims the tail and runs the validator inside the transaction that owns the deletion, so an `Err` rolls it back. `EncryptedSession::pop_item` passes `extract_timestamp` (no TTL), so authenticated-but-expired and plaintext rows still drain while an unauthenticatable envelope fails the pop and keeps its row id and position. Other backends keep the trait default (pop-then-validate), the same split Python draws. | `tests/sessions_tests.rs::a_pop_that_cannot_authenticate_leaves_the_envelope_in_place`, `src/memory/sqlite_session.rs::tests::a_rejected_pop_leaves_the_row_in_place` |
| 2 | `prune_prefix` reclamation | New defaulted `Session::replace_compaction_suffix_pruning(handles, items, PrunePrefix)`. Implemented in `SqliteSession` and `SqlxSession` (Python supports it on native SQLite and SQLAlchemy); batched `LIMIT 256` prefix scan inside the replacement transaction, stopping at the first rejected or unparsable row, exactly as `sqlite_session.py` L410-434. `EncryptedSession` now keeps the raw handles / positions / retained boundary of its last snapshot (Python captures them in the closure) and translates a handle slice back to the contiguous stored range, so expired envelopes *inside* the suffix are replaced instead of making the store refuse a tail it cannot match. | `tests/sessions_tests.rs::a_successful_replacement_reclaims_expired_envelopes`, `src/memory/sqlite_session.rs::tests::{a_successful_replacement_reclaims_the_accepted_prefix,a_refused_replacement_reclaims_nothing}` |
| 3a | `max_rollback_items` | Confirmed present in Python 0.24 (`openai_responses_compaction_session.py` L122, L158-159, L548-556). Added `OpenAIResponsesCompactionSession::with_max_rollback_items(usize) -> Result<Self>` (zero rejected) and `all_underlying_items()` which reads `budget + 1` and fails with Python's message. Enforced on the manual path before any work and at the top of `compact_whole_history` (Python L309-312 and L448-449). Automatic suffix replacement through a native snapshot is unaffected, as in Python. | `tests/sessions_tests.rs::{compaction_refuses_a_history_over_the_rollback_budget,compaction_within_the_rollback_budget_still_runs}` |
| 4 | `drop_orphan_function_calls` at the compaction call site | **Premise corrected:** Python does *not* prune here. `openai_responses_compaction_session.py` L408-419 is `if drop_orphan_function_calls(...) != session_items: return` — it refuses the suffix, same as the port. The real delta was the orphan *definition*: `has_complete_tool_groups` ignored `program` / `program_output` items, so a complete program group was wrongly refused (its output's `call_id` had no matching call) and an orphan program was wrongly accepted. Now programs are a group of their own: a program needs its output or a retained owned item, a program output needs its program, and a program-owned item needs its program. Reasoning items preceding a dropped call are irrelevant because nothing is dropped — the whole suffix is refused. | `src/memory/openai_responses_compaction_session.rs::tests::program_groups_must_survive_orphan_pruning` |
| 5 | Redis `TYPE` check atomic with the write | `KEY_GUARD_PRELUDE` (a Lua `guard()` + `now()`) is prepended to three scripts — `ADD_ITEMS_SCRIPT`, `CLEAR_SESSION_SCRIPT`, `REPLACE_SUFFIX_SCRIPT` — and every write is now one `EVAL`. The separate `TYPE`/`TIME` round trip (`check_keys_and_server_time`) and the Rust `validate_key_types` are gone. `WRONGTYPE …` message text unchanged; server clock still comes from `TIME`, now read inside the script (effect replication, Redis ≥ 5). | `src/memory/redis_session.rs::tests::every_write_runs_the_key_type_guard_inside_one_script` (structural) plus the existing env-gated `redis_round_trip`, which already asserts `WRONGTYPE` from `add_items` and `clear_session` against a real server. Per the task constraints Redis integration tests stay env-gated. |
| 6 | `SqlxSession` on SQLite | **The dependency objection was wrong.** `cargo tree -i libsqlite3-sys --all-features` resolves `rusqlite 0.39` and `sqlx-sqlite 0.9` to one `libsqlite3-sys 0.37`; a scratch crate using both links and runs. Added `SqlxDialect::Sqlite` (`?` placeholders, `INSERT OR IGNORE`, the same schema `SqliteSession` creates) and `from_url` now accepts `sqlite:`. | `tests/sessions_tests.rs::sqlx_session_round_trips_on_sqlite` (real temp-file database, not gated), `src/memory/sqlx_session.rs::tests::{dialects_come_from_the_url_scheme,placeholders_follow_the_dialect}` |
| 7a | Table-ownership claims | `claim_structure_tables` + `resolve_base_table_owners` + `resolve_table_identifier` ported, including the SQLite authorizer (`rusqlite::Connection::authorizer`, feature `hooks`) that reports which table an identifier resolves to, and `identifiers_equal`, which asks SQLite `SELECT ? = ? COLLATE NOCASE` rather than reimplementing its folding. Run after creation and instead of creation when `create_tables` is cleared. | `tests/sessions_tests.rs::{structure_tables_reject_a_second_base_table_pair,opening_an_unclaimed_file_without_creating_tables_is_refused}` |
| 7b | Cross-process `branch_reservations` | Table created and back-filled (`ensure_branch_reservations`); `create_branch_from_turn` now takes `BEGIN IMMEDIATE` before claiming, and the claim is `INSERT OR IGNORE` on the primary key (`reserve_branch_id`), so two processes cannot take one name and `clear_session` does not release a reservation. Generated names follow Python's `branch_from_turn_{n}_{epoch}` with a `_2`, `_3`… suffix walk. | `tests/sessions_tests.rs::advanced_sqlite_reads_any_branch_and_reserves_branch_ids` |
| 7c | `session_clear_generations` | Branch state is now `(branch, generation)`. `clear_session` bumps the durable generation in its transaction and resets the pointer under the connection lock; reads call `refresh_branch_after_external_clear`; `switch_to_branch` and `create_branch_from_turn` commit through `commit_branch_pointer`, which loses to a clear that committed meanwhile. | `tests/sessions_tests.rs::another_instance_clearing_the_session_resets_a_stale_branch_pointer` |
| 8c | `get_items(limit, branch_id=…)` | New `AdvancedSqliteSession::get_items_for_branch(limit, Option<&str>)`; `Session::get_items` delegates with `None`, which resolves the current branch after synchronizing an external clear. | `tests/sessions_tests.rs::advanced_sqlite_reads_any_branch_and_reserves_branch_ids` |
| 8e | `create_tables` parameter | `AdvancedSqliteSession::open_with(session_id, path, create_tables)` and `from_session_with(base, create_tables)`. `open` / `in_memory` / `from_session` keep creating (unchanged behaviour). | `tests/sessions_tests.rs::{structure_tables_reject_a_second_base_table_pair,opening_an_unclaimed_file_without_creating_tables_is_refused}` |
| 9 | `OpenAIConversationsSession::session_id()` on an uninitialized session | `Session::session_id` returns `&str` and cannot fail, so it keeps the documented `""`. Added the fallible inherent form `try_session_id() -> Result<String>` carrying Python's exact message ("Session ID not yet available. The session is lazily initialized on first API call…"). `conversation_id()` remains the async form that creates the conversation. | `tests/sessions_tests.rs::an_uninitialized_conversations_session_reports_why_it_has_no_id` |
| 10 | `await_session_mutation` | `memory::await_session_mutation(fut)` spawns the mutation, so a dropped caller cannot abort a half-written mutation. Used in `EncryptedSession::add_items` and `EncryptedSession::replace_compaction_suffix`. `SqliteSession`'s mutations already run on `spawn_blocking`, whose task survives a dropped handle. | `src/memory/mod.rs::tests::a_shielded_mutation_outlives_its_caller` |
| 12 | `MongoDBSession::pop_item` generation re-checks | The atomic `$slice` claim is **not** equivalent on its own: both Python re-check loops were missing. Ported — (a) a claim that finds nothing re-reads the generation and retries in the new one instead of reporting an empty session, (b) a successful claim re-reads the generation and, if a `clear_session` committed meanwhile, deletes the claimed document and retries rather than returning an item from history that is no longer visible. | Covered by the existing env-gated Mongo integration test path only (`MONGODB_URI`); Mongo tests stay env-gated per the task constraints. See "Concerns". |
| 13 | `OpenAIResponsesCompactionSession::new` fallible | `new` now returns `Result<Self>` and rejects a store that manages its own history remotely, as Python's `__init__` does. `try_new` removed. All call sites are in files owned here. **Breaking change** — see below. | `tests/sessions_tests.rs::compaction_session_refuses_a_server_managed_store` (updated) |
| 14 | `_ignore_ids_for_matching` wired on both sides | Verified. Runner: `src/run_internal/mod.rs` L1396-1410 gates on `Session::wants_model_exchange()` and strips the item id when `session.manages_history_remotely()`. Session: `openai_responses_compaction_session.rs` `ignore_ids_for_matching()` reads the same predicate off the wrapped store and passes it to `match_model_exchange_suffix`. Both read one predicate, so they cannot disagree. | existing `suffix_matching_finds_the_longest_tail_present_in_the_exchange` + runner tests |
| 15 | `chacha20poly1305` unused dependency | **Report was stale.** `grep chacha Cargo.toml` finds nothing; `encrypt = ["dep:hkdf", "dep:base64", "dep:aes", "dep:cbc", "dep:hmac"]`. The only `chacha` hits in `Cargo.lock` are `rand_chacha`, a transitive dependency of `rand` (needed for the Fernet IV). Nothing to drop. | n/a (verification, no code change) |
| 16 | Dapr HTTP-vs-gRPC, `async_sqlite_session` | Re-verified as parity, not gaps. Dapr: Python uses the gRPC Dapr SDK; there is no Rust Dapr SDK, and the HTTP state API this port uses exposes the same operations (ETag, consistency, `metadata.ttlInSeconds`), so the observable contract matches. `async_sqlite_session.py` exists only because Python's `SQLiteSession` is synchronous; the Rust `SqliteSession` is already async (`spawn_blocking` + `Mutex<Connection>`), so a second async variant would be duplicate code with no behavioural difference. | n/a (documented in the type docs) |

## Rows left open, with the inherent reason

- **3b — Python's incremental candidate cache.** Not ported; proved from the source rather than
  reimplemented. `_ensure_compaction_candidates` (`openai_responses_compaction_session.py`
  L814-850) consults the cache **only** when `read_items is None` (so never through the
  encrypted wrapper) and `limit is None`. Every wrapper mutation maintains it exactly:
  `add_items` extends it with the items just written (L765-771), `pop_item` and a failed write
  clear it (L755-764, L773-792), `clear_session` sets it to `[]` (L808-809), and a successful
  compaction sets it to the output it just stored (L517-520). For wrapper-only traffic it
  therefore holds exactly what a fresh read returns — pure memoization. The one case where it
  differs is an *external* writer mutating the wrapped store directly, and then the cache is
  stale while this port's fresh read is correct. Documented on the type.
- **11 — the "context opt-in" (`session.py` L209-266).** Inherent. Python decides per method,
  at call time, with `inspect.signature(method).parameters`, whether a session accepts a
  `wrapper` keyword. Rust has no runtime signature reflection. The second half is also
  inherent: `RunContext<C>` is generic over the user's context type (`src/run_context.rs:180`),
  while `Session` must stay object-safe for `Arc<dyn Session>`, so a trait method could not
  name the caller's context type. The only Rust shape would be defaulted `*_with_context`
  methods taking an erased context, and the call sites that would have to pass it live in
  `src/run_internal/**`, which is not owned here. Not added: dead defaulted methods nothing
  calls would be worse than reporting it.
- **10 (runner half).** `memory::await_session_mutation` exists and is used by the wrappers
  owned here. The runner-side call sites that persist a run's items to the session live in
  `src/run_internal/**` and are not owned here — they still await the session mutation
  directly, so a cancelled run can drop a session write in flight. Needs the runner owner to
  wrap those awaits.
- **14 (follow-up, not a regression of the tested behaviour).** Now that `new` rejects any
  store with `manages_history_remotely() == true`, that same predicate — which doubles as the
  ignore-ids flag on both sides — can no longer be true for a wrapped store, so the ignore-ids
  path is unreachable in-tree. Python's rejection is narrower (`isinstance(…,
  OpenAIConversationsSession)`) than its `_ignore_ids_for_matching` attribute lookup, so a
  custom Python store can opt into id-insensitive matching without being rejected. Separating
  the two would need a new `Session::ignore_ids_for_matching()` hook consumed in
  `src/run_internal/mod.rs`, which is not owned here.

## Breaking changes

1. **`OpenAIResponsesCompactionSession::new` returns `Result<Self>`** (was infallible) and
   rejects a wrapped store that manages its history remotely, matching Python's `__init__`.
   **`try_new` is removed** — it is now exactly `new`.
2. **`SqlxDialect` gained a `Sqlite` variant.** The enum was already non-`#[non_exhaustive]`,
   so an exhaustive external `match` on it stops compiling.
3. **`SqlxDialect::from_url("sqlite://…")` now succeeds** instead of returning the
   "use SqliteSession" error. Any caller relying on that error loses it.
4. **`Session` gained two defaulted methods** (`pop_item_validated`,
   `replace_compaction_suffix_pruning`). Defaulted, so existing implementations keep
   compiling; only a `#[deny]`-style exhaustive re-implementation is affected.
5. **`SqliteSession::pop_item` now skips a row whose JSON no longer parses** and keeps looking
   for a readable item, instead of returning `None`. This is what Python does
   (`sqlite_session.py` L508-513).

New public API (additive): `memory::await_session_mutation`, `memory::PrunePrefix`,
`memory::PopValidator`, `Session::pop_item_validated`,
`Session::replace_compaction_suffix_pruning`,
`OpenAIResponsesCompactionSession::with_max_rollback_items`,
`OpenAIConversationsSession::try_session_id`,
`AdvancedSqliteSession::{open_with, from_session_with, get_items_for_branch}`,
`SqlxDialect::Sqlite`.

## Shared-file edits

Two lines in `Cargo.toml`, both additive, no version changes:

```toml
# was: features = ["bundled"]
rusqlite = { version = "0.39", features = ["bundled", "hooks"], optional = true }

# was: features = [..., "postgres", "mysql", "any", ...]
sqlx = { version = "0.9", default-features = false, features = ["runtime-tokio", "tls-rustls-aws-lc-rs", "postgres", "mysql", "sqlite", "any", "json", "chrono"], optional = true }
```

`rusqlite/hooks` enables `Connection::authorizer` (row 7a). `sqlx/sqlite` enables
`SqlxDialect::Sqlite` (row 6); it unifies on the `libsqlite3-sys 0.37` `rusqlite` already
pulls, verified with `cargo tree -i libsqlite3-sys --all-features`. **No `src/lib.rs` edit was
needed or made.**

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/misc2 \
       CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features --lib memory          # 58 passed
cargo test --all-features --test sessions_tests # 37 passed
```

Clippy findings in files **not** owned here (reported, not fixed):

- `src/models/openai_chatcompletions.rs:1352` and `:1381` — `unused_assignments` on
  `reasoning_index`.
- `src/sandbox/events.rs:505` — `manual_is_multiple_of`.
- `src/sandbox/events.rs:510` — `match_like_matches_macro`.

Nothing in `src/memory/**`, `tests/sessions_tests.rs` or `examples/sessions_encrypted.rs`.
