# Session Backends Port Report

Port of the remaining Python `openai-agents` 0.24 session backends into
`openai-agents-sdk` (`src/memory/**`). All work is confined to `src/memory/**`,
`tests/sessions_tests.rs` and `examples/sessions_encrypted.rs`. The `Session`
trait, `SessionRunCompletion` and the existing types' public APIs are unchanged.

## Ported

| Python source | Rust module | Feature |
| --- | --- | --- |
| `memory/openai_responses_compaction_session.py` | `src/memory/openai_responses_compaction_session.rs` | default |
| `extensions/memory/encrypt_session.py` | `src/memory/encrypted_session.rs` | `encrypt` |
| `extensions/memory/redis_session.py` | `src/memory/redis_session.rs` | `redis` |
| `extensions/memory/advanced_sqlite_session.py` | `src/memory/advanced_sqlite_session.rs` | `sqlite` |
| `extensions/memory/dapr_session.py` | `src/memory/dapr_session.rs` | default |
| `extensions/memory/mongodb_session.py` | `src/memory/mongodb_session.rs` | `mongodb` |
| `extensions/memory/sqlalchemy_session.py` | `src/memory/sqlx_session.rs` (`SqlxSession`) | `sqlx` |

`src/memory/sqlite_session.rs` gained three `pub(super)` accessors
(`connection`, `sessions_table`, `messages_table`) so `AdvancedSqliteSession`
extends the same connection and base schema. These are module-private, not
public API.

## Not ported, and why

- **`extensions/memory/async_sqlite_session.py`** — no port needed. The Python
  class exists only because `SQLiteSession` is synchronous; the Rust
  `SqliteSession` is already async (`spawn_blocking` + `Mutex<Connection>`),
  so a second async variant would be duplicate code.
- **`SqlxSession` on SQLite** — intentionally unsupported. This crate already
  links `sqlite3` through `rusqlite` for `SqliteSession`, and linking it again
  through `sqlx-sqlite` is not sound. `SqlxDialect::from_url` rejects `sqlite:`
  URLs with a message pointing at `SqliteSession`.
- **Compaction: atomic suffix replacement.** Python can replace a bounded,
  model-visible suffix atomically on stores that support it, matching stored
  items against the exact model exchange (reasoning-id policies, orphan
  function-call checks, hosted-MCP approval pairing). This port always replaces
  the complete stored history (`clear_session` + `add_items`, restoring the
  previous items if the replacement fails) — the behaviour Python itself uses
  for stores without suffix replacement and for manual `run_compaction` calls.
  `max_rollback_items` and the candidate cache are not implemented.
- **`EncryptedSession` wire format.** Python uses Fernet (AES-128-CBC +
  HMAC-SHA256), which none of this crate's dependencies provide. The envelope
  shape is identical (`{"__enc__":1,"v":1,"kid":"hkdf-v1","payload":…}`) and the
  HKDF-SHA256 derivation uses the same salt (`session_id`) and info string
  (`agents.session-store.hkdf.v1`), but the payload is
  `base64(nonce || XChaCha20-Poly1305(nonce, be_u64(created_at) || json))`.
  Histories are **not** interchangeable with Python's; this is documented on the
  type. The timestamp is inside the authenticated plaintext, so an item's age
  cannot be forged. Python's SQLite-transactional `pop_item` authentication is
  not reproduced: unreadable envelopes encountered by `pop_item` are removed.
- **`RedisSession`**: appends/clears use one `MULTI`/`EXEC` pipeline instead of
  a `WATCH`-guarded transaction — `WATCH` is per-connection state and unsafe on
  the shared `MultiplexedConnection`. Python's key-type validation under `WATCH`
  and its use of server `TIME` for expiry are therefore not ported (expiry uses
  the client clock). Redis Cluster is not supported.
- **`DaprSession`**: Python talks gRPC through the Dapr SDK; this port issues
  the equivalent HTTP state-API requests with `reqwest` (there is no Rust Dapr
  SDK dependency here). TTL goes out as the `metadata.ttlInSeconds` query
  parameter, which the HTTP API requires. ETag conflicts are detected from
  HTTP 409 and from 500 responses whose message names a mismatch, mirroring
  Python's message-substring detection.
- **`AdvancedSqliteSession`**: the Python table-ownership claims (SQLite
  authorizer based), cross-process `branch_reservations`, and
  `session_clear_generations` bookkeeping are not ported; this session assumes
  it is the only writer of the structure tables for its database. The
  `message_structure` and `turn_usage` schemas and indexes are identical to
  Python's. `store_run_usage` is an explicit method taking a run's `Usage`
  rather than a runner hook.
- **`MongoDBSession`**: schema, batch documents (`message_data` array + `seq` +
  `generation`), legacy single-item documents, and the generation-based
  `clear_session` are ported faithfully. Python's extra generation re-check
  loops inside `pop_item` are simplified to the atomic `$slice` claim plus
  empty-marker cleanup. The `_generation` and `_seq` counters are read through a
  helper that accepts BSON `Int32` as well as `Int64`, because pymongo stores
  small Python ints as `Int32`; without it a Python-created session would read
  as generation 0 and Rust appends would become invisible.

## Public API summary

All items below are re-exported from `openai_agents_sdk::memory`.

**Compaction (default features)**
- `OpenAIResponsesCompactionSession` — `new(session_id, Arc<dyn Session>)`,
  `with_client`, `with_model` (validated), `with_compaction_mode`,
  `with_should_trigger_compaction`, `underlying_session`,
  `run_compaction(args) -> Result<Usage>`. Implements `Session`, and
  `on_run_completed` performs the automatic compaction: it skips when there is
  no `response_id`, defers (recording the response id) when the turn produced
  local tool outputs, forces a deferred compaction on the next settled turn, and
  returns the compaction request's `Usage` so the runner adds it to the run.
- `CompactionMode` (`PreviousResponseId` | `Input` | `Auto`),
  `ResolvedCompactionMode`, `OpenAIResponsesCompactionArgs`
  (`response_id`, `store`, `compaction_mode`, `force`), `CompactionContext`,
  `ShouldTriggerCompaction`, `DEFAULT_COMPACTION_THRESHOLD` (10),
  `default_should_trigger_compaction`, `select_compaction_candidate_items`,
  `is_openai_model_name`.

**`EncryptedSession` (feature `encrypt`)**
- `new(session_id, Arc<dyn Session>, encryption_key, ttl_seconds)`,
  `with_max_scan_items`, `underlying_session`, `ttl`, `encrypt_items`.
- Does **not** forward `on_run_completed`: the only supported composition is
  `Compaction(Encrypted(store))`, so the compaction API always receives
  plaintext. Documented on the type and covered by a test.

**`RedisSession` (feature `redis`)**
- `from_url(session_id, url)`, `with_connection`, `with_key_prefix`,
  `with_ttl`, `with_settings`, `session_key`, `messages_key`, `ping`.
- `DEFAULT_KEY_PREFIX` (`agents:session`), `redis_session_keys(prefix, id)`.

**`AdvancedSqliteSession` (feature `sqlite`)**
- `in_memory`, `open`, `from_session`, `with_settings`.
- Branching: `current_branch`, `switch_to_branch`, `list_branches`,
  `create_branch_from_turn`, `create_branch_from_content`, `delete_branch`.
- Analytics: `get_conversation_turns`, `get_conversation_by_turns`,
  `find_turns_by_content`, `get_tool_usage`, `store_run_usage`,
  `get_turn_usage`, `get_session_usage`.
- Types: `BranchInfo`, `ConversationTurn`, `ToolUsage`, `TurnUsage`,
  `SessionUsage`, `MAIN_BRANCH`.

**`DaprSession` (default features)**
- `new(session_id, state_store_name, endpoint)`, `from_env` (reads
  `DAPR_HTTP_ENDPOINT`, `DAPR_HTTP_PORT`, `DAPR_API_TOKEN`), `with_api_token`,
  `with_ttl`, `with_consistency`, `with_settings`, `with_http_client`,
  `messages_key`, `metadata_key`, `ping`.
- `DaprConsistency`, `DEFAULT_DAPR_HTTP_ENDPOINT`, `dapr_session_keys`.

**`MongoDBSession` (feature `mongodb`)**
- `from_uri(session_id, uri, database)`, `with_client`, `with_collections`,
  `with_settings`, `ping`.
- `DEFAULT_SESSIONS_COLLECTION`, `DEFAULT_MESSAGES_COLLECTION`,
  `generation_filter`.

**`SqlxSession` (feature `sqlx`)**
- `from_url(session_id, url, create_tables)`, `with_pool`, `with_tables`,
  `with_settings`, `pool`, `dialect`, `create_tables`.
- `SqlxDialect` (`Postgres` | `MySql`) with `from_url` and `placeholder`,
  `DEFAULT_SESSIONS_TABLE`, `DEFAULT_MESSAGES_TABLE`.

## Tests

- 37 unit tests in `src/memory/**` (`cargo test --features … --lib memory`):
  candidate selection, model-name validation, orphan-id stripping, compaction
  output normalization, auto-mode resolution, encryption round trip / key
  handling / scan budget, Redis and Dapr key layouts and decoding, Dapr ETag
  conflict classification and backoff, Mongo generation filter and document
  decoding including `Int32` counters from other drivers, sqlx per-dialect SQL
  generation, SQLite message classification and tool-name extraction, SQLite
  branching / turns / usage / pop / clear.
- 10 integration tests in `tests/sessions_tests.rs` (5 of them under default
  features):
  - encryption round trip asserting the stored envelope shape and that no
    plaintext appears in the underlying store;
  - TTL expiry (`ttl = 1`, 2.1 s sleep): the envelope stays stored but is no
    longer returned;
  - wrong key and wrong session id (the HKDF salt) both read as empty history;
  - advanced SQLite branching / turns / tool usage / per-turn and session usage
    over a temp file, including reopening the file and deleting a fork;
  - compaction request body in `previous_response_id` mode plus history
    replacement and returned `Usage`;
  - compaction declining below the default threshold, then honouring `force`;
  - compaction `on_run_completed` skipping when `response_id` is `None`,
    deferring when `has_local_tool_outputs` is set, and then forcing the
    deferred compaction on the next settled turn even though the decision hook
    declines;
  - compaction over an encrypted store: the request carries plaintext and no
    `__enc__` envelopes, and the replacement is written back encrypted;
  - an end-to-end `Runner` run with `testing::FakeModel`, proving
    `on_run_completed` fires with the model's `resp_fake_1` id and that the
    compaction usage lands in `RunResult::usage`;
  - `DaprSession` against a stateful mock sidecar: key names, the stored value
    shape, `created_at` preservation in metadata, `consistency=strong` on every
    read, `metadata.ttlInSeconds` and `options.consistency` on every write,
    `options.concurrency = first-write` on every conditional write, and a
    seeded 409 that the session retries successfully.
- Redis / MongoDB / sqlx integration tests are present but skip unless
  `REDIS_URL`, `MONGODB_URI` or `DATABASE_URL` is set; no services are
  available in this environment.
- Mock servers are raw `tokio::net::TcpListener` handlers, as in
  `tests/openai_http_tests.rs`. No real network or external services.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/sessions

cargo test --features sqlite,encrypt,redis,mongodb,sqlx --lib memory          # 37 passed
cargo test --features sqlite,encrypt,redis,mongodb,sqlx --test sessions_tests # 10 passed
cargo test --test sessions_tests                                             # 5 passed (default features)
cargo test --features sqlite,encrypt,redis,mongodb,sqlx                      # whole suite passed
cargo doc  --features sqlite,encrypt,redis,mongodb,sqlx --no-deps            # no warnings in my files

cargo clippy --features sqlite,encrypt,redis,mongodb,sqlx --all-targets -- -D warnings  # FAILS
cargo clippy --all-targets -- -D warnings                                               # FAILS
```

**Both clippy gates currently fail, and every reported diagnostic is in a file
another agent owns — zero are in `src/memory/**`, `tests/sessions_tests.rs` or
`examples/sessions_encrypted.rs`** (verified with
`cargo clippy … --all-targets 2>&1 | grep -c -E 'memory/|tests/sessions|examples/sessions'`
→ `0` for both feature sets). See "Requested core changes" below; items 1 and 3
list the exact locations. The default-feature gate passed cleanly earlier in
this session, so item 3 appeared while parallel work landed.

`rustfmt --edition 2024` applied to every file touched.

## Requested core changes (outside my scope)

These are the only reasons the two clippy gates do not pass.

1. **`clippy::large_enum_variant` fails only when the `mongodb` feature is on.**
   `bson 2` (default features) enables `serde_json/preserve_order`, which makes
   `serde_json::Value` IndexMap-backed and larger. That pushes two existing
   enums past the lint threshold, so
   `cargo clippy --features …,mongodb --all-targets -- -D warnings` reports:
   - `src/result.rs:111` `StreamMsg::Event(StreamEvent<C>)`
   - `src/run_internal/local_tools.rs:91` `Gate::Pending(ToolApprovalItem)`

   Both are one-line fixes (box the large variant) in files another agent owns.
   Reproduce with `cargo clippy --features mongodb --lib -- -D warnings`;
   every other single feature is clean. Alternative: pull `bson 2` in as a
   direct dependency with `default-features = false` to drop `preserve_order`
   (a `Cargo.toml` change, also outside my scope).

2. **Optional `Cargo.toml` entry for the new example.** `examples/sessions_encrypted.rs`
   guards its `main` with `#[cfg(feature = "encrypt")]` / `#[cfg(not(…))]` so it
   compiles under default features. Adding

   ```toml
   [[example]]
   name = "sessions_encrypted"
   required-features = ["encrypt"]
   ```

   would let the guards be dropped, matching `sessions_sqlite`.

3. **`clippy::type_complexity` under default features** (appeared while parallel
   work landed; all in other agents' files):
   - `src/extensions/codex/tool.rs:72` and `:73`
   - `src/run_config.rs:113`
   - `src/run_internal/mod.rs:162`

   Each is fixable with a `type` alias.

4. **No changes needed** to `src/run_internal/**`: the existing
   `Session::on_run_completed` hook and `SessionRunCompletion`
   (`response_id`, `store`, `has_local_tool_outputs`) carry exactly what
   automatic compaction needs, and the returned `Usage` is already folded into
   the run.
