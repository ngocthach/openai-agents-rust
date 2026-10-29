# Session Backends Port Report

Port of the remaining Python `openai-agents` 0.24 session backends into
`openai-agents-sdk` (`src/memory/**`). All work is confined to `src/memory/**`,
`tests/sessions_tests.rs` and `examples/sessions_encrypted.rs`. The `Session`
trait, `SessionRunCompletion` and the existing types' public APIs are unchanged.

## Ported

| Python source | Rust module | Feature |
| --- | --- | --- |
| `memory/openai_responses_compaction_session.py` | `src/memory/openai_responses_compaction_session.rs` | default |
| `extensions/memory/encrypt_session.py` | `src/memory/encrypted_session.rs` + `src/memory/fernet.rs` | `encrypt` |
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
- **`EncryptedSession` is now wire-compatible with Python** (phase 2). `src/memory/fernet.rs`
  implements Fernet itself — version `0x80`, 64-bit big-endian timestamp, 128-bit IV,
  AES-128-CBC with PKCS7, HMAC-SHA256 over version ‖ timestamp ‖ IV ‖ ciphertext, url-safe
  base64 with padding, and the spec's signing-key/encryption-key split of the 32-byte key.
  The session derives that key exactly as Python does (`_ensure_fernet_key_bytes` then
  HKDF-SHA256 with `salt = session_id` and `info = agents.session-store.hkdf.v1`) and writes
  the same envelope. Master-key decoding reproduces `base64.urlsafe_b64decode`'s leniency
  rather than using a strict url-safe decoder: Python only *translates* `-`/`_` to `+`/`/` and
  then decodes with `validate=False`, so it also accepts standard-alphabet `+` and `/` and
  discards non-alphabet bytes such as a trailing newline, while still requiring padding. This
  matters in practice — a key from `openssl rand -base64 32` usually contains `+` or `/`, and a
  strict decoder would fall back to the raw 44 string bytes and derive a *different* key than
  Python, silently breaking interop. Verified against `python3`: `+`/`/`, a trailing newline and
  surrounding whitespace all decode to 32 bytes, while an unpadded 43-character key raises
  `Incorrect padding` and therefore takes the raw-bytes fallback on both sides. TTL handling matches too, including the one-minute
  `_MAX_CLOCK_SKEW` rejection of future-dated tokens and the age-before-signature check order.
  Python's SQLite-transactional `pop_item` authentication is still not reproduced: unreadable
  envelopes encountered by `pop_item` are removed rather than left in place.
  - **The XChaCha20-Poly1305 variant from phase 1 was removed, not kept as an option.** Two
    on-disk formats would fragment stored history, and because Python's
    `_is_encrypted_envelope` only checks that `kid` is *present*, a Rust-written non-Fernet
    envelope would be handed to Python's `_unwrap`, fail, and be silently skipped as if the
    item never existed. One interoperable format avoids that data-loss foot-gun. As a result
    `chacha20poly1305` is no longer referenced anywhere in `src/` (see core change 2).

- **`RedisSession`**: appends/clears use one `MULTI`/`EXEC` pipeline instead of a
  `WATCH`-guarded transaction — `WATCH` is per-connection state and unsafe on the shared
  `MultiplexedConnection`. Phase 2 closed the two gaps this left: the key-type validation
  (`TYPE` on the metadata, messages and counter keys, rejecting a collision with
  `WRONGTYPE …`) and the server clock (`TIME`, used for `created_at`/`updated_at`) are now
  read in the round trip immediately before the write, and `clear_session` refuses to delete
  keys occupied by another type. The residual difference is that this check is not atomic with
  the write: a type changed in the instant between them is not caught, where Python's `WATCH`
  would abort. Expiry deliberately uses a relative `PEXPIRE` rather than Python's `PEXPIREAT`
  computed from server `TIME`, which makes it independent of clock skew. Redis Cluster is not
  supported.
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
  `with_max_scan_items`, `underlying_session`, `ttl`, `encrypt_items`,
  `envelope_timestamp` (the authenticated creation time of an envelope, ignoring the TTL).
- The Fernet primitive lives in a private `src/memory/fernet.rs`; it is deliberately not part
  of the public API, since nothing outside this module needs it.
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

- 49 unit tests in `src/memory/**` (`cargo test --features … --lib memory`):
  candidate selection, model-name validation, orphan-id stripping, compaction
  output normalization, auto-mode resolution, encryption round trip / key
  handling / scan budget, Redis and Dapr key layouts and decoding, Dapr ETag
  conflict classification and backoff, Mongo generation filter and document
  decoding including `Int32` counters from other drivers, Redis key-type validation
  (missing keys accepted, collisions rejected, first mismatch reported), sqlx per-dialect SQL
  generation, SQLite message classification and tool-name extraction, SQLite
  branching / turns / usage / pop / clear.
- Fernet conformance (`src/memory/fernet.rs`): the official spec vector from
  <https://github.com/fernet/spec> (`verify.json`) is decrypted **and** reproduced byte for
  byte (the vector fixes the IV, so the whole token must match); all eight cases from the
  spec's `invalid.json` are rejected with the right reason (incorrect MAC, too short, invalid
  base64, payload size not a multiple of the block size, payload padding error, far-future
  timestamp, expired TTL, incorrect IV); plus the TTL boundary and no-TTL paths, a wrong
  version byte, a wrong key, and round trips at every length around the 16-byte block
  boundary.
- Python interop vectors (`src/memory/encrypted_session.rs`), generated with `cryptography` 48
  through the real Python `EncryptedSession` key-derivation code: the HKDF output is pinned for
  a raw-string master key and a url-safe base64 32-byte key, both envelopes decrypt to the
  exact items, the authenticated timestamp is recovered, and the same vector is shown to expire
  on the same boundary Python uses. A second set pins the derived key for the same 32 bytes
  spelled four ways — standard alphabet with `+` and `/`, url-safe, url-safe with a trailing
  newline, and unpadded — where the first three must agree and the unpadded one must take the
  raw-bytes fallback, exactly as Python does.
- 12 integration tests in `tests/sessions_tests.rs` (6 of them under default
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
  - reading a history written by the Python SDK end to end through
    `EncryptedSession::get_items`, then appending from Rust and checking that both envelopes
    stay readable and authenticated under the same key;
  - compaction retaining history when another writer appends while the compaction request is in
    flight (driven by a `Session` wrapper that injects an append on the snapshot read): the
    request is still billed, but the replacement is skipped and the injected item survives;
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
- The reverse direction was verified empirically rather than in the suite, because the test
  binary cannot call Python: a Rust-written envelope was decrypted by `cryptography` 48 through
  the Python derivation path, yielding `{'content': 'hello from rust', 'role': 'user'}` and the
  expected timestamp. The Redis key-type rejection is likewise covered by the env-gated
  integration test (it needs a real server to hold a wrong-typed key).
- Redis / MongoDB / sqlx integration tests are present but skip unless
  `REDIS_URL`, `MONGODB_URI` or `DATABASE_URL` is set; no services are
  available in this environment.
- Mock servers are raw `tokio::net::TcpListener` handlers, as in
  `tests/openai_http_tests.rs`. No real network or external services.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/sessions

cargo clippy --all-targets -- -D warnings                                               # clean
cargo clippy --features sqlite,encrypt,redis,mongodb,sqlx --all-targets -- -D warnings  # clean
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx --lib memory          # 49 passed
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx --test sessions_tests # 12 passed
cargo test   --test sessions_tests                                             # 6 passed (default)
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx                       # whole suite passed
cargo doc    --features sqlite,encrypt,redis,mongodb,sqlx --no-deps             # no warnings in my files
```

Both clippy gates pass as of this report. `cargo doc` emits one pre-existing warning for an
unresolved `[`mcp`]` link in `src/lib.rs:9`, which is expected without the `mcp` feature and is
not in a file I own.

`rustfmt --edition 2024` applied to every file touched.

## Requested core changes (outside my scope)

1. **Compaction suffix replacement needs two frozen contracts to change.** Python replaces a
   bounded, model-visible *suffix* atomically, which needs both halves of a mechanism this
   port cannot build from `src/memory/**`:
   - the runner must pass the latest model exchange (item digests plus the reasoning-id
     policy) into the session, which means new fields on `SessionRunCompletion`;
   - the session must ask the concrete store for an atomic snapshot/replace handle, which means
     either a new `Session` trait method or an `Any` supertrait so `Arc<dyn Session>` can be
     downcast.

   Until then, whole-history replacement plus the post-request staleness check described above
   is the faithful subset. The staleness check is what makes this safe rather than lossy.

2. **`chacha20poly1305` is now an unused dependency of the `encrypt` feature.** Phase 2 replaced
   the XChaCha20 envelope with Fernet, and nothing in `src/` references the crate any more
   (`grep -rn chacha src/` → no matches). It can be dropped from both the feature list and the
   `[dependencies]` block. An unused optional dependency produces no warning, so this is
   cleanliness, not a failure. `rand` is still needed — it supplies the 128-bit Fernet IV.

3. **No changes needed** to `src/run_internal/**` for the automatic-compaction trigger itself:
   the existing `Session::on_run_completed` hook and `SessionRunCompletion`
   (`response_id`, `store`, `has_local_tool_outputs`) carry exactly what the trigger needs, and
   the returned `Usage` is already folded into the run. Item 1 is about the *replacement*
   strategy, not the trigger.

## Phase 2 summary

- Added `src/memory/fernet.rs` (Fernet primitive, spec-vector verified in both directions).
- Rewrote `EncryptedSession`'s payload to Fernet; derivation, envelope, TTL and clock-skew
  behaviour now match Python byte for byte. XChaCha20 removed.
- Fixed master-key decoding to reproduce Python's lenient `urlsafe_b64decode`, which a strict
  decoder would have broken for the common `openssl rand -base64 32` key shape.
- Added Python interop vectors generated with `cryptography` 48, and empirically verified the
  reverse direction (Rust token → Python `Fernet.decrypt`).
- Redis: added key-type validation and server-clock timestamps; documented the one residual
  non-atomicity.
- Compaction: added the post-request staleness check so a concurrent append cannot be lost.
- Example simplified now that `Cargo.toml` carries
  `[[example]] sessions_encrypted / required-features = ["encrypt"]`.
