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
- **Compaction suffix replacement is now ported** (phase 3). Automatic compaction replaces only
  the tail of history that the latest model exchange produced and keeps older history, as
  Python does. `max_rollback_items` and Python's candidate cache remain unimplemented, and the
  reasoning-id/digest machinery is simplified (see below).

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

**Core additions (phase 3, `src/memory/mod.rs`)**
- `compaction_item_digest(&InputItem) -> String` — the canonical form used to match stored
  history against the last model exchange. Normalizes the cosmetic differences storage and the
  API introduce: drops `created_by`, expands an assistant message's plain-string content to the
  `output_text` list the API returns, and drops a `completed`/absent `status`. It is canonical
  JSON rather than a hash, so unlike Python's SHA-256 digest it cannot collide.
- `SessionRunCompletion::model_exchange: Vec<String>` — digests of everything the model was sent
  and produced, in order. **Backward compatibility:** the struct keeps `Default`, so existing
  construction with `..Default::default()` is unaffected; an exhaustive struct literal needs the
  new field, and an empty `model_exchange` falls back to whole-history replacement. `Eq` is
  preserved precisely because the field is digests rather than `InputItem`s.
- `SnapshotHandle(pub String)` — an opaque, store-specific id for one item in a snapshot.
- `CompactionSnapshot { items, handles, complete }` — a bounded window of the end of history.
- `Session::compaction_snapshot(limit) -> Result<Option<CompactionSnapshot>>` and
  `Session::replace_compaction_suffix(handles, items) -> Result<bool>`, both **defaulted**
  (`None` / `false`), so every existing `Session` implementation keeps compiling and simply
  falls back to coverage-guarded whole-history replacement.
- `Session::wants_model_exchange() -> bool`, defaulted to `false`. Digesting the exchange costs
  one serialization per item of the run's history, so the runner only does it for sessions that
  compact; `OpenAIResponsesCompactionSession` returns `true` and nothing else pays. This mirrors
  Python's gating on `is_openai_responses_compaction_aware_session`.

**Compaction (default features)**
- `OpenAIResponsesCompactionSession` — automatic compaction now follows Python's suffix
  algorithm: it asks the store for a window of `model_exchange.len() + 1` items, matches the
  stored tail against the exchange, and replaces only that tail. A partial suffix forces `input`
  mode (a response chain cannot scope its summary to a suffix), starts at a user message so a
  retained reasoning item keeps the model item that must follow it, is refused when it would
  split a tool-call or MCP-approval group, and is re-approved by the decision hook when the
  reloaded window differs from the one the hook first saw. Anything unmatched, unsafe or
  concurrently changed retains history instead of replacing it.
  `new(session_id, Arc<dyn Session>)`,
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

## Snapshot hook support

| Store | `compaction_snapshot` | Handle | Atomicity of the replacement |
| --- | --- | --- | --- |
| `InMemorySession` | yes | monotonic item id | the session mutex |
| `SqliteSession` | yes | `AUTOINCREMENT` row id | one transaction |
| `AdvancedSqliteSession` | yes (branch-scoped) | `message_structure` row id | one transaction |
| `SqlxSession` | yes | row id | one transaction |
| `RedisSession` | yes | the entry's stored bytes | one `EVAL` Lua script |
| `DaprSession` | yes | the entry's stored bytes | `first-write` ETag on the one key |
| `EncryptedSession` | forwards, decrypting | the wrapped store's | the wrapped store's |
| `MongoDBSession` | no, by design | — | — |
| `OpenAIConversationsSession` | no (server-managed) | — | — |

Python implements the hook only for SQLite, async SQLite, SQLAlchemy and Encrypted, so SQLite,
`SqlxSession` and `EncryptedSession` are the same set of backends — behaviourally equivalent for
the suffix mechanism itself, with the two documented simplifications below (the conservative
orphan check, and no `prune_prefix` reclamation of expired envelopes). Redis and Dapr go further than Python,
which has no suffix hook for them, because each can do the check-and-replace atomically in one
round trip — Redis inside a Lua script, Dapr under a first-write ETag. MongoDB deliberately
keeps the default: a suffix would span several batch documents, and multi-document atomicity
needs a transaction that standalone deployments do not support. That is safe because a store
without the hook is **not** simply handed the whole history — see the coverage guard below. `AdvancedSqliteSession` scopes both halves to the current branch and
only removes an underlying message when no other branch still references it.

Two invariants hold for every implementation: exactly one handle per item, and a replacement
that returns `false` without changing anything when the stored tail no longer matches the
handles. A row or envelope that cannot be decoded (corrupt JSON, an envelope this key cannot
authenticate) holds itself and everything before it out of the replaceable window and clears
`complete`, so compaction never summarizes away an item it could not read. An *authenticated but
expired* envelope is different: it is already invisible as history, so it is skipped without
shrinking the window, matching Python's `is_expired` branch. Treating it as unreadable would
force `input` mode on every TTL'd session as soon as anything aged out.

### Coverage guard for stores without the hook

When `compaction_snapshot` returns `None`, the automatic path does **not** replace the whole
history outright. It first re-reads a bounded window of `model_exchange.len() + 1` items and
refuses unless that window is complete *and* the exchange covers all of it, exactly as Python
does under its "backends without atomic suffix replacement still require complete coverage"
branch. Without this, a store whose retained history is longer than the run loaded — any
`SessionSettings::limit`, or a store that keeps more than it serves — would have its older items
summarized away by a response chain that never contained them. This is the data-loss case the
whole suffix mechanism exists to prevent, so it is guarded on every path, not only the ones with
a hook.

## Tests

- 52 unit tests in `src/memory/**` (`cargo test --features … --lib memory`):
  candidate selection, model-name validation, orphan-id stripping, compaction
  output normalization, auto-mode resolution, encryption round trip / key
  handling / scan budget, Redis and Dapr key layouts and decoding, Dapr ETag
  conflict classification and backoff, Mongo generation filter and document
  decoding including `Int32` counters from other drivers, Redis key-type validation
  (missing keys accepted, collisions rejected, first mismatch reported), sqlx per-dialect SQL
  generation, SQLite message classification and tool-name extraction, SQLite
  branching / turns / usage / pop / clear.
- Compaction matching (phase 3): the suffix matcher over a full match, a partial tail, an
  exchange padded with items storage does not hold, a disjoint exchange, an empty exchange and a
  reordered tail; digest normalization of assistant content, terminal vs. non-terminal `status`
  and `created_by`; and tool-group completeness for function calls and hosted MCP approvals in
  both directions (orphan call, orphan output).
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
- 22 integration tests in `tests/sessions_tests.rs` (12 of them under default
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
  - compaction replacing **only** the suffix the model exchange covers: the earlier turn is
    retained verbatim, the request switches to `input` mode carrying exactly the replaceable
    items, and the summary replaces just the tail;
  - compaction retaining history it cannot match (a disjoint exchange) and history it cannot
    safely split (a suffix with no user message to start from) — no request is sent in either
    case;
  - compaction refusing a suffix that would split a tool call away from its pending output;
  - an end-to-end `Runner` run against a session with `SessionSettings::with_limit(2)`, the case
    where the suffix path actually earns its keep: the run only loads the last two items, the
    compaction request provably excludes the older ones, and they survive in storage while the
    rest becomes the summary;
  - a store with no snapshot hook refusing compaction when the exchange covers only part of its
    history, and still compacting when it covers all of it (the data-loss guard above);
  - an authenticated-but-expired envelope being skipped without shrinking the window, while an
    unauthenticated one still clears it — asserted offline with the 2023 Python vector, so no
    sleep is needed;
  - `SqliteSession` snapshot/replace: a bounded window reports `complete: false`, a replacement
    keeps the retained prefix, and both a replayed handle set and an append after the snapshot
    are refused without changing anything;
  - `AdvancedSqliteSession` suffix replacement scoped to one branch, leaving `main` and the
    shared first turn intact;
  - `EncryptedSession` forwarding snapshots as plaintext and re-encrypting the replacement, and
    an unreadable envelope holding everything before it out of the window;
  - `DaprSession` snapshot/replace under an ETag, including refusal of stale handles;
  - `DaprSession` against a stateful mock sidecar: key names, the stored value
    shape, `created_at` preservation in metadata, `consistency=strong` on every
    read, `metadata.ttlInSeconds` and `options.consistency` on every write,
    `options.concurrency = first-write` on every conditional write, and a
    seeded 409 that the session retries successfully.
- Redis suffix replacement **was** exercised against a real server: a throwaway
  `redis:7-alpine` container on a spare port, with `REDIS_URL` set, ran the env-gated test
  covering the Lua script, the retained prefix, the bounded window's `complete: false`, refusal
  of a stale tail, and that a full replacement keeps the TTL on all three keys. The TTL
  assertion was confirmed to have teeth by disabling the script's `PEXPIRE` and watching it fail
  with `PTTL = -1`. The container was removed afterwards; the test stays skipped by default.
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
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx --lib memory          # 52 passed
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx --test sessions_tests # 22 passed
cargo test   --test sessions_tests                                             # 12 passed (default)
cargo test   --features sqlite,encrypt,redis,mongodb,sqlx                       # whole suite passed
cargo doc    --features sqlite,encrypt,redis,mongodb,sqlx --no-deps             # no warnings in my files
```

Both clippy gates pass as of this report. `cargo doc` emits one pre-existing warning for an
unresolved `[`mcp`]` link in `src/lib.rs:9`, which is expected without the `mcp` feature and is
not in a file I own.

`rustfmt --edition 2024` applied to every file touched.

## Core changes made (phase 3, authorized)

1. **`src/memory/mod.rs`** — added `compaction_item_digest`, `SnapshotHandle`,
   `CompactionSnapshot`, the `model_exchange` field on `SessionRunCompletion`, and the two
   defaulted `Session` methods. All additive: `Session` implementations outside this crate keep
   compiling, and `SessionRunCompletion` keeps `Default`, `Clone`, `Debug`, `PartialEq` and `Eq`.
   The one source-breaking edge is an exhaustive struct literal of `SessionRunCompletion`, which
   now needs the new field or `..Default::default()`; the in-tree call sites were updated to the
   latter.
2. **`src/run_internal/mod.rs`** — 26 added lines inside `notify_session_completed` only, which
   build `model_exchange` from `model_prefix` plus `items_to_input(&model_items)`. For a
   non-server-managed run the model input is cumulative, so that set is exactly "the last input
   plus the last response", matching Python's `prepare_compaction_model_input` /
   `record_compaction_model_response` pair without threading new state through the runner. Both
   spellings of a reasoning item are recorded (raw and with the id stripped per
   `ReasoningItemIdPolicy`), which is how Python's two-digest comparison is reproduced.
   The work is gated on `session.wants_model_exchange()`, so a non-compacting session is
   unaffected. No other function and no call site needed changing.

### Remaining simplifications

- `max_rollback_items` and Python's incremental candidate cache are still not implemented; the
  cache is a performance optimization and the budget is an opt-in guard neither of which changes
  observable behaviour by default.
- Python digests items with SHA-256 to bound memory; this port compares canonical JSON, which is
  exact rather than probabilistic but holds the whole exchange in memory for the duration of the
  call.
- Python's encrypted snapshot passes a `prune_prefix` predicate down so a contiguous prefix of
  authenticated expired envelopes is physically reclaimed during a successful replacement. This
  port skips expired envelopes when reading but does not reclaim their rows; they are removed by
  `clear_session` or by a whole-history replacement, exactly as before phase 3.
- Python's `drop_orphan_function_calls` also prunes reasoning items that precede a dropped call
  and reasons about program-owned items. The port checks the invariant that actually gates
  compaction — that call/output and MCP approval request/response sets match within the suffix —
  and refuses the suffix rather than pruning it, which is strictly more conservative.
- `chacha20poly1305` is still an unused dependency of the `encrypt` feature (phase 2 removed the
  XChaCha envelope); `grep -rn chacha src/` returns nothing. It can be dropped from `Cargo.toml`.
  `rand` is still needed for the Fernet IV.

## Phase 3 summary

- Added the snapshot/replace hook to `Session` (defaulted) plus `CompactionSnapshot`,
  `SnapshotHandle`, `compaction_item_digest` and `SessionRunCompletion::model_exchange`.
- Implemented the hook for in-memory, SQLite, advanced SQLite (branch-scoped), sqlx, Redis
  (atomic Lua), Dapr (ETag) and Encrypted (forwarding); MongoDB keeps the default by design.
- Ported Python's automatic suffix algorithm: exchange matching, the partial-suffix mode switch
  to `input`, the user-message boundary, tool-group and MCP-approval integrity, and re-approval
  of a reloaded window.
- Added the coverage guard so a store *without* the hook still refuses to replace history the
  exchange does not fully cover — the data-loss case the mechanism exists to prevent.
- Fixed the Redis script to re-apply the TTL after a full replacement (trimming a list to empty
  deletes the key, which would otherwise come back without an expiry), verified against a real
  Redis 7.
- Fixed encrypted snapshots to distinguish an expired envelope from an unauthenticated one.
- 26-line runner edit confined to `notify_session_completed`, gated on
  `Session::wants_model_exchange()` so non-compacting sessions pay nothing.

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
