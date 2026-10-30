# Sessions/memory, voice, codex and MCP — final parity report

Date: 2026-10-30. Python reference: `openai-agents` 0.24
(`src/agents/{memory,voice,mcp}/**`, `src/agents/extensions/{memory,experimental/codex}/**`).

Area owned: `src/memory/**`, `src/voice/**`, `src/extensions/codex/**`, `src/mcp/**`,
`tests/{session*,voice*,codex*,mcp*}.rs`, `tests/fixtures/{mock_mcp_server.py,fake_codex.sh}`,
`examples/{sessions_encrypted,voice_static,codex_tool}.rs`.

Work was split four ways. This file is the index and the MCP detail; the three sub-areas have
their own reports, linked per section and not repeated here:

- `plans/reports/sessions-memory-final-parity-report.md`
- `plans/reports/voice-final-parity-report.md`
- `plans/reports/codex-final-parity-report.md`

Every row below was re-verified against the current Rust source and the current Python source
before being changed; three rows turned out to be **stale report claims** and are called out as
such (sessions 4, 6, 15).

---

## 1. MCP (`src/mcp/**`, `tests/mcp_tests.rs`)

Audit rows 62, 67, 103, 104 plus the "Not done" list of
`mcp-memory-testing-parity-report.md`. New files: `src/mcp/http.rs`, `src/mcp/handler.rs`,
`src/mcp/stdio.rs`.

### Closed

| Row | What | Test |
| --- | --- | --- |
| `max_retry_attempts == -1` | `MCPServerConfig::max_retry_attempts` is now `i32`; a negative value retries forever, exactly as Python. `call_tool` reproduces Python's budget accounting verbatim: the budget counts *retries*, not backoffs, is not advanced while unlimited, and a failed isolated retry charges it on its own; backoffs are counted separately so the delay follows the schedule either way. | `mcp_tests::a_negative_retry_budget_retries_forever` (3 consecutive failures: `with_max_retry_attempts(1)` fails, `-1` succeeds) |
| Isolated-session retry | `_isolated_client_session` / `_call_tool_with_isolated_retry` / `_should_retry_in_isolated_session` ported as `Core::call_tool_once` + `Core::build_connection`. A transport-class failure (`ServiceError::{TransportSend, TransportClosed, Cancelled, Timeout}`, plus a client-side timeout — Python's `is_http_timeout_error`) replays the call on a connection opened only for it, which is then closed; the shared session is untouched either way. A server-side JSON-RPC error is **not** retried this way. **Divergence:** Python retries on an HTTP status ≥ 500 only; `rmcp` folds every non-2xx POST failure into `ServiceError::TransportSend`, so this port also replays a 4xx on an isolated session where Python would not. | `mcp_tests::a_transport_failure_is_retried_on_an_isolated_session` (asserts the server saw a *second* `initialize`) |
| `_credential_safe_exception_group` | New `mcp::redact_credentials`: rewrites every URL inside an error string, stripping userinfo, query and fragment, for any scheme (not just http/https — a transport error can name a `ws://` or proxy URL). Applied in `svc_err` and on every `src/mcp/sse.rs` error path (`GET`, endpoint parse, endpoint join, origin mismatch, stream error, `POST`). **Deliberate improvement over Python:** Python cannot rewrite the URLs inside an `httpx` exception, so it throws the whole exception away and substitutes a fixed string; a Rust transport error is already a string, so the same invariant (no credential reaches a log, span or error) is met while keeping the diagnostic text. | `mcp::logging::tests::redacts_urls_inside_error_text`, `mcp_tests::transport_errors_never_carry_credentials` (both transports, asserts neither the password nor the query token survives) |
| Streamable HTTP params | `timeout`, `sse_read_timeout`, `terminate_on_close`, `ignore_initialized_notification_failure`, `auth`, `http_client` (`httpx_client_factory`) added to `MCPServerStreamableHttpParams`. Implemented with a new `ObservedHttpClient` (`src/mcp/http.rs`) that implements `rmcp::transport::streamable_http_client::StreamableHttpClient` over the caller's (or a configured) `reqwest::Client`: it suppresses the session-ending `DELETE` when `terminate_on_close` is false, and turns a failed `notifications/initialized` POST into `Accepted` when `ignore_initialized_notification_failure` is set. | `mcp_tests::streamable_http_records_the_session_id_and_sends_the_auth_header`, `mcp_tests::streamable_http_can_keep_the_session_and_tolerate_a_rejected_initialized` |
| `session_id()` on Streamable HTTP | The same wrapper records the `Mcp-Session-Id` the server returns on each POST response, so `MCPServer::session_id()` is no longer always `None`. `cleanup` clears it. (The previous report called this impossible because `rmcp` moves the transport into the service — true of `StreamableHttpClientTransport`, but the *client* it is built from is ours.) | same test (`Some("sess-abc")`, then `None` after cleanup) |
| SSE `auth` / `httpx_client_factory` | `MCPServerSseParams::{auth, http_client}`. `MCPAuthProvider` is a callback consulted when the transport is built (so reconnecting refreshes a short-lived credential), returning the bearer token; a value already prefixed with `Bearer ` is accepted and not doubled. | `mcp_tests::streamable_http_records_the_session_id_and_sends_the_auth_header` asserts the exact header the server received |
| `message_handler` | `MCPServerConfig::message_handler` + `MCPServerXxx::with_message_handler`. `rmcp` dispatches each server notification to its own hook instead of one handler, so new `src/mcp/handler.rs::MCPClientHandler` implements all ten `ClientHandler` notification hooks and re-assembles the JSON-RPC shape `{"jsonrpc","method","params"}` for a single callback. | `mcp_tests::server_notifications_reach_the_message_handler` (the mock server emits `notifications/resources/updated` mid-tool-call) |
| Per-server `failure_error_function`, `tool_input_guardrails`, `tool_output_guardrails`, `custom_data_extractor` | New `MCPToolPolicy<C>`, stored type-erased in `MCPServerConfig::tool_policy` and recovered by downcast in `to_function_tool` for the run's concrete `C`. `MCPServer` is not generic over `C` and making it so would change every `Arc<dyn MCPServer>` in the crate; Python types all four fields `Any` for the same reason. Python's `_UNSET` sentinel (`_get_failure_error_function`) is reproduced by `failure_error_function_set`, so a server that did not set one still inherits the agent's `MCPConfig::failure_error_function`. | `mcp_tests::a_server_tool_policy_overrides_the_agent_failure_formatter` (the mock server gained an always-failing `fail` tool) |
| Async dynamic tool filters and approvals | `ToolFilter::Dynamic` and `RequireApproval::Dynamic` now return `BoxFuture<'static, bool>`; `ToolFilter::{dynamic, dynamic_async}` and `RequireApproval::{dynamic, dynamic_async}` wrap sync or async closures. Both evaluation sites are in `src/mcp/util.rs`, so no file outside the area changed. | `mcp_tests::async_filters_and_approvals_are_awaited` |
| `ToolFilterContext.run_context` / `.agent` | Both added, type-erased (`Arc<dyn Any + Send + Sync>`) with typed accessors `run_context::<C>()` / `agent::<C>()` — the Rust equivalent of Python's `Any`-typed fields, for the same object-safety reason as above. `run_context` is populated today; `agent` needs the caller (see "Requested elsewhere"), so a new entry point `get_all_function_tools_for_agent` takes it without breaking the existing signature. | same test (the filter asserts `ctx.run_context::<()>().is_some()`) |
| stdio `encoding` / `encoding_error_handler` | `MCPServerStdioParams::{encoding, encoding_error_handler}` + `EncodingErrorHandler{Strict,Ignore,Replace}`. A non-UTF-8 `encoding` is rejected at connect with an explicit message. `Ignore`/`Replace` spawn the child here and wrap its stdout in a new `src/mcp/stdio.rs::Utf8Sanitizer` (an `AsyncRead` that drops or replaces invalid sequences and carries a split multi-byte character across reads) before the JSON-RPC codec sees it; `Strict` keeps the existing `TokioChildProcess` path untouched. | `mcp_tests::a_stdio_server_rejects_a_non_utf8_encoding`, `mcp::stdio::tests::{recognizes_utf8_aliases, drops_or_replaces_invalid_sequences, carries_a_split_multibyte_character_across_reads, flushes_an_incomplete_tail_at_eof}` |
| `collect_mcp_list_tools_metadata` (audit 104) | Ported as `mcp::collect_mcp_list_tools_metadata(&[Value])` + `extract_mcp_tool_metadata_value`, accepting both a raw `mcp_list_tools` payload and a run item whose `raw_item` holds one, keyed `(server_label, tool name)`. | `mcp::logging::tests::indexes_hosted_mcp_list_tools_metadata` |

### Flaky legacy-SSE test — root cause and fix

`tests/mcp_tests.rs::agent_calls_tools_over_legacy_sse` failed intermittently under parallel
load with a transport send error. **Cause:** the in-test mock answered the JSON-RPC `POST` with
`202 Accepted` and no `Connection: close`, then returned and dropped the socket. HTTP/1.1
default is keep-alive, so hyper legally pooled that connection; the next POST could be written
onto it in the window between the server's FIN and hyper observing EOF, and a POST already on
the wire is not retried. Parallel load widened that window.

Fixed at the cause, not with a sleep: the mock is now a real HTTP/1.1 keep-alive server —
one task per accepted connection reading requests in a loop until EOF, with the consumed
request drained from the buffer each round. Two latent bugs in the same fixture were fixed
alongside: `read` never checked for `n == 0`, so a client close before the headers completed
spun the task at 100% CPU, and `unwrap()`/`assert!` inside the spawned tasks died silently
instead of surfacing as a client error.

Verified: 40 isolated runs of the single test and 48 runs of the whole `mcp_tests` binary
(8 rounds × 6 concurrent processes, `--test-threads 16`) — **0 failures**.

### `src/voice/result.rs` redaction constant

The private `const REDACTED_TRACE_ERROR_MESSAGE` is gone. `voice::result::get_trace_error` now
delegates to `crate::tracing::model_error_text`, which is built on the crate-wide
`crate::tracing::REDACTED_TRACE_ERROR_MESSAGE`, so voice spans carry byte-identical text to
model and tool spans and the error is only stringified when it will actually be exported.
`src/voice/models/openai_stt.rs` (the other caller) is unchanged and goes through the same
function.

### MCP rows left open

Each needs a file this area does not own, or is inherent.

- **`mcp_title` on the derived function tool** (`mcp/util.py:598`). Needs a `mcp_title` field on
  `FunctionTool` in `src/tool.rs`. The value itself is already computed
  (`mcp::resolve_mcp_tool_title`).
- **`reserved_tool_names` plumbing.** `mcp::util::get_all_function_tools_with_reserved` exists
  and is tested; the caller that would pass the agent's own tool names is
  `src/run_internal/tools.rs`.
- **`ToolFilterContext.agent` population.** The field and the entry point
  (`get_all_function_tools_for_agent`) exist; the caller in `src/run_internal/tools.rs` must
  pass the `Arc<Agent<C>>` it already holds.
- **`collect_mcp_list_tools_metadata` consumer.** Implemented, but `src/mcp/**` is behind the
  `mcp` feature while its Python consumer (`run_internal/turn_resolution.py:3204`, hosted MCP)
  is not feature-gated. Using it from `src/run_internal/mod.rs` needs the three metadata
  helpers moved to an ungated module.
- **`MCPToolCancellationError`.** No `AgentsError` variant (`src/error.rs`), and inherent:
  Python needs it because `asyncio.wait` can observe a cancelled task, whereas a dropped Rust
  future simply stops and produces no "the call was cancelled" result to translate.
- **Non-UTF-8 stdio `encoding`.** Inherent. The MCP stdio transport mandates UTF-8 and a Rust
  `str` is UTF-8; the port rejects any other value explicitly rather than silently transcoding.
  `encoding_error_handler` *is* ported (above).
- **`_ServerWorker`** (one task per server so connect and cleanup share a task). Deliberate:
  anyio requires it, tokio and `rmcp` do not. Documented on `MCPServerManager`.
- **`MCPServerManager`'s async-context-manager guard** (`__aenter__`/`__aexit__`). Inherent:
  Rust has no async `Drop`, so cleanup cannot run from a guard's destructor; callers call
  `cleanup_all`.

### MCP breaking changes

1. `MCPServerConfig::max_retry_attempts` is `i32` (was `u32`); `with_max_retry_attempts` takes
   `i32`. Negative means unlimited.
2. `MCPServerConfig` gained `message_handler` and `tool_policy`; it is constructed via
   `Default` + builders in-tree, but an exhaustive struct literal breaks.
3. `MCPServerStdioParams`, `MCPServerSseParams` and `MCPServerStreamableHttpParams` gained
   public fields. Struct literals need `..Default::default()` (every in-tree one already had
   it). `MCPServerStdioParams` now has a hand-written `Default` (`encoding = "utf-8"`), and the
   two HTTP param structs have hand-written `Debug` impls that print the URL through
   `get_mcp_server_log_name` so a `Debug` dump cannot leak credentials.
4. `ToolFilter::Dynamic` and `RequireApproval::Dynamic` carry async callbacks now; use the new
   `dynamic` / `dynamic_async` constructors.
5. `RequireApproval::requires` is `async`.
6. `ToolFilterContext` gained two fields and has a hand-written `Debug`; build it with
   `ToolFilterContext::new`.
7. `mcp::util::to_function_tool` and `to_function_tool_with_failure` are `async` (they await the
   approval policy).

New public API (additive): `mcp::redact_credentials`, `mcp::collect_mcp_list_tools_metadata`,
`mcp::extract_mcp_tool_metadata_value`, `mcp::MCPToolPolicy`, `mcp::MCPMessageHandler`,
`mcp::MCPAuthProvider`, `mcp::EncodingErrorHandler`,
`MCPServerConfig::{with_tool_policy, tool_policy}`,
`MCPServerXxx::{with_message_handler, with_tool_policy}`,
`mcp::util::get_all_function_tools_for_agent`, `ToolFilterContext::{new, run_context, agent}`,
`ToolFilter::{dynamic, dynamic_async}`, `RequireApproval::{dynamic, dynamic_async}`.

`tests/fixtures/mock_mcp_server.py` gained a `fail` tool (always returns a JSON-RPC error) and a
`notify` tool (emits a server notification before replying). The two tool-count assertions in
`tests/mcp_tests.rs` were updated (4 → 6 listed, 3 → 5 after filtering).

---

## 2. Sessions / memory — see `sessions-memory-final-parity-report.md`

18 rows settled (3 of them verification-only), each code change with a test. Headlines:

- `EncryptedSession::pop_item` now **leaves** an unauthenticatable envelope in place (new
  defaulted `Session::pop_item_validated`, validated inside the SQLite transaction that owns the
  deletion).
- `prune_prefix` reclamation ported (new defaulted
  `Session::replace_compaction_suffix_pruning`, implemented for SQLite and sqlx — the same two
  backends Python supports).
- `max_rollback_items` ported with Python's message and both enforcement points.
- Redis key-type validation is now **atomic** with the write: a shared Lua `guard()` prelude
  prepended to the add/clear/replace scripts, so the separate `TYPE`/`TIME` round trip is gone.
- `SqlxSession` on SQLite enabled — the previous "linking sqlite3 twice is unsound" objection
  was wrong; `rusqlite 0.39` and `sqlx-sqlite 0.9` unify on one `libsqlite3-sys 0.37`.
- `AdvancedSqliteSession`: table-ownership claims (via `rusqlite`'s SQLite authorizer),
  cross-process `branch_reservations`, `session_clear_generations`, `get_items_for_branch`,
  `create_tables` constructor parameter.
- `MongoDBSession::pop_item`: both generation re-check loops ported — the atomic `$slice` claim
  alone was **not** equivalent.
- `memory::await_session_mutation` added and used by the wrappers in this area.
- `OpenAIResponsesCompactionSession::new` is now fallible and rejects a server-managed store, as
  Python's `__init__` does (`try_new` removed).

**Three stale report claims corrected:** Python does *not* prune orphan function calls at the
compaction call site (it refuses the suffix, as this port already did — the real delta was that
`program`/`program_output` items were missing from the group definition, now fixed); the
`chacha20poly1305` "unused dependency" does not exist in `Cargo.toml`; the SQLite linking
objection above.

Left open: Python's incremental candidate cache (proved from the source to be pure memoization
for wrapper-only traffic, and *stale* where it differs), the `inspect.signature`-based context
opt-in (inherent — no runtime signature reflection, and `Session` must stay object-safe), and
the runner-side half of `await_session_mutation` (call sites in `src/run_internal/**`).

---

## 3. Voice — see `voice-final-parity-report.md`

Row 123 (`voice/testing.py`) closed in full, plus six secondary rows:

- `VoiceScriptError` / `UnexpectedVoiceCall { operation }` / `UnconsumedVoiceSteps
  { remaining_steps }`, shaped like `testing::ModelScriptError`, with Python's verbatim
  messages.
- `assert_complete()` on the session, STT model, TTS model and workflow, including the
  workflow's three distinct messages.
- `with_close_error`, close idempotence, and "a closed session stops yielding".
- The `_START_NOT_CONFIGURED` semantic (configured tracked separately from pending).
- `pcm16_samples` and `Scripted*` aliases so the Python names resolve; `Fake*` names kept.
- `VoicePipelineConfig::tracing_api_key` — verified from the Python that `TracingConfig`'s other
  field never reaches the voice pipeline and that there is no precedence rule against the flat
  flags.
- `OpenAIVoiceModelProvider::with_agent_harness_id` — the audit's correction confirmed; the
  behaviour was already in `models::resolve_openai_harness_id`.
- `VoiceWebSocketOptions { ping_interval, ping_timeout, max_size }` on the STT websocket, with
  the `websockets` defaults Python inherits (20 s / 20 s / 1 MiB), and client `default_query`
  merged into the handshake URL.
- A panicking producer now reports an error instead of a silent `session_ended`.

Left open: consumer-side cancellation bookkeeping (inherent — a cancelled Rust consumer is a
drop; there is no cancellation exception to observe and `Drop` cannot be async; every other exit
path now delivers a terminal event and is tested), NumPy dtypes, `coerce_dataclass_config`, the
legacy positional-arg shim, and `websocket_base_url` / `refresh_openai_client_api_key_if_supported`
(need `src/models/openai_client.rs`).

---

## 4. Codex — see `codex-final-parity-report.md`

9 of 11 rows closed plus audit row 105:

- Result string is now byte-identical to CPython `json.dumps` (new `py_json.rs`: `", "`/`": "`
  separators, `ensure_ascii` with surrogate pairs).
- `_enforce_span_data_budget` ported verbatim (new `span_budget.rs`): `span_data_max_chars` is a
  **whole-payload** budget now, not a per-value limit. Every test expectation is the Python
  functions' own output on the same input.
- `TurnOptions.signal` → `CancellationToken`, selected against the line stream, killing the
  child; the idle timeout fires it too.
- `codex_subprocess_stream_limit_bytes` + env var, with a bounded reader mirroring asyncio's two
  `LimitOverrunError` branches and texts.
- `on_stream` backpressure via an unbounded channel + consumer task, matching
  `run_producer_consumer`'s drain-then-propagate.
- Custom `params_json_schema`: `additionalProperties: false` and `required` are now honoured.
- Path resolution confirmed against Python's order, plus the `shutil.which` executable-bit check.
- **Audit row 105 closed**: `ToolContext::run_config` is now wired by `run_internal`, so the
  Codex spans honour the run's `tracing_disabled` / `trace_include_sensitive_data` exactly as
  Python does. (The codex port report called this impossible; it was re-verified and is no
  longer true.)

Left open: `use_run_context_thread_id` (`RunContext<C>` still has no run-scoped key/value slot —
`set_custom_data` is per-call; the `with_thread_id_store` closures remain), `_is_codex_tool`
(no marker slot on `FunctionTool`), and `_DictLike` / the kwarg overload (inherent Python sugar).

---

## Breaking changes — all four sub-areas

MCP (detail and rationale in §1):

- `MCPServerConfig::max_retry_attempts` is `i32`; `with_max_retry_attempts(i32)`.
- `MCPServerConfig` gained `message_handler` + `tool_policy`.
- `MCPServerStdioParams` / `MCPServerSseParams` / `MCPServerStreamableHttpParams` gained public
  fields; the two HTTP ones swapped `derive(Debug)` for URL-redacting manual impls, and
  `MCPServerStdioParams` swapped `derive(Default)` for a manual one (`encoding = "utf-8"`).
- `ToolFilter::Dynamic` / `RequireApproval::Dynamic` carry async callbacks;
  `RequireApproval::requires` is `async`.
- `ToolFilterContext` gained two fields + a manual `Debug`; build it with `::new`.
- `mcp::util::to_function_tool` / `to_function_tool_with_failure` are `async`.

Sessions / memory:

- `OpenAIResponsesCompactionSession::new` returns `Result<Self>` and rejects a server-managed
  store; **`try_new` removed**.
- `SqlxDialect` gained a `Sqlite` variant (exhaustive external `match` breaks), and
  `SqlxDialect::from_url("sqlite://…")` now succeeds instead of erroring.
- `Session` gained two defaulted methods (`pop_item_validated`,
  `replace_compaction_suffix_pruning`).
- **Behavioural:** `SqliteSession::pop_item` now skips a row whose JSON no longer parses and
  keeps looking, instead of returning `None` (Python's behaviour).

Voice:

- **Behavioural, live traffic:** the STT transcription websocket now caps an incoming message at
  1 MiB (was tungstenite's 64 MiB) and sends a keepalive ping every 20 s (was none) — the
  `websockets` defaults Python inherits. Both are tunable via `VoiceWebSocketOptions`;
  `with_max_size(None)` / `with_ping_interval(None)` restore the old behaviour.
- `VoicePipelineConfig` gained a public field (`tracing_api_key`).
- **Behavioural, test fakes:** `FakeTranscriptionSession::next_turn` returns `None` after
  `close()` even with turns left; `FakeVoiceWorkflow::with_start` marks the start *configured*,
  so a second `on_start` now fails; scripted error messages gained Python's trailing `.`.
- `examples/voice_static.rs` no longer compiles without the `voice` feature.

Codex:

- **Behavioural:** `span_data_max_chars` is now a whole-payload budget, not a per-value limit.
- **Behavioural:** `CodexToolResult::to_string()` bytes changed (CPython `json.dumps` spacing +
  `ensure_ascii`); the parsed value is unchanged.
- **Behavioural:** a custom `params_json_schema` with `additionalProperties: false` / `required`
  is now enforced instead of ignored; signal-killed exits report `-signal`.
- `TurnOptions` lost `PartialEq`; `TurnOptions` / `CodexExecArgs` / `CodexOptions` gained public
  fields.

## Shared-file edits

`src/lib.rs`: **not touched** by any of the four sub-areas.

`Cargo.toml`: two additive feature-list edits, both from the sessions sub-area, no version
changes and no new dependency:

```toml
rusqlite = { version = "0.39", features = ["bundled", "hooks"], optional = true }
sqlx = { version = "0.9", default-features = false, features = [
  "runtime-tokio", "tls-rustls-aws-lc-rs", "postgres", "mysql", "sqlite", "any", "json", "chrono",
], optional = true }
```

`rusqlite/hooks` enables `Connection::authorizer` (table-ownership claims); `sqlx/sqlite`
enables `SqlxDialect::Sqlite` and unifies on the `libsqlite3-sys 0.37` `rusqlite` already pulls
(`cargo tree -i libsqlite3-sys --all-features`).

## Requested changes in files this area does not own

| File | Change | Why |
| --- | --- | --- |
| `src/tool.rs` | `FunctionTool::mcp_title: Option<String>` | `mcp/util.py:598` sets it; the value is already computed |
| `src/tool.rs` | `pub(crate) is_codex_tool: bool` on `FunctionTool` | Python's `_is_codex_tool` marker |
| `src/agent.rs` | port `_validate_codex_tool_name_collisions` (exact text in the codex report) | duplicate-Codex-tool fail-fast |
| `src/run_internal/tools.rs` | pass the agent's own tool names to `mcp::util::get_all_function_tools_with_reserved`, and the `Arc<Agent<C>>` to `get_all_function_tools_for_agent` | closes `reserved_tool_names` and `ToolFilterContext.agent` |
| `src/run_internal/**` | wrap the session-write awaits in `memory::await_session_mutation` | a cancelled run can still drop a session write in flight |
| `src/models/openai_client.rs` | `with_websocket_base_url` / `websocket_base_url()`; `pub(crate) authorize(...)`; `post_multipart` + raw byte-stream POST | voice STT/TTS re-implement `url()` and the error mapping, and the websocket handshake never runs the `RequestAuth` hook |
| `src/error.rs` | a variant for `STTWebsocketConnectionError`, and one for `MCPToolCancellationError` if that path is ever wanted | callers currently match on the message |
| `src/tracing/mod.rs` | `parent: Option<&Span>` on `speech_span` / `transcription_span` / `speech_group_span` | `src/voice/result.rs` calls `create_span` directly to parent from a spawned task |
| ungated module | move `MCPToolMetadata` + `extract_mcp_tool_metadata_value` + `collect_mcp_list_tools_metadata` out from behind `feature = "mcp"` | hosted MCP in `src/run_internal/mod.rs` is not feature-gated |
| `src/lib.rs` (optional) | re-export the voice entry points under `#[cfg(feature = "voice")]` | consistency with `models::*` |

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/misc2 \
       CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0

cargo clippy --all-features --all-targets          # see below: 4 findings, none in this area
cargo test --all-features --test mcp_tests        # 13 passed
cargo test --all-features --test sessions_tests   # 37 passed
cargo test --all-features --test voice_tests      # 32 passed
cargo test --all-features --test codex_tests      # 18 passed
cargo test --all-features --lib mcp               # 15 passed
cargo test --all-features --lib memory            # 76 passed
cargo test --all-features --lib voice             # 28 passed
cargo test --all-features --lib extensions::codex # 63 passed
cargo test --all-features --doc                   # 46 passed (incl. the 2 new mcp doctests)

cargo test --all-features --no-fail-fast          # 32 targets, 0 failed, 1084 passed
```

`cargo clippy --all-features --all-targets` reports **zero** findings in this area's files, so
the `-D warnings` gate is clean for everything owned here. It does **not** pass crate-wide: four
findings remain, all in other areas' files, and were **not** touched (per the "do not fix
another area's findings" rule):

- `src/run_state/wire.rs:795`, `:798` — `nonminimal_bool`
- `src/sandbox/events.rs:506` — `manual_is_multiple_of`
- `src/sandbox/events.rs:511` — `match_like_matches_macro`

All tests use dummy `sk-test` keys and local `tokio::net::TcpListener` mock servers (a new
Streamable HTTP MCP mock, the fixed keep-alive SSE mock, the stdlib-only Python stdio MCP
server, the `fake_codex.sh` fixture, and the voice mock HTTP/websocket servers) plus
`InMemoryTraceProcessor`. No network, no real `codex` binary, no paid API. Redis / MongoDB /
sqlx-Postgres / sqlx-MySQL integration tests remain env-gated; `sqlx` on SQLite is now covered
by an ungated temp-file test.

`rustfmt --edition 2024` applied to every file this area owns. `src/lib.rs` was never formatted.

## Concurrency note

Four sibling areas edited the crate in parallel throughout. Several verification runs were
blocked by transient compile errors in `src/run_state/wire.rs`,
`src/run_internal/tool_invocation.rs`, `src/models/openai_chatcompletions.rs`,
`src/extensions/models/{mod,anthropic}.rs` and `src/sandbox/**`; each was re-run after the
owning agent settled. The figures above are from runs where the whole crate compiled.

---

Status: DONE_WITH_CONCERNS
Summary: Every deferred/partial row in the sessions, voice, codex and MCP reports is closed with
a focused test, or left open with a concrete inherent reason or a named change in a file this
area does not own. MCP gained isolated-session retry, unlimited retries, credential-safe
transport errors, full Streamable-HTTP/SSE params plus `session_id`, a message handler, a
per-server tool policy, async filters/approvals and stdio decoding policy; the flaky legacy-SSE
test was fixed at its cause (keep-alive mock) and stress-verified 88×.
Concerns/Blockers:
- Crate-wide `cargo clippy --all-features --all-targets -- -D warnings` does **not** pass: 4
  findings remain in other areas' files (`src/run_state/wire.rs:795,798` `nonminimal_bool`;
  `src/sandbox/events.rs:506,511` `manual_is_multiple_of`, `match_like_matches_macro`). Zero in
  this area; not fixed per the "do not fix another area's findings" rule.
- 8 rows are complete on this side but need a caller or a field in a file owned elsewhere — see
  "Requested changes in files this area does not own". The highest-value ones are
  `FunctionTool::mcp_title`, the `reserved_tool_names` / `ToolFilterContext::agent` call in
  `src/run_internal/tools.rs`, wrapping the runner's session writes in
  `memory::await_session_mutation`, and `FunctionTool::is_codex_tool` + the duplicate-Codex-tool
  fail-fast.
- Two `Cargo.toml` feature-list lines changed (`rusqlite/hooks`, `sqlx/sqlite`); no new
  dependency, no version change.
- `collect_mcp_list_tools_metadata` is implemented but sits behind `feature = "mcp"`, while its
  hosted-MCP consumer in `src/run_internal/mod.rs` is not feature-gated.
