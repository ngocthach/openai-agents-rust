# MCP / memory / testing-helper parity report

Date: 2026-10-30
Scope: `src/mcp/**`, `src/memory/**`, `src/testing.rs`, `src/voice/testing.rs`,
`tests/{mcp,sessions,voice,testing_model}_tests.rs`, `tests/fixtures/mock_mcp_server.py`.

Python reference: `/Users/ngocthach/myGit/openai-agents-python/src/agents` (0.24).
Audit rows addressed: 46, 62, 67, 73, 103, 104, 123, 125, 126, 127.

---

## What was implemented

### MCP

**New `src/mcp/logging.rs`** (`agents.mcp._logging` + `agents._mcp_tool_metadata`)

- `get_mcp_server_log_name` — hand-written URL splitter that mirrors Python's `urlsplit`
  semantics rather than using the `url` crate (which normalizes away `:443`, adds a trailing
  `/` and resolves `..`). Strips userinfo, query and fragment; keeps the port verbatim;
  brackets IPv6 hosts; honors the `sse: ` / `streamable_http: ` / `streamable-http: ` prefixes;
  returns `<prefix><invalid-url>` for a rejected URL and the name unchanged for a non-HTTP name.
- `get_mcp_server_log_message` — leaves the server name out entirely in redacted mode.
- `MCPToolMetadata`, `resolve_mcp_tool_title`, `resolve_mcp_tool_description`,
  `resolve_mcp_tool_description_for_model`, `extract_mcp_tool_metadata`.

**`src/mcp/mod.rs`**

- Resources API as defaulted trait methods: `list_resources(cursor)`,
  `list_resource_templates(cursor)`, `read_resource(uri)`. The default returns a
  "does not support …" `UserError`, matching Python's `NotImplementedError`; the three
  built-in transports override them through `rmcp`. Results are raw `Value`, so `nextCursor`
  is available to the caller, as the existing prompts API already does.
- `cached_tools()`, `server_initialize_result()` (from `rmcp`'s `peer_info`) and a defaulted
  `session_id()`.
- `call_tool_with_meta(name, args, meta)` — defaulted, delegates to `call_tool` when `meta` is
  `None` and errors otherwise; overridden by the built-in transports, which send `_meta` on
  the wire.
- `MCPToolMetaContext` + `MCPToolMetaResolver` and `MCPServerConfig::tool_meta_resolver`
  (builder: `with_tool_meta_resolver`). `MCPServerConfig` now has a manual `Debug`.
- `retry_backoff_seconds_max` (caps the exponential backoff) and `max_list_pages`
  (bounds automatic `list_tools`/`list_prompts` pagination). Tools and prompts are now paged
  explicitly, with a repeated-cursor guard; exceeding the page budget fails instead of
  returning a partial list.
- All server-facing log lines and error messages now use the sanitized log name.
- Retry logging goes through `crate::logger::log_tool_action_warning`, so an error that may
  carry tool data is suppressed in redacted mode.

**`MCPServerManager`**

- `with_connect_in_parallel(bool)` (default `false`, as in Python) — `join_all` over the
  servers, active/failed lists assigned in input order so runs stay reproducible; in strict
  mode the first failure in input order is returned.
- `with_cleanup_timeout(Option<Duration>)` (default 10 s) — cleanup failures and timeouts are
  recorded instead of silently dropped.
- `all_servers()` and `errors()` (connect + cleanup failures, by server name). Cleanup errors
  are tracked separately from `failed_servers()`, so `reconnect(failed_only = true)` does not
  try to reconnect a server that merely failed to close.
- Python's `_ServerWorker` was deliberately **not** ported: it exists because anyio requires a
  transport's connect and cleanup to run in the same task, which neither tokio nor `rmcp`
  requires. This is documented on the struct.

**`src/mcp/util.rs`**

- `invoke_mcp_tool(server, tool, input_json, meta, tool_display_name)` is now a public
  function, no longer inlined in the tool closure. It resolves the server's `tool_meta_resolver`,
  merges it with the explicit `meta` (`merge_mcp_meta`, explicit keys win), honors
  `DONT_LOG_TOOL_DATA` in the invalid-JSON message, and logs failures through
  `crate::logger`.
- `merge_mcp_meta` is public. A `UserError` or `ModelBehaviorError` from the server call
  propagates with its own kind; only other failures are wrapped in the generic
  "Error invoking MCP tool …", matching Python, which re-raises `UserError`/`MCPError` as-is.
- **Blank-description bug fixed**: `to_function_tool` now uses
  `resolve_mcp_tool_description_for_model` (description → `title` → `annotations.title` → `""`),
  so a title-only MCP tool no longer reaches the model as a blank function.
- `f.mcp_server_name` is now the sanitized log name (it only feeds span `mcp_data.server`;
  nothing looks servers up by it — verified by grep).
- **Hash disambiguation** replaces the hard `Duplicate tool names` error when
  `include_server_in_tool_names` is on: `shorten_tool_name` /
  `build_prefixed_tool_name_overrides` allocate bounded (≤ 64 char), collision-free names
  deterministically, honoring a `reserved` set. New public
  `get_all_function_tools_with_reserved(..., reserved)`; `get_all_function_tools` delegates
  with an empty set.

**`tests/fixtures/mock_mcp_server.py`** gained `resources/list`, `resources/templates/list`,
`resources/read`, a title-only tool (`titled_only`) and an `echo_meta` tool that echoes the
`_meta` it received, which is the only way to prove `_meta` reaches the server.

### Memory

**`src/memory/mod.rs`**

- `resolve_session_limit(explicit, settings)` (public), mirroring
  `agents.memory.session_settings.resolve_session_limit`.
- Public `SessionInputCallback` alias (same shape as the previously private alias in
  `run_config.rs`), so downstream users can name the type.
- Defaulted `Session::manages_history_remotely()`.

**Every backend now resolves its own limit inside `get_items`** — `InMemorySession`,
`SqliteSession`, `AdvancedSqliteSession`, `SqlxSession`, `RedisSession`, `DaprSession`,
`MongoDBSession`, `EncryptedSession`, `OpenAIConversationsSession` — matching Python, where
each backend calls `resolve_session_limit(limit, self.session_settings)`.

**`OpenAIConversationsSession` (rewritten)**

- 20-item request batching (`MAX_ITEMS_PER_REQUEST`), so a turn producing more than 20 items no
  longer fails at the API.
- A mutation lock serializes get/add/pop/clear through one instance.
- `clear_session` now resets the cached conversation id, so a later call creates a fresh
  conversation instead of targeting a deleted one. The id lives in a `tokio::Mutex<Option<String>>`;
  `session_id() -> &str` reads a leaked `&'static str` view that is republished whenever the id
  changes (one small leak per conversation created — documented on the field).
- `session_settings` support, including `limit == 0 → []`.

**`OpenAIResponsesCompactionSession`**

- `try_new` rejects wrapping a store that manages its history remotely; `new` keeps its
  infallible signature but every `Session` operation on such a wrapper now fails with the same
  `UserError` (see "API decisions" below).
- `normalize_content_part` is now priority-exclusive and fallible, exactly like Python's
  `_normalize_compaction_input_image` / `_normalize_compaction_input_file`:
  `image_url` **else** `file_id`; `file_data` **else** `file_url` **else** `file_id`; extras
  (`filename`, `detail`) kept; all-missing is an error instead of a bare
  `{"type": "input_image"}`.

**`AdvancedSqliteSession`**

- `ConversationTurn` gained `full_content` and `can_branch`; `content` is now the 100-character
  preview Python produces. `find_turns_by_content` searches `full_content`.
- `get_turn_usage` selects and returns `input_tokens_details` / `output_tokens_details`
  (both columns were already written but never read back).
- New `get_turn_structure(branch_id) -> BTreeMap<i64, Vec<TurnMessage>>` — the structure view
  Python's `get_conversation_by_turns` returns.

### Testing helpers

**`src/testing.rs`** (`FakeModel` left untouched; `ScriptedModel` added beside it)

- `ModelScriptError` { `InvalidModelStep` (with `ModelStepReason`), `UnexpectedModelCall`,
  `UnconsumedModelSteps` }, `impl std::error::Error`, `From<ModelScriptError> for AgentsError`.
- `ModelCall { request, streamed }`, `ScriptedStep` (output / usage / response_id /
  request_id / raw_usage / error / responder / stream_events / retry_advice) with
  `output`, `raise_error`, `respond`, `stream` constructors and
  `DEFAULT_SCRIPTED_RESPONSE_ID = "resp-789"` (Python's default; `FakeModel` keeps `resp_fake_{n}`).
- `ScriptedModel` with strict step matching: `enqueue`, `extend`, `with_steps`,
  `set_default_usage`, `calls`, `first_call`, `last_call`, `remaining_steps`,
  `assert_complete`, `with_emit_traces` (generation span per call, with
  `REDACTED_TRACE_ERROR_MESSAGE` honored), and `get_retry_advice`.
  Step validation rejects `conflicting_outcomes` and `invalid_retry_advice` before queueing;
  the other Python reasons are unrepresentable in the typed Rust step.
- `scripted_stream_events` — full stream synthesis, all 20 event types in Python's order with
  monotonic `sequence_number`: `response.created`, `response.in_progress`, per item
  `output_item.added` (with the in-progress shape) → the item's events → `output_item.done`,
  then `response.completed`. Covers `content_part.added/done`, `output_text.delta/done`,
  `output_text.annotation.added`, `refusal.delta/done`,
  `function_call_arguments.delta/done` and all six `reasoning_*` events.
- Richer helpers: `assistant_message_with_id`, `function_call_full(name, args, call_id,
  item_id, namespace)`, `DEFAULT_SCRIPTED_ITEM_ID`.

**`src/voice/testing.rs`**

- `STTSessionCall` + `FakeSTTModel::session_calls()`, so a test can assert the settings and the
  two privacy flags passed to `create_session`.
- `FakeSTTModel::created_sessions()` — `FakeTranscriptionSession` is now `Clone` over shared
  state, so a session the pipeline owns stays observable (`close_calls`, `remaining_turns`).
- `TTSCall.settings` (dropped `PartialEq`/`Eq` on `TTSCall`, because `TTSModelSettings` is not
  `PartialEq`) and `TTSResult { chunks }` with `From<Vec<Bytes>>`, so `with_result` still takes
  a plain chunk vector.

---

## Tests

- `tests/mcp_tests.rs::exposes_resources_metadata_and_request_meta` — resources, templates,
  read, `server_initialize_result`, `cached_tools`, title-only description fallback, and a
  round-trip proving the resolver's `_meta` plus explicit `_meta` reach the server.
- `src/mcp/logging.rs` unit tests — credential/query/fragment stripping, IPv6, prefixes,
  non-URL names, invalid URLs, metadata resolution.
- `src/mcp/util.rs` unit tests — name disambiguation (collisions, over-long names, reserved
  names, determinism) and part sanitization.
- `tests/sessions_tests.rs` — `conversations_session_batches_writes_and_resets_on_clear`
  (asserts the 20/5 request split and the fresh conversation after clear),
  `conversations_session_honors_its_own_settings_limit` (including `limit == 0`),
  `compaction_session_refuses_a_server_managed_store`,
  `advanced_sqlite_reports_turn_structure_and_usage_details`.
- `tests/testing_model_tests.rs` (new, 6 tests) — strict matching, unexpected call, unconsumed
  steps, conflicting outcomes, responders, the full synthesized event sequence (asserted
  type-by-type and on `sequence_number`), a streamed `Runner` run proving the Rust consumer
  accepts the new events, and exact-event passthrough.
- `tests/voice_tests.rs::session_creation_and_tts_settings_are_recorded`.

---

## API decisions worth a second opinion

1. **`OpenAIResponsesCompactionSession::new` kept its infallible signature.** Python raises in
   `__init__` when wrapping an `OpenAIConversationsSession`. Changing `new` to return `Result`
   is a breaking public change, so `try_new` was added and `new` defers the same `UserError` to
   first use. If a breaking change is acceptable, `new` should become `try_new`.
2. **Derived MCP tool names keep the `{server}_{tool}` base format.** Python uses
   `mcp_{server}__{tool}`. Switching would rename every persisted/prompted tool name, so it is
   left as a user decision; only the *disambiguation* behaviour was brought to parity. Two
   smaller name changes **do** happen, both matching Python: the server part is now the
   sanitized log name (so a URL-derived server name loses its credentials and query), and
   leading/trailing `_`/`-` are trimmed from each part (a server literally named `srv_` now
   yields `srv_tool`, not `srv__tool`).
3. **Tool-name hash suffix is FNV-1a, not SHA-1.** Python truncates a SHA-1 hex digest. No hash
   crate is available under the `mcp` feature and `Cargo.toml` is off-limits, so a deterministic
   built-in hash is used. Suffix *values* therefore differ from Python; collision behaviour and
   length bounds match. If SHA-1 parity matters:
   `NEEDS_CONTEXT: Cargo.toml line 26 — mcp = ["dep:rmcp", "dep:sha1"]` (plus a `sha1` dependency entry).
4. **`ConversationTurn.content` is now a 100-char preview** (was the full text) and the struct
   gained two fields. `full_content` carries what `content` used to. In-repo callers were
   updated.
5. **`TTSCall` lost `PartialEq`/`Eq`** so it could carry `TTSModelSettings`.
6. **`ScriptedStep` is not `Clone`** because `AgentsError` is not `Clone`; a step's error is
   moved out when it is raised.
7. **`get_items(None)` is now bounded by a session's own settings.** One existing test
   (`runner_compacts_only_what_the_run_loaded_from_a_limited_session`) read the whole store that
   way and was changed to pass `Some(usize::MAX)`. Internal full reads elsewhere
   (`EncryptedSession`, `OpenAIResponsesCompactionSession`) now go through the underlying
   store's limit — which is exactly what Python does, since its `encrypt_session` and every
   backend resolve the limit too.

---

## Not done — out of scope or blocked

Reported rather than worked around.

- **`_ignore_ids_for_matching` propagation (audit row 127b).** Blocked: the digests the
  compaction session compares against come from `SessionRunCompletion::model_exchange`, which
  the runner fills in `src/run_internal/mod.rs` (another agent's file). Both sides must agree on
  whether `id` is part of the digest, so this needs a `Session::ignore_ids_for_matching()` hook
  consumed in `run_internal` + `memory::compaction_item_digest`. Half of it would be worse than
  none.
- **`failure_error_function`, `tool_input_guardrails`, `tool_output_guardrails`, `mcp_title`,
  `custom_data_extractor` / `MCPToolCustomDataContext`** on MCP servers — these need
  `src/tool.rs` / `src/agent.rs` fields that are owned by other agents.
- **`ToolFilterContext.run_context` / `agent`, and async dynamic filters / approvals** — a
  breaking signature change to a public type plus an async callback redesign.
- **`reserved_tool_names` plumbing from the agent** — `get_all_function_tools_with_reserved`
  exists, but the agent-side caller that would pass the agent's own tool names lives in
  `src/run_internal/tools.rs`.
- **`collect_mcp_list_tools_metadata`** — serves hosted MCP (`items.rs` / `tool.rs`) and does
  not belong behind the `mcp` feature.
- **`max_retry_attempts == -1` (infinite retry)** — `u32` cannot express it; changing the field
  type is a breaking change.
- **`MCPToolCancellationError` path** — Python needs it because `asyncio.wait` can observe a
  cancelled task. A dropped Rust future simply stops; there is no observable "the call was
  cancelled" result to translate. No `AgentsError` variant exists for it either.
- **`session_id()` on Streamable HTTP** — `rmcp` 3.8 moves the transport into the service and
  exposes no session-id accessor afterwards. The trait method exists and returns `None`.
- **Stdio `encoding` / `encoding_error_handler`, Streamable HTTP `timeout`,
  `sse_read_timeout`, `terminate_on_close`, `ignore_initialized_notification_failure`, `auth`,
  `httpx_client_factory`, and `message_handler`** — no counterpart in
  `rmcp::transport::StreamableHttpClientTransportConfig`; adding dead fields would be worse
  than reporting them.
- **MCP v1/v2 cursor naming compat (`next_cursor` vs `nextCursor`)** — not applicable; `rmcp`
  speaks one protocol.
- **Isolated-session retry (`_isolated_client_session` / `_call_tool_with_isolated_retry`) and
  `_credential_safe_exception_group`** — need a second `rmcp` service instance per call and an
  exception-group analogue; large and low value, not started.
- **Disabling client retries for multi-request Conversations writes** — `OpenAIClient` performs
  no automatic retries, so there is nothing to disable.
- **`testing/sandbox.py` (audit row 45)** — explicitly outside this scope (`src/sandbox/**` is
  another agent's).
- **`ScriptedModel` async responders, `ModelStepSpec` dict form, stream *factories*** — the
  Rust responder is sync (consistent with the rest of the SDK's callbacks) and a typed struct
  replaces the dict form. `ScriptedStep::stream` takes a concrete event vector, not a factory.
- **`Usage::requests` default for scripted steps** — Python's `ModelStep` defaults to a zero
  `Usage`; `ScriptedStep::usage = None` falls back to `set_default_usage` and then
  `Usage::default()`, which is the same zero usage.

---

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/mcpmem \
       CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
cargo clippy --features mcp,sqlite,encrypt,redis,voice --all-targets -- -D warnings
cargo test   --features mcp,sqlite,encrypt,redis,voice
```

Also run: `cargo check --offline --features mongodb,sqlx` (the two backends edited for
`resolve_session_limit` that are outside the verification feature set) — clean.

Result: all 21 test targets pass (including 14 doctests). Clippy reports **no diagnostics in
any file in this scope**; it fails overall only on pre-existing issues in
`src/run_internal/{history,items,mod,tool_caller}.rs`, which belong to another agent. Because
`-D warnings` makes the lib build fail there, the integration-test targets are skipped under
the exact command; they were linted separately without `-D warnings` and are clean.
