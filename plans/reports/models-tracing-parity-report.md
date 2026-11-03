# Models / tracing / logging-redaction parity report

Date: 2026-10-30. Scope: the `logger.py` + `util/_json.py` + `util/_tool_errors.py` +
`util/_error_tracing.py` + `exceptions.py` redaction row, and the `models/**` and `tracing/**`
rows of `plans/reports/parity-audit-261030-0505-python-to-rust-report.md`.

Build (final state): `CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/models cargo clippy
--features websocket --all-targets -- -D warnings` is clean across the whole crate, and
`cargo test --features websocket --no-fail-fast` passes: 21 test binaries, 248 tests, plus 11
doctests. `cargo check --features mcp,voice,realtime,sandbox --lib` is also clean. Build dir:
1.5 GB.

---

## 1. Redaction layer (new)

### `src/logger.rs` (new, declared by hand in `src/lib.rs`)

Port of `agents.logger`. Public, logging to target `openai.agents`:

- `log_model_action_{error,warning,debug}` — text appears only when `debug::dont_log_model_data()`
  is false.
- `log_tool_action_{error,warning,debug}` — gated on `dont_log_tool_data()`.
- `log_model_and_tool_action_{error,warning,debug}` and `log_model_and_tool_data_warning` — gated
  on **both** flags, as in Python.
- `may_log_model_data(contains_tool_data)` and `model_data_payload(payload, contains_tool_data)`
  — the `validate_json` rule from `util/_json.py`, plus `REDACTED_DATA_PLACEHOLDER`
  (`"<redacted>"`), so a call site that embeds a model/tool payload in an error message can
  redact it.
- `LOG_TARGET` for call sites that need `log::warn!(target: …)` directly.

### `src/tracing/error_tracing.rs` (new, re-exported from `crate::tracing`)

Port of `util/_error_tracing.py` + `util/_tool_errors.py`:

- `REDACTED_TRACE_ERROR_MESSAGE` (`"Error details are redacted."`),
  `REDACTED_TOOL_ERROR_MESSAGE` (`"Tool execution failed. Error details are redacted."`).
- `get_trace_error`, `get_trace_tool_error`, `model_error_text` — the error is only stringified
  when the text will actually be exported.
- `record_model_error_on_span`, `record_tool_error_on_span` — set the span error with the
  *message* intact and the `data.error` detail redacted when
  `trace_include_sensitive_data` (`ModelTracing::include_data()`) is off.
- `attach_error_to_current_span` — the "no active span" fallback, itself redacted.

### Applied at every model span-error site in my scope

`openai_responses.rs` (×4), `openai_chatcompletions.rs` (×3), `openai_responses_ws.rs` (×4) now
call `record_model_error_on_span(..., request.tracing.include_data())` and
`logger::log_model_action_error` instead of `span.set_error(..., e.to_string())` /
`log::warn!("… {err}")`. `run_internal/model_call.rs` retry logging now uses
`logger::log_model_action_warning`.

Covered by `tests/models_redaction_tests.rs`.

---

## 2. Models

| Gap | Status |
|---|---|
| **`chatcmpl_stream_handler` tool-call id/name append bug** | Fixed: the id and the name are now **assigned** (and an empty incoming value is ignored), matching `_accumulate_tool_call_delta`. Test `assigns_repeated_tool_call_id_and_name`. |
| `models/_trace.py` metadata leak | New `src/models/trace.rs`: `model_config_for_trace` drops `metadata` when data is excluded and adds a `base_url` key sanitized by `sanitize_url_for_trace` (userinfo, query and fragment stripped). Wired into the generation span. |
| `models/multi_provider.py` fallbacks | `litellm/` and `any-llm/` now resolve to `extensions::models::{LitellmProvider, AnyLLMProvider}` without registration (explicitly registered providers still win). Added `MultiProviderMap` with `has_prefix`/`get_mapping`/`set_mapping`/`get_provider`/`add_provider`/`remove_provider`, `MultiProvider::with_provider_map`, and `BUILT_IN_FALLBACK_PREFIXES`. |
| `models/_response_terminal.py` + `error`/`response.error` stream events | New `src/models/response_terminal.rs` with `format_response_terminal_failure`, `format_response_error_event`, `terminal_failure_error`, `is_terminal_failure_event`. Both the HTTP and the WebSocket Responses streams now handle `response.failed`, `response.incomplete`, `error` and `response.error` with Python's `status=…; error=…; incomplete_details=…` / `code=…; message=…; param=…` detail format. Test `responses_stream_terminal_events_become_errors`. |
| `openai_chatcompletions.py` `tool_choice` | `tool_choice` is now sent whenever it is set, not only when tools are present (`parallel_tool_calls` stays gated on tools, as in Python). |
| `openai_chatcompletions.py` duplicate `extra_args` | `extra_args` keys that collide with the body now raise `UserError` with Python's `chat.completions.create() got multiple values for keyword argument 'x'` wording. |
| `strict_feature_validation` | `OpenAIChatCompletionsModel::with_strict_feature_validation` + `OpenAIProvider::with_strict_feature_validation`. Rejects (or warns about) `prompt`, `previous_response_id` and `conversation_id` on a Chat model, and empty/non-text tool outputs. |
| `chatcmpl_converter` url citations + logprobs | `convert_url_citations`, `convert_logprobs_for_output_text`, `convert_logprobs_for_text_delta` (public). Applied on the non-streaming path (`annotations`, `logprobs` on the output text part) and on the streaming path (`logprobs` on `response.output_text.delta`, accumulated annotations and logprobs on the final part and on `response.output_text.done`). Tests `parses_annotations_and_logprobs`, `streams_logprobs_and_url_citations`. |
| `preserve_tool_output_all_content` + `_OMITTED_TOOL_OUTPUT_PLACEHOLDER` | `ChatConversionOptions::preserve_tool_output_all_content` / `strict_feature_validation`, `OpenAIChatCompletionsModel::with_preserve_tool_output_all_content`, `OMITTED_TOOL_OUTPUT_PLACEHOLDER`. Non-text-only tool output now degrades to the placeholder (or errors under strict validation) instead of silently flattening. |
| Unpaired `function_call_output` | Now a `UserError` with Python's wording, instead of emitting `tool_call_id: null`. **Breaking** (see below). |
| `openai_client_utils.is_official_openai_base_url` | Implemented in `openai_client.rs` with the `websocket` flag; `OpenAIClient::is_openai_api()` now compares scheme + host instead of the whole base URL. |
| `openai_provider.py` `websocket_base_url` | `OpenAIProvider::with_websocket_base_url`, with the `OPENAI_WEBSOCKET_BASE_URL` env fallback, forwarded to `OpenAIResponsesWSModel`. |
| `models/interface.py` `ModelProvider.aclose()` | `ModelProvider` is now `#[async_trait]` with a default no-op `close()`. `OpenAIProvider::close` closes its cached WebSocket models; `MultiProvider::close` forwards to the OpenAI provider, every registered provider and every lazily created fallback. |
| `run_internal/model_retry.py` `apply_retry_attempt_usage` | `Usage::apply_retry_attempt_usage(failed_attempts)` (zero-token `RequestUsage` entry per failed attempt, `requests` bumped). Applied in `run_internal/model_call.rs`, so `Usage.requests` no longer undercounts retries. |
| `models/_openai_retry.py` `ReplaySafety::Unsafe` | `get_openai_retry_advice` now returns `suggested: Some(false)` + `ReplaySafety::Unsafe` for the ambiguous WebSocket error ("Responses websocket connection closed before a terminal response event."), which the Rust WS model already raises. |
| `extensions/models` litellm gaps | `fix_tool_message_ordering` + `requires_strict_tool_ordering` ported from `LitellmModel._fix_tool_message_ordering`, applied automatically for `anthropic`/`claude`/`gemini` models through the new `OpenAIChatCompletionsModel::with_message_transform` hook. Tests `pairs_tool_calls_with_their_results`, `strict_tool_ordering_matches_anthropic_and_gemini`. **Deliberately not ported:** Python's `preserve_tool_output_all_content=True` for LiteLLM models. Python can set it because LiteLLM translates the content parts into each provider's native shape; this adapter posts straight to the provider's OpenAI-compatible endpoint with nothing in between, and those endpoints are not guaranteed to accept array content on a `role: tool` message. The flag exists (`with_preserve_tool_output_all_content`) for callers who know their endpoint accepts it. |

---

## 3. Tracing

| Gap | Status |
|---|---|
| `TraceProvider` as a trait + `set_trace_provider` | `TraceProvider` is now a **trait**; the old struct is `DefaultTraceProvider`. `provider()` returns `Arc<dyn TraceProvider>`. Added `set_trace_provider`, `reset_trace_provider`, `default_provider`, `shutdown_tracing`. The export API key moved to its own static (`provider::set_export_api_key`), so a custom provider does not have to carry it. **Breaking** (see below). |
| `tracing/context.py` disabled masking | `create_trace_for_run(...)` returns `None` when a trace is already active, and **always** returns a disabled trace when the run disables tracing, so an outer trace is masked. `masked_run_span()` + `MASKED_RUN_SPAN_NAME` do the same for the active span. Call site is **not** wired (see **Hand-off**). |
| `tracing/create.py` `task_span` / `turn_span` | Public `task_span(name)` and `turn_span(turn, agent_name)`. |
| `tracing/spans.py` accessors | `Span::started_at()` / `Span::ended_at()`. |
| `tracing/processors.py` `Retry-After` | `BackendSpanExporter::send` now parses `retry-after-ms` / `retry-after` (reusing `retry::parse_retry_after`) and sleeps `max(backoff, retry_after)` bounded by `max_delay`, and honours `x-should-retry: false` over the status classification. |
| `tracing/processors.py` configurable backoff + setters | `with_max_retries`, `with_base_delay`, `with_max_delay`, `with_organization`, `with_project`. |
| `tracing/processors.py` recursive truncation | `truncate_value` is now recursive over objects and arrays (largest child first, with cached child sizes and an incrementally updated total), so a generation span's message list keeps its structure instead of collapsing into one truncated string. Tests `truncates_structures_recursively`, `truncation_keeps_nested_structure_within_the_limit`. |
| `tracing/processors.py` `_omit_generation_reasoning` | Ported (test `omits_reasoning_from_generation_spans`): generation span `input`/`output` drop `reasoning` items and the `reasoning`/`reasoning_content`/`thinking_blocks`/`thinking` keys plus inline `thinking`/`redacted_thinking` content blocks on assistant messages, including inside a nested `object: "response"` item. Only the OpenAI export path is affected; custom processors still see the untouched span. |

Covered by `tests/tracing_provider_tests.rs` (custom provider, masking, task/turn spans) and
`tests/tracing_exporter_tests.rs` (`Retry-After` honoring, the `x-should-retry: false` override,
no retry on a client error).

---

## 4. Visualization (`src/extensions/visualization.rs`)

- `get_all_nodes` and `get_all_edges` are now **public**.
- Stable node ids: a label is used as the DOT id only when it is unique and does not collide with
  `__start__` / `__end__`; otherwise `__agents_graph_{kind}_{n}__` is generated. Two tools, or an
  agent and a tool, sharing a name are no longer conflated, and an agent literally named
  `__start__` no longer collides with the terminal.
- `escape_label` now escapes `\r\n`, `\r` and `\n` to `\n`, so a multi-line name cannot emit
  malformed DOT.
- A handoff target that was already drawn no longer re-emits its node (Python's visited guard).

---

## Breaking changes

1. **`tracing::TraceProvider` is a trait, not a struct.** The concrete type is now
   `DefaultTraceProvider`, and `tracing::provider()` returns `Arc<dyn TraceProvider>` instead of
   `&'static TraceProvider`. `TraceProvider::set_export_api_key` moved to the free function
   `tracing::provider::set_export_api_key`; `set_tracing_export_api_key` is unchanged.
2. **DOT node ids changed** when labels are ambiguous or shadow the terminals. Unique-label
   graphs (the common case) render identically.
3. **The Chat Completions model now sends `tool_choice` without tools**, so
   `tool_choice: "none"` with zero tools reaches the provider instead of being dropped.
4. **An unpaired `function_call_output` is now a `UserError`** when converting to Chat
   Completions, instead of producing `tool_call_id: null`. Matches `chatcmpl_converter.py:875`.
5. **Colliding `extra_args` keys now error** instead of silently overwriting the body.
6. **`OpenAIClient::is_openai_api()`** compares scheme + host, so a non-`/v1` path on
   `api.openai.com` now counts as official (affects the `store` and `stream_options` defaults).
7. **`ModelProvider` is now `#[async_trait]`** with a default `close()`. Existing impls compile
   unchanged; implementors that want cleanup must add `#[async_trait]` to their impl block.
8. **Generation span `model_config` gained a `base_url` key** and loses `metadata` under
   `ModelTracing::EnabledWithoutData`.
9. **The streaming terminal-failure error text changed.** `response.failed` /
   `response.incomplete` used to raise `"Responses API returned terminal status failed: …"` and
   now raise `"Responses stream ended with terminal event \`response.failed\`. status=…"`.
   Callers matching on the message text must be updated. The non-streaming `parse_response`
   message is unchanged.
10. **Multimodal tool output on the Chat Completions wire.** A `ToolOutput::Parts` output used
    to be flattened into one joined string; it now becomes an array of `{"type":"text",…}`
    parts (Python's shape), or `"[tool output omitted]"` when it has no text at all. Plain
    `ToolOutput::Text` and `ToolOutput::Json` outputs are unaffected, so only multimodal tool
    results change shape.
11. **`logprobs` now appears on `response.output_text.delta` and `response.output_text.done`**
    for Chat Completions streams (empty when the provider sends none).

---

## Hand-off: call sites outside my scope

These need a one-line change in files owned by other agents:

1. **`src/run_internal/mod.rs:403-433`** — the run-trace creation should call
   `tracing::create_trace_for_run(workflow_name, trace_id, group_id, metadata, tracing_api_key,
   disabled)` and, when `disabled`, also enter `tracing::masked_run_span()` for the duration of
   the run. Today a run with `tracing_disabled = true` nested inside an enabled trace still
   exports its spans. The helpers and their test exist; only the wiring is missing.
2. **`src/run_internal/tools.rs:374`** — `span.set_error("Error running tool",
   json!({"error": e.to_string()}))` should become
   `tracing::record_tool_error_on_span(&span, "Error running tool", &e, include_data)` so a tool
   error respects `trace_include_sensitive_data` (Python writes
   `"Tool execution failed. Error details are redacted."`). Same for `tools.rs:313`,
   `local_tools.rs:{392,507,674,759}`, `run_internal/mod.rs:{468,694,1938}`, `mcp/util.rs:142`
   and `voice/models/openai_stt.rs:126`.
3. **`src/tool.rs:481`** — `"Invalid JSON input for tool {}: {e}"` embeds the raw payload.
   Python's `validate_json` replaces it with `<redacted>`; use
   `logger::model_data_payload(payload, /* contains_tool_data */ true)`.
4. **`src/voice/result.rs:27`** — has its own private `REDACTED_TRACE_ERROR_MESSAGE`; it can now
   use `crate::tracing::REDACTED_TRACE_ERROR_MESSAGE`.
5. **`src/extensions/hosted_multi_agent.rs:619-641`** — can use
   `models::response_terminal::terminal_failure_error` for its terminal-event handling.

---

## Deferred (with reasons)

- **`OpenAIResponsesWebSocketOptions`** (`ping_interval`, `ping_timeout`, `max_size`) and the
  `buffer_streamed_tool_calls` provider flag. Keepalive pings need a background task per
  connection in `openai_responses_ws.rs`; `max_size` needs a `WebSocketConfig` plumbed through
  `connect_async_with_config`. Note: Rust's Chat Completions stream **already** behaves as
  `buffer_streamed_tool_calls = True` (tool-call arguments are emitted once in `finish()`), so
  the flag's permissive direction — incremental argument streaming — is the missing half, not
  the buffering.
- **`_openai_shared.py` / `_config.py` globals**: `OpenAIResponsesTransport`,
  `set_default_openai_responses_transport`, `set_use_responses_websocket_by_default`, the
  `get_default_*` getters and `OPENAI_HARNESS_ID_TRACE_METADATA_KEY`. Pure global-config surface;
  the WS transport remains per-provider.
- **`models/_run_context.py`** (`model_run_context`, `get_model_run_owner`,
  `Model._cleanup_on_run_end`). The runner owns the lifetime, and `src/run.rs` /
  `src/run_internal/mod.rs` are not mine. The cheap, meaningful half —
  `ModelProvider::close()` + `MultiProvider` forwarding — **is** done.
- **Streaming `request_id`** (`openai_responses.rs`): `post_sse` returns it, but the
  `ModelResponse` is built by `run_internal/model_call.rs` from the `response.completed` event,
  so threading it through needs a change to the raw-event stream contract.
- **`_mark_error_to_drain_stream_events`**: Rust's stream returns `Err` and the consumer drops
  the stream, which closes the transport. The remaining events are not drained first. No
  observable difference today because nothing reads post-terminal events.
- **`_is_never_sent_websocket_error` / `_did_start_websocket_response` / `_get_websocket_timeout_phase`**
  and the rest of the WS replay-safety machinery. Only the ambiguous-replay branch is ported.
- **`_should_disable_provider_managed_retries`, `_should_disable_websocket_pre_event_retry`,
  `websocket_pre_event_retries_disabled`, session rewind on a stateful retry** — the last one
  belongs to `run_internal/session_persistence`, which is another agent's row.
- **`extensions/models` provider reach**: Bedrock, Vertex/Azure, Watsonx, Sagemaker,
  HuggingFace, Databricks, Nebius, Sambanova and per-provider auth (AWS SigV4, Azure `api-key`
  / AD tokens, GCP OAuth). These need new dependencies (a SigV4 signer, a GCP token source) —
  see **Dependencies** below. `preserve_thinking_blocks` and
  `_convert_gemini_extra_content_to_provider_specific_fields` also remain.
- **`tracing/traces.py` `TraceState` / `reattach_trace` / `ReattachedTrace`** — trace continuity
  across a serialized `RunState`; it belongs with the `run_state.rs` row.
- **`atexit` shutdown hook**: Rust has no process-exit hook; `flush_traces()` /
  `shutdown_tracing()` must be called by the application. Documented in `tracing/processors.rs`.
- **Per-processor error isolation** (`_processor_diagnostic_extra`): Rust still propagates a
  processor panic. Isolating it needs `catch_unwind` plus `UnwindSafe` bounds on
  `TracingProcessor`, which would be a trait-level breaking change.
- **`_refresh_disabled_flag`**: the env var is read once at provider creation, not re-read.
- **Span creator signature gaps** (`span_id`, `parent`, `disabled` on every creator) and the
  `metadata` slot on `Agent`/`Task`/`Turn` span data. Export output is byte-identical today, so
  this is API surface only.
- **`TracingExporter::export` typed items**: still `Vec<ExportItem>` (pre-serialized), not live
  `Trace | Span`. Changing it would break every custom exporter.
- **`models/_openai_websocket.py`**: `default_query` merging on `OpenAIClient`, the websocket
  logger adapter's debug suppression, `refresh_openai_client_api_key_if_supported`, and
  `Omit`/`NotGiven` header-removal semantics.

---

## Dependencies I would need (none added)

Nothing was added to `Cargo.toml`. For the deferred any-llm provider reach, the missing pieces
would be an AWS SigV4 signer (`aws-sigv4` or `aws-sdk-bedrockruntime`) and a GCP token source
(`gcp_auth`), both substantial new dependency trees. URL sanitization in `models/trace.rs` and
host matching in `openai_client.rs` were written by hand rather than pulling in `url`, which is
only a transitive dependency of `reqwest`.

---

## Notes on other agents' files

During the session `cargo check` / `cargo clippy` / `cargo test` intermittently failed in
`src/testing.rs`, `src/memory/openai_responses_compaction_session.rs`, `src/result.rs`,
`src/run_internal/{mod,history,items,tool_caller}.rs` and `src/run_state/wire.rs` while other
agents were mid-edit, and `sessions_tests` and `runner_tests` each failed at one point. **All of
those are resolved in the final state**: the full `cargo clippy --features websocket
--all-targets -- -D warnings` and `cargo test --features websocket --no-fail-fast` runs above are
clean across the whole crate.

## Extra hardening done after review

- `truncate_value_for_limit` caches child sizes and updates the running total incrementally
  instead of re-serializing the container once per removal, so a long generation input does not
  stall the export thread.
- The `prompt` and `previous_response_id`/`conversation_id` warnings on a Chat Completions model
  fire once per model instance (`AtomicBool`), matching Python's `_has_warned_*` flags.
- `tests/models_redaction_tests.rs`'s log test filters on the SDK target plus a unique probe
  string, so the tracing test running in parallel in the same binary cannot make it flake. The
  duplicate `logger::tests` unit test (which flipped process-global flags inside the shared lib
  test binary) was removed.
