# Models / extensions-models / tracing — final parity report

Date: 2026-10-30. Area: `src/models/**`, `src/extensions/models/**`, `src/tracing/**`,
`src/logger.rs`, `src/extensions/hosted_multi_agent.rs`. Closes the deferred list of
`plans/reports/models-tracing-parity-report.md`, the any-llm/LiteLLM row of
`plans/reports/parity-audit-261030-0505-python-to-rust-report.md` (~line 118), and the
`terminal_failure_error` wiring row.

Build dir `/Users/ngocthach/.cache/oars-build/mdl2`.

---

## 1. Deferred rows from `models-tracing-parity-report.md`

| Row | Status |
|---|---|
| **WS keepalive options** | `OpenAIResponsesWebSocketOptions { ping_interval, ping_timeout, max_size }` with Python's defaults (20 s / 20 s / unlimited). `max_size` goes to `connect_async_with_config` (`max_message_size` + `max_frame_size`; tungstenite's own default is 64 MiB, so `None` is set explicitly to mean unlimited as in Python). Keepalive pings are driven from the receive loop (`recv_next`), since tungstenite has no background pinger: a ping goes out after `ping_interval` of silence and the request fails after `ping_timeout` without a frame; `ping_timeout: None` keeps pinging and never fails, as in Python. A connection cached *between* requests has no reader, so it is not pinged while idle — a server close during idle is caught on the next send and triggers the one reconnect. Builders on `OpenAIResponsesWSModel`, `OpenAIProvider` and `ResponsesWebSocketSession`. |
| **`_openai_shared` / `_config` globals** | `OpenAIResponsesTransport`, `set_/get_default_openai_responses_transport`, `set_/get_use_responses_websocket_by_default`, `set_/get_use_responses_by_default`, `get_default_openai_api`, `get_default_openai_key`, `get_default_openai_client`, `OPENAI_HARNESS_ID_TRACE_METADATA_KEY`. `OpenAIProvider::use_responses_websocket` is now `Option<bool>` and falls back to the global. Without the `websocket` feature, a Websocket transport warns once and falls back to HTTP. |
| **model run owner + `_cleanup_on_run_end`** | New `src/models/run_context.rs`: `ModelRunOwner`, `new_model_run_owner`, `model_run_context`, `model_run_context_sync`, `get_model_run_owner`, `model_run_context_stream` (a `Stream` adapter that polls inside the task-local scope, Python's `aclosing` wrapper). `Model::cleanup_on_run_end(owner)` added with a no-op default; `OpenAIHostedMultiAgentModel` implements it (and now refuses to resume a paused response owned by another run, as Python does). **Runner wiring is a hand-off** (below). |
| **streaming `request_id`** | `attach_stream_request_id` puts the SSE response's `x-request-id` on every event's `response` object under `STREAM_REQUEST_ID_KEY` (`_request_id`, Python's attribute name), and `parse_response` falls back to it, so the terminal event yields `ModelResponse::request_id` without changing the raw-event contract. |
| **rest of the WS replay-safety machinery** | Ported into `OpenAIResponsesWSModel`: a single reconnect-and-resend when the send on a *cached* connection fails (gated on `should_disable_websocket_pre_event_retries()`), the pre-event-disconnect wrap with safe/unsafe replay classification, the response-started marker, close invalidation (a `close()` racing the request aborts instead of retrying), the four timeout phases (`request lock wait` / `connect` / `send` / `receive`), the never-sent classification, and the `server_is_overloaded` / untyped `server_error` error-frame retry. `websocket_retry_advice` is the port of `OpenAIResponsesWSModel.get_retry_advice` and falls back to `retry::get_openai_retry_advice`. Rust errors carry no attributes, so the markers are deterministic message prefixes/suffixes (`WS_AMBIGUOUS_REPLAY_MESSAGE`, `WS_CLOSE_INVALIDATED_MESSAGE`, `WS_PRE_EVENT_DISCONNECT_MESSAGE`), which is what Python's own `startswith` checks already relied on. |
| **WS timeout granularity** | Python derives the four phases from an `httpx.Timeout` (distinct `pool`/`connect`/`write`/`read`) *or* from a scalar applied to all four. `ModelSettings::timeout` is a scalar, so only the scalar form is implemented; per-phase timeouts would need a new settings field, which is not this area's file. |
| **`_mark_error_to_drain_stream_events`** | `models::response_terminal::should_drain_stream_events_before_raising(&AgentsError)` recognizes a terminal-failure error by the message shape this module produces. **`src/result.rs` must call it** (hand-off). |
| **`TraceState` / `reattach_trace`** | New `src/tracing/trace_state.rs`: `TraceState` (serde, `from_trace`/`from_json`/`to_json(include_tracing_api_key)`, `extra` round-trip, `matches_effective_settings`), `hash_tracing_api_key` (SHA-256 fingerprint), the 4096-entry started-trace-id LRU + `trace_id_was_started`, and `reattach_trace`. `Trace` gained a `reattached` flag: it keeps its id and metadata, spans flow into it, and no second `trace.start` is exported. `Trace::to_json` added. `tracing::create_trace_for_run_with_state(..., trace_state, reattach_resumed_trace)` is the resume-aware creator; **`run_internal` must call it** (hand-off). |
| **atexit-style flush** | `tracing::shutdown_guard() -> TracingShutdownGuard`; its `Drop` calls `shutdown_tracing()`. Rust has no process-exit hook and statics are never dropped, so a scope guard held in `main` is the idiomatic equivalent; it runs on return, on `?` and while unwinding. Documented that `process::exit`/abort still skip it. |
| **per-processor panic isolation** | `DefaultTraceProvider` wraps every processor callback in `catch_unwind(AssertUnwindSafe(..))` and logs `Error in trace processor during {action} (trace_processor: ...)`, matching `_processor_diagnostic_extra`. No trait change was needed (`AssertUnwindSafe` suffices). Documented that `panic = "abort"` bypasses it. |
| **span-creator signature gaps** | `tracing::span(data) -> SpanBuilder` with `.span_id()`, `.parent_span()`, `.parent_trace()`, `.parent()`, `.disabled()`, `.metadata()`, `.build()`, plus `SpanParent`. One builder covers the `span_id`/`parent`/`disabled` keywords every Python creator takes, instead of 13 near-duplicate `*_with` functions. `Span::data_metadata()` / `set_data_metadata()` are the `metadata` slot of `AgentSpanData`/`TaskSpanData`/`TurnSpanData`; like Python's, it is **not** exported, so the wire payload is unchanged. The existing named creators keep their signatures (`SpanData::Agent/Task/Turn` are built by struct literal in `run_internal`, so no enum field was added). |
| **`_refresh_disabled_flag`** | `DefaultTraceProvider` now reads `OPENAI_AGENTS_DISABLE_TRACING` lazily and caches it, with `set_disabled` as a manual override that always wins — Python's exact rule. |
| **`TracingExporter::export` typed items** | `ExportItem` gained `object: Option<ExportedObject>` (`Trace`/`Span`), plus `ExportItem::new`, `.span()`, `.trace()`, `.span_data()`. A custom exporter can read typed span data instead of re-parsing the payload. `PartialEq` is hand-written over `payload` + `api_key`. |
| **`_openai_websocket.py`** | `OpenAIClient::with_default_query` / `default_query()` (merged into every HTTP request and into the WebSocket URL, per-request `extra_query` winning), and `OpenAIClient::without_header` as the `Omit` header-removal semantics. `refresh_openai_client_api_key_if_supported` is subsumed by the new `RequestAuth` hook, which runs per request and is where a refreshed credential belongs (see `GoogleOAuthAuth`). The websocket logger's DEBUG suppression has **no Rust equivalent** (see §5). |
| **`_should_disable_provider_managed_retries` / `_should_disable_websocket_pre_event_retry` / `websocket_pre_event_retries_disabled`** | New `src/models/retry_runtime.rs`: task-locals with the same `false` defaults, `provider_managed_retries_disabled(flag, fut)` and `websocket_pre_event_retries_disabled(flag, fut)` scopes, and `should_disable_*` getters. The WS model honours the pre-event flag. |
| **`buffer_streamed_tool_calls`** | Implemented, **off by default as in Python**: Chat Completions streams now announce a function call as soon as its id and name arrive and forward each argument fragment as `response.function_call_arguments.delta`. `OpenAIChatCompletionsModel::with_buffer_streamed_tool_calls(true)` and `OpenAIProvider::with_buffer_streamed_tool_calls(true)` restore the single-complete-call behaviour. Output indices are now handed out in announcement order, so a message that starts after a tool call cannot reuse that call's index. |

---

## 2. any-llm / LiteLLM row

`src/extensions/models.rs` became `src/extensions/models/{mod,auth}.rs`.

### Provider reach

`ProviderEndpoint` gained `chat_path`, `auth` (`ProviderAuthKind`) and `default_query`; base URLs
may contain `${VAR}` / `${VAR:default}` and `{model}` placeholders. New entries:

- **AWS Bedrock** — `https://bedrock-runtime.${AWS_REGION:us-east-1}.amazonaws.com/openai/v1`.
  Uses `AWS_BEARER_TOKEN_BEDROCK` when set, otherwise signs with SigV4 (service `bedrock`).
  **Unverified from here:** the test proves the SigV4 headers are built correctly, not that the
  `/openai/v1` route accepts SigV4 rather than only the long-lived API key. See §5.
- **AWS SageMaker** — `https://runtime.sagemaker.${AWS_REGION}.amazonaws.com/endpoints/{model}`
  with `chat_path = "invocations"`, SigV4 (service `sagemaker`). `/invocations` has no SSE route
  (a streamed SageMaker response is AWS binary event-stream framed, not SSE), so its models are
  wrapped in `NonStreamingModel`: `stream_response` runs the ordinary request and replays the
  finished response as Responses events, instead of failing mid-parse. any-llm refuses to stream
  there; replaying keeps the run working, at the cost of the text arriving in one piece.
- **Google Vertex AI** — `https://${VERTEXAI_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEXAI_PROJECT}/locations/${VERTEXAI_LOCATION}/endpoints/openapi`, OAuth bearer.
- **Azure OpenAI** — `${AZURE_OPENAI_ENDPOINT}/openai/deployments/{model}`, `api-key` header,
  `api-version` default query, `AZURE_OPENAI_AD_TOKEN`/`AZURE_AD_TOKEN` taking precedence as a
  bearer. Plus **Azure AI** (`${AZURE_AI_ENDPOINT}/models`).
- **IBM watsonx.ai** — `https://${WATSONX_REGION:us-south}.ml.cloud.ibm.com`,
  `chat_path = "ml/v1/text/chat"`, `version` query, IAM token exchanged from the API key, and a
  body transform supplying `model_id` + `project_id`/`space_id` (the only non-OpenAI envelope in
  the table).
- OpenAI-compatible additions: `huggingface`, `databricks`, `nebius`, `sambanova`, `cohere`,
  `llama`, `deepinfra`, `hyperbolic`, `novita`, `baseten`, `portkey`, `vercel`.
- Aliases: `vertexai`/`vertex`, `aws`/`amazon`/`bedrock_converse`, `aws_sagemaker`/`sagemaker_chat`,
  `azure_openai`, `watsonx_text`/`ibm`, `huggingface_hub`/`hf`.

### Auth (`src/extensions/models/auth.rs`, feature `provider-auth`)

- `AwsSigV4Auth` + `AwsCredentials::from_env` — hand-written SigV4 over `sha2`/`hmac`, verified
  against AWS's published `get-vanilla` test vector (unit test `signs_the_aws_test_vector`).
- `GoogleOAuthAuth` + `GoogleCredentials::from_env` — Application Default Credentials order:
  explicit access token, `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_APPLICATION_CREDENTIALS`, the
  gcloud ADC file, then the metadata server. Service-account keys are signed RS256 via
  `aws-lc-rs`; `authorized_user` uses the refresh-token grant. Tokens are cached with a 60 s
  safety margin.
- `IbmIamAuth` — IBM Cloud API key → IAM bearer, cached.
- `StaticHeaderAuth` — the generic non-bearer header case.
- Hook point: `OpenAIClient::with_request_auth(Arc<dyn RequestAuth>)` plus
  `with_bearer_auth(false)`. `RequestAuth::authorize(method, url, body, headers)` runs after the
  client's own headers and sees the final URL (query included) and the exact body, which is what
  a signature covers. `post_json`/`post_sse`/`get_json`/`delete` now serialize the body
  themselves and build the full URL, so the signed bytes are the sent bytes.

### Other items of the row

- **Anthropic native messages**: new `src/extensions/models/anthropic.rs` —
  `AnthropicMessagesModel` speaks `POST /v1/messages` directly (`x-api-key` +
  `anthropic-version: 2023-06-01`), which is what any-llm's `anthropic` provider does. It
  converts Responses input items into Anthropic messages (merging consecutive turns, replaying
  signed `thinking` blocks *in front of* the `tool_use` block they belong to, `tool_result`
  blocks on a user turn, data-URL and URL images), maps `reasoning.effort` onto a
  `thinking.budget_tokens` budget (clamped to Anthropic's 1024 floor), converts `tool_choice`,
  and turns the reply back into a Responses `response` (thinking/redacted_thinking →
  one reasoning item carrying the whole block sequence, `text` → a message, `tool_use` →
  function calls, cache-read/-write tokens folded into `Usage`). Streaming consumes the native
  SSE event set (`message_start`, `content_block_start/delta/stop`, `message_delta`,
  `message_stop`), forwards `response.output_text.delta`, `response.reasoning_text.delta` and
  `response.function_call_arguments.delta`, and reassembles the complete message for the
  terminal `response.completed`. Reachable from the adapters through the `anthropic_native/` /
  `anthropic_messages/` prefixes; plain `anthropic/` keeps the OpenAI-compatible route, which is
  what LiteLLM's own `anthropic/` does.
- **Model listing**: `OpenAIClient::list_models()`, `LitellmModel::list_models()`,
  `AnyLLMModel::list_models()` and `extensions::models::list_provider_models(provider, base_url, api_key)`.
- **`preserve_thinking_blocks`**: ported into `ChatConversionOptions` and
  `OpenAIChatCompletionsModel::with_preserve_thinking_blocks`; the adapters turn it on and, as in
  LiteLLM, it only takes effect when `reasoning.effort` is set. Both halves are implemented: the
  response side records the whole signed block sequence in the reasoning item's
  `provider_data.thinking_blocks` (plus the normalized `reasoning_text` + `encrypted_content`
  view), including the streaming accumulator (`accumulate_thinking_block`, Anthropic's
  text-deltas-then-signature run and whole `redacted_thinking` blocks); the request side replays
  a native block list in `thinking_blocks`, or reconstructs blocks from `content` +
  `encrypted_content` and prepends them to the message content.
- **Gemini `extra_content`**: the Chat Completions converter now preserves Google's thought
  signature in both directions (`extra_content.google.thought_signature` ⇄
  `provider_data.thought_signature`), which was being dropped outright.
  `convert_gemini_extra_content_to_provider_specific_fields` is the port of
  `_convert_gemini_extra_content_to_provider_specific_fields` and is applied only when the target
  is a LiteLLM proxy — Gemini's own OpenAI-compatible endpoint wants the `extra_content` shape it
  emitted, the proxy wants `provider_specific_fields`.
- **`_normalize_google_tool_result_roles`** — **not applicable.** It patches any-llm's
  *Gemini-native* `contents` converter so `role: "function"` entries become `role: "user"`. This
  crate never builds Gemini-native payloads; Gemini and Vertex AI are called through their
  OpenAI-compatible routes, where a tool result is a `role: "tool"` message. Documented in the
  module header.
- **`_clone_provider_without_retries`** — **not applicable.** It clones the any-llm provider's
  HTTP client with `max_retries=0`. `OpenAIClient` performs no retries of its own, so
  `should_disable_provider_managed_retries()` has nothing to switch off and the SDK retry policy
  is always the only one running. The task-local exists for symmetry. Documented.
- **`preserve_tool_output_all_content=True` for LiteLLM models** — the previous report's
  deliberate decision is kept: this adapter posts straight to the provider's OpenAI-compatible
  endpoint, which is not guaranteed to accept array content on a `role: tool` message. The flag
  is available (`with_preserve_tool_output_all_content`).

---

## 3. `terminal_failure_error` in `hosted_multi_agent.rs`

The `response.failed` / `response.incomplete` / `error` / `response.error` branch now calls
`models::response_terminal::terminal_failure_error(&ty, &event)` after the output normalization,
matching `model.py:960-967`.

---

## 4. Tests

- `tests/extensions_models_provider_tests.rs` (new, `provider-auth`): Azure deployment path +
  `api-key` + `api-version`, Entra ID token precedence, Bedrock SigV4 headers and signed-header
  list, Bedrock API-key path, Vertex AI refresh-token exchange **and token caching**, watsonx IAM
  exchange + `model_id`/`project_id` body + `version` query, `GET /models`, the provider table,
  the `anthropic_native/` prefix reaching `/messages` with the native headers and body (and the
  same prefix listing models), SageMaker's `/invocations` route with its SigV4 scope and the
  replayed stream, and the Gemini `provider_specific_fields` rewrite.
- `tests/models_thinking_blocks_tests.rs` (new): Claude thinking blocks round trip (native list
  replayed verbatim), the "only when reasoning is requested" rule, Gemini thought signature round
  trip, and a two-turn native Anthropic Messages round trip (request shape, usage with cached
  tokens, thinking block replayed before the tool use, `tool_result` on the user turn).
- `tests/tracing_lifecycle_tests.rs` (new): panicking processor does not stop the others, the
  shutdown guard, the span builder (`span_id` / explicit parent / `disabled` / non-exported
  `metadata`), and trace reattachment.
- `tests/openai_http_tests.rs`: streamed `x-request-id`, incremental vs buffered tool-call
  argument streaming. The SSE mock now also sends `x-request-id`.
- Unit tests: the WebSocket keepalive over a real paired socket (repeated pings while the read
  stays pending with `ping_timeout: None`; an unanswered ping failing the receive with a finite
  one) and the keepalive-deadline fallback; AWS SigV4 vector + body/session-token signing + URL splitting + `amz_date`;
  WS options defaults, the four timeout phases, replay classification, overloaded error frames;
  `retry_runtime` scoping; `run_context` scoping; `TraceState` round trip / matching /
  reattachment; the streamed-`request_id` fallback.

---

## 5. Rows that stay open, with the inherent reason

- **WebSocket dependency DEBUG-log suppression** (`_OpenAIWebSocketLoggerAdapter`). Python
  replaces the `websockets.client` logger with an adapter that drops DEBUG records. Rust's `log`
  facade gives a *library* no way to filter another crate's target — only the installed logger
  implementation can, and it belongs to the application. There is no Rust-idiomatic equivalent a
  library can provide; the substitute is documented configuration
  (`RUST_LOG=tungstenite=info,tokio_tungstenite=info`). The SDK itself installs no logger and
  emits nothing at DEBUG from the WebSocket path.

Everything else in both rows is closed.

**Unverified rather than open:** whether AWS Bedrock's `/openai/v1` route accepts SigV4-signed
requests (as opposed to only `AWS_BEARER_TOKEN_BEDROCK`). Both paths are implemented and the
signature is verified against AWS's published vector, but confirming which the service honours
needs a real account, which the rules forbid.

---

## 6. Breaking changes

1. **`ExportItem` gained a public field `object`.** Struct-literal construction needs
   `object: None` (or `ExportItem::new`). `PartialEq` ignores it.
2. **Chat Completions streams now emit incremental
   `response.function_call_arguments.delta` events** (Python's default). Consumers that assumed
   exactly one delta per call should set `with_buffer_streamed_tool_calls(true)`.
3. **Chat Completions stream output indices** are assigned in announcement order instead of
   being derived; the only observable change is when an assistant message starts *after* a tool
   call, where it no longer collides with that call's index.
4. **`OpenAIResponsesWSModel` error messages changed** for transport failures: a pre-event
   disconnect is now `"Responses websocket connection closed before any response events were
   received. …"`, an error frame is `"Responses websocket error: {payload-json}"` (Python's
   format) instead of `"Responses websocket error ({code}): {message}"`, and timeouts/receive
   failures have their own prefixes.
5. **`Model` gained `cleanup_on_run_end`** (default no-op) — existing impls compile unchanged.
6. **`ProviderEndpoint` gained `chat_path`, `auth`, `default_query` and `default_headers`.**
   Code constructing one by struct literal must add them (`openai_compatible(..)` is the
   convenience constructor).
7. **`src/extensions/models.rs` is now `src/extensions/models/mod.rs`** — the module path
   `extensions::models` is unchanged.
8. **`OpenAIClient` request bodies are serialized by the client**, and query parameters are now
   appended to the URL (sorted) rather than through `reqwest`'s query builder. Same wire result,
   different parameter ordering.
9. **`hosted_multi_agent` terminal `error` / `response.error` events** now raise
   `ModelBehaviorError` with the `Responses stream ended with terminal event …` text instead of
   `AgentsError::Other("Hosted multi-agent {ty}: {msg}")`.
10. **A hosted multi-agent model with a paused response now rejects a second run's turn**
    (`UserError`), matching Python, instead of silently resuming it.
11. **`OpenAIResponsesWSModel::supports_default_prompt_cache_key`** now consults
    `websocket_base_url` when one is set (Python's `_supports_default_prompt_cache_key`), so a
    custom WebSocket URL on a non-OpenAI host no longer gets a generated `prompt_cache_key`.

---

## 7. Shared-file edits

- `Cargo.toml`: one new feature `provider-auth = ["dep:hmac", "dep:base64", "dep:aws-lc-rs"]`
  and one new optional dependency `aws-lc-rs = { version = "1", optional = true }`
  (already in the tree as the rustls crypto provider, so no new crates are downloaded).
  `hmac` and `base64` were already optional dependencies.
- `src/lib.rs`: **no edits.**

---

## 8. Hand-offs (files outside this area)

1. **`src/run_internal/mod.rs`** — wrap the run's model calls in
   `models::run_context::model_run_context(owner, fut)` (and model streams in
   `model_run_context_stream`) with an owner from `new_model_run_owner()`, and call
   `Model::cleanup_on_run_end(owner)` for every model used when the run ends
   (Python's `cleanup_models_after_run`). Without it the hosted multi-agent model's paused
   response is only released by `close()`.
2. **`src/run_internal/mod.rs`** — switch the run-trace creation to
   `tracing::create_trace_for_run_with_state(..., trace_state, reattach_resumed_trace)` so a
   resumed `RunState` keeps exporting into its original trace.
3. **`src/run_state.rs` / `src/result.rs`** — persist/restore `tracing::TraceState`
   (`TraceState::from_trace` / `from_json`), which is what feeds hand-off 2.
4. **`src/result.rs`** — before surfacing a stored exception, drain the queued stream events when
   `models::response_terminal::should_drain_stream_events_before_raising(&err)` is true. Python
   (`result.py:1060-1067`) drains for `MaxTurnsExceeded` **unconditionally** as well, so the
   condition there is `is MaxTurnsExceeded || should_drain_stream_events_before_raising(..)`.
5. **`src/run_internal/*`, `src/mcp/util.rs`, `src/tool.rs`, `src/voice/result.rs`** — the
   redaction hand-offs listed in the previous report are still open where they have not been
   taken up.

---

## 9. Verification

Build dir `/Users/ngocthach/.cache/oars-build/mdl2`, 3.3 GB.

- `cargo clippy --all-features --all-targets -- -D warnings`: **clean across the whole crate**
  (no remaining findings in any file, this area's or anyone else's).
- `cargo test --all-features --no-fail-fast`: **0 failures** — 619 lib tests, every integration
  binary (including the four new/extended ones: `extensions_models_provider_tests` 11,
  `models_thinking_blocks_tests` 4, `tracing_lifecycle_tests` 4, `openai_http_tests` 7), and 46
  doctests.
- Every file owned by this area was formatted with `rustfmt --edition 2024`. `src/lib.rs` was
  not touched or formatted.

---

Status: DONE_WITH_CONCERNS
Summary: Every deferred row of `models-tracing-parity-report.md`, the any-llm/LiteLLM row of the
parity audit (provider reach with real auth, model listing, `preserve_thinking_blocks`, the
Gemini `extra_content` conversion, native Anthropic Messages) and the `terminal_failure_error`
wiring are closed, with tests against mock servers only; `cargo clippy --all-features
--all-targets -- -D warnings` and `cargo test --all-features` are clean crate-wide.
Concerns: (1) whether AWS Bedrock's `/openai/v1` route honours SigV4 rather than only
`AWS_BEARER_TOKEN_BEDROCK` cannot be confirmed without a real account — both paths are
implemented and the signature matches AWS's published vector. (2) The websocket dependency's
DEBUG-log suppression has no library-side equivalent in Rust's `log` facade (documented in §5).
(3) Five hand-offs in `run_internal`, `run_state` and `result.rs` are needed before the model
run-owner cleanup, resumed-trace reattachment and stream-event draining actually run (§8).
(4) Eleven breaking changes, the user-visible ones being incremental Chat Completions tool-call
argument streaming (Python's default) and the new WebSocket transport error texts (§6).
