# Py test parity: models / responses (final)

Rust test files (all green, clippy clean except one foreign lint, rustfmt'd):
`tests/py_models_responses_{converter,core,http,ws,ws_retry}.rs` = 58 + 38 + 24 + 46 + 10 = 176 Rust tests.
Python grep counts match the table (149 / 59 / 10 / 2 / 11 / 2 = 233). Parametrize expansions not counted.

## 1. Per-file table

| file | #tests | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_openai_responses.py | 149 | 116 (http.rs, core.rs `tools::`, ws.rs, ws_retry.rs) | 1: `src/models/openai_responses_ws.rs::tests::{pings_a_live_connection_while_the_read_is_pending, an_unanswered_ping_fails_the_receive, the_keepalive_deadline_falls_back_to_the_interval, websocket_options_default_to_pythons}` for keepalive options | 32: stream-close bookkeeping aclose/aexit/cancel (8); omit/NOT_GIVEN sentinels (7); httpx URL/Timeout + client default timeout (6); openai-python `with_options(max_retries)` (2); asyncio loop affinity / lazy Lock (2); asyncio close-handshake blocking + transport.abort (2); fallback `responses.create()` stream path (2); dataclass fields/replace (1); unresolved computer initializer (1); fake ws connection abort fallback (1) | yes, see section 2 |
| test_openai_responses_converter.py | 59 | 58 (converter.rs; includes `ToolSearch`/namespace/tool_choice validation) | 0 | 1: `WebSearchTool` positional params (Python signature) | yes |
| test_remove_openai_responses_api_incompatible_fields.py | 10 | 10 (core.rs; `is list_input` identity check ported as equality) | 0 | 0 | none |
| test_model_payload_iterators.py | 2 | 0 | 0 | 2: monkeypatched single-use iterators; Rust builds owned `serde_json::Value`s | n/a |
| test_responses_websocket_session.py | 11 | 5 (ws.rs: shared run config, openai prefix alias, `model_id` modes, unknown-prefix mode, close) | 0 | 6: dict run_config coercion (2); monkeypatched `Runner.run/run_streamed` injection (2); `run_config` override rejection (1); `run_sync` absence (1) -- Rust session exposes `run_config()` and callers pass it to `Runner` | yes |
| test_trace_config.py | 2 | 2 (core.rs) | 0 | 0 | yes |
| **Totals** | **233** | **191** | **1** | **41** | |

## 2. Rust bugs fixed

`src/models/openai_responses.rs`
- `tool_choice` validated against the request's tools/handoffs with Python's exact `UserError` text (`required` + deferred without `ToolSearchTool`, hosted tool_search choice, namespace wrapper / bare inner name, deferred-only function).
- `shell` / `apply_patch` choices select the builtin tool (were function selectors); computer selector follows the effective model, resolved to `None` when no computer tool.
- Added tool-search, programmatic-tool-calling, one-computer-tool, function-tool lookup (reserved synthetic namespace, qualified-name collision, duplicate deferred names) and `file_search` `max_num_results` (0..=50, 0 omitted) validation.
- Non-streamed `failed`/`incomplete` response status now raises `Responses stream ended with terminal event `response.{status}`...` (was custom text).
- Streaming now yields the terminal failure event first, then raises the error, and stops without polling the transport again (`deliver_terminal_failures`); a transport failure after `response.completed` is no longer seen.
- Response `usage.total_tokens` is the reported value (null -> 0), not derived from input + output.
- Public additions: `with_explicit_model`, `ToolSurfaces`, `convert_tool_choice_for_request`, `parse_response`, `clean_input_for_openai`.

`src/models/openai_responses_ws.rs`
- Non-streamed `get_response` raises the terminal failure for `response.failed` / `response.incomplete`.
- `close_connection` no longer waits for an in-flight request: a watch channel aborts it with `WS_CLOSE_INVALIDATED_MESSAGE`; in-flight connection is discarded if close happens while connecting or replacing a cached socket.
- A stream dropped or cancelled before its terminal event drops the cached connection (`InFlight` guard); one dropped after the terminal event keeps it.
- Cached connection is liveness-checked before reuse (server close -> reconnect); one retry on a dropped handshake, honouring `websocket_pre_event_retries_disabled`; a send/receive failure after the frame was written is classified pre-event-disconnect + `(the request frame was already sent)`.
- URL builder keeps the base URL's own query, merges client default query then request `extra_query` (later wins); header names compared case-insensitively for connection identity.
- Retry advice: removed two arms that advised on rejected handshakes (Python gives no advice).
- `ResponsesWebSocketSession::run_config()` now routes through a `MultiProvider` (prefix routing preserved); added `with_openai_prefix_mode` / `with_unknown_prefix_mode`.

`src/models/trace.rs`
- `model_config_for_trace_with_extra` (Python `extra_config`).

## 3. Cross-area bugs deferred

1. `src/extensions/models/mod.rs:1016` (not mine): clippy `manual_flatten` on `for source in [&mut settings.extra_body, &mut settings.extra_args] { if let Some(map) = source {...} }` fails `-D warnings` for every test target. Minimal fix: `for map in [&mut settings.extra_body, &mut settings.extra_args].into_iter().flatten() { map.remove("reasoning_effort"); }`. My targets are clean with `-A clippy::manual_flatten`.
2. `src/models/openai_client.rs:647-660` (SSE parser): an `error` frame is turned into `AgentsError::ApiError{status:0}` before the model sees it, so the stream yields no `error` event and raises `ApiError` instead of Python's `ModelBehaviorError` after yielding the event (`code=...; message=...`). Fix: return the frame as `SseEvent::Data` and let the model's `terminal_failure_error` raise it. Tests assert the current behaviour and carry a comment.
3. `src/usage.rs:205` `to_span_usage` / tracing: response span `usage` is `{input_tokens, output_tokens, details{requests,total_tokens,cached_input_tokens,reasoning_output_tokens}}`; Python exports flat `{requests, input_tokens, output_tokens, total_tokens, input_tokens_details, output_tokens_details}`. Also `SpanData::Response` carries no `response`/`input` (Python records both when tracing includes data). Fix is in tracing, outside my ownership.
4. `src/run_internal`: a model turn with empty output and no tool calls loops until `MaxTurnsExceeded` (observed with a mock replying `output: []`); Python ends the run. Not changed.

## 4. Shared-file edits needed

None. No `src/lib.rs` or `Cargo.toml` changes.

## 5. `#[ignore]`d tests

None.

## 6. Verification

- `cargo test --all-features --test <each of the five>`: converter 58, core 38, http 24, ws_retry 10, ws 46 (single binary run, one hung test fixed: the session mocks answered with empty output, so the runner kept calling the model; they now answer with a message). All pass.
- Related existing binaries still green: `openai_http_tests`, `responses_ws_tests`, `models_redaction_tests`, `core_prompt_cache_key`, `core_model_run_lifecycle`, `core_tracing_masking`, `models_thinking_blocks_tests`, `extensions_models_provider_tests`; lib `models::` 51 pass.
- `cargo clippy --all-features --test <each> -- -D warnings`: clean except item 3.1 (foreign file).
- Build dir `/Users/ngocthach/.cache/oars-build/tp-models`: 2.7 GB.
- Stray file observed, not mine: `tests/py_scratch_dbg.rs`.
