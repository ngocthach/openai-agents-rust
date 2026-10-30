# Voice final parity report

Date: 2026-10-30. Scope: `src/voice/**`, `tests/voice_tests.rs`, `examples/voice_static.rs`.
Python reference: `openai-agents-python` 0.24, `src/agents/voice/**`.

## Closed rows

| Row | What | Test |
| --- | --- | --- |
| 1 | `VoiceScriptError` enum (`UnexpectedVoiceCall { message, operation }`, `UnconsumedVoiceSteps { message, remaining_steps }`) + `Display` + `std::error::Error` + `From<…> for AgentsError`, shaped like `testing::ModelScriptError`. Operation ids in `voice::testing::operation` (`static_transcription`, `streamed_session`, `tts`, `workflow_start`, `workflow_turn`). | `voice_tests::an_exhausted_script_fails_with_pythons_messages`, `src/voice/testing.rs::a_script_error_converts_to_an_agents_error_with_the_same_message` |
| 2 | `assert_complete()` on `FakeTranscriptionSession`, `FakeSTTModel` (transcriptions + sessions, then every created session), `FakeTTSModel`, `FakeVoiceWorkflow` (three verbatim branches: startup-only / turns-only / mixed). Returns `Result<(), VoiceScriptError>` like `ScriptedModel::assert_complete`. | `voice_tests::assert_complete_reports_every_unconsumed_step`, `src/voice/testing.rs::{a_configured_start_must_be_consumed, unconsumed_turns_and_a_pending_start_report_steps, an_unconfigured_start_is_never_pending}` |
| 3 | `FakeTranscriptionSession::with_close_error`, `with_turn_error`, `is_closed()`; close is idempotent (second close is a no-op, still counted by `close_calls`); `next_turn` stops yielding after close (Python's `while self._turns and not self.closed`). `FakeSTTModel::with_session_instance` so a pre-built session can be scripted. | `voice_tests::a_scripted_session_close_failure_reaches_the_pipeline_and_close_is_idempotent`, `voice_tests::a_closed_session_stops_yielding_its_remaining_turns`, `voice_tests::a_failing_turn_ends_a_streamed_session` |
| 4 | `_START_NOT_CONFIGURED` semantics: `StartState { configured, pending, step }`; `on_start` fails only when a start was configured *and* already consumed; an unconfigured start yields nothing and never fails. `with_start_error` added. | `voice_tests::a_configured_workflow_start_is_consumed_exactly_once` |
| 5 | Python names: `pcm16_samples` (fn) and `ScriptedSTTModel` / `ScriptedTTSModel` / `ScriptedTranscriptionSession` / `ScriptedVoiceWorkflow` type aliases. `Fake*` names unchanged. | `src/voice/testing.rs::the_python_names_resolve_to_the_same_types` |
| 6 | `VoicePipelineConfig::tracing_api_key` + `with_tracing_api_key`, passed to the pipeline trace. Verified in Python: `TracingConfig` is `{api_key, include_task_and_turn_spans}`, and `TraceProvider.create_trace` reads **only** `api_key` (`tracing/provider.py:401`); `include_task_and_turn_spans` never reaches the voice pipeline. There is no precedence rule against the flat flags — `tracing_disabled` independently makes the trace a no-op. Field name copies `RunConfig::tracing_api_key`. | `voice_tests::the_pipeline_trace_carries_the_configured_tracing_api_key` |
| 7 | `OpenAIVoiceModelProvider::with_agent_harness_id` + `agent_harness_id()`, resolving explicit → global default → `OPENAI_AGENT_HARNESS_ID` through `models::resolve_openai_harness_id`. Verified: in Python the voice provider's `agent_registration` is a stored property no voice code reads — the harness id only reaches trace metadata via the *run's* provider (`run_internal/agent_runner_helpers.py:332`). Same in Rust. | `src/voice/models/openai_model_provider.rs::resolves_the_configured_agent_harness_id` |
| 8a | `VoiceWebSocketOptions { ping_interval, ping_timeout, max_size }` on `OpenAISTTModel::with_websocket_options` and `OpenAIVoiceModelProvider::with_websocket_options`. `max_size` → `connect_async_with_config` (`max_message_size` + `max_frame_size`); the keepalive is a ping driven from the event loop (tungstenite sends none) with a pong deadline. Defaults are the `websockets` defaults Python inherits: 20 s / 20 s / 1 MiB. | `voice_tests::the_websocket_max_size_rejects_an_oversized_event`, `voice_tests::an_unanswered_keepalive_ping_fails_the_session` |
| 8b | Client `default_query` is now merged into the realtime transcription handshake URL, with `intent=transcription` forced last (Python `_prepare_websocket_url`). Possible because another agent made `OpenAIClient::default_query()` public. | `src/voice/models/openai_stt.rs::the_handshake_url_carries_the_client_default_query` |
| 9 | Covered by row 3. | — |
| 10a | A panic in the producer task now reports an error instead of a silent `session_ended`: `SessionGuard::drop` checks `std::thread::panicking()`. (Before: the consumer saw a clean `session_ended` for a crashed pipeline.) | `voice_tests::a_panicking_workflow_still_ends_the_stream` |
| 10b | Drop-cancellation: dropping `StreamedAudioResult` aborts the producer, so later turns never run. | `voice_tests::dropping_the_result_stops_the_pipeline` (plus the pre-existing `dropping_the_result_closes_the_transcription_websocket`) |
| 12.1 | `[[example]] voice_static` with `required-features = ["voice"]` is already in `Cargo.toml` (lines 96-98). The now-dead `#[cfg(not(feature = "voice"))]` stub `main` and the `#[cfg(feature = "voice")]` attribute were removed from `examples/voice_static.rs`. | `cargo build --features voice --example voice_static` |

## Remaining rows

- **10c — consumer-side cancellation.** Python's `StreamedAudioResult.stream()` / `_cancel()` / `_cleanup_tasks()` and `transcribe_turns()` reconcile `asyncio.CancelledError` against producer, consumer and cleanup exceptions. In Rust a cancelled consumer is a *drop*: there is no cancellation exception to observe or re-raise, and `Drop` cannot be async, so an aborted producer cannot report "I was cancelled" to anyone. Inherent. Every other exit path (normal end, error, close failure, TTS failure, panic, drop) now delivers a terminal event and is covered by a test.
- **11 — NumPy dtypes.** `npt.DTypeLike` and the `float32` reshape to `(-1, 1)` have no meaning without NumPy; `AudioFormat` already covers the only two dtypes Python accepts. Inherent.
- **11 — `coerce_dataclass_config`.** Accepting a `dict` where a dataclass is declared is a Python-only affordance; Rust has the typed builder. Inherent.
- **11 — legacy positional-arg shim** (`_preserve_legacy_positional_arguments`). Rust has no positional struct literals to keep compatible. Inherent.
- **8c — `websocket_base_url`.** Needs `OpenAIClient` (not mine). See below.
- **8d — `refresh_openai_client_api_key_if_supported`.** Needs `OpenAIClient` (not mine). See below.

## Requested changes in files I do not own

1. `src/models/openai_client.rs` — add `with_websocket_base_url(impl Into<String>)` + `websocket_base_url() -> Option<&str>` (Python `AsyncOpenAI.websocket_base_url`, env `OPENAI_WEBSOCKET_BASE_URL`). `src/voice/models/openai_stt.rs::websocket_url` currently derives the URL from `base_url` (`http`→`ws`, `https`→`wss`); it would prefer the explicit value in one line. Note `OpenAIProvider` already has a Responses-only `with_websocket_base_url`, so the client-level one is the missing tier.
2. `src/models/openai_client.rs` — make `OpenAIClient::authorize(method, url, body, &mut HeaderMap)` `pub(crate)`. It is the Rust counterpart of `refresh_openai_client_api_key_if_supported`: `headers()` does **not** run the `RequestAuth` hook, so a dynamic/refreshed credential never reaches the voice (or realtime) websocket handshake, which uses `client.headers(None)`.
3. `src/models/openai_client.rs` — `post_multipart` and a raw byte-stream POST (or `pub(crate) http()` / `url()`). Still needed: `src/voice/models/openai_stt.rs` and `openai_tts.rs` build their own `reqwest::Client` and re-implement `url()` and the `ApiError` mapping in `src/voice/models/mod.rs::check_response`. (`headers()` is already `pub(crate)` and is reused.)
4. `src/error.rs` — a dedicated variant (e.g. `AgentsError::VoiceWebsocket`) for Python's `STTWebsocketConnectionError`. Still needed: `openai_stt.rs::websocket_error` returns `AgentsError::other`, so callers must match on the message.
5. `src/tracing/mod.rs` — a `parent: Option<&Span>` parameter on `speech_span` / `transcription_span` / `speech_group_span`. Still needed: `src/voice/result.rs:329` calls `create_span(SpanData::Speech { .. }, Some(&group_span), disabled)` directly to parent a segment span from a spawned task.
6. `src/lib.rs` — optional: re-export the voice entry points from the crate root under `#[cfg(feature = "voice")]`, the way `models::*` is re-exported.
7. `src/models/mod.rs` — optional de-duplication: `OpenAIResponsesWebSocketOptions` (same three knobs, same semantics) is gated on `feature = "websocket"`, which `voice` does not enable, so `voice::VoiceWebSocketOptions` is a second copy. Widening the gate to `any(feature = "websocket", feature = "voice")` would let voice share the type — only worth it if the two are meant to stay identical.

## Breaking changes

- `VoicePipelineConfig` gained a public field (`tracing_api_key`). Struct-literal construction breaks; `Default` + builders are unaffected.
- `FakeVoiceWorkflow::with_start` now marks the startup step *configured*: calling `on_start` twice on such a workflow fails where it used to re-yield the fragments.
- `FakeTranscriptionSession::next_turn` returns `None` after `close()` even with turns left (Python's behaviour).
- The scripted error messages gained Python's trailing `.` (`"… no scripted transcriptions remain."`). No test matched the old text.
- The streamed transcription websocket now caps an incoming message at 1 MiB (Python's `websockets` default) instead of tungstenite's 64 MiB, and sends a keepalive ping every 20 s. Both are configurable via `VoiceWebSocketOptions`; `with_ping_interval(None)` restores the old no-keepalive behaviour and `with_max_size(None)` removes the cap.
- `examples/voice_static.rs` no longer compiles without the `voice` feature (it is skipped by `required-features`).

## Shared-file edits

None. `Cargo.toml` and `src/lib.rs` were not touched.

## Verification

`CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/misc2`:

- `cargo test --all-features --lib voice` — 27 passed.
- `cargo test --all-features --test voice_tests` — 32 passed.
- `cargo test --all-features --doc voice` — 3 passed.
- `cargo build --features voice --example voice_static` — ok.
- `cargo clippy --all-features --all-targets -- -D warnings` — zero findings in `src/voice/**`, `tests/voice_tests.rs`, `examples/voice_static.rs`. Findings remain in other agents' in-flight files (`src/extensions/codex/*`, `src/sandbox/events.rs`, `src/sandbox/entries.rs`); not touched.
- `rustfmt --edition 2024` on every owned file — clean.

Tests use `sk-test` and local `TcpListener` mock HTTP/websocket servers plus `InMemoryTraceProcessor`; nothing reaches the network.
