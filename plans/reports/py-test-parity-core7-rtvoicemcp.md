# py-test-parity core7: realtime / voice / mcp / trimmer / fastapi (fixb)

Scope: python-only audit findings and the python-only groups of the realtime, voice, mcp, tool-output-trimmer and fastapi test files.

## Source bugs fixed (behaviour now matches Python)

- `src/realtime/event_validation.rs` (new, private module) plus `src/realtime/openai_model.rs`, `src/realtime/mod.rs`:
  - Every server event is validated against the full OpenAI server-event union (required scalar fields, nested `error.type/message`, object/array fields). Python validates all of them; Rust validated only `response.output_audio.delta`.
    - An event with an unknown or missing `type`, or a missing or mistyped required field, now emits `RealtimeModelEvent::Error { message, context: "Failed to validate server event" }`. It is no longer forwarded as `Other`.
    - The error text names the field and expected type but never a value.
    - The log line is `Failed to validate server event`, with the event appended only when `dont_log_model_data` is off (Python's redaction rule).
    - A `response.output_item.added/done` message item still bypasses validation, as in Python.
    - Known-but-unhandled types still surface as `Other` (Rust extra, kept).
  - A raw client message is validated against the client-event union. Unknown type or missing required field: it is dropped, logged as `Failed to convert raw message` (detail only when model data logging is on), and `send_event` returns `Ok(())`. Previously every raw message went to the wire verbatim.
- This fixes the malformed `response.output_audio_transcript.done` gap of `test_app_server_debug.py`.

## Breaking changes (behavioural, no signature change)

- Server events of an unknown type, or missing required fields, are now `Error` events instead of `Other` or silently accepted. A custom mock server must send complete events (`event_id`, `output_index`, and so on), as the real API does.
- A raw message whose `type` is not a client event type is dropped instead of sent.
- Python's `RealtimeModelErrorEvent` carries the full pydantic error, including the offending value. The Rust error event names the field and expected type only; the payload stays available through the preceding `RawServerEvent`. This is a deliberate divergence: the redaction test asserts no log line holds the value.
- The audio-delta error message is now `Invalid response.output_audio.delta event: missing field ...`. It was `Invalid response.output_audio.delta event`.

## Cases ported (15 python-only cases moved to ported)

| Python test | Rust test | file |
|---|---|---|
| `test_openai_realtime_conversions::test_try_convert_raw_message_invalid_returns_none` | `raw_message_of_an_unknown_type_is_dropped` | `tests/py_core7_rtvoicemcp_realtime_wire.rs` |
| `test_conversion_helpers::test_try_convert_raw_message_{invalid_type,malformed_data,missing_type}` (3) | `raw_message_of_an_unknown_type_is_dropped`, `raw_message_missing_a_required_field_is_dropped` | same |
| `test_app_server_debug::test_forwarded_provider_errors_have_payload_free_debug_summaries[malformed=True]` (2) | `malformed_transcript_done_is_reported_with_model_data_{redacted,logged}` | same |
| `test_openai_realtime::test_handle_invalid_event_schema_{redacts_event_from_logs,preserves_diagnostics_when_enabled}` (2) | `invalid_event_log_omits_all_event_data_when_redacted`, `invalid_event_log_keeps_diagnostics_when_model_data_is_logged` | same |
| every-event-type validation (extra) | `every_known_server_event_type_is_validated`, `a_complete_known_event_is_not_reported`, `raw_message_of_a_known_type_is_forwarded` | same |
| `test_pipeline::test_streamed_audio_result_odd_length_buffer_int16`, `..._preserves_cross_chunk_sample_boundaries` (2) | `odd_length_buffer_is_padded_for_int16`, `cross_chunk_sample_boundaries_are_preserved` | `tests/py_core7_rtvoicemcp_voice_pipeline.rs` |
| `test_mcp_auth_params::test_sse_with_httpx_client_factory`, `test_sse_with_auth_and_factory`, `test_streamable_http_with_auth_and_factory` (3) | `sse_uses_the_caller_supplied_http_client`, `sse_combines_headers_auth_and_the_caller_supplied_http_client`, `streamable_http_combines_auth_and_the_caller_supplied_http_client` | `tests/py_core7_rtvoicemcp_mcp_transport.rs` |
| `test_mcp_version_compat::test_stdio_connects_to_legacy_server` | `stdio_connects_to_a_server_that_reports_an_older_protocol_version` (fixture `tests/fixtures/legacy_mcp_server.py`) | same |
| `fastapi/test_streaming_context::test_streaming_context` | `a_stream_started_in_one_task_is_consumed_in_another` (run started in one tokio task and consumed in a spawned one: 11 events as in Python's snapshot, one `trace_start` and one `trace_end`, no panic) | `tests/py_core7_rtvoicemcp_streaming_context.rs` |

Fixture updates forced by the stricter validation: the realtime test fixtures in `py_realtime_transport.rs`, `py_realtime_openai_{events,interrupt,sequencing,connect}.rs` and `realtime_tests.rs` now carry the fields the API always sends (`event_id`, `output_index`, `usage`, `audio_end_ms`). The barrier event is now `input_audio_buffer.cleared` instead of an unknown `py.test.sentinel`. `handle_unknown_event_type_ignored` now asserts the Python outcome, a validation error and no exception.

## Remaining python-only, one reason per group

- Trimmer (4 of 60): `bytes` rejection, negative `recent_turns` / `preview_chars` and `frozenset` coercion are unrepresentable under `usize` and generic-iterator types. `trimmable_tools_from_string` is covered by `a_single_allowlist_entry_matches_a_whole_tool_name`.
- Realtime session, testing, handoffs, openai transport, model settings, playback tracker: asyncio task and cancellation internals, listener callbacks, `__traceback__` and `capture_locals`, `caplog` on non-redaction paths, Mock arbitrary types, tools or handoffs in `initial_model_settings`, `max_size=None`, the 1009 close code. All earlier-round reasons were spot-checked and hold.
- Realtime `examples/` servers (`test_app_server_security`, `test_twilio_*`, `test_app_server_debug` non-malformed parts, `test_session_payload_and_formats`): Python `examples/` code that Rust does not ship.
- Voice: asyncio cancellation and cleanup ordering, positional or dict `VoicePipelineConfig`, numpy dtype spellings, the scripted-testing doubles' snapshot semantics. Behavioural cases were checked and are already covered under different names: `get_sentence_based_splitter` (inline tests in `src/voice/utils.rs`), speech-span instruction redaction (`py_voice_pipeline_tracing.rs`), `transform_data` (`voice_tests.rs`), failing close reaching the consumer (`py_voice_pipeline.rs`).
- MCP: worker-task, anyio and `CancelledError` policy, `ExceptionGroup` and httpx attachment scrubbing (the no-secret-in-error-text contract is asserted in `py_mcp_server_errors.rs`), typed timeout and page-limit validation, httpx factories or kwargs on patched clients (Rust takes a `reqwest::Client`, now tested), MCP v2 / httpx2, import-path tests, `FakeMCPPromptServer` self-tests, `_get_session_id` internals.

Remaining python-only after this pass: 523 (per file in `py-test-parity-core7-rtvoicemcp.csv`).

## Rule deviations

- One `git diff --stat` was run by mistake (read-only, nothing modified).

## Status

Status: DONE_WITH_CONCERNS
Summary: 15 python-only cases moved to ported (python-only 538 to 523 for the area's ledger rows); the realtime transport now validates every server event and raw client message as Python does. Area test binaries pass with `--all-features`; clippy `-D warnings --all-targets` is clean for both all-features and default features (no diagnostics in my files); rustfmt applied.
Concerns: the doc-test run for realtime did not finish within the time limit and was not confirmed (no doc comments were changed); behaviour changes listed above are breaking for custom mock servers that send incomplete events.
