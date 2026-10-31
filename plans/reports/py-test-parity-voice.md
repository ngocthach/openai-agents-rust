# Python test parity: voice (`tests/voice/`)

Counted by grep `def test_`, parametrize not expanded. 166 tests in 15 files.
Ledger: `/private/tmp/claude-501/-Users-ngocthach-myGit-openai-agents-rust/df34ec6b-67cb-4052-99ee-9636fbfd755b/scratchpad/tp/ledger-voice.tsv` (one row per test, every cited Rust test verified to exist).

## Per-file table

| file | #tests | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_input.py | 15 | 13 | 1 | 1 | 0 |
| test_utils.py | 4 | 0 | 4 | 0 | 0 |
| test_tts_voice_types.py | 3 | 2 | 0 | 1 | 0 |
| test_testing.py | 27 | 16 | 4 | 7 | 0 |
| test_pipeline.py | 53 | 19 | 9 | 25 | 0 |
| test_pipeline_config.py | 3 | 1 | 0 | 2 | 0 |
| test_pipeline_tracing.py | 4 | 3 | 0 | 1 | 0 |
| test_workflow.py | 2 | 2 | 0 | 0 | 0 |
| test_openai_stt.py | 25 | 15 | 0 | 10 | 4 |
| test_openai_stt_session_config.py | 8 | 6 | 0 | 2 | 1 |
| test_openai_stt_client_config.py | 8 | 6 | 0 | 2 | 2 |
| test_openai_stt_eof.py | 3 | 3 | 0 | 0 | 1 |
| test_openai_stt_api_key_refresh.py | 1 | 1 | 0 | 0 | 1 |
| test_openai_tts.py | 3 | 3 | 0 | 0 | 0 |
| test_openai_model_provider.py | 7 | 4 | 0 | 3 | 0 |
| **Total** | **166** | **94** | **18** | **54** | |

94 + 18 + 54 = 166. "failures found" counts Rust behaviors that differed from Python (all fixed in src/voice, below).

New test files (94 tests, all green with `--all-features`):
- `tests/py_voice_misc.rs` (31): input, tts voice types, testing fakes.
- `tests/py_voice_pipeline.rs` (22): pipeline lifecycle, split words, float32, on_start, close/error ordering, config defaults, single-agent workflow.
- `tests/py_voice_pipeline_tracing.rs` (3): caller trace/span, privacy flags, disabled pipeline, with real `OpenAISTTModel`/`OpenAITTSModel` against mock HTTP + websocket servers.
- `tests/py_voice_openai_models.rs` (38): STT/TTS/provider on the wire, websocket handshake (URL, headers, auth refresh), session config, EOF drain, timeouts, error events.

Also green: `tests/voice_tests.rs` (32), inline `src/voice` unit tests.

## Rust bugs fixed (all in `src/voice/models/openai_stt.rs`)

1. Non-JSON websocket frame was silently skipped; Python raises `STTWebsocketConnectionError("Error parsing events")`. Now a session error, both during setup and while streaming.
2. Server `Close` frame ended the event loop silently, so a disconnect while draining the final transcription never produced "Websocket closed before final transcription completed". Close frames now take the same path as a dropped socket.
3. Handshake ignored the client's `websocket_base_url` (and any query already on it); Python `_prepare_websocket_url` honours it, merges client `default_query`, forces `intent=transcription`. Now ported.
4. Handshake used `client.headers()` only, so a `RequestAuth` hook (Rust form of Python's `_refresh_api_key` before the handshake) never ran for the transcription websocket. Now uses `client.websocket_headers(url)`.
5. Session/inactivity timeouts were hard-coded; Python exposes them as module constants tests patch. Added `session_creation_timeout`, `session_update_timeout`, `event_inactivity_timeout` (seconds) to `VoiceWebSocketOptions` with builders; constants `SESSION_CREATION_TIMEOUT`, `SESSION_UPDATE_TIMEOUT`, `EVENT_INACTIVITY_TIMEOUT` are now `pub f64` (defaults unchanged: 10 / 10 / 1000). Additive on the public struct (new pub fields: struct-literal users must add `..Default::default()`).
6. The first turn's transcription span was opened inside the audio task, so a session closed right after setup recorded no span. It is now opened at connect time (Python opens it at the start of the session).
7. Removed the now-unused `request_headers` helper in `src/voice/models/mod.rs` (crate-private).

Note: another agent also touched `openai_stt.rs` to convert the f64 timeout constants with `Duration::from_secs_f64`; this is consistent with the change above (call sites now go through `seconds(options.*)`).

## Deferred cross-area bugs

None found. Observations, not bugs:
- `OpenAIClient` has no omit sentinel for default *query* params (Python `default_query={"remove": omit}`), so that sub-assertion of `test_streaming_stt_websocket_url_merges_client_default_query` is ported as "client default overrides the base-URL param" only. File: `src/models/openai_client.rs` (`default_query`). Minimal fix if wanted: `without_default_query(name)` mirroring `without_header`.

## Python-only groups (54)

- asyncio CancelledError / `aclose` / task-cleanup / queue-blocking semantics (25: pipeline cancellation, STT consumer cancellation); Rust uses Drop. Post-drop observable ported where one exists (`voicepipeline_dropped_consumer_closes_the_session_without_further_tts`); existing `voice_tests.rs::dropping_the_result_*` also cover it.
- Python type coercion: bare `str` vs `list[str]` scripted inputs, dict -> dataclass settings normalization, positional/legacy `VoicePipelineConfig` constructor, dataclass field order/subclassing (12).
- numpy dtype spellings/rejection (4); `__all__` membership (1); httpx2 type (1); falsy `__bool__` client (1); kwargs-conflict validation, unrepresentable in the Rust builder (1).
- Exception identity / deepcopy-failure snapshot paths, list aliasing/freeze (5).
- Logging record introspection (caplog) (3); weakref/gc retention (1).
- Python `TraceProvider` subclass + ContextVar introspection (1).
- `close()` during setup / listener-task pre-setup buffering: Python builds the session before connecting; Rust `create_session` returns only after setup (2); `asyncio.TimeoutError` vs builtin class identity (1).

## Notes

- Ports that adapt Python's mocks: websocket tests that patched `websockets.connect` run against a scripted local tokio-tungstenite server instead (no network). `wait_for_event` (private, queue-based in Python) is covered through the real handshake path, including the one-deadline-across-unrelated-events behaviour.
- `#[ignore]`: none.
- Shared files: no edits to `src/lib.rs` or `Cargo.toml`.
- Clippy: `cargo clippy --all-features --all-targets` reports nothing in `src/voice/**` or `tests/py_voice_*`; `-D warnings` currently fails only in other areas (`src/extensions/models/mod.rs:1016`, `src/models/openai_responses_ws.rs:553/556`, `src/sandbox/sandboxes/unix_local.rs:1149`), not touched.
- rustfmt applied to the four new test files only.
