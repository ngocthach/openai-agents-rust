# Voice package port report

Port of the Python `agents.voice` package (openai-agents 0.24) to `src/voice/**` in
`openai-agents-sdk`, behind the existing `voice` feature.

Date: 2026-10-29

## Files added

| File | Python counterpart |
| --- | --- |
| `src/voice/mod.rs` | `voice/__init__.py` |
| `src/voice/input.rs` | `voice/input.py` |
| `src/voice/events.rs` | `voice/events.py` |
| `src/voice/utils.rs` | `voice/utils.py` |
| `src/voice/model.rs` | `voice/model.py` |
| `src/voice/result.rs` | `voice/result.py` |
| `src/voice/pipeline.rs` | `voice/pipeline.py` |
| `src/voice/pipeline_config.rs` | `voice/pipeline_config.py` |
| `src/voice/workflow.rs` | `voice/workflow.py` |
| `src/voice/testing.rs` | `voice/testing.py` |
| `src/voice/models/mod.rs` | `voice/models/__init__.py` |
| `src/voice/models/openai_stt.rs` | `voice/models/openai_stt.py` |
| `src/voice/models/openai_tts.rs` | `voice/models/openai_tts.py` |
| `src/voice/models/openai_model_provider.rs` | `voice/models/openai_model_provider.py` |
| `tests/voice_tests.rs` | `tests/voice/*` |
| `examples/voice_static.rs` | — |

No file outside `src/voice/**`, `tests/voice_tests.rs` and `examples/voice_static.rs` was
touched.

## Ported

- **Audio input**: `AudioInput` (buffer, `frame_rate` 24000, `sample_width` 2, `channels` 1),
  `to_base64`, `to_wav_bytes` / `to_audio_file` (port of `_buffer_to_audio_file`, including the
  1–4 byte sample widths and the validation errors), `StreamedAudioInput` with `add_audio`,
  `add_audio_f32`, `end` and `take_receiver`, plus `f32_to_i16` / `i16_to_f32`.
- **Models**: `STTModel`, `TTSModel`, `StreamedTranscriptionSession`, `VoiceModelProvider`,
  `STTModelSettings` (prompt, language, temperature, turn_detection, languages, keywords),
  `TTSModelSettings` (voice, buffer_size, dtype, transform_data, instructions, text_splitter,
  speed), `TTSVoice` with the 13 built-in voices plus `TTSVoice::Custom` (Python's
  `TTSCustomVoice`, serialized as `{"id": ...}`).
- **OpenAI models**: `OpenAIVoiceModelProvider` (defaults `gpt-transcribe` and
  `gpt-4o-mini-tts`), `OpenAISTTModel::transcribe` (multipart POST `/audio/transcriptions`,
  with the `gpt-transcribe` / `gpt-live-transcribe` `languages[]` special case),
  `OpenAITTSModel::run` (POST `/audio/speech`, `response_format: "pcm"`, streamed chunks,
  `ash` default voice, instructions in the body), and `OpenAISTTTranscriptionSession` over the
  realtime transcription websocket: `session.created` / `session.update` / `session.updated`
  handshake, `audio/pcm` at 24 kHz, `semantic_vad` default turn detection,
  `input_audio_buffer.append`, the final `commit` + `clear` with the expected
  `input_audio_buffer_commit_empty` no-op, committed/cleared/completed/failed bookkeeping
  (including legacy `input_audio_transcription_completed` without `item_id`), the inactivity and
  setup timeouts, and per-turn transcription spans.
- **Pipeline**: `VoicePipeline` with model instances or model names, `VoicePipelineConfig`,
  static and streamed input (`VoiceInput`), `on_start` greeting turn, ordered audio dispatch,
  `VoiceStreamEvent` (audio / `turn_started` / `turn_ended` / `session_ended` / error), text
  splitting through `get_sentence_based_splitter` (exact port, trailing-whitespace carry
  included), `buffer_size` chunk counting with the odd-byte carry and the final pad, and
  `transform_data`.
- **Tracing**: a pipeline trace (reusing the caller's, or masking it when `tracing_disabled`),
  `speech_group_span` per turn, a `speech_span` per text segment parented to it with
  `first_content_at` and base64 audio output, `transcription_span` for both static and streamed
  transcription, and `trace_include_sensitive_data` /
  `trace_include_sensitive_audio_data` honored everywhere (including the redacted span error
  message, Python's `get_trace_error`).
- **Workflows**: `VoiceWorkflowBase` (`run`, default `on_start`), `VoiceWorkflowHelper::stream_text_from`,
  `SingleAgentVoiceWorkflow` (history + last agent carried across turns, `Runner::run_streamed`,
  text deltas), `SingleAgentWorkflowCallbacks`.
- **Testing fakes**: `voice::testing` with `FakeSTTModel`, `FakeTranscriptionSession`,
  `FakeTTSModel`, `FakeVoiceWorkflow`, `pcm16_bytes`, and recorded `STTCall` / `TTSCall`.

## Not ported, and why

- **NumPy dtypes**. `TTSModelSettings::dtype` is the `AudioFormat` enum (`Int16`, `Float32`)
  instead of `npt.DTypeLike`; Python only accepts those two anyway and raises `UserError`
  otherwise, so the invalid-dtype error cannot occur. Audio buffers are `Vec<i16>` / `Vec<f32>`
  (`AudioData`), and the `float32` reshape to `(-1, 1)` has no meaning without NumPy.
- **`agent_registration`** on `OpenAIVoiceModelProvider`: the core crate has no
  `OpenAIAgentRegistrationConfig` equivalent.
- **Shared HTTP client / `websocket_base_url` / `default_query` plumbing**
  (`_openai_websocket.py`): `OpenAIClient` has no websocket base URL or default query, so the
  websocket URL is derived from `base_url` (`http`→`ws`, `https`→`wss`) plus
  `/realtime?intent=transcription`. Dynamic API-key refresh
  (`refresh_openai_client_api_key_if_supported`) has no core counterpart.
- **`coerce_dataclass_config` / legacy positional-argument shims** in `VoicePipelineConfig`:
  dict-shaped config and the pre-tracing positional order are Python-only concerns. There is
  also no `TracingConfig` in this crate, so `VoicePipelineConfig::tracing` is absent;
  `tracing_disabled`, `workflow_name`, `group_id` and `trace_metadata` cover the rest.
- **Python's `ScriptedTranscriptionSession` close-error injection**: the fake session's `close`
  always succeeds; the pipeline's close-failure path is covered by the real websocket session.
- **Python's cancellation bookkeeping** in `StreamedAudioResult.stream()` /
  `_cancel()` / `_cleanup_tasks()` and in `transcribe_turns()`: those ~150 lines reconcile
  `asyncio.CancelledError` against producer, consumer and cleanup exceptions. In Rust, dropping
  the `StreamedAudioResult` aborts the producer, the dispatcher and the segment tasks, and a
  `SessionGuard` closes the session on any producer exit (including a panic or an abort), so the
  consumer always sees a terminal event.
- **`UnexpectedVoiceCall` / `UnconsumedVoiceSteps` / `assert_complete`** in `testing.py`: the
  fakes return `AgentsError::other` with the same messages instead of a script-error hierarchy,
  and unconsumed steps are asserted directly in tests (`calls()`, `transcriptions()`).

## Deviations worth knowing

- `StreamedTranscriptionSession` exposes `async fn next_turn(&mut self) -> Result<Option<String>>`
  rather than an async-iterator `transcribe_turns()`. It is the idiomatic Rust shape (like
  `Receiver::recv`), avoids handing out a borrowed stream that must be dropped before `close()`,
  and keeps the pipeline loop trivial.
- `turn_started` is routed through the ordered segment channel, so it cannot overtake the
  previous turn's audio. Python writes it straight to the output queue.
- The speech-group span is finished in `turn_done` (after every segment task of the turn has
  completed) rather than inside the dispatcher.
- `VoiceStreamEvent::Error` carries `Arc<AgentsError>` because `AgentsError` is not `Clone`.
- `STTWebsocketConnectionError` maps to `AgentsError::other` with the same messages (see the
  requested core change below).

## Public API summary

`openai_agents_sdk::voice` re-exports:

- Input: `AudioInput`, `StreamedAudioInput`, `AudioData`, `AudioFormat`, `DEFAULT_SAMPLE_RATE`,
  `f32_to_i16`, `i16_to_f32` (and `input::AudioQueueReceiver`).
- Events: `VoiceStreamEvent`, `VoiceLifecycleEvent`.
- Models: `STTModel`, `TTSModel`, `StreamedTranscriptionSession`, `VoiceModelProvider`,
  `STTModelSettings`, `TTSModelSettings`, `TTSVoice`, `AudioByteStream`, `AudioTransform`,
  `DEFAULT_TTS_INSTRUCTIONS`, `DEFAULT_TTS_BUFFER_SIZE`.
- OpenAI: `OpenAIVoiceModelProvider`, `OpenAISTTModel`, `OpenAISTTTranscriptionSession`,
  `OpenAITTSModel`, `DEFAULT_STT_MODEL`, `DEFAULT_TTS_MODEL`, `openai_tts::DEFAULT_VOICE`.
- Pipeline: `VoicePipeline`, `VoiceInput`, `VoicePipelineConfig`, `StreamedAudioResult`.
- Workflow: `VoiceWorkflowBase`, `VoiceWorkflowHelper`, `SingleAgentVoiceWorkflow`,
  `SingleAgentWorkflowCallbacks`, `TextStream`.
- Utils: `get_sentence_based_splitter`, `TextSplitter`, `DEFAULT_MIN_SENTENCE_LENGTH`.
- `voice::testing`: `FakeSTTModel`, `FakeTranscriptionSession`, `FakeTTSModel`,
  `FakeVoiceWorkflow`, `STTCall`, `TTSCall`, `pcm16_bytes`.

Typical use:

```rust
let pipeline = VoicePipeline::new(Arc::new(SingleAgentVoiceWorkflow::new(agent)))
    .with_config(VoicePipelineConfig::default());
let mut result = pipeline.run(AudioInput::new(samples)).await?;
let mut events = std::pin::pin!(result.stream());
while let Some(event) = events.next().await { /* ... */ }
```

## Robustness fixes beyond a literal port

Two defects found while reviewing the first working version, each now covered by a test that
was confirmed to fail without the fix:

- `OpenAISTTTranscriptionSession` now aborts its audio and event tasks in `Drop`. Dropping a
  `JoinHandle` detaches the task, so a dropped session (the normal "stop listening" path, via
  `StreamedAudioResult::drop` aborting the producer) left the websocket open until the
  1000 second inactivity timeout. Test:
  `dropping_the_result_closes_the_transcription_websocket`.
- A failed TTS segment now stops the session. `AudioResultWriter` records the failure,
  `turn_done` returns `Err`, and the pipeline stops instead of transcribing and running the
  workflow (real LLM calls under `SingleAgentVoiceWorkflow`) for turns whose audio could never
  be delivered. Python stops here too, through `_turn_done`'s `gather`. Tests:
  `a_failing_tts_model_ends_the_stream_with_an_error`,
  `a_failing_tts_model_stops_a_streamed_session`.

A `SessionGuard` in the producer task also closes the session on any exit path, including a
panic or an abort, so a consumer can never wait forever on a session that stopped producing.

## Requested core changes (not made here)

1. **`Cargo.toml`**: add

   ```toml
   [[example]]
   name = "voice_static"
   required-features = ["voice"]
   ```

   The example currently compiles without the feature through a `#[cfg(not(feature = "voice"))]`
   fallback `main` that prints a hint; `required-features` would remove that stub.
2. **`OpenAIClient` HTTP helpers**: `post_multipart` and a raw byte-stream POST (or making
   `http()` / `url()` `pub(crate)` next to the already `pub(crate)` `headers()`). The voice
   models build their own `reqwest::Client` and re-implement `url()` and the `ApiError` mapping
   from `check()`. They do reuse `OpenAIClient::headers()` (already `pub(crate)` in HEAD), so
   organization, project and default headers are sent; if it ever becomes private again,
   `src/voice/models/mod.rs` is the single place to adapt.
3. **A dedicated error variant** (e.g. `AgentsError::VoiceWebsocket` or a reusable
   `STTWebsocketConnectionError`) so callers can match on a failed STT websocket instead of
   matching `AgentsError::Other` by message.
4. **Tracing**: `speech_span` / `transcription_span` / `speech_group_span` take no `parent`
   argument, so `src/voice/result.rs` calls `tracing::create_span(SpanData::Speech { .. },
   Some(&group_span), disabled)` directly to parent a segment span from a spawned task. A
   `parent: Option<&Span>` parameter on those helpers would be tidier.
5. Optional: re-export the voice entry points from the crate root under `#[cfg(feature = "voice")]`,
   the way `models::*` is re-exported today.

## Verification

Run with `CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/voice`:

- `rustfmt --edition 2024 src/voice/*.rs src/voice/models/*.rs tests/voice_tests.rs examples/voice_static.rs` — clean.
- `cargo clippy --features voice --all-targets -- -D warnings` — no warnings.
- `cargo clippy --all-targets` (no features) — no warnings; the example and test are gated.
- `cargo test --features voice --test voice_tests` — 20 passed.
- `cargo test --features voice --lib voice` — 20 passed.
- `cargo test --features voice --doc` — passed, including the `voice` module and
  `voice::testing` doctests.
- `cargo doc --features voice --no-deps` — the only warning is the pre-existing unresolved
  `[\`mcp\`]` link in `src/lib.rs` (that module is feature-gated off).
- `cargo run --example voice_static --features voice` — prints the lifecycle events, the spoken
  text and 360 audio samples.

Tests use a dummy `sk-test` key and local `TcpListener`-based mock HTTP and websocket servers.
Tracing is redirected to an `InMemoryTraceProcessor` (which replaces the OpenAI exporter), so
nothing touches the network and spans stay observable:
`speech_spans_are_grouped_per_turn_and_honor_the_privacy_flags` runs the pipeline inside an
explicit trace and asserts, for both settings of the two privacy flags, that each speech span is
parented to the turn's speech-group span and that the spoken text, the TTS instructions and the
base64 audio are present or empty accordingly.

Note for the controller: when this port was finished, `cargo clippy --features voice --all-targets
-- -D warnings` was clean. It currently reports findings in other agents' in-flight files
(`src/extensions/codex/*`, `src/run_config.rs`, `src/run_internal/mod.rs`,
`tests/codex_tests.rs`). No finding is in `src/voice/**`, `tests/voice_tests.rs` or
`examples/voice_static.rs`; `--message-format short` filtered by path confirms zero findings in
voice files.
