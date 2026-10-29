# Realtime port report

Date: 2026-10-29
Scope: port of Python `agents.realtime` (openai-agents 0.24) into `openai_agents_sdk::realtime`
(feature `realtime`).

## Files

Created (all under the allowed scope):

| File | Lines | Contents |
| --- | --- | --- |
| `src/realtime/mod.rs` | 76 | Module docs, submodules, re-exports |
| `src/realtime/items.rs` | 421 | `RealtimeItem`, message/tool-call items, history merge helpers |
| `src/realtime/config.rs` | 470 | `RealtimeSessionModelSettings`, `RealtimeRunConfig`, turn detection, tracing, user input |
| `src/realtime/model_events.rs` | 216 | `RealtimeModelEvent` and payload types |
| `src/realtime/model_inputs.rs` | 88 | `RealtimeModelSendEvent` |
| `src/realtime/model.rs` | 150 | `RealtimeModel` trait, `RealtimeModelConfig`, audio length math |
| `src/realtime/agent.rs` | 217 | `RealtimeAgent<C>`, `RealtimeInstructions<C>` |
| `src/realtime/handoffs.rs` | 271 | `RealtimeHandoff<C>`, `realtime_handoff` |
| `src/realtime/events.rs` | 421 | `RealtimeSessionEvent<C>`, `RealtimeEventInfo<C>` |
| `src/realtime/session.rs` | 1288 | `RealtimeSession<C>`: history, tools, approvals, handoffs, guardrails |
| `src/realtime/openai_model.rs` | 1286 | `OpenAIRealtimeWebSocketModel` (GA protocol over `tokio-tungstenite`) |
| `src/realtime/runner.rs` | 107 | `RealtimeRunner<C>` |
| `src/realtime/testing.rs` | 192 | `FakeRealtimeModel` (port of `testing.py`) |
| `tests/realtime_tests.rs` | 919 | 11 integration tests |
| `examples/realtime_text.rs` | 91 | Text-only realtime session example |

Nothing outside `src/realtime/**`, `tests/realtime_tests.rs` and `examples/realtime_text.rs` was
touched.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/realtime
cargo clippy --features realtime --all-targets -- -D warnings   # clean
cargo test --features realtime --test realtime_tests            # 11 passed
cargo test --features realtime --lib realtime                   # 23 passed
cargo test --features realtime --doc                            # 4 realtime doctests compile
cargo doc --features realtime --no-deps                         # no warnings in src/realtime
cargo check --all-targets                                       # clean without the feature
rustfmt --edition 2024 --check <all files above>                # clean
```

`cargo doc` still reports one pre-existing warning in `src/lib.rs` (`unresolved link to 'mcp'`,
because the `mcp` feature is off); it is not in this module.

## Public API

```rust
// agents, handoffs
RealtimeAgent<C>            // name, handoff_description, instructions (static or async fn),
                            // prompt, tools, handoffs, output_guardrails, model_settings
RealtimeInstructions<C>     // Static | Dynamic
RealtimeHandoff<C>, realtime_handoff(Arc<RealtimeAgent<C>>)

// running
RealtimeRunner<C>           // new, with_model, with_config, with_model_config,
                            // run(context), run_with_context(RunContext<C>)
RealtimeSession<C>          // send_message, send_audio, interrupt, update_agent,
                            // approve_tool_call, reject_tool_call, recv, event_stream,
                            // history, current_agent, context, model, is_closed, close

// events
RealtimeSessionEvent<C>     // agent_start, agent_end, handoff, tool_start, tool_end,
                            // tool_approval_required, raw_model_event, audio, audio_end,
                            // audio_interrupted, error, history_updated, history_added,
                            // guardrail_tripped, input_audio_timeout_triggered
RealtimeEventInfo<C>

// transports
trait RealtimeModel { connect(config) -> RealtimeModelEvents; send_event(..); close(); }
RealtimeModelConfig, RealtimeModelEvents (= mpsc::UnboundedReceiver<RealtimeModelEvent>)
RealtimeModelEvent, RealtimeModelSendEvent, RealtimeModelToolCallEvent,
RealtimeModelAudioEvent, RealtimeConnectionStatus
OpenAIRealtimeWebSocketModel, FakeRealtimeModel, fake_realtime_model()

// data
RealtimeItem, RealtimeMessageItem, RealtimeToolCallItem, RealtimeContent, RealtimeMessageRole
RealtimeSessionModelSettings, RealtimeRunConfig<C>, RealtimeFunctionToolSpec,
RealtimeTurnDetectionConfig, RealtimeInputAudioTranscriptionConfig,
RealtimeNoiseReductionConfig, RealtimeModelTracing(+Config), RealtimeGuardrailsSettings,
RealtimeToolExecutionConfig, RealtimeUserInput(+Content)
DEFAULT_REALTIME_MODEL ("gpt-realtime-2.1"), DEFAULT_REALTIME_VOICE,
DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL, DEFAULT_DEBOUNCE_TEXT_LENGTH
calculate_audio_length_ms
```

## Deliberate design differences from Python

1. **Channel instead of listeners.** `RealtimeModel::connect` returns an
   `mpsc::UnboundedReceiver<RealtimeModelEvent>` rather than taking listener objects. This keeps
   the trait object safe, removes the re-entrancy Python guards against, and means a slow
   consumer can never block the transport's read loop. Applications still see every transport
   event through `RealtimeSessionEvent::RawModelEvent`.
2. **Non-generic model layer.** `RealtimeSessionModelSettings` carries
   `RealtimeFunctionToolSpec` (name, description, parameters) instead of `Tool<C>`/`Handoff<C>`.
   The session resolves and filters the agent's tools and handoffs, then hands the model plain
   specs. Keeping `Tool<C>` in the settings would have forced `RealtimeModel` and the WebSocket
   transport to be generic over `C`.
3. **`RealtimeHandoff<C>` is a separate type.** The core `Handoff<C>` resolves to
   `Arc<Agent<C>>`, and its invoke function is private, so realtime handoffs could not reuse it.
   `RealtimeHandoff` reuses `handoffs::default_tool_name`/`default_tool_description` and the same
   `{"assistant": "<name>"}` transfer message, verified by a unit test.
4. **Output guardrails see a stand-in `Agent`.** `OutputGuardrail::run` requires
   `Arc<Agent<C>>`; the session passes `Agent::<C>::new(realtime_agent.name)` with the transcript
   as `Value::String`. This mirrors Python's `cast(Agent[Any], agent)` (which the upstream source
   marks `TODO (rm) Remove this cast, it's wrong`). The guardrail therefore sees the right agent
   name but not realtime-specific fields.
5. **`RealtimeRunner::run` connects before returning.** Python returns an un-entered session you
   later `async with`. The Rust session is live when `run` resolves, and `close()` tears it down.
6. **Tool calls run inline on the session's event pump,** not on a detached task
   (`async_tool_calls` in Python). Because the transport feeds an unbounded channel, a long tool
   never blocks the WebSocket reader, and event ordering stays deterministic.
   `approve_tool_call` likewise runs the tool before it resolves, so `tool_start`/`tool_end` are
   already queued when it returns. This is documented on the method.
   **Output guardrails do run on spawned tasks**, as in Python: a slow, LLM-backed guardrail would
   otherwise stall delivery of every queued `Audio` event roughly every `debounce_text_length`
   characters. The handles live in `SessionInner::guardrail_tasks` and `close()` aborts them. The
   "already interrupted" bookkeeping uses a `HashSet::insert` return value, so concurrent
   guardrail runs on one response still emit `guardrail_tripped` exactly once.
7. **Tri-state settings.** `turn_detection` and `input_audio_noise_reduction` are
   `Option<Option<T>>` so "unset" (use the transport default) stays distinct from "explicitly
   disabled" (send JSON `null`), which is what Python expresses with `"key" in settings` plus a
   `None` value.
8. **Clone/Debug are hand-written** for every type generic over `C`, following `src/agent.rs`, so
   no `C: Clone`/`C: Debug` bound leaks into the public API.

## Intentionally not ported

- **`OpenAIRealtimeSIPModel`** and `build_initial_session_payload`. `RealtimeModelConfig::call_id`
  is supported (it switches the connect URL to `?call_id=` and stops defaulting the audio
  formats), but the SIP subclass and its payload helper add no behavior the trait does not.
- **`RealtimePlaybackTracker`.** The transport keeps the equivalent of Python's default tracker
  internally (last audio item, received length, arrival time) so `interrupt()` can send
  `conversation.item.truncate` with a sensible `audio_end_ms`. A user-supplied tracker for
  delayed or remote playback is not exposed.
- **`_ResponseCreateSequencer`'s generation ordering.** Python orders many concurrent manual and
  automatic `response.create` requests by generation and target version. The port keeps a smaller
  state machine on `WsState` (`ongoing_response`, `create_in_flight: Option<event_id>`,
  `pending_response_create`, `create_counter`) with I/O-free transitions, unit-tested in `--lib`:
  - a create is sent only when no response is ongoing *and* none is in flight, otherwise it is
    deferred — without the in-flight flag, parallel tool outputs both see an idle connection after
    `response.done` and the second `response.create` is rejected as "conversation already has an
    active response";
  - `response.created` clears in flight and sets ongoing; `response.done` clears ongoing and
    flushes one deferred create;
  - each create carries `event_id: "agents_rs_response_create_{n}"`, and only an error whose
    `event_id` matches that id (or, with no `event_id`, whose code is `bad_response_create` or
    whose message mentions `response.create`) releases the in-flight create. Python matches *any*
    error carrying an `event_id`, which drops a deferred create when an unrelated failure such as
    a barge-in truncate error arrives; this port is deliberately stricter.
  - a send failure also releases the in-flight slot, so a transient socket error cannot wedge the
    session.
- **Response-generation guardrail branching.** Python scopes guardrail interrupts by
  `output_response_generation` and `cancel_response_only`/`playback_only`. The port keeps the
  interrupted-response-id de-duplication and sends a plain forced interrupt, which matches the
  single-response case that `RealtimeModelSendInterrupt` models here.
- **Invocation-identity and data-redaction machinery** (`_tool_identity`, `_tool_invocation`,
  `_clear_data_redacted_error_traceback`, duplicate-call-id detection). None of it exists in the
  Rust core; approvals are keyed by `(agent, tool, call_id)` through `RunContext` as elsewhere in
  this crate.
- **`send_event_if`** (conditional commit at the transport boundary), **`RealtimeStep`/
  `ScriptedRealtimeModel` strict scripting** (the fake model records sends and emits on demand
  instead), **audio-storage history config** (a `TODO` upstream) and **`_debug` log redaction**.
- **Hosted tools on realtime agents.** `RealtimeAgent::function_tools` silently skips non-function
  tools instead of raising, because the realtime API only accepts function tools.
- **Tool-guardrail `RaiseException`** emits a `RealtimeSessionEvent::Error` instead of aborting the
  session: a realtime session has no caller to raise into. `RejectContent` behaves as in Python.

## Behavior worth calling out

- **`RealtimeToolExecutionConfig::pre_approval_tool_input_guardrails` is implemented.** With the
  flag on, a tool that needs approval first runs its input guardrails; a `RejectContent` verdict
  answers the model and emits `tool_end` without ever asking a human, and the same guardrails run
  again before the tool executes after approval. Covered by
  `pre_approval_input_guardrails_block_a_call_before_a_human_sees_it`.
- **Barge-in parity (`input_audio_buffer.speech_started`).** The transport emits
  `audio_interrupted`, truncates at `floor(elapsed_ms)` clamped to the audio actually received
  (never past it), skips the truncate when playback has caught up and no response is still
  producing audio, resets its audio tracker, and sends `response.cancel` when the negotiated
  session does not cancel automatically (`turn_detection.interrupt_response != true`).

## Requests for core changes (not made here)

1. **`Cargo.toml`: add an example entry.**

   ```toml
   [[example]]
   name = "realtime_text"
   required-features = ["realtime"]
   ```

   Worked around for now by gating the example body behind `#[cfg(feature = "realtime")]` with a
   fallback `main` that prints how to enable the feature, so `cargo check --all-targets` passes
   without the feature. The entry would let the fallback be deleted.

2. **`src/lib.rs`: re-export the realtime surface** alongside the other modules, e.g.
   `pub use realtime::{RealtimeAgent, RealtimeRunner, RealtimeSession};` under
   `#[cfg(feature = "realtime")]`. Not required — `openai_agents_sdk::realtime::*` already works.

3. **No `pub`/`pub(crate)` visibility changes were needed.** `crate::util::BoxFuture`,
   `crate::models::default_openai_key()` and the rest were reachable from inside the crate.

4. **Optional cleanup, unrelated to this module:** `cargo doc` warns about the `[`mcp`]`
   intra-doc link in `src/lib.rs` when the `mcp` feature is off.

## Test coverage

`tests/realtime_tests.rs` (11 tests, all event waits wrapped in `tokio::time::timeout`, no sleeps,
no network beyond `127.0.0.1`, dummy key `sk-test`):

1. `tool_call_runs_end_to_end_and_sends_function_call_output` — tool advertised in the connect
   settings, `tool_start` before `tool_end`, `function_call_output` sent with
   `start_response: true`, call recorded in the history.
2. `handoff_switches_agent_and_sends_session_update` — agent switches, `session.update` with the
   new agent's instructions and tools is sent *before* the tool output, transfer message matches
   `{"assistant":"Billing"}`.
3. `tool_approval_is_requested_then_approved_and_rejected` — nothing is sent while a call waits,
   approval runs the tool, re-approving is a no-op, rejection returns the custom message without
   starting the tool.
4. `transcript_deltas_build_history_and_trip_output_guardrails` — debounce threshold, tripwire,
   forced interrupt plus the `guardrail triggered: <names>` message, transcript in the history.
5. `history_and_audio_events_are_forwarded` — initial empty `history_updated`, `history_added`,
   `audio`, `audio_end`, `input_audio_timeout_triggered`, and stream termination after `close()`.
6. `update_agent_pushes_new_settings`.
7. `dynamic_instructions_and_run_config_settings_reach_the_model`.
8. `unknown_tool_reports_an_error_to_the_model` — error event plus `start_response: false`.
8b. `pre_approval_input_guardrails_block_a_call_before_a_human_sees_it` — a rejected call never
   reaches `tool_approval_required` or `tool_start`, while an allowed one still does.
9. `websocket_model_speaks_the_ga_protocol_against_a_mock_server` — `accept_hdr_async` on
   `127.0.0.1:0` captures the request URI and `Authorization: Bearer sk-test`; asserts the full
   `session.update` payload (`type`, `model`, `output_modalities`, `instructions`, GA
   `audio.input.format`/`transcription`/`turn_detection`, `audio.output.voice`, `tools`); checks
   that a server `response.output_audio_transcript.delta` is forwarded raw and mapped to
   `TranscriptDelta`; checks that `session.created` triggers the tracing `session.update`.
10. `custom_headers_replace_the_bearer_token` — supplying headers suppresses `Authorization`.

Plus 23 unit tests in `src/realtime/**`, covering the response-create state machine (deferral,
flush on `response.done`, unrelated errors keeping a deferred create, code/message matching) and
truncation clamping, as well as the default model name and URL
(`wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1`), the session payload defaults, audio
format mapping, usage conversion, GA message parsing, history merging and transcript
preservation, settings merge semantics, duplicate tool-name rejection and audio length math.
