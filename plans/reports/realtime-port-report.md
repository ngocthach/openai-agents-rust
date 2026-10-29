# Realtime port report

Date: 2026-10-29
Scope: port of Python `agents.realtime` (openai-agents 0.24) into `openai_agents_sdk::realtime`
(feature `realtime`).

Phase 1 ported the agent, runner, session, events, items, handoffs, the OpenAI WebSocket transport
and the fake transport. Phase 2 closed the SIP / call-accept flow, the playback tracker, and
Python's generation-scoped guardrail interrupt branching. **Phase 3 closed the rest: strict step
scripting, history audio storage, debug log redaction, and the invocation-identity behavior that is
observable in realtime events and approvals.** All three phases are reflected below.

## Files

Created (all under the allowed scope):

| File | Lines | Contents |
| --- | --- | --- |
| `src/realtime/mod.rs` | 106 | Module docs, submodules, re-exports |
| `src/realtime/items.rs` | 461 | `RealtimeItem`, message/tool-call items, history merge helpers |
| `src/realtime/debug.rs` | 169 | Model/tool data redaction for realtime logs |
| `src/realtime/config.rs` | 492 | `RealtimeSessionModelSettings`, `RealtimeRunConfig`, turn detection, tracing, user input |
| `src/realtime/model_events.rs` | 216 | `RealtimeModelEvent` and payload types |
| `src/realtime/model_inputs.rs` | 224 | `RealtimeModelSendEvent`, `RealtimeInterrupt` |
| `src/realtime/model.rs` | 339 | `RealtimeModel` trait, `RealtimeModelConfig`, `RealtimePlaybackTracker`, audio length math |
| `src/realtime/agent.rs` | 217 | `RealtimeAgent<C>`, `RealtimeInstructions<C>` |
| `src/realtime/handoffs.rs` | 271 | `RealtimeHandoff<C>`, `realtime_handoff` |
| `src/realtime/events.rs` | 421 | `RealtimeSessionEvent<C>`, `RealtimeEventInfo<C>` |
| `src/realtime/session.rs` | 1779 | `RealtimeSession<C>`: history, tools, approvals, handoffs, guardrails |
| `src/realtime/openai_model.rs` | 2093 | `OpenAIRealtimeWebSocketModel`, `OpenAIRealtimeSIPModel`, `ModelAudioTracker` |
| `src/realtime/runner.rs` | 107 | `RealtimeRunner<C>` |
| `src/realtime/testing.rs` | 766 | `FakeRealtimeModel`, `ScriptedRealtimeModel` (port of `testing.py`) |
| `tests/realtime_tests.rs` | 1920 | 28 integration tests |
| `examples/realtime_text.rs` | 83 | Text-only realtime session example |

Nothing outside `src/realtime/**`, `tests/realtime_tests.rs` and `examples/realtime_text.rs` was
touched.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/realtime
cargo clippy --features realtime --all-targets -- -D warnings   # clean
cargo test --features realtime --test realtime_tests            # 28 passed
cargo test --features realtime --lib realtime                   # 39 passed
cargo test --features realtime --doc                            # 6 realtime doctests compile
cargo doc --features realtime --no-deps                         # no warnings in src/realtime
cargo check --all-targets                                       # clean without the feature
rustfmt --edition 2024 --check <all files above>                # clean
```

`cargo doc` still reports one pre-existing warning in `src/lib.rs` (`unresolved link to 'mcp'`,
because the `mcp` feature is off); it is not in this module.

## Phase 2: the three closed gaps

### SIP / call-accept flow

`OpenAIRealtimeSIPModel` attaches to a call the Realtime Calls API already created. It wraps
`OpenAIRealtimeWebSocketModel` by composition (Rust has no inheritance) and delegates every trait
method, overriding `connect` to require a non-empty `call_id`.

- With a `call_id` and no URL override the transport connects to
  `wss://api.openai.com/v1/realtime?call_id=<id>` instead of `?model=<name>`; the URL choice is a
  pure `connect_url` helper, unit-tested in `--lib`.
- A call has already negotiated its audio formats, so with a `call_id` set the session payload
  omits `audio.input.format` and `audio.output.format` unless the settings name one explicitly —
  the same carve-out Python makes in `_get_session_config`.
- `OpenAIRealtimeSIPModel::build_initial_session_payload(agent, context, model_config, run_config,
  overrides)` builds the accept-call body, applying run config, connect-time settings, the agent's
  instructions/tools/handoffs, and then `overrides`. It is a port of Python's
  `build_initial_session_payload`.
- The HTTP POST to `/v1/realtime/calls/{call_id}/accept` stays in application code, exactly as in
  Python, where it lives in the Twilio SIP example rather than the SDK. The `OpenAIRealtimeSIPModel`
  doc comment shows the two-step flow. Adding an HTTP surface here would duplicate the client and
  base-URL resolution the controller owns in `src/models/**`.

### Playback tracker

Both halves of Python's audio bookkeeping are now ported:

- `RealtimePlaybackTracker` (public, from `model.py`): `on_play_bytes`, `on_play_ms`,
  `on_interrupted`, `set_audio_format`, `audio_format`, `state() -> RealtimePlaybackState`, plus
  `shared()` for the `Arc` the config wants. It is interior-mutable so the application can report
  progress through a shared handle while the transport reads it. Supply it through
  `RealtimeModelConfig::playback_tracker`.
- `ModelAudioTracker` (private, from `_default_tracker.py`): per-`(item_id, content_index)` state
  holding the first arrival time, the accumulated audio length and the owning response, plus the
  last audio item and the per-response index lists. This replaces Phase 1's single-item tracker.
- The transport learns the negotiated output format from `session.created`/`session.updated` and
  sets it on both trackers, so `on_play_bytes` converts bytes with the right sample rate.
- `_get_playback_state`, `_ongoing_response_owns_audio`, `_get_audio_limits` and
  `_interrupt_audio_playback` are ported, including the re-read of the playback position when the
  first read falls outside the response being interrupted, and the skip of a truncate for audio the
  user has finished hearing that no running response still owns.
- `close()` resets the model's tracker and calls `on_interrupted()` on the application's, so a
  reused model never reports a closed session's item.

### Generation-scoped guardrail interrupts

`RealtimeModelSendEvent::Interrupt` now carries a `RealtimeInterrupt` with Python's four modes
(`force_response_cancel`, `response_id`, `cancel_response_only`, `playback_only`) and a `validate()`
that rejects the contradictory combinations Python raises `ValueError` for. The session tracks
`latest_output_response_generation` (an `AtomicU64`, so the send-boundary check needs no lock),
`active_output_response_generation`, `active_response_id` and `active_response_agent`, and ports
`_reconcile_output_response` so text and transcript deltas open a generation even when they arrive
before `turn_started`.

When a guardrail trips, the branch matches Python exactly:

| Situation | Action |
| --- | --- |
| No generation known | forced interrupt |
| Generation superseded by a newer response, audio output | `playback_only` for that response, no feedback message |
| Generation superseded, text output | nothing; the newer response is left alone |
| Generation is the active one | forced interrupt scoped to the response, `cancel_response_only` when the output was text |
| Generation is still the latest but no longer active (`turn_ended` cleared it, nothing newer started), audio output | `playback_only` for that response |

The `guardrail triggered: <names>` feedback goes out through the new
`RealtimeModel::send_event_if`, which rechecks the generation at the transport's write boundary —
the WebSocket model serializes the payload, rechecks, then writes, and declines when a
`response.create` is still pending. The default trait implementation sends nothing and returns
`false`, as Python's does.

Guardrail lifecycle is ported too: in-flight runs are counted per response
(`guardrail_runs_by_response`), `turn_ended` either releases the response's transport audio at once
or records it in `responses_awaiting_guardrail_cleanup`, and the last guardrail run to finish
triggers `RealtimeModel::retire_response_audio` (a new trait method, default no-op) plus the
`interrupted_response_ids` discard.

## Phase 3: the remaining gaps

### Strict step scripting (`ScriptedRealtimeModel`)

Alongside `FakeRealtimeModel`, which records and emits on demand, the port now has
`ScriptedRealtimeModel`: every outbound event must match the next `RealtimeStep`, which then
decides what the transport delivers back or how the send fails.

- `RealtimeSendMatcher` covers Python's three matcher forms: an exact event, a kind (the
  `RealtimeModelSendEvent::event_type` name, standing in for Python's class matcher) and a
  predicate. `From<RealtimeModelSendEvent>` and `From<&'static str>` keep steps terse.
- `RealtimeStep::new(expect).emitting([...])` / `.failing("...")`, with the same "cannot both emit
  and fail" rule Python enforces in `__post_init__`.
- `RealtimeScriptError::{UnexpectedSend, UnconsumedSteps}` with `Display`/`Error` and
  `From<RealtimeScriptError> for AgentsError`, plus `assert_complete()`, `remaining_steps()`,
  `sent_events()`, `close_calls()`, `emit()` and `lenient()` for Python's `strict=False`.
- `RealtimeConnectCall` is the credential-free connect snapshot: `api_key_provided`,
  `headers_provided`, a URL stripped of its query string and userinfo, the initial settings, the
  call id, and whether a playback tracker was supplied. The URL sanitizer ports
  `sanitize_url_for_trace`.
- Note for test authors: connect-time settings travel with `connect`, not as a `session.update`
  send, so the first scripted step is the session's first *real* send. The snapshot's
  `initial_model_settings` is where to assert them.
- **Not ported:** Python's delivery queue (`_RealtimeDeliveryResult`, `pending_deliveries`,
  re-entrancy detection). That machinery exists because Python's listeners run synchronously inside
  `send_event`; here emitting is a channel push, so a step's events cannot re-enter the send that
  produced them.

### History audio storage

Python exposes **no** API here — `config.py` and `session.py` carry only
`TODO (rm) Add history audio storage config`. There was nothing to port, so this is the port's
implementation of that intent, marked as such in the doc comment:
`RealtimeRunConfig::history_audio_storage` takes `RealtimeAudioStorage::Keep` (the default, which is
what Python does today) or `DiscardAudio`, which drops the base64 bytes from history items while
keeping transcripts and every other field. It is applied where the transport's items enter the
history.

### Debug log redaction

The crate had no equivalent of `agents._debug`, and `src/**` outside `src/realtime` is not in scope,
so the switch lives in `realtime::debug`: `dont_log_model_data()` / `dont_log_tool_data()`, both
defaulting to `true` as in Python, read from `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` /
`OPENAI_AGENTS_DONT_LOG_TOOL_DATA`, with `set_*` overrides that beat the environment and `reset_*`
to go back to it.

Internal helpers mirror `logger.py`'s redacting wrappers (`log_model_action_error`,
`log_model_action_warning`, `log_model_data_debug`, `log_tool_action_error`,
`log_tool_action_warning`): with redaction on they log the message alone, otherwise the message and
its detail. They are applied at every realtime site whose detail is model or tool data — server
payload parse failures, undecodable audio deltas, unknown audio formats, guardrail errors, tool
failures and tool output send failures. **Event payloads are never redacted**, since
`RealtimeSessionEvent` is the session's API rather than its log; Python likewise puts the detail in
events.

If the crate later grows a shared `_debug` equivalent, this module should delegate to it; see the
requests below.

### Invocation identity, as far as it is observable

The full `_tool_identity` / `_tool_invocation` machinery (lookup keys, namespaces, SHA-256 semantic
fingerprints, hosted-MCP approval scopes) has no counterpart in this crate's core and is not
observable through realtime events. What *is* observable is ported, with identity reduced to the
fields the fingerprint covers for a realtime function call — name, dispatch role and arguments —
compared directly instead of hashed:

- **A redelivered call runs once.** A second `function_call` for a call id that is dispatching,
  awaiting approval, or already answered is dropped. Approving or rejecting a pending call reserves
  its id under the same lock that takes it off the pending list and releases it on every exit path,
  so a redelivery arriving while an approved tool is still running is ignored rather than running
  the tool twice or drawing a spurious "already executed" error.
- **A reused call id is an error.** The same id with a different tool, role or arguments raises
  `ModelBehaviorError("Model reused a Realtime tool call ID for a different invocation...")`,
  surfaced as a `RealtimeSessionEvent::Error`. An unresolved role matches anything, so a call that
  arrives before its tool is enabled is not mistaken for a different invocation. The route is
  recorded at admission and kept for the session's life — Python's `_tool_invocation_routes` — so a
  call answered *without* running a tool (rejected, guardrail-blocked or not found) still owns its
  id.
- **An executed call whose output never committed cannot be retried.** It raises
  `"A Realtime tool call already executed, but its output was not committed..."`.
- **An undelivered output is cached and resent.** Function tool outputs, handoff transfer messages
  and "tool not found" replies all go through one `commit_tool_output` path: on a send failure the
  output is kept under its call id, the session emits
  `"Tool output send failed; cached output will be retried"`, and a retry of that call id resends
  it instead of running the tool again. A handoff's cached entry carries the `session.update` it
  depends on, which is resent before the output, so the new agent's settings still land first.

Handoffs are tracked the same way, so a handoff call id cannot later be reused for a function tool.

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
trait RealtimeModel {
    connect(config) -> RealtimeModelEvents;
    send_event(..);
    send_event_if(event, condition) -> bool;   // default: sends nothing, returns false
    retire_response_audio(response_id);        // default: no-op
    close();
}
RealtimeModelConfig, RealtimeModelEvents (= mpsc::UnboundedReceiver<RealtimeModelEvent>)
RealtimeModelEvent, RealtimeModelSendEvent, RealtimeInterrupt, RealtimeModelToolCallEvent,
RealtimeModelAudioEvent, RealtimeConnectionStatus, RealtimeSendCondition
RealtimePlaybackTracker, RealtimePlaybackState
OpenAIRealtimeWebSocketModel, OpenAIRealtimeSIPModel
FakeRealtimeModel, fake_realtime_model(), ScriptedRealtimeModel, RealtimeStep,
RealtimeSendMatcher, RealtimeScriptError, RealtimeConnectCall

// logging
realtime::debug::{dont_log_model_data, dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data, reset_dont_log_model_data, reset_dont_log_tool_data}

// data
RealtimeItem, RealtimeMessageItem, RealtimeToolCallItem, RealtimeContent, RealtimeMessageRole
RealtimeSessionModelSettings, RealtimeRunConfig<C>, RealtimeFunctionToolSpec,
RealtimeTurnDetectionConfig, RealtimeInputAudioTranscriptionConfig,
RealtimeNoiseReductionConfig, RealtimeModelTracing(+Config), RealtimeGuardrailsSettings,
RealtimeToolExecutionConfig, RealtimeUserInput(+Content), RealtimeAudioStorage
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
   characters. The handles live in `SessionInner::guardrail_tasks`; `close()` aborts them and
   clears the response bookkeeping (`guardrail_runs_by_response`,
   `responses_awaiting_guardrail_cleanup` and the active generation), mirroring
   `_clear_response_bookkeeping`. The
   "already interrupted" bookkeeping uses a `HashSet::insert` return value, so concurrent
   guardrail runs on one response still emit `guardrail_tripped` exactly once.
7. **Tri-state settings.** `turn_detection` and `input_audio_noise_reduction` are
   `Option<Option<T>>` so "unset" (use the transport default) stays distinct from "explicitly
   disabled" (send JSON `null`), which is what Python expresses with `"key" in settings` plus a
   `None` value.
8. **Clone/Debug are hand-written** for every type generic over `C`, following `src/agent.rs`, so
   no `C: Clone`/`C: Debug` bound leaks into the public API.

## Intentionally not ported

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
- **The internals of `_tool_identity` / `_tool_invocation`** (lookup keys, namespaces, SHA-256
  semantic fingerprints, hosted-MCP approval scopes) and Python's redacted-traceback plumbing
  (`_clear_data_redacted_error_traceback`, `_detach_data_redacted_error_traceback`). None of it
  exists in this crate's core, and none of it is observable through realtime events; the observable
  duplicate-call and cached-output behavior *is* ported, see **Phase 3** above. Approvals stay
  keyed by `(agent, tool, call_id)` through `RunContext`, as elsewhere in this crate.
- Nothing from `testing.py`, the audio-storage TODO or `_debug` remains outstanding; see
  **Phase 3** above. `send_event_if` is ported too, as the generation-scoped feedback needs it.
- **`RealtimePlaybackTracker` assertion semantics.** Python's `get_state` asserts that elapsed time
  is set once an item is current; the port returns `elapsed_ms: None` instead of panicking.
- **Hosted tools on realtime agents.** `RealtimeAgent::function_tools` silently skips non-function
  tools instead of raising, because the realtime API only accepts function tools.
- **Python's `_stored_exception` does not end the session here.** In Python a failed tool-call task,
  a transport `exception` event, or a tool-guardrail `RaiseException` sets `_stored_exception`, and
  the next iteration of `__aiter__` closes the session and raises it at the caller. This port emits
  a `RealtimeSessionEvent::Error` (or, for the transport, a `RawModelEvent` carrying
  `RealtimeModelEvent::Exception`) and keeps the session running, because `recv()` returns
  `Option<Event>` and has nowhere to raise. Applications decide whether an `Error` is fatal and call
  `close()` themselves. `RejectContent` behaves as in Python.

## Behavior worth calling out

- **`RealtimeToolExecutionConfig::pre_approval_tool_input_guardrails` is implemented.** With the
  flag on, a tool that needs approval first runs its input guardrails; a `RejectContent` verdict
  answers the model and emits `tool_end` without ever asking a human, and the same guardrails run
  again before the tool executes after approval. Covered by
  `pre_approval_input_guardrails_block_a_call_before_a_human_sees_it`.
- **Barge-in (`input_audio_buffer.speech_started`) is its own path**, as in Python, not a reuse of
  the requested-interrupt path. It reports `audio_interrupted` for the *tracker's last audio item*,
  prefers the server's `audio_end_ms` over a local estimate when that value is above zero, rounds
  rather than floors the truncate point, clamps it to the audio actually received, skips the
  truncate when playback has caught up and no running response still owns that audio, resets both
  trackers, and cancels the response when the negotiated session does not cancel automatically
  (`turn_detection.interrupt_response != true`).
- **`response.cancel` is gated like Python's `begin_cancel_response`.** A cancel is only sent when
  a response is actually running, no cancel is already outstanding, and the target response is the
  active one; a failed send releases the slot. Without this, a barge-in landing after a response
  ended, a duplicate interrupt, or a cancel aimed at a superseded response would each draw a
  server error.

## Requests for core changes (not made here)

1. **`Cargo.toml` example entry — done by the controller.** `[[example]] name = "realtime_text"`
   with `required-features = ["realtime"]` is present, so the example's `cfg` fallback `main` was
   removed.

2. **`src/lib.rs`: optionally re-export the realtime surface**, e.g.
   `pub use realtime::{RealtimeAgent, RealtimeRunner, RealtimeSession};` under
   `#[cfg(feature = "realtime")]`. Not required — `openai_agents_sdk::realtime::*` already works.

3. **No `pub`/`pub(crate)` visibility changes were needed.** `crate::util::BoxFuture`,
   `crate::models::default_openai_key()` and the rest were reachable from inside the crate.

4. **A crate-wide `_debug` equivalent would be worth having.** `realtime::debug` is scoped to this
   module because `src/**` outside `src/realtime` is off limits, but
   `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` / `OPENAI_AGENTS_DONT_LOG_TOOL_DATA` are crate-wide concerns
   in Python. If a shared module appears, `realtime::debug` should delegate to it and keep its
   current names as re-exports.

5. **Optional cleanup, unrelated to this module:** `cargo doc` warns about the `[`mcp`]`
   intra-doc link in `src/lib.rs` when the `mcp` feature is off.

## Test coverage

`tests/realtime_tests.rs` (28 tests, all event waits wrapped in `tokio::time::timeout`, no sleeps,
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

9. `a_guardrail_on_text_output_cancels_the_response_without_touching_audio` — a text-output
   guardrail on the active response sends `cancel_response_only`, leaving audio untouched.
10. `a_guardrail_on_a_superseded_response_only_stops_that_response_audio` — a guardrail gated on a
   semaphore so a newer response starts first; asserts `playback_only` for the old response, that
   the newer response is *not* told about the older guardrail, and that the response's transport
   audio is retired only once the guardrail settles.
11. `a_response_with_no_guardrails_releases_its_audio_immediately` — the other lifecycle branch.
12. `the_sip_model_builds_an_accept_payload_from_the_agent` — instructions, tools, turn detection,
   `overrides` winning, and run-config base settings.
13. `the_sip_model_requires_a_call_id`.
14. `the_sip_model_attaches_to_a_call_and_forwards_events` — mock server; asserts the attach
   payload omits the formats the call already negotiated and that server events still map.
15. `the_playback_tracker_decides_where_audio_is_truncated` — mock server negotiates `audio/pcm`
   and sends 100 ms of audio; the test reports 30 ms played through the tracker and asserts the
   resulting `conversation.item.truncate` carries `audio_end_ms: 30` (not a wall-clock value), that
   the format reached the tracker, and that the interruption reset it.

16. `a_duplicate_tool_call_id_runs_the_tool_only_once` — a redelivered `function_call` is dropped;
   the tool body counts its own runs.
17. `reusing_a_call_id_for_a_different_invocation_is_an_error` — the same id with a different tool
   raises the reuse error and sends no second output.
18. `an_undelivered_tool_output_is_cached_and_resent` — a scripted transport fails the first tool
   output; the session reports it, then a retry of that call id resends the cached output without
   starting the tool again.
19. `the_scripted_model_enforces_the_expected_exchange` — a predicate step, the credential-free
   connect snapshot, `assert_complete`, and the refusal of an unscripted send.
20. `the_scripted_model_reports_unconsumed_steps` — `UnconsumedSteps` and the "expected an event of
   kind" message.
21. `discarding_history_audio_keeps_transcripts` — both `RealtimeAudioStorage` modes over the same
   item.
22. `log_redaction_overrides_beat_the_environment` — the override round-trips and the reset falls
   back to the environment.
23. `a_call_redelivered_while_its_approved_tool_runs_is_ignored` — the approved tool is gated on a
   semaphore; the redelivery produces no `Error` and the tool body runs once.
24. `reusing_a_rejected_call_id_for_another_tool_is_an_error` — an id answered without running a
   tool still owns its route.
25. `an_undelivered_handoff_output_resends_its_session_update` — a scripted transport fails the
   handoff's `session.update`; the agent still switches, and the retry resends the update before
   the output.

Plus 39 unit tests in `src/realtime/**`, covering URL sanitizing, the send matchers, the
emit-and-fail rule, the redaction flags, audio stripping, the cancel gate (not running, mismatched response
id, duplicate cancel, released after a failed send, allowed again for the next response), the
playback tracker's accumulation and byte conversion, the `ModelAudioTracker`'s per-response indexes and their selective release, audio
ownership across responses, playback position with and without an application tracker, the
`call_id` URL branch, the call carve-out for audio formats, `RealtimeInterrupt` validation, covering the response-create state machine (deferral,
flush on `response.done`, unrelated errors keeping a deferred create, code/message matching) and
truncation clamping, as well as the default model name and URL
(`wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1`), the session payload defaults, audio
format mapping, usage conversion, GA message parsing, history merging and transcript
preservation, settings merge semantics, duplicate tool-name rejection and audio length math.
