# Realtime final parity report

Date: 2026-10-30
Area: `src/realtime/**`, `tests/realtime_tests.rs`, realtime parts of `src/usage.rs`.
Python reference: `/Users/ngocthach/myGit/openai-agents-python/src/agents/realtime/**`.

## Re-verification of the assigned rows

Every row named in the task was re-checked against the current tree first. Most had already
landed in phase 4 / commit `0a89b04`; those are listed as **already closed** with the evidence, and
no code was rewritten for them.

| Row | State | Evidence |
| --- | --- | --- |
| Nested audio config (`RealtimeAudioConfig` / `Input` / `Output`) + Python precedence over the flat legacy keys | already closed | `config.rs` `RealtimeAudioInputConfig` / `RealtimeAudioOutputConfig` / `RealtimeAudioConfig`; precedence applied in `openai_model.rs::session_payload_for_settings` (`audio.input.*` / `audio.output.*` `.or_else(flat)`); test `the_nested_audio_config_wins_over_the_flat_keys` |
| Transcription `keywords`, `languages`, `delay` | already closed | `RealtimeInputAudioTranscriptionConfig` fields; asserted on the wire by the same test |
| Turn-detection `model_version` | already closed | `RealtimeTurnDetectionConfig::model_version` |
| Legacy `modalities` fallback | already closed | `RealtimeSessionModelSettings::modalities`, used only when `output_modalities` is unset |
| `RealtimeCustomVoice` (`{id}`) | already closed | `RealtimeVoice::Custom { id }`, serialized as `{"id": ...}` |
| `RealtimeRunConfig.tool_error_formatter` | already closed, now **aliased** (below) | `config.rs`, consulted on the approval-rejection and tool-not-found paths in `session.rs::format_tool_error` |
| Usage modality details (input / output / cached token details) | already closed | `RealtimeModelCachedTokensDetails` / `RealtimeModelInputTokensDetails` / `RealtimeModelOutputTokensDetails` on `RealtimeModelEvent::Usage`, parsed in `openai_model.rs::convert_response_usage`; test `usage_events_carry_the_modality_breakdown` |
| `RealtimeResponse` type | already closed | `items.rs::RealtimeResponse` (`id`, `output`), exported. Unused in Python too — it is a public type only |
| `audio_formats.py` mapping branch incl. the non-24000 warn-and-coerce | already closed | `RealtimeAudioFormat::Object` branch in `config.rs`; `as_f64` accepts the int and float spellings and a missing `rate` coerces silently to 24000, as in Python |
| Round-tripping unknown server fields (`extra="allow"`) | **partly open → closed** | items already had `extra`; the five content-part types did not. Closed below |

### usage.rs

**No change made, and none is needed.** Python's shared `Usage` (`usage.py:197-218`) carries
`input_tokens_details` (`cached_tokens`, `cache_write_tokens`) and `output_tokens_details`
(`reasoning_tokens`) — exactly what `src/usage.rs` already has. The realtime modality breakdown
lives on `RealtimeModelUsageEvent`, not on `Usage`, in both SDKs.

## Rows closed in this pass

1. **`RealtimeToolErrorArgs` is now the core type.** Python's realtime config imports
   `ToolErrorFormatter` from `agents.run_config`, so the two are the same by construction.
   `RealtimeToolErrorArgs<'a, C>` is now a `pub type` alias of `ToolErrorFormatterArgs<'a, C>` and
   `RealtimeToolErrorKind` an alias of `ToolErrorKind`; `RealtimeToolErrorFormatter<C>` is an
   `Arc<dyn for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Option<String> + Send + Sync>`. One
   formatter now serves a `RunConfig` and a `RealtimeRunConfig`. The realtime paths only ever pass
   `ApprovalRejected` / `ToolNotFound` and `error: None`; `ToolException` comes from the core
   runner, so the kind enum is a superset, as in Python.
   Test: `the_tool_error_formatter_is_the_core_one`.

2. **`RealtimeRunConfig::async_tool_calls` (new field, default `true`).** Was absent from the crate
   (`grep async_tool_calls src/` → 0 hits) while Python defaults it to `True`
   (`session.py:261`). The session awaited every tool call inside the transport event loop, so a
   slow tool stalled audio, transcript and interrupt delivery. `handle_tool_call` is now split:
   the duplicate-call reservation (`admit_tool_call`) stays synchronous and in arrival order, and
   the work (`run_admitted_tool_call`) runs on its own task when the flag is on. Tasks are tracked
   and aborted by `close()`, like the guardrail tasks.
   Tests: `a_tool_call_runs_off_the_event_loop_by_default`,
   `async_tool_calls_off_serializes_a_call_with_the_event_stream`.

3. **`extra="allow"` on content parts.** `items.py` marks all five content classes
   (`InputText`, `InputAudio`, `InputImage`, `AssistantText`, `AssistantAudio`) `extra="allow"`;
   only the item types were ported. Each `RealtimeContent` variant now carries a flattened
   `extra: serde_json::Map`, and `openai_model.rs::parse_message_item` fills it from the wire part
   (shared helper `extra_fields`).
   Test: `unknown_content_part_fields_survive_a_round_trip`.

4. **A hosted tool on a realtime agent is now an error, not a silent skip.**
   `openai_realtime.py:1884` raises `UserError("Tool {name} is unsupported. Must be a function
   tool.")`. `RealtimeAgent::get_all_tools` now returns that same `UserError`, so the session fails
   to start instead of connecting with the tool missing. `function_tools()` keeps its filtering
   behavior and documents the difference.
   Test: `a_hosted_tool_on_a_realtime_agent_is_rejected`.

5. **`_stored_exception` now ends the session** (was listed "intentionally not ported"). Python
   stores the failure on a transport `exception` event (`session.py:643`) and on a failed tool-call
   task (`session.py:1984`), then closes and raises out of `__aiter__`. The port now does the
   Rust-idiomatic equivalent: `SessionInner::end_with_error` queues the matching
   `RealtimeSessionEvent::Error`, then drops the event sender, so `recv()` delivers every queued
   event and then returns `None`; that `None` closes the session. The failure itself is handed to
   the caller by the new `RealtimeSession::take_error() -> Option<AgentsError>`, which is where
   Python raises. Dropping the sender (rather than checking a flag in `recv`) makes the end of the
   stream deterministic instead of racing the pump.
   Test: `a_transport_exception_ends_the_session`.

6. **A tool guardrail tripwire now ends the session.** `session.py:828` / `:851` raise
   `ToolInputGuardrailTripwireTriggered` / `ToolOutputGuardrailTripwireTriggered` from inside the
   tool task, which Python stores and raises. The port emitted an `error` event and returned
   `Ok(())`, which was the right call when there was no way to end the session; with
   `end_with_error` in place the three raise sites now return the matching `AgentsError`, so the
   session ends exactly as in Python. Nothing had exercised `RaiseException` on a realtime session
   before.
   Test: `a_tool_guardrail_tripwire_ends_the_session`, which also covers the tool-task half of
   row 5 (the spawned task calling `end_with_error` while the pump may be mid-`emit`).

7. **`get_api_key` is public and exported**, matching Python's `realtime.__init__` export. It was
   the private `resolve_api_key`. It takes the crate's `RealtimeApiKey`, which generalizes Python's
   `str | Callable[[], MaybeAwaitable[str]] | None`. No new test: this is a visibility change on
   code the WebSocket connect tests already exercise.

## Rows that stay open, with the inherent reason

These supersede the "Intentionally not ported" bullets in `realtime-port-report.md` for hosted
tools and `_stored_exception`, which are now closed.

- **Python's delivery queue** (`_RealtimeDeliveryResult`, `pending_deliveries`, re-entrancy
  detection). It exists because Python's listeners run synchronously inside `send_event`; here a
  session event is a channel push, so a step's events cannot re-enter the send that produced them.
  Structurally absent, not deferred.
- **`_ResponseCreateSequencer` generation ordering.** The port keeps an equivalent I/O-free state
  machine on `WsState` and is deliberately *stricter* on one point: only an error whose `event_id`
  matches the in-flight create (or an untagged `bad_response_create` / `response.create` error)
  releases it, where Python releases on any error carrying an `event_id` and so drops a deferred
  create when an unrelated failure arrives. Verified by the `--lib` unit tests on the transition
  table; not reversed, per the verified-decision rule.
- **`RealtimePlaybackTracker::get_state` assertion.** Python asserts elapsed time is set once an
  item is current; the port returns `elapsed_ms: None`. A panic in a library accessor has no Rust
  equivalent worth porting.
- **`_tool_identity` / `_tool_invocation` internals** (lookup keys, namespaces, SHA-256 semantic
  fingerprints, hosted-MCP approval scopes) and the redacted-traceback plumbing. None of this
  exists in this crate's core, so it is not a realtime row — it belongs to the
  `src/tool.rs` / `src/run_context.rs` row of the parity audit. The observable duplicate-call and
  cached-output behavior is ported.
- **A duplicate or mismatched call id reported by `admit_tool_call` emits an `error` event and
  leaves the session running**, where Python's `ModelBehaviorError` would travel through
  `_handle_tool_call_task_failure` into `_stored_exception` and end it. The reservation is a
  deliberate, tested phase-3 behavior (`reusing_a_call_id_for_a_different_invocation_is_an_error`,
  `reusing_a_rejected_call_id_for_another_tool_is_an_error`): the model retrying a call id is a
  model-side mistake the session can answer and survive, not a transport failure. Only failures
  raised *after* admission end the session.
- **`approve_tool_call` always awaits the call**, even with `async_tool_calls` on, because it is
  called from application code rather than the transport event loop, and its `Result` is the
  caller's only channel for the failure. Python's `approve` enqueues instead. Documented on the
  method; the flag governs the event loop, which is what it protects.

## Breaking changes

- `RealtimeToolErrorArgs` gained a lifetime parameter, `run_context` became `&'a RunContext<C>`,
  and the struct gained the core `error: Option<&'a AgentsError>` field.
  `RealtimeToolErrorKind` is now `ToolErrorKind`, which has a third variant (`ToolException`), so a
  formatter matching on it needs a catch-all arm.
- `RealtimeRunConfig` gained `async_tool_calls` (it is not `#[non_exhaustive]`, so struct literals
  need `..Default::default()` or the new field). Tool calls now run off the event loop by default.
- `RealtimeContent`'s five variants gained an `extra` field, so struct-literal construction needs
  it; the `RealtimeContent::input_text` / `text` / `audio_transcript` constructors are unaffected.
- `RealtimeAgent::get_all_tools` (and therefore `RealtimeRunner::run`) now returns `Err` for a
  hosted tool instead of skipping it.
- A transport exception, a failed tool task, or a tool guardrail tripwire now ends the session:
  `recv()` returns `None` and `is_closed()` becomes true, where the session previously kept
  running. This includes a *single malformed server event*: `openai_model.rs` emits
  `RealtimeModelEvent::Exception` per undecodable message, and Python does the same
  (`_listen_for_messages` turns a per-message `ValidationError` into
  `RealtimeModelExceptionEvent(context="Failed to validate server event")`, which is stored and
  raised), so one bad frame ends the session in both SDKs.
- `ToolGuardrailBehavior::RaiseException` on a realtime tool's input or output guardrail now
  propagates as `AgentsError::Tool{Input,Output}GuardrailTripwireTriggered` instead of only
  emitting an `error` event.

## Shared-file edits

None. `src/lib.rs` and `Cargo.toml` were not touched. The new `get_api_key` export is on
`src/realtime/mod.rs`.

## Requests for other areas

- `src/run_config.rs`: `type ToolErrorFormatter<C>` (line 197) is private. `RealtimeToolErrorFormatter`
  spells the same HRTB out by hand instead. Making the core alias `pub` would let realtime
  `pub use` it; purely cosmetic.

## Verification

Green, with the realtime feature in isolation:

- `cargo test --all-features --test realtime_tests` — 40 passed (33 before this pass, 7 new),
  reached on a retry loop during a window when the rest of the tree compiled.
- `cargo test --no-default-features --features realtime --test realtime_tests` — 40 passed.
- `cargo test --no-default-features --features realtime --lib realtime` — 41 passed.
- `cargo clippy --no-default-features --features realtime --lib --test realtime_tests -- -D warnings`
  — clean (run before the guardrail change; every later retry, under both feature sets, failed
  only on `src/extensions/codex/**` and `tests/core_python_run_state.rs` and never reported a
  finding under `src/realtime/**` or `tests/realtime_tests.rs`).
- `cargo build --no-default-features --features realtime --example realtime_text` — clean.
- Files touched were formatted with `rustfmt --edition 2024`.

`--all-features` could **not** be run to completion: the shared tree was broken throughout this
pass by the agents working in `src/mcp/**`, `src/sandbox/**`, `src/models/**`,
`src/extensions/codex/**` and `tests/core_python_run_state.rs` (e.g. `ToolFilterContext` missing
`agent_any` / `run_context_any`, `CodexExecArgs` missing `cancel`, unresolved `mcp::handler` /
`mcp::stdio` / `mcp::http`, `OpenAIResponsesWebSocketOptions` absent, a dead `TRACE_ID` const).
None of those errors are in realtime files, and every typeck run that got that far reported no
error under `src/realtime/**` or `tests/realtime_tests.rs`. The `--all-features` gate should be
re-run once those areas land. `mcp_tools_reach_the_realtime_session` passes under the realtime
feature (its body is `#[cfg(feature = "mcp")]`, so the mcp half of it is still unverified here).

Status: DONE_WITH_CONCERNS
Summary: Re-verified every assigned row; ten were already closed by phase 4 / `0a89b04`, and seven
remained open and are now closed (core `ToolErrorFormatterArgs` alias, `async_tool_calls`,
`extra="allow"` on content parts, hosted-tool rejection, `_stored_exception` ending the session,
tool-guardrail tripwires ending the session, public `get_api_key`), with seven focused tests
added. `src/usage.rs` needed no change: Python's
shared `Usage` has the same details and the realtime modality breakdown lives on the model event.
Concerns/Blockers: `cargo test --all-features --test realtime_tests` is green (40 passed), but
`cargo clippy --all-features --all-targets -- -D warnings` could not be completed — `src/mcp/**`,
`src/sandbox/**`, `src/models/**`, `src/extensions/codex/**` and `tests/core_python_run_state.rs`
were broken by parallel work for most of the session. No clippy run ever reported a finding under
`src/realtime/**` or `tests/realtime_tests.rs`; re-run the full gate once those areas compile. Seven breaking changes on the realtime public API
are listed above.
