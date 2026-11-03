# py-test-parity core18 r18b

Work list: 7 files, 170 python-only cases (ledger counts). Method: Python test names matched against Rust `fn` names and the Rust files' "Python: `test_...`" citations; every name-unmatched body was read and decided against the latest report's remaining-group description for that file (r16b for `realtime/test_session.py`, r16f for voice/mongo, r15c/r16e for testing/codex, r16a for scripted model).

| file | python_only before | ported | covered | remaining | read depth | reason per remaining group |
|---|---|---|---|---|---|---|
| tests/realtime/test_app_server_security.py | 29 | 0 | 0 | 29 | full | every case drives `examples/realtime/app/server.py` (FastAPI demo) through `TestClient` and monkeypatches its module globals; Python-only example app (Rust has no such example) |
| tests/realtime/test_session.py | 22 | 2 | 2 | 18 | full for the 25 python-only candidates; heads plus Rust counterparts for the rest | asyncio tracked/cleanup-task and `__aiter__` cancellation (aiter cancel, close-sets-closing, tracked task re-entering cleanup, in-flight/raw-event monkeypatched `_put_event`), private bookkeeping (`_put_event_nowait` x2, response bookkeeping after failed close), traceback frame locals / `__cause__` (6; the `str(error)` redaction half is already asserted by `py_core2_toolerrors_logging::realtime_handoff_input_validation_error_follows_mixed_data_policy`), `AsyncMock` returning `123` from a handoff, cross-group enablement cancellation and dispatch-stage failures injected into private resolvers (Rust `IsEnabled::evaluate` returns `bool`, so an enablement callback cannot fail) |
| tests/realtime/test_testing.py | 21 | 0 | 0 | 21 | full | `ScriptedRealtimeModel` listener-callback semantics (startup delivery, broadcast quiescing, reentrant delivery, listener errors/cancellation, delivery worker); Rust transports deliver through an mpsc channel. Deep-copy/mutation isolation of configured steps and matcher arguments (Rust moves owned values; matchers get `&event`); raising matchers / `__deepcopy__` failures |
| tests/voice/test_pipeline.py | 26 | 0 | 2 | 24 | full | asyncio cancellation and `aclose()` of the result generator (11 cases), private `_tasks`/`_set_task`/`_dispatcher_event` wiring, `__bool__`-falsy exception, dict config coercion and settings subclasses (5), numpy dtype spellings (8), test-support `QueuedSTTModel` queue sharing, segment-buffer retention probe (refcount; the observable "no audio in spans" half is covered by `voice_tests.rs`) |
| tests/extensions/experiemental/codex/test_codex_tool.py | 25 | 0 | 0 | 25 | full | `use_run_context_thread_id` setattr/mapping key derivation on frozen/slots/pydantic/mapping contexts and key-suffix validation (Rust replaces it with the explicit `ThreadIdStore` closures, `src/extensions/codex/tool.rs` `with_thread_id_store`), dict/kwargs option coercion and `inspect.signature` parity, circular list value, lazy resolver identity, `dataclasses.replace`, BaseException/cancel from `on_stream` (Rust `on_stream` returns `()`) |
| tests/extensions/memory/test_mongodb_session.py | 21 | 0 | 0 | 21 | full | non-string session ids (8, typed `String`), `close()` lifecycle group (see GAP), asyncio cancellation waits (4), caller-supplied `driver` kwarg (Rust `from_uri` takes no client kwargs; the URI cannot carry driver info), serialization failure via subclassed `_serialize_item` (Rust items are JSON values) |
| tests/test_scripted_model.py | 26 | 0 | 0 | 26 | full | 14 unreferenced names (plus params): `__deepcopy__` snapshot failure, dict step keys, mutable aliasing at queue time / in emitted events (Rust moves and clones), `aclose()` failures of exact streams, `__str__`/`BaseException` formatting, monkeypatched span `set_error`, `contextvars` current-span scope |
| **total** | 170 | 2 | 4 | 164 | | |

## Ported (tests/py_core18_r18b_realtime_session.rs, feature `realtime`)

1. `test_on_guardrail_task_done_emits_error_event` -> `a_failed_guardrail_interrupt_is_reported_as_a_guardrail_task_failure`: a tripped output guardrail whose interrupt send fails is reported as `Guardrail task failed: interrupt transport down`, sends no feedback, and the session keeps running.
2. `test_realtime_tool_contexts_share_session_tool_state` -> `tool_guardrails_and_tools_see_the_session_approval_state`: after `session.approve_tool_call`, the tool input guardrail's and the tool's `ToolContext::run_context()` and `session.context()` all report the approval. This is the approvals half of the Python test; the `_tool_invocations` half has no public Rust read API (invocation records are crate-private).

## Covered (moved, verified by reading)

- `realtime/test_session.py::test_on_tool_call_task_done_emits_error_event_immediately[tool failed, send failed]` -> `tests/py_realtime_session_history.rs::async_tool_failure_emits_an_error_event_and_ends_the_session` ("Tool call task failed") and `::tool_output_send_failure_is_retried_without_rerunning_the_tool` ("Tool output send failed; cached output will be retried").
- `voice/test_pipeline.py::test_speech_span_redacts_tts_instructions[True, False]` -> `tests/voice_tests.rs::speech_spans_are_grouped_per_turn_and_honor_the_privacy_flags` (instructions string/null and input by flag) and `::session_creation_and_tts_settings_are_recorded` (TTS receives the real instructions).

## Source fix

`src/realtime/session.rs` `run_output_guardrails`: a failed interrupt or feedback send after a tripwire emitted the bare transport error. Python lets it escape the guardrail task and `_on_guardrail_task_done` reports `Guardrail task failed: <error>`; Rust now uses that message for both sites. Behaviour change: the text of that error event only. No public API change.

## GAPs found, not fixed (decision needed)

1. `MongoDBSession` (and every Rust memory session) has no `close()` / terminal closed state: Python's close group (owned client closed, external client left open, closed operations raise `MongoDBSession is closed`, failed/cancelled close retryable, ping on closed) has no Rust surface. "API absent" is not a valid python-only reason on its own; adding `close()` is a cross-backend design decision (r16b kept the advanced_sqlite close cases python-only for the same reason), so it is recorded here rather than added to one backend.
2. Rust `IsEnabled` callbacks are infallible, so Python's "enablement raised" paths (cross-group cancellation, dispatch failure keeps the call id) cannot occur; API difference, noted only.

## Not moved although a Rust test exists

`codex/test_codex_tool.py::test_codex_tool_persisted_thread_mismatch_raises` (asserted by `src/extensions/codex/tool.rs::tests::persisted_sessions_reuse_one_thread`) and `test_codex_tool_result_stringifies` (`serializes_the_tool_result_like_python_json_dumps`): the r15c remaining-group list accounts for all 25 python-only cases without them, so they are most likely already counted as covered. Same for `voice/test_pipeline.py::test_voicepipeline_failing_close_after_a_clean_run_reaches_the_consumer` (`voice_tests.rs::a_scripted_session_close_failure_reaches_the_pipeline_and_close_is_idempotent`).

## Gates

`cargo test --all-features --test py_core18_r18b_realtime_session --test py_realtime_session_guardrails` (2 + 34 passed), `py_realtime_session_core`, `py_realtime_session_history`, `realtime_tests` green; default-feature build of the new file compiles (0 tests, gated). `cargo clippy --lib --test py_core18_r18b_realtime_session -D warnings` clean with `--all-features` and default features. `rustfmt --edition 2024` on the new file only. No edits under `src/sandbox/**` or `src/run_internal/**`.

## Read-depth caveats

- Bucket membership is reconstructed: the ledger holds counts, not case names. For `realtime/test_session.py` the python-only set reconstructed from r16b's description is about 25 collected cases against a ledger count of 22, so the 18 remaining is the ledger count minus this round's moves, not a recount.
- For `realtime/test_session.py` cases that r16b already matched to named Rust tests (event transformation, timeouts, handoffs, settings precedence, call-id checks), only heads were re-read and the Rust counterpart names were checked; their bodies were not compared line by line.
- Rust-side `clippy --all-targets` was not run over the whole workspace (only lib + this test binary), so other agents' in-flight test files are not checked here.

Status: DONE_WITH_CONCERNS
Summary: 170 python-only cases across 7 files read; 2 ported (tests/py_core18_r18b_realtime_session.rs), 4 covered, 164 remain. One source fix in src/realtime/session.rs (guardrail-task failure message).
Concerns: MongoDBSession (and all Rust memory sessions) lack close()/terminal state, so Python's close group has no Rust surface and "API absent" is not a valid python-only reason on its own; needs a cross-backend decision. IsEnabled callbacks are infallible, so Python's enablement-raised paths cannot be reached.
