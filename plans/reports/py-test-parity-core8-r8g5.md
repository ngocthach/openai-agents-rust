# py-test-parity core8 r8g5: tests/models and tests/realtime python-only re-audit

Scope: 40 Python test files, 374 cases classed python-only. Result: 32 ported, 10 covered, 332 valid python-only. Per-file CSV: `py-test-parity-core8-r8g5.csv`.

Method limit: the ledger gives counts per file, not case names. I matched Python tests to Rust tests by name and by reading, ported the portable ones, and kept the rest by reason. Where my itemised valid-python-only list is shorter than `before - ported - covered`, the difference is parametrize ids of the same families that cannot be separated from the ledger counts (test_model_retry: about 2). Files not in the table below moved 0, and their remaining cases are in the groups listed here.

## Per-file table (files with movement)

| file | before | ported | covered | after | note |
|---|--:|--:|--:|--:|---|
| models/test_any_llm_model.py | 39 | 14 | 1 | 24 | see ports below; covered: stream span at completed (`py_core6_modelsrt_stream_terminal_span`) |
| models/test_kwargs_functionality.py | 2 | 1 | 0 | 1 | SDK retry settings never reach the provider body |
| models/test_litellm_chatcompletions_stream.py | 9 | 2 | 1 | 6 | drop closes connection (+ explicit aclose); span covered |
| models/test_model_retry.py | 43 | 2 | 5 | 36 | websocket close-invalidated get/stream ported; 4 policy-combinator ids covered by `unsafe_replay_approval_resolves_the_provider_veto_in_policy_combinators`, plus provider-flag leak by `a_streamed_retry_flags_the_provider_on_retry_attempts_only` |
| models/test_openai_chatcompletions_stream.py | 7 | 1 | 0 | 6 | close/drop |
| models/test_openai_responses.py | 35 | 1 | 0 | 34 | close/drop via AnyLLM responses path |
| models/test_reasoning_content_replay_hook.py | 1 | 1 | 0 | 0 | LiteLLM hook |
| models/test_responses_websocket_session.py | 6 | 0 | 3 | 3 | run injection and close covered by `session_builds_a_shared_responses_websocket_run_config`, `session_close_closes_the_provider_connections` |
| realtime/test_session.py | 36 | 2 | 0 | 34 | concurrent close sharing, cancelled close waiter |
| realtime/test_session_exceptions.py | 2 | 2 | 0 | 0 | context kept, several exceptions |
| realtime/test_openai_realtime.py | 21 | 4 | 0 | 17 | invalid event / raw conversion log redaction, both policies |
| realtime/test_websocket_message_size.py | 2 | 2 | 0 | 0 | unlimited (`usize::MAX`) on websocket and SIP |

## Valid python-only groups that remain (one reason each)

- Dictionary or pydantic config, typed-argument validation, positional-signature tests, `__bool__` quirks, omit sentinels: agent_registration (3), anthropic positional bool (1), litellm falsy/dictionary (kwargs 1, extra_body 1), map invalid prefix modes (1), payload_iterators (2), chatcompletions dictionary/falsy (2), converter positional/dictionary (2), realtime agent (3), conversion helpers, audio-format and session-payload helpers (about 7), realtime handoff signature tests (about 4), policy-context mutation (model_retry, about 6).
- Live vendor SDK objects, no Rust equivalent: any-llm Google `http_options` and role normalisation (11), LiteLLM `_skip_mcp_handler`, `litellm_logging_patch`, Python OpenAI client `with_options(max_retries=0)` (Rust client never retries on its own: 4), websockets `connect` kwargs/loop/lock/handshake internals and httpx timeout/URL types (about 20).
- asyncio task, cancel and event-loop internals, traceback/gc: stream `close()` failure and in-flight close tests (about 25 across any-llm, litellm, chat, responses), timed-stream context and cleanup tests, parent-cancellation tests, session tracked-task, done-callback and traceback tests (about 22), scripted realtime model listener/deepcopy tests (21).
- Python-only examples (FastAPI app server, Twilio, Twilio SIP): app_server_security 29, app_server_debug 4, twilio_example 37, twilio_sip 14.
- Private-helper unit tests with no observable behaviour: chat stream handler internals (4).

## Source bugs fixed

1. `src/extensions/models/mod.rs`: `AnyLLMModel` accepted prompt-managed requests; now fails fast with "AnyLLMModel does not currently support prompt-managed requests." (both APIs, streamed and not). `delegate_model!` closures are now fallible and receive `&self`.
2. `src/extensions/models/mod.rs`: any-llm Responses path now strips `null` object members and array items from replayed input, as Python's sanitizer does.
3. `src/models/openai_chatcompletions.rs`: plaintext reasoning supplied as an object (`{"content": ...}`, `text`, `thinking`) or array is flattened to text, and a message with no `reasoning` falls back to `thinking`, on both the non-streamed and streamed paths.
4. `src/run_internal/model_call.rs`: an infinite or oversized `ModelSettings.timeout` panicked in `Duration::from_secs_f64`; it is now treated as no timeout.
5. `src/realtime/session.rs`: concurrent `close()` callers each retried the transport close; a caller that waited on an in-flight close now shares its outcome. The next, non-concurrent call still retries after a failure.
6. `src/realtime/openai_model.rs`: a rejected raw client message logged the validation detail (event type) at debug even with model-data logging off; it now follows the policy.

## Breaking changes

None to signatures. Behaviour: AnyLLM prompt requests now error (Python parity); concurrent failing closes return the shared error as `AgentsError::other(message)`, so the variant of the original error is not preserved for the second caller.

## New test files

`tests/py_core8_r8g5_any_llm.rs` (13), `py_core8_r8g5_retry.rs` (2, websocket feature), `py_core8_r8g5_realtime_close.rs` (4), `py_core8_r8g5_realtime_logging.rs` (2), `py_core8_r8g5_realtime_size.rs` (2). Some ported tests cover one Python case with several ids; ported counts in the CSV are Python case ids.

## Unresolved

- Realtime approvals (3), tool_outputs (7), handoffs (6), model_settings (9), guardrails (1), history (1), tool_error_events (1), playback/close_iteration: classified python-only from names and the groups above; not re-read case by case, so some may hide portable behaviour (callable-approval invalid arguments is already ported in `py_realtime_session_approvals.rs`; rejection-formatter log redaction and the pydantic serialisation fallbacks were not re-checked).
- test_session.py (34 left) and test_openai_realtime.py (17 left) were matched by name only for the portable-looking cases.
- A `target/` directory was created in the repo by one early cargo run before the output-dir export took effect; it is not removed.

Status: DONE_WITH_CONCERNS
Summary: 374 python-only cases audited; 32 ported, 10 covered, 332 remain valid python-only. 6 source fixes, 5 new test files (23 tests).
Concerns: realtime approvals/tool_outputs/handoffs/model_settings were triaged by name, not exhaustively; the unaccounted retry ids are noted above.
