# Python test parity: realtime, voice, codex

Area: `realtime-voice-codex`. Build dir `tp-rtv` (deleted at end).

## Scope and counting

- Python `tests/realtime/` (31 files, 521 tests), `tests/voice/` (15 files, 166 tests), `tests/extensions/experiemental/codex/` (3 files, 132 tests; directory misspelled upstream). Total 819.
- Counted with `grep -o "def test_"` per file, one row per match. Parametrize was **not** expanded; the 521/166/132 split matches the task numbers.
- Every count below is derived from the ledgers (one row per Python test with bucket, Rust reference and note). Ledgers are in the scratchpad `tp/` dir (`ledger-realtime-final.tsv`, `ledger-voice.tsv`, `ledger-codex.tsv`).
- Final totals: **819 rows, 0 TODO.**

| Area | Rows | PORTED | COVERED | PYTHON-ONLY |
|---|---|---|---|---|
| realtime | 521 | 353 | 23 | 145 |
| voice | 166 | 94 | 18 | 54 |
| codex | 132 | 54 | 33 | 45 |
| **Total** | **819** | **501** | **74** | **244** |

Per-area detail reports (written by the sub-agents, each with its own per-file table):
`py-test-parity-realtime-openai.md`, `py-test-parity-realtime-session.md`, `py-test-parity-realtime-testing.md`, `py-test-parity-voice.md`, `py-test-parity-codex.md`.

## Per-file table, realtime (521)

| file | # | ported | covered | python-only |
|---|---|---|---|---|
| test_openai_realtime.py | 120 | 96 | 2 | 22 |
| test_session.py | 85 | 51 | 7 | 27 |
| test_testing.py | 38 | 20 | 0 | 18 |
| test_session_guardrails.py | 27 | 23 | 2 | 2 |
| test_conversion_helpers.py | 24 | 20 | 0 | 4 |
| test_session_approvals.py | 24 | 18 | 4 | 2 |
| test_session_history.py | 22 | 20 | 1 | 1 |
| test_realtime_handoffs.py | 19 | 13 | 0 | 6 |
| test_openai_realtime_conversions.py | 18 | 15 | 1 | 2 |
| test_realtime_model_settings.py | 16 | 6 | 1 | 9 |
| test_session_tool_outputs.py | 13 | 7 | 0 | 6 |
| test_twilio_example.py | 17 | 0 | 0 | 17 |
| test_app_server_security.py | 10 | 0 | 0 | 10 |
| test_runner.py | 8 | 8 | 0 | 0 |
| test_audio_formats_unit.py | 8 | 7 | 0 | 1 |
| test_session_exceptions.py | 7 | 5 | 0 | 2 |
| test_session_tool_output_guardrails.py | 6 | 6 | 0 | 0 |
| test_agent.py | 6 | 4 | 0 | 2 |
| test_model_events.py | 6 | 6 | 0 | 0 |
| test_tracing.py | 6 | 4 | 2 | 0 |
| test_twilio_sip_server.py | 6 | 0 | 0 | 6 |
| test_item_parsing.py | 5 | 5 | 0 | 0 |
| test_session_payload_and_formats.py | 4 | 0 | 0 | 4 |
| test_app_server_debug.py | 3 | 0 | 0 | 3 |
| test_session_close_iteration.py | 3 | 3 | 0 | 0 |
| test_openai_realtime_sip_model.py | 2 | 0 | 2 | 0 |
| test_playback_tracker.py | 14 | 12 | 1 | 1 |
| test_playback_tracker_manual_unit.py | 1 | 1 | 0 | 0 |
| test_session_tool_error_events.py | 1 | 1 | 0 | 0 |
| test_ga_session_update_normalization.py | 1 | 1 | 0 | 0 |
| test_websocket_message_size.py | 1 | 1 | 0 | 0 |
| **Total** | **521** | **353** | **23** | **145** |

Voice (166) and codex (132) per-file tables are in their area reports.

## Test files added

- Realtime: `tests/py_realtime_units.rs`, `py_realtime_transport.rs`, `py_realtime_openai_{connect,events,interrupt,sequencing,env}.rs`, `py_realtime_session_{core,dispatch,approvals,guardrails,history,settings,tool_outputs}.rs`, `py_realtime_testing.rs`, `py_realtime_handoffs.rs`. Shared mock-websocket harness in `tests/py_realtime_support/mod.rs`.
- Voice: `tests/py_voice_misc.rs`, `py_voice_pipeline.rs`, `py_voice_pipeline_tracing.rs`, `py_voice_openai_models.rs`.
- Codex: `tests/py_codex_exec.rs`, `py_codex_tool.rs`, `py_codex_tool_spans.rs`, fixture `tests/py_codex_support/fake_codex.sh`.

## Rust bugs fixed (src)

Realtime:
- `src/realtime/openai_model.rs`: `conversation.item.*` now defaults status to `in_progress` (Python `item.status or "in_progress"`). Concurrent `connect()` could open two sockets; a late `response.done` could free the active response slot; raw `response.create` bypassed sequencing; a dropped send future left the create/cancel slot set forever; abnormal close codes ended the stream silently; malformed audio deltas and non-JSON frames were fatal; a failed initial `session.update` left the model marked connected; error wording aligned with Python.
- `src/realtime/session.rs`: `EndOfStream` now ends the event stream (recv hung before); dispatch rechecks tool and handoff `is_enabled` at call time; `close()` retries after a failed model close, serializes concurrent callers, and no longer aborts the caller's own task; empty tool call id rejected before dispatch; approval ordering (standing "always" decision checked before the dynamic rule and rechecked after it and after pre-approval guardrails); an output guardrail on both agent and run config ran twice (now deduped by name).
- `src/realtime/testing.rs`: unexpected-send error no longer leaks payload (redacted); duplicate connect message; connect events plus connect error rejected at connect time; step and unconsumed-step wording; added public `is_connected()` / `is_closed()` (additive).
- `src/realtime/model_inputs.rs`: interrupt validation messages.

Voice (`src/voice/models/openai_stt.rs`):
- Non-JSON websocket frame is now a session error (was silently skipped).
- Server close while draining the final transcription now reports the same error as a dropped socket.
- Handshake now follows Python's URL building (client websocket base URL and existing query preserved).
- Handshake now runs through `client.websocket_headers`, so a rotating API key is refreshed.
- Session-creation, session-update and inactivity timeouts configurable via three new `VoiceWebSocketOptions` fields with the same defaults. **Breaking for struct-literal construction: add `..Default::default()`.**
- First turn's transcription span now opens at connect time, so a session closed right after setup still records it.
- The f64 timeout constants are converted with `Duration::from_secs_f64` (fixed a compile break left by an interrupted helper).

Codex (`src/extensions/codex/**`):
- `tool.rs`: blank `thread_id` rejected with Python's message (was silently `None`).
- `tool.rs`: argument parse errors no longer echo the value when tool-data logging is redacted.
- `mod.rs`: removed a doc claim that the duplicate-Codex-tool check is not performed (it is).

## Decisions applied after review

1. **Responses-only tool rejection on the OpenAI realtime transport (closes the two ignored tests).**
   `RealtimeFunctionToolSpec` (`src/realtime/config.rs`) gained `namespace: Option<String>` and
   `defer_loading: bool` (both `serde(default)`, skipped when empty, so existing serialized specs still
   load), builders `with_namespace` / `with_defer_loading`, `responses_only_features()` and
   `ensure_realtime_supported()`. `build_model_settings_from_agent` (`session.rs`) now carries the tool's
   namespace and `defer_loading` into the spec. The OpenAI transport (`openai_model.rs`,
   `ensure_tools_supported`) rejects such a spec in `send_session_update` (used by connect and by session
   updates) and in `OpenAIRealtimeSIPModel::build_initial_session_payload`, with Python's message
   ("The following function-tool features are only supported with OpenAI Responses models: ... Tool `x`
   cannot be used with Realtime models."). The rejection is deliberately **not** in
   `RealtimeAgent::get_all_tools` or the session layer: Python's session and approval tests run
   namespaced tools on realtime agents against a non-OpenAI model and connect fine, and
   `py_realtime_session_approvals` (20 tests) stays green. Both tests are un-ignored and pass; a
   connect-path test (`connect_rejects_namespaced_and_deferred_tool_specs`) was added.
   **Breaking change (accepted):** code building `RealtimeFunctionToolSpec` with a struct literal must
   add the two fields or use `RealtimeFunctionToolSpec::new`.
2. **`VoiceWebSocketOptions` timeout fields** accepted as a breaking change for struct-literal users
   (add `..Default::default()`).
3. **Tool failure wording.** A failed tool task now publishes Python's fixed `"Tool call task failed"`
   error event (`session.rs`, `end_with_tool_task_error`), logs the detail through the redaction-aware
   tool-data logger, and keeps the real error for `RealtimeSession::take_error`. The transport
   `Exception` path still carries its detail, as in Python. Updated the tests that pinned the old text:
   `tests/realtime_tests.rs::a_tool_guardrail_tripwire_ends_the_session`,
   `py_realtime_session_history.rs`, `py_realtime_session_core.rs`, `py_realtime_session_dispatch.rs`
   (the event now asserts the fixed text and the detail is asserted through `take_error`).

## Open items and deferred

Realtime (no decision pending; the three items previously raised are closed, see below):
- Sticky approvals keyed by agent name: two distinct agents with the same name share decisions.
- Guardrail dedupe by name, not identity (exact dedupe needs an identity accessor in `src/guardrail.rs`, another area).
- Duplicate tool-name message in `session.rs::validate_tool_names` differs from Python's wording.

Voice:
- `OpenAIClient` cannot omit a default query parameter, so one assertion in `test_streaming_stt_websocket_url_merges_client_default_query` is ported only as "client default overrides the base-URL parameter".

Codex:
- A custom schema built from `codex_tool_params_schema(true)` marks `thread_id` required, unlike Python; a call omitting it fails. The ported test sends `"thread_id": null`. Schema contract left unchanged.

Cross-area bugs found by the sub-agents, not fixed here: none others recorded.

## Gates

- Realtime: `realtime_tests` 41, `py_realtime_transport` 15, `py_realtime_units` 23, `py_realtime_testing` 28, `py_realtime_handoffs` 13, `py_realtime_session_*` 145, `py_realtime_openai_*` 129, 0 ignored. Lib realtime tests 47 pass.
- Voice: `py_voice_misc` 31, `py_voice_openai_models` 38, `py_voice_pipeline` 22, `py_voice_pipeline_tracing` 3, `voice_tests` 32, pass.
- Codex: `py_codex_exec` 16, `py_codex_tool` 32, `py_codex_tool_spans` 5, `codex_tests` 20, pass; lib codex/voice/realtime 136 pass.
- Clippy: `cargo clippy --all-features --all-targets --keep-going -- -D warnings` is clean for every realtime, voice and codex file (library and tests). The workspace as a whole still fails in other areas' test files (`py_tracing_core`, `py_mcp_*`, `py_sandbox_a_*`, `py_models_retry_*`, `py_deferred_writes_c`, `py_tracing_reasoning_export`), which were not touched. Using `--keep-going` mattered: without it cargo stopped at the first failing target and hid 11 diagnostics in this area's tests (unused imports, `.err().expect()`, large `Err` in handshake callbacks), now fixed; `py_codex_tool_spans.rs` also needed an async `tokio::sync::Mutex` because its guard is held across awaits.
- The library was repeatedly non-compiling for minutes at a time because other agents were editing core files concurrently (`ToolApprovalItem.tool_origin`, `persist_skip_registry`, `pending_session_write`, ...). All 26 test binaries of this area and the 136 lib tests were run to green once the lib stabilized.
- rustfmt `--edition 2024` run on our own test files.

## Shared-file edits

None. `src/lib.rs` and `Cargo.toml` are unchanged.

## Rule deviations to note

- No `#[ignore]` remains in any file of this area.
- One sub-agent ran a read-only `git diff --stat` despite the no-git rule; nothing was modified by it.
- `py_voice_openai_models.rs` (~1650 lines) and `py_realtime_session_guardrails.rs` (~1520 lines) exceed the ~1500-line target.

Status: DONE
Summary: All 819 Python tests in scope are classified with no TODO left (501 ported, 74 covered, 244 python-only) and every ported test runs and passes, with no ignored tests. About 25 real Rust bugs were fixed, plus the three review decisions (Responses-only tool rejection on the OpenAI realtime transport, `VoiceWebSocketOptions` fields, Python's fixed tool-failure message).
Concerns/Blockers: Two accepted breaking changes for struct-literal users (`RealtimeFunctionToolSpec`, `VoiceWebSocketOptions`). Workspace-wide clippy `-D warnings` still fails in other areas' test files, none in realtime, voice or codex. Remaining documented deviations: sticky approvals keyed by agent name, guardrail dedupe by name, one voice query-parameter assertion ported partially, and a codex schema nuance (`thread_id` required in a custom schema).
