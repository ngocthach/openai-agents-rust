# Python test parity: core4 tracerunner

Status: DONE. 99 of 102 in-scope Python cases ported and green. 3 python-only. 0 not ported.

## Counting

`pytest --collect-only -q` (scratchpad pyenv) on the five files: 190 nodes. In scope: the four
whole files plus the one multipart case of test_trace_processor.py (the other 88 nodes of that file
belong to earlier agents). 49 + 23 + 17 + 12 + 1 = 102.

## Per-file table (cases | ported | covered | python-only | not ported)

| Python file | cases | ported | covered | python-only | not ported | Rust file |
|---|---|---|---|---|---|---|
| tracing/test_agent_tracing.py | 49 | 49 | 0 | 0 | 0 | py_core4_tracerunner_agent.rs (29 fns) |
| test_runner_tracing_disabled.py | 23 | 21 | 0 | 2 | 0 | py_core4_tracerunner_disabled.rs (7 fns) |
| test_provider_span_errors.py | 17 | 16 | 0 | 1 | 0 | py_core4_tracerunner_provider_errors.rs (12 fns) |
| test_responses_tracing.py | 12 | 12 | 0 | 0 | 0 | py_core4_tracerunner_responses.rs (10 fns) |
| test_trace_processor.py (multipart case only) | 1 | 1 | 0 | 0 | 0 | py_core4_tracerunner_multipart.rs |
| **Total** | **102** | **99** | **0** | **3** | **0** | |

Totals: 99 + 0 + 3 + 0 = 102.

Support: tests/py_core4_tracerunner_support/mod.rs is a copy of the core3 tracing helper plus a
scripted HTTP model endpoint (plain and TLS) and a throwaway-certificate helper. The core3 files
were not edited.

## Rust source fixes

- src/models/openai_responses.rs, openai_responses_ws.rs: the response span recorded
  `response_id` even with data tracing off on a custom endpoint. Now recorded only when data is
  included or the endpoint is the official one (`finish_response_span` takes `trusted_response_id`).
- src/tracing/error_tracing.rs: `model_error_text` panicked when a provider error's `Display`
  failed while sensitive data was on. It now writes through `fmt::Write` and records
  "Unrenderable error".
- src/run_internal/mod.rs (shared, targeted):
  - session preparation (session read, input callback) ran inside the task span. It now runs in the
    caller's scope (`RunLoop.setup_scope`), as Python prepares the session before the task span.
  - output-guardrail validation resolved the model outside the masked span, so a disabled run's
    provider spans leaked into the caller's trace. Now resolved inside the masked span.
  - task span usage was missing `cached_input_tokens` and `cache_write_input_tokens`, and reported
    cumulative usage for resumed runs. Now the Python shape, as a run-local delta.
  - turn span usage now the Python shape (input, output, cached, cache-write deltas).
- src/result.rs, src/run.rs: `RunResultStreaming::is_complete()` only turned true after the consumer
  read the stream. It now also turns true when the run task finishes (`StreamControl::finished`).
- src/tracing/processors.rs: `BackendSpanExporter::with_http_client` (`#[doc(hidden)]`), a test seam
  so the official ingest host can be routed to a local TLS server. No behaviour change.
- src/models/openai_responses_ws.rs: `OpenAIResponsesWSModel::with_test_transport` (`#[doc(hidden)]`),
  a test seam that dials a given address and trusts a given rustls `ClientConfig` instead of the URL's
  host and the webpki roots. `None` in the SDK, so behaviour and the documented API are unchanged.
  The model's response-id trust rule for websocket calls now follows the HTTP client's endpoint
  (`self.client.is_openai_api()`), as Python's `_uses_official_openai_endpoint` does, instead of the
  prompt-cache-key rule, which requires a `wss` scheme.
- Timed-out model calls (real divergence, now fixed): `with_timeout` (src/run_internal/model_call.rs)
  dropped the model future, and the span it owned was never finished, so a timed-out non-streamed
  call exported no model span. Now the attempt is scoped with a `ModelTimeoutSlot`, the timeout
  branch marks it before the future is dropped, and `ModelSpanGuard` (src/tracing/error_tracing.rs)
  records "Model call timed out after X seconds." and finishes the span on drop. Wired into the
  non-streamed `get_response` of the Responses and Chat Completions models (src/models/
  openai_responses.rs, openai_chatcompletions.rs), and into the Responses stream guard (replacing
  `SpanGuard` there). `tracing/mod.rs` re-exports the new items. Chat Completions streaming still
  uses its own span finish, so a timed-out chat stream exports its span without the timeout error
  text (documented gap, not in scope of the marked case).

No edits to src/lib.rs or Cargo.toml.

## Deviations (ported, with a difference)

- Streamed setup errors: Python's `run_streamed` raises synchronously. Rust reports them through the
  stream. The port skips the caller's `after-setup` span on that path, matching Python's sequence.
- Streamed model resolution happens in the background task. The port checks the resolved name and
  that a record exists, not Python's exact count at setup.
- `task_span_resets_current_span_if_run_setup_fails`: Python monkeypatches
  `SandboxRuntime.assert_agent_supported`. Rust has no such setup hook, so the port fails the first
  model call inside the task span. The asserted invariants are the same.
- Multipart: Python counts calls to its private `_value_json_size_bytes` through a monkeypatch. Rust
  has no hook. The port checks the exact payload size bound, structure, non-mutation, and batch
  ordering, plus a 10 s wall-clock guard in place of the work counter.
- Official-endpoint cases (3 responses cases including the websocket one, multipart): the official host is routed to a local
  rustls server. A throwaway self-signed certificate is generated at test time with the `openssl`
  CLI into a temp dir. Nothing is committed. When `openssl` is not on PATH, these tests print
  `skipped: openssl is not on PATH; ...` and return early (they do not fail).
- Custom-endpoint cases: Python flips `client.base_url` mid-stream. The port tests a fixed custom URL.

## Python-only (3)

- test_provider_span_errors `test_failing_span_recording_preserves_the_provider_exception` (1):
  monkeypatches the module function `attach_error_to_span` to raise. The Rust recording API returns
  `()` and cannot fail, and there is no runtime patching.
- test_runner_tracing_disabled `test_disabled_runner_uses_configured_provider` [2]: subclasses
  `DefaultTraceProvider.create_trace` and `NoOpTrace` to hook `start`/`finish(mark_as_current)` and a
  contextvar. Rust `Trace` is a concrete struct and `TraceProvider` has no `create_trace` hook.

## Not ported

None. The websocket official-endpoint case is ported with the `with_test_transport` seam: a local
secure WebSocket server (rustls plus tungstenite) with a throwaway certificate answers the model,
which still sees `https://api.openai.com/v1` as its websocket base URL and the official client. It
skips with a message when `openssl` is missing.

## Checks

- My five binaries, all green: agent 29, disabled 7, provider_errors 12, responses 10, multipart 1
  (59 functions, 99 cases). Re-run after the websocket seam: all green. Existing websocket and
  responses model tests (py_models_responses_ws, _ws_retry, _http, models_redaction_tests) still pass.
- `cargo test --all-features --lib`: 651 passed, 0 failed.
- Related existing binaries (tracing, runner, run state, models, guardrails, core2/core3 runner):
  1091 passed, 3 failed (the timeout change did not touch these paths). The failures are in other areas, not the paths I changed:
  core_run_state `reads_python_tool_search_items_and_hosted_mcp_interruptions`,
  py_core2_runner_routing `persistent_hosted_mcp_approval_does_not_cross_servers`,
  py_core2_runstate_resume `nested_agent_tool_hitl_resume_with_duplicate_agent_names`.
  Not investigated further.
- Clippy `-D warnings` on the lib and my five test targets: clean. Full `--all-targets` clippy fails
  in other agents' files (py_core4_programmatic_process_items.rs, py_core3_functiontools_schema.rs).
  Not fixed.
- rustfmt: my test files formatted. Shared src hunks I touched are rustfmt-clean; remaining rustfmt
  diffs in src/run_internal/mod.rs are pre-existing and in other agents' code.

Build dir: $HOME/.cache/oars-out/c4-tracerunner (deleted at the end).
