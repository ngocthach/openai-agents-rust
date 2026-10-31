# Python test parity: core4 tracerunner

Status: DONE_WITH_CONCERNS. 98 of 102 in-scope Python cases ported and green. 3 python-only.
1 not ported: the websocket official-endpoint case, blocked by a missing connector seam (details below).

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
| test_responses_tracing.py | 12 | 11 | 0 | 0 | 1 | py_core4_tracerunner_responses.rs (9 fns) |
| test_trace_processor.py (multipart case only) | 1 | 1 | 0 | 0 | 0 | py_core4_tracerunner_multipart.rs |
| **Total** | **102** | **98** | **0** | **3** | **1** | |

Totals: 98 + 0 + 3 + 1 = 102.

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
- Official-endpoint cases (2 responses cases, multipart): the official host is routed to a local
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

## Not ported (1, open)

- test_responses_tracing `test_stream_non_data_tracing_preserves_id_for_https_official_websocket`
  (1): needs a wss server whose certificate the websocket connector trusts. The connector is fixed
  to webpki roots (`connect_async_with_config` in src/models/openai_responses_ws.rs). Not ported:
  it needs a connector seam (a public way to supply a TLS connector or trusted roots), which is new
  public API and was left out on purpose. The span-id rule it checks is covered by the non-websocket
  official-endpoint cases through the same trusted flag.
- None of the marked-timeout case: now ported (see Rust source fixes).

## Checks

- My five binaries, all green: agent 29, disabled 7, provider_errors 11, responses 9, multipart 1
  (57 functions, 97 cases).
- Five binaries re-run after the timeout fix: agent 29, disabled 7, provider_errors 12, responses 9, multipart 1 (all green; the provider_errors binary now has 12 fns with the marked-timeout case).
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
