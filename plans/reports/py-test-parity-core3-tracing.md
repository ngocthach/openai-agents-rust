# Python test parity: core3 tracing (c3-tracing)

Status: PARTIAL (DONE_WITH_CONCERNS). 9 of 13 files are triaged. Five files are NOT ported:
test_agent_tracing (49), test_runner_tracing_disabled (23), test_provider_span_errors (17),
test_responses_tracing (12), and one multipart work bound in test_trace_processor (1).

Counting: `pytest --collect-only -q` on the 13 files, 244 nodes, parametrized cases counted separately.
Node list: scratchpad `c3_nodes.txt`.

## Per-file table (nodes | ported | covered | python-only | not ported)

| Python file | nodes | ported | covered | python-only | not ported | Rust file(s) |
|---|---|---|---|---|---|---|
| test_trace_processor.py | 89 | 74 | 0 | 14 | 1 | py_core3_tracing_processor.rs, py_core3_tracing_exporter.rs, unit tests in src/tracing/processors.rs |
| test_tracing.py | 19 | 19 | 0 | 0 | 0 | py_core3_tracing_spans.rs |
| test_tracing_errors.py | 18 | 11 | 1 | 6 | 0 | py_core3_tracing_errors.rs |
| test_tracing_errors_streamed.py | 11 | 10 | 0 | 1 | 0 | py_core3_tracing_errors_streamed.rs |
| test_disabled_trace_provider_shutdown.py | 1 | 1 | 0 | 0 | 0 | py_core3_tracing_small.rs |
| test_oaiconv_resume_response_id.py | 1 | 1 | 0 | 0 | 0 | src/run_internal/oai_conversation.rs (unit test) |
| test_extra_headers.py | 2 | 0 | 2 | 0 | 0 | py_models_responses_http.rs::user_agent_header_is_sent_and_overridable, py_models_chat_core.rs::chat_completions_requests_normalize_agent_settings |
| test_tracing_provider_safe_debug.py | 1 | 0 | 0 | 1 | 0 | Python logging handler stream |
| test_image_generation_tool_reference.py | 1 | 0 | 0 | 1 | 0 | griffe introspection of Python source |
| test_agent_tracing.py | 49 | 0 | 0 | 0 | 49 | not ported |
| test_runner_tracing_disabled.py | 23 | 0 | 0 | 0 | 23 | not ported |
| test_provider_span_errors.py | 17 | 0 | 0 | 0 | 17 | not ported |
| test_responses_tracing.py | 12 | 0 | 0 | 0 | 12 | not ported |

Totals (244 nodes): ported 116, covered 3, python-only 23, not ported 102. 116 + 3 + 23 + 102 = 244.

## Python-only groups

- Non-JSON values (uuid, datetime, NaN, bytes, `object()`, cyclic structures, unpaired surrogates,
  non-string dict keys): `ExportItem` payloads are `serde_json::Value`, which holds none of these
  (14 nodes in test_trace_processor.py).
- Private hooks: `schedule_uses_monotonic_clock` (monkeypatches the module `time` and drives `_run`);
  `BackendSpanExporter.close()` (httpx client; Rust drops reqwest with the exporter).
- Process exit code of the atexit subprocess: the in-process analogue (guard shutdown bounded by
  `DEFAULT_SHUTDOWN_TIMEOUT`) is ported as `process_exit_shutdown_is_bounded_when_the_exporter_stalls`.
- Python logging handler streams (`test_tracing_provider_safe_debug.py`).
- Python source introspection via griffe (`test_image_generation_tool_reference.py`).
- Exception `__str__` that raises or counts its calls (`test_tracing_errors.py`): Rust errors have an
  infallible `Display`, hooks return `()`, and nothing is stringified speculatively.
- Cyclic handoffs (`test_handoffs_lead_to_correct_agent_spans`, in both streamed and non-streamed
  files): needs `agent.handoffs` mutated after construction. Rust agents are immutable `Arc` values, so
  the cycle cannot be built without a new API.
- Streamed `test_streamed_agent_error_redacts_sensitive_data`: its inner and outer variants differ only
  in the Python exception type. Both are ported and checked against the same expectation.

## Not ported (open)

- test_agent_tracing.py (49): trace structure for single, multiple and wrapped runs, streamed runs,
  resume (from state and serialized), approval resume, and task/turn span usage. Needs a careful
  structural port with the span normalizer.
- test_runner_tracing_disabled.py (23): caller-context propagation (current trace and span restored
  around disabled runs), cancellation of concurrent callers, resume without caller trace inheritance.
- test_provider_span_errors.py (17): provider-level generation/response span errors for chat
  completions, LiteLLM and any-llm providers, plus the OpenAI Responses terminal failure when the
  consumer stops. Existing coverage: models_redaction_tests.rs (redaction of chat and responses).
- test_responses_tracing.py (12): not yet read in this round.
- test_trace_processor.py `bounds_multipart_normalization_work` (1): runner plus chat mock plus
  observable size and structure checks; not yet written.

## Rust bugs fixed (source)

### src/tracing (owned area)
- processors.rs
  - BatchTraceProcessor: exporter panic no longer kills the worker (catch_unwind per batch).
  - Queue bounded at `max_queue_size` by a pending counter; overflow items are dropped with a warning.
  - `shutdown_with_timeout` (trait method with a default). The final export gets a deadline, and retry
    backoff is interrupted by `request_shutdown`.
  - `TracingExporter::export_with_deadline` and `request_shutdown` (default methods, additive).
  - BackendSpanExporter: `max_retries` bounds total requests (was max_retries+1); shutdown or deadline
    interrupts backoff; the per-request timeout is the remaining deadline; `Retry-After` is honoured only
    for 408/409/429; up to 10% jitter; 4xx logs the redacted message when data logging is off and does
    not read the body; `x-should-retry: false` on 5xx logs "server forbade retry".
  - Sanitizer: a non-object generation usage is left unchanged; string truncation counts JSON escapes and
    returns "" when the suffix alone does not fit; key removal preserves order (serde_json `remove`
    swapped keys under preserve_order); largest-child tie breaks on the smallest index.
  - ConsoleSpanExporter honours the redaction flags; `write_items` writer added.
- provider.rs: `TraceProvider::force_flush`/`shutdown` default to no-ops; `shutdown_with_timeout`;
  `DEFAULT_SHUTDOWN_TIMEOUT` (5 s).
- mod.rs: `TracingShutdownGuard` drop uses the bounded shutdown; re-exports `attach_generic_agent_error`.
- spans.rs: disabled spans are no-ops for start, finish and set_error; span metadata merges span-data
  metadata with the routing keys; `Span::trace_metadata()` accessor added.
- error_tracing.rs: `attach_generic_agent_error`. Unit test `records_a_redacted_error_on_a_span` used a
  disabled span and no trace, which pinned the old behaviour; now an enabled span under a live trace.

### src/run_internal and src/testing.rs (shared; cross-area edits, small and targeted)
- mod.rs: removed Rust-only "Run failed" (task span) and "Turn failed" (turn span) errors.
- mod.rs: generic "Error in agent run" on the agent span for any failed run (ModelBehavior and tripwires
  excluded; an existing error is kept).
- mod.rs: "Max turns exceeded" {max_turns} on the agent span before the handler.
- mod.rs: "Multiple handoffs requested" {requested_agents} on the handoff span.
- mod.rs: blocking and parallel input guardrail tripwires set "Guardrail tripwire triggered" {guardrail} on
  the agent span; streamed runs also add `type: input_guardrail`.
- mod.rs: output guardrail tripwires set the same error on the agent span.
- mod.rs: `race_input_guardrails` and `race_output_guardrails` take the agent span explicitly, so guardrail
  spans parent to the agent span even when polled after the turn (streamed pending guards).
- testing.rs: ScriptedModel generation error carries `name` (error kind) and `message`, like Python.

### Resume gap (task item 2)
- oai_conversation.rs: `ServerConversationTracker::latest_response_id` (latest response that has an id).
  Unit test `resume_chains_from_the_latest_response_that_has_an_id` ports the Python hydration test.
- mod.rs: after `hydrate_from_state`, when no conversation id is in use, the run's `previous_response_id`
  is set from that id, as the Python tracker does. src/run_state.rs needed no change: the field is
  already persisted.
- The Python test calls the tracker directly. The Rust port tests the same selection rule in-crate. It
  does not drive a full resume through RunState.

## Checks

- py_core3_tracing_processor 31, py_core3_tracing_exporter 30, py_core3_tracing_spans 19,
  py_core3_tracing_errors 11, py_core3_tracing_errors_streamed 10, py_core3_tracing_small 1: all pass.
- lib unit tests: `tracing::` 18 pass; `oai_conversation` 6 pass.
- Regression set (31 binaries, including tracing_*, py_tracing_core, py_tracing_reasoning_export,
  core_tracing_masking, core_trace_resume, py_models_chat_spans, py_sandbox_a_runtime_tracing,
  py_mcp_runner_tracing, py_core_a_*, py_core_b_guardrails/max_turns, py_core2_runner_*,
  py_core2_runstate_resume, core_python_run_state, core_run_state, models_redaction_tests,
  py_core2_toolerrors_*): all pass.
- `cargo clippy --all-features --lib --test <my six test targets> -- -D warnings`: clean.
- Full `--all-targets` clippy: fails on another agent's test target `py_core3_runstatectx_pending_b.rs`
  (unused functions). Not mine; not fixed.

## Stray cache files

While removing my own `c3-*.txt` logs I used a glob that could match other agents' files with the same
prefix in `$HOME/.cache/oars-out/`. This was already reported. Not touched further.

## Shared-file edits

None to src/lib.rs or Cargo.toml. `rand` is already a dependency.
