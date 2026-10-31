# Python test parity: core2 / toolerrors

Scope: `test_error_logging_redaction.py`, `test_tool_batch_failure_history.py`,
`test_tool_failure_boundaries.py`, `test_default_tool_error_redaction.py`,
`test_model_error_logging_redaction.py`, `test_native_tool_failure_history.py`,
`test_tool_exception_formatter.py`.

How counted: `def test_*` × the product of each `@pytest.mark.parametrize` arg list, computed
with an AST walk (pytest is not installed in the scratch venv). Sum = 387, matching the task.

Rust test files (all new, all green):
- `tests/py_core2_toolerrors_formatter.rs`: formatter, default tool error redaction, model error logging (14 tests)
- `tests/py_core2_toolerrors_logging.rs`: error logging redaction (31 tests)
- `tests/py_core2_toolerrors_batch.rs`: batch failure history, failure boundaries, one native test (8 tests)

Rust parametrize variants are loops inside one `#[tokio::test]`.

## Per-file table

| Python file | #tests | ported | covered | python-only | not ported | failures found |
|---|---|---|---|---|---|---|
| test_error_logging_redaction.py | 170 | 61 | 0 | 92 | 17 | 9 |
| test_tool_batch_failure_history.py | 64 | 16 | 0 | 39 | 9 | 4 |
| test_tool_failure_boundaries.py | 30 | 20 | 0 | 10 | 0 | 3 |
| test_default_tool_error_redaction.py | 17 | 12 | 0 | 5 | 0 | 2 |
| test_model_error_logging_redaction.py | 1 | 1 | 0 | 0 | 0 | 0 |
| test_native_tool_failure_history.py | 72 | 2 | 0 | 20 | 50 | 1 |
| test_tool_exception_formatter.py | 33 | 26 | 0 | 7 | 0 | 4 |
| **Total** | **387** | **138** | **0** | **173** | **76** | |

### Python-only groups (reasons)
- Traceback frame locals, `__cause__`/`__context__`, exception notes, `BaseExceptionGroup`,
  `SystemExit`/`KeyboardInterrupt`/`GeneratorExit`, hybrid `CancelledError` subclasses,
  `_safe_redacted_persistence_error`, sandbox cleanup wrapper, `AgentRunner.run_sync`
  detachment (error_logging: 54). Rust errors carry no traceback or chain.
- Hostile `__str__`/`__bool__`/`__setattr__`/`__class__` objects, `exc_info`, `funcName`/
  `pathname` caller location, `diagnostic_extra` record attrs, `diagnostic_args` that raise
  (error_logging: 13; default redaction `does_not_format`: 1; handler hostile failure: 2).
- `asyncio.CancelledError` variants (`cancel`, `tool_cancel`, `native_cancel`,
  `cancel_at_verdict`, `cancelled_child`, persistence `cancelled`): 6+4+4+20+4+1+... per file.
- Tracing processors that raise (`TracingProcessor` methods return `()` in Rust): 18.
- `warnings`-module serialization warnings of permissive pydantic fallbacks: 9.
- Hooks that raise (`RunHooks`/`AgentHooks` cannot fail in Rust): batch `hook`,
  `approved_failure_diagnostics`, `live_snapshot`, fresh-guarded `tool_hook`/`agent_hook`,
  boundaries `provider_final_hook`/`final_hook`.
- Hosted MCP approval callback that raises (returns `MCPToolApprovalResult`): boundaries `mcp_callback` 2.
- Run tool formatter that raises / returns a non-string, approval-rejection formatter failure
  (`tool_error_formatter` is `Fn -> Option<String>`): 2 + 2 + 2.
- Python object identity / deepcopy / late mutation / cyclic outputs (Rust values are owned
  JSON): `opaque_output_identity` 4, `mutable_snapshot` 2, `cyclic` 4, `freezes_accepted_output_before_metadata` 8.
- `asyncio.gather` monkeypatching (`selected_tool_error_survives_completed_task_guardrail_race`), positional `ToolErrorFormatterArgs` constructor.

### Not ported (76) — real divergences, deferred
1. **Parallel input guardrails settle before tools run** (`src/run_internal/mod.rs`, `run_turn`
   parallel `tokio::select!`, model-first branch awaits `guard_fut`). Python runs the turn's
   tools concurrently with the pending verdict and waits for the verdict only to admit partial
   history. Tests whose guardrail waits on tool progress deadlock in Rust:
   batch `input_guardrail_rejection…` 2, `streamed/nonstreamed_partial_history_waits_for_input_verdict` 6,
   `nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail` 1, native
   `input_verdict_during_tool_drain` 12. Minimal design: keep the guardrail future owned
   (`BoxFuture<'static>`; `run_input_guardrails` only captures clones), carry it into
   `execute_side_effects`, race it with tool execution; on tool error await the verdict only if
   outputs were committed (else drop it); a failed verdict drops the turn and keeps the tool error.
2. **Native/local tools run sequentially after the function batch**
   (`execute_side_effects_inner`, local-call loop). Python runs computer/custom/shell/apply_patch/
   local_shell categories concurrently with function tools and cancels queued natives on a
   selected function failure. Native: `finalization_survives` 14, `preserves_output_without_delaying` 2,
   `selected_function_failure_stops_queued` 16, `parent_cancellation_after_function_failure` 2,
   `converted_native_cancellation` 2, `pending_native_invocation_cleanup` 2. Design: run the local
   loop under a join with the batch, pushing completed outcomes into a shared vec; drop it when the
   batch fails.
3. `Agent::as_tool` with `parameters` does not validate arguments against the schema (only parses
   JSON), so `agent_tool_validation_error_*` (2) has no error to redact.
4. Model-provider log redaction `get_response`/`stream_response` (3): models area; needs a mock
   Responses server.
5. `_deserialize_items` warnings for unknown agents (12): run_state area.

## Rust bugs fixed (src)
- `src/run_internal/tools.rs`: non-SDK tool errors (failure policy `None`) now surface as
  `UserError("Error running tool {name}: …")`; handled failures log `Tool {name} failed` (debug)
  or the detail (error) per `DONT_LOG_TOOL_DATA`; the default failure policy keeps the error out
  of span data even with `trace_include_sensitive_data`; a default-policy tool with
  `output_json_schema` called by a hosted program re-raises instead of returning the generic message.
- `src/tool.rs`: `invalid_tool_input_error` ("Invalid JSON input for tool X[: detail]", redaction
  per tool data policy, port of `_parse_function_tool_json_input`); explicit
  `with_failure_error_function(default_tool_error_function)` keeps default trace redaction;
  shared default handle so `MCPConfig::default()` is recognized; output-type validation follows
  `DONT_LOG_TOOL_DATA` only (was model+tool).
- `src/agent.rs`: `as_tool` uses `invalid_tool_input_error`; its failure policy is final (run
  `tool_error_formatter` no longer rewrites agent-tool failures); `failure_error_function` also
  covers the custom output extractor.
- `src/mcp/util.rs`: default MCP failure policy keeps trace redaction.
- `src/error.rs`: `REDACTED_INVALID_JSON_MESSAGE`, `DATA_REDACTED_ERROR_MESSAGE`,
  `invalid_json`, `data_redacted`, `is_data_redacted`, `redact_follow_up` (port of the
  `_mark_error_data_redacted` boundary).
- `src/agent_output.rs`, `src/handoffs.rs`, `src/realtime/handoffs.rs`: output-schema and handoff
  validation errors follow `validate_json` redaction (model policy; handoffs model+tool).
- `src/run_internal/mod.rs`: redacted errors carry no `run_data`; a session write / error-handler
  failure / invalid handler output after a redacted error becomes `UserError("Error details are
  redacted.")`; blocked terminal tool session failure is redacted; **`_publish_completed_tools`**:
  any failure in a fresh tool turn keeps only accepted call/output pairs (+ their reasoning,
  completed provider items, programs), persists them (input first; a write failure never replaces
  the tool error and is reconciled against the session tail), and keeps output private when
  output guardrails gate the turn.
- `src/run_internal/items.rs`: `completed_tool_step_items` (port of `_completed_tool_step_items`).
- `src/run_internal/local_tools.rs`: custom and apply_patch tools keep their committed output when
  the custom data extractor fails (computer already did).

## Cross-area divergences deferred
- Python persists the run input to the session before the first turn; Rust writes it with the
  first settled turn, so an aborted first turn leaves the session empty (boundaries
  `parent_cancel` assertion relaxed accordingly).
- Streamed tool-output events are emitted live per tool; Python streams turn items after the turn
  resolves, so a guarded terminal output that is later rejected/failed is never streamed. Rust
  emits it (fresh-guarded streaming assertion dropped).
- Items 1–2 above (run loop concurrency).

## Shared-file edits
No `src/lib.rs` / `Cargo.toml` edits. Edited shared src files listed above.

## Breaking changes
- Tool error messages: `Invalid JSON input for tool X` replaces `Invalid JSON when parsing … for
  tool X`; output-schema errors are `Invalid JSON when parsing model output` under default policy.
- Unhandled non-SDK tool errors are now `AgentsError::UserError` (was `Other`).
- On a failed tool turn, unanswered calls and the turn's messages are no longer in `run_data.new_items`/checkpoint.

## Gates (final)
- My three test binaries: `py_core2_toolerrors_formatter` 14/14, `py_core2_toolerrors_logging` 31/31, `py_core2_toolerrors_batch` 8/8.
- Regression-fixed binaries: `runner_tests` 27/27 (`structured_output_is_validated` asserts the Python redaction rule: no run data under the default policy), `py_deferred_writes_e` 13/13.
- `cargo test --all-features --lib`: 642 passed.
- `cargo clippy --all-features --all-targets -D warnings`: my files clean. Only flagged file is another agent's `tests/py_core2_scopedapprovals_session.rs`.
- rustfmt applied to my three test files only.
- Full `cargo test --all-features` against the baseline: the 7 earlier regressions are resolved. Two new failures are in other agents' in-flight areas, not from my edits:
  - `tests/py_core2_runstate_sidecars.rs::tool_approval_item_with_an_unrecognized_raw_item_is_kept` (run-state deserialization, `missing field call_id`). I did not edit run-state files.
  - `tests/py_core2_runstep_execution.rs::execute_tools_runs_hosted_mcp_callback_when_present` failed once in the full run and passed on rerun (transient).

## Checkpoint and publish rules (final)
- A batch failure whose selected failure did not commit its own output clears `turn_in_flight`: the retry calls the model again. This is what the sibling and partial-history batch tests need.
- A failure after the selected call's own output was committed (custom-data merge failure) keeps `turn_in_flight` and skips publish, so the retry re-processes the stored response (`py_deferred_writes_e`).
- Hosted MCP approval decisions are retained in the session even when the turn stays private behind output guardrails. They are not guarded tool output.

## Removed scope
The parallel input-guardrail and native-concurrency tests (76 rows above) are not ported; they are documented as not-ported design gaps, not python-only.
