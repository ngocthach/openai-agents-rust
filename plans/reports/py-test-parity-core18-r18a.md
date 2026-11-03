# Python-only re-audit, round 18 (r18a)

Rust crate 0.20.0. Python source: `openai-agents-python/tests`. The `before` counts come from `py-test-parity-ledger.csv`.

## Per-file result

| file | before | ported | covered | after | remaining groups (reason) |
|---|--:|--:|--:|--:|---|
| test_error_logging_redaction.py | 81 | 0 | 0 | 81 | `logging.LogRecord` fields and caller location (`pathname`/`funcName`), exception `__bool__`/hostile `__str__`, `__cause__`/`__notes__`, traceback frame locals, `BaseException`/`BaseExceptionGroup`/hybrid cancellation, `_safe_redacted_persistence_error` class topology, pydantic serializer `warnings`, monkeypatched `start_streaming`/`attach_error_to_current_span`. The behavioural cases (streamed output-validation redaction, handler-failure redaction, session failure after a guardrail) are already ported under other names in `py_core2_toolerrors_logging.rs` (`run_surfaces_redacted_output_validation_error` streamed loop, `invalid_final_output_handler_failure_preserves_redaction`, `streamed_session_error_after_output_guardrail_respects_redaction`). |
| test_run_state_compatibility_corpus.py | 61 | 0 | 0 | 61 | Unit tests of the Python fixture harness (`_find_subset_errors`, `validate_*_fixture` with monkeypatched `to_json`/`from_json`), foreign `Mapping`/`RunContextWrapper` subclasses and descriptors, `json.loads` monkeypatch, plus traceback/slots/`__notes__`/exception-group/cancellation scrubbing. |
| test_tool_batch_failure_history.py | 24 | 7 | 1 | 16 | `cancel`/`cancel_at_verdict`/`tool_cancel` verdicts (asyncio cancellation and a tool raising `CancelledError`, 4); opaque `__deepcopy__`/`is` identity (4) and cyclic outputs (4), which `serde_json` values cannot represent; live-snapshot `single` and `tuple` shapes (4): Rust has one structured-parts type, so a lone `ToolOutputText` and a tuple are not separate cases. |
| test_function_tool.py | 23 | 1 | 0 | 22 | `**kwargs` tools (7 ids) and the `kwargs` variadic row; `copy.copy`/subclass constructors/bound-method metadata/`__post_init__` (5); `CancelledError` normalization; monkeypatched module default handler; positional `tool_namespace` call (TypeError); timeout validation of str/NaN/inf/unsupported-behavior strings/non-callables and sync-handler timeouts (typed API); `get_all_tools` failing enablement check (the Rust `is_enabled` predicate returns `bool` and cannot fail). |
| test_agent_as_tool.py | 21 | 0 | 1 | 20 | Monkeypatched `Runner.run` with dummy results: falsey extractor/builder (`__bool__`), builder returning `123` (dynamically typed); ToolContext-level ambiguous approval routing through the private `record_agent_tool_run_result` registry (x4; the Rust nested resume rebuilds approvals from the nested state's own JSON, and the RunState-level fail-closed check is ported in `py_core17_r17a_run_state.rs`); legacy deferred grant reconstruction through private `_approvals`; tool-state scope ids (x2) and the nested `ToolContext` namespace (the Python nested run takes the ToolContext object as its context); `BaseException`/`CancelledError` from `on_stream` (x2); monkeypatched `RunResultStreaming.stream_events` close failures (x4); negative backlog limit (`usize`); handler-raised `CancelledError` backlog row. |
| models/test_model_retry.py | 21 | 0 | 0 | 21 | asyncio timeout cleanup and parent-cancellation ordering, owner-task traceback locals, ContextVar/`aclose` context, cooperative `aclose`, falsey policy (`__bool__`), policies mutating provider advice (Rust passes immutable references), raw `AbortError` inferred from the Python exception class (Rust has no raw abort kind; the `is_abort \|=` rule exists in `src/retry.rs`). The stream-timeout message is ported (`py_models_retry_stream.rs`). |
| models/test_openai_responses.py | 27 | 0 | 0 | 27 | Monkeypatched `websockets.connect`, omit/NOT_GIVEN sentinels, legacy httpx `Timeout`/`URL`, httpx client default timeout, event-loop/request-lock/close-handshake internals. |
| **total** | **258** | **8** | **2** | **248** | |

### test_tool_batch_failure_history.py detail

New file `tests/py_core18_r18a_batch_snapshots.rs`, 3 tests:

- `partial_history_freezes_accepted_output_before_metadata`: all 8 ids (function and native custom tool × approved × streaming). A sibling fails while the extractor is blocked. The run fails with the sibling's `UserError` within the post-invoke wait, and the extractor future is dropped. One output is kept with `custom_data == None` and the right wire output, and the tool-output guardrail saw the output. The state round-trips through JSON unchanged, the session saved the output, and a continuation run does not re-run the tool.
- `approved_mutable_snapshot_recovery_uses_publication_value` (2): the retry from the restored state returns the published output via `StopOnFirstTool`. Effects are `ticket, audit`, each once.
- `approved_live_snapshot_preserves_structured_output_type`, `list` shape (2): **this exposed a source bug** (below).
- Covered: `selected_tool_error_survives_completed_task_guardrail_race` is `py_core6_nativetools_failure_history.rs::input_verdict_during_tool_drain_preserves_selected_tool_error`. Its `function`/`error` row has the same gating: the end hook waits, the tool fails, and a late guardrail error loses to the primary tool failure.
- Accounting: the 24 could not be traced id by id. The unported candidates I found add up to 29 ids, so the ledger already credited 5 of them somewhere, probably the native non-approved freeze covered by `native_finalization_preserves_output_after_drain_timeout`. I count the 16 valid ids as remaining and the other 8 as moved (7 ported, 1 covered), although the file tests 12 ids.
- Stale note: r16g said the `*_waits_for_input_verdict` cases depend on Rust awaiting the guardrail before tools. That is wrong. They are ported in `py_core6_nativetools_guardrail_race.rs` (the pass, reject and error verdicts), and r8g8 had already shown that `run_turn` keeps the input guardrail pending while tools run.

### Other moves

- `test_function_tool.py::test_variadic_field_constraints_validate_before_invocation[args]`: new `tests/py_core18_r18a_variadic_tool.rs`. A `Vec` of a range-checked integer with `#[serde(default)]` rejects `-1` and `11` with `ModelBehaviorError` before the body runs. `[0,10]` returns 10, and both `{}` and `[]` return 0.
- `test_agent_as_tool.py::test_agent_as_tool_rejected_nested_approval_resumes_run`: covered by `py_core2_scopedapprovals_hitl.rs::nested_agent_tool_resumes_after_rejection`, which rejects the nested approval and checks that the nested run resumes with the rejection applied and is not short-circuited.

## Source bug fixed (in `src/run_internal/**`, another agent's area: minimal edit)

`src/run_internal/mod.rs`, where a turn recovers already-committed function outputs (resuming after a failure): the `FunctionToolResult.output` handed to a custom `ToolUseBehavior` was rebuilt as `Text` for a string and `ToolOutput::Json` for anything else. A tool that returned typed parts (`ToolOutput::Parts`) therefore reached the behavior flattened to `Json([...])` on the retry, while Python keeps the structured type. The fix is a 7-line match arm: a non-empty list that parses as `Vec<ToolOutputPart>` is rebuilt as `ToolOutput::Parts`, and anything else falls back to `Json` as before.

One side effect: a `Json` output that is itself a list of valid typed parts is now also recovered as `Parts`. That matches what the model saw, because `tool_call_output_item` already sends such JSON as structured parts. There is no public API change.

## Verification

- `py_core18_r18a_batch_snapshots` (3) and `py_core18_r18a_variadic_tool` (1) pass.
- Regression check after the source edit (all pass, all features): `py_core2_toolerrors_batch`, `py_core15_r15b_batch_failures`, `py_core6_nativetools_failure_history`, `py_core6_nativetools_guardrail_race`, `py_core6_streamedleft_batch`, `py_core_a_tool_behavior`, `py_core_b_tool_output_result`, `py_deferred_writes_e`, `py_core6_streamedleft_tool_behavior`, `py_core6_resume_legacy`, `py_core6_streamedleft_items`, `py_core8_r8g3_trimmer_replay`, `py_mcp_util`, `py_sandbox_b_capabilities`, `py_sandbox_b_view_image`, `py_core2_runstate_resume`, `runner_tests`. The full workspace suite was not run.
- clippy `-D warnings` is clean for `--lib` plus both new test binaries, with default features and `--all-features`. rustfmt was run on both new files, and `rustfmt --check src/run_internal/mod.rs` passes (only the edited region was formatted by hand).

## Read depth (honest)

- Read every body: `test_tool_batch_failure_history.py` (whole file), the 11 name-unmatched tests of `test_agent_as_tool.py` and its parametrized rows, all 33 name-unmatched tests of `test_function_tool.py`.
- `test_error_logging_redaction.py`: bodies of 22 of the 31 name-unmatched tests. The rest were read by name, and their behavioural counterparts were checked in the Rust files.
- `test_run_state_compatibility_corpus.py`: bodies of 7 of 33; the rest by name only (r16e read all of them).
- `models/test_model_retry.py`: bodies of 6 of the r16d python-only groups, plus a source check of the abort and timeout paths. The other 85 name-unmatched names were taken as renamed ports, following r16d/r14c, without re-reading.
- `models/test_openai_responses.py`: names only, relying on r16b's rename map to `py_models_responses_*.rs`. No body was re-read in this round.

## Unresolved questions

- Should the ledger's 5 untraceable credits in `test_tool_batch_failure_history.py` be re-derived from the earlier reports, so every id has a single owner?
- Rust evaluates `is_enabled` predicates one by one and they cannot fail, while Python runs them concurrently and lets them fail. Is that intended?

Status: DONE_WITH_CONCERNS
