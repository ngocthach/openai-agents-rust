# py-test-parity core19 / r19a

Crate 0.21.0. Closes two known gaps, fixes the memory rollout input of a resumed sandbox run, and
re-reads the Python-only cases of 10 core files. CSV: `py-test-parity-core19-r19a.csv` (columns
path,moved_to_ported,moved_to_covered,remaining_python_only; before - ported - covered =
remaining). No live model call was made.

## Source changes

1. **Sandbox `process_context` on the original input** (`src/run.rs`, `src/run_internal/mod.rs`,
   `src/sandbox/runtime.rs`). Python's per-turn `SandboxRuntime.prepare_agent` applies the
   capability chain to the run's `original_input` only and writes the result back (`run.py`
   1084-1107, `run_loop.py` 1406-1425). Rust now does the same through a crate-private
   `RunOptions::process_input` hook that the sandbox runner installs (it replaces the previous
   `call_model_input_filter` wrapper):
   - the rewrite replaces `original_input` (and the model prefix), so it reaches
     `RunResult::input`, `to_input_list()`, `to_state()` and the streamed result;
   - items the run generated are never passed through the chain; the chain runs once per turn,
     after the agent-start hooks and the first-turn session load: parallel input guardrails, the
     caller's `call_model_input_filter` and the model see the processed input, while the
     blocking first-turn guardrails saw the raw input (in `run_inner`, before preparation), as
     in Python. A resumed run also applies it before running its approved tools (Python
     prepares before resolving the interrupted turn), so a resume that ends on those tools
     carries the rewrite too;
   - server-managed conversations: the prefix is rewritten each turn, but the initial input is
     sent once (`take_initial_input`), so a later-turn rewrite of the prefix is not resent
     (untested);
   - a plain-text run input is left alone (Python's `prepare_sandbox_input` returns a string
     unchanged); a handoff filter that changes the history, or a nested handoff, turns it into a
     list from then on;
   - with an SDK session Rust's model input is history + run input while `original_input` is the
     run input only; the rewrite then applies to the model input only (Python's
     `original_input` is history + input there). A resumed state loses Python's
     "input was a string" marker (Rust `RunState.original_input` is a list), so a resumed
     text-input run is processed where Python would skip it.
   - Nested-history ownership needs no rebase: Rust revalidates owned refs by value against
     `RunResult::input` (`active_owned_occurrences`), so a rewrite that drops a forwarded item
     releases it (`py_sandbox_a_runtime_4`, `py_core16_r16g` keep passing).
2. **Detached post-invoke siblings** (`src/run_internal/tools.rs`). Each function-tool call in
   a batch now owns its environment (`OwnedToolEnv`: cloned `RunContext`, `Arc<Agent>`,
   `Arc<RunConfig>`, hooks, key, approval sets), so the per-call future is `Send + 'static`.
   When the post-invoke drain wait expires, the calls still finalizing are no longer cancelled:
   their committed output is kept (as before) and the futures are moved to `tokio::spawn`, in
   the captured trace scope and model-run owner. A late failure is logged as Python's
   "Background function tool post-invoke task raised after failure propagation." (detail only
   when the tool-data log policy allows). Nothing else changed: calls not yet in post-invoke are
   still cancelled at the failure, natives still cancel after their wait, a run dropped during
   the wait still drops everything. No public bound changed.
3. **Memory rollout input** (`src/sandbox/runtime.rs`, coordinator request). Port of Python's
   `_sandbox_memory_input`: the segment's `input` is computed from the caller's input before the
   run: `[]` for a `RunInput::State` with an SDK session and no `conversation_id` /
   `previous_response_id` / `auto_previous_response_id`; otherwise the caller's input items (a
   resumed state's saved input). Both the non-streamed and streamed paths use it. Because it is
   taken before the run, neither a handoff filter nor the new `process_context` rewrite of
   `RunResult::input` reaches the memory segment (Python records `original_user_input`, not the
   rewritten `original_input`). The two `#[ignore]`d tests in
   `tests/py_core19_r19c_sandbox_memory.rs` are un-ignored and pass.

## Tests

- `tests/py_core19_r19a_sandbox_process_context.rs` (sandbox): rewrite kept in
  `RunResult::input` / `to_input_list` and the model's turn-2 input = rewritten prefix +
  unprocessed generated items, non-streamed and streamed; text input not processed; `Compaction`
  truncates an input compaction item but not one the model returns.
- `tests/py_core19_r19a_post_invoke_detach.rs`: second half of
  `test_multiple_tool_calls_do_not_wait_indefinitely_for_sleeping_post_invoke_sibling`
  (non-streamed, streamed, multi-thread runtime): the run fails with the sibling error within
  the wait, the blocked guardrail is released afterwards and finishes.
- `tests/py_core19_r19a_runner_audit.rs`: streamed cancelled-tool analogs (2 ported); the
  changed-pending-invocation case is written and `#[ignore]`d (see Concerns).
- Updated tests whose premise was the old Rust behaviour:
  `tests/sandbox_tests.rs::process_context_runs_on_every_turn` renamed
  `a_generated_compaction_item_does_not_truncate_the_next_turn` (asserts Python semantics);
  `tests/py_core18_r18a_batch_snapshots.rs` (function-tool extractor now keeps running
  detached until released, as Python's `not settled` / `settled` assertions say; the native
  extractor is still dropped).

## Per file

| file | before | ported | covered | after | notes |
|---|--:|--:|--:|--:|---|
| tests/test_agent_runner.py | 16 | 0 | 0 | 16 | changed-pending-invocation written but ignored (Rust bug, see Concerns). Remaining: dict hash/eq impostor and BaseException snapshot helpers, Python falsy runner/agent, legacy positional args, model_dump/pydantic, invalid filter types, `prepare_input_with_session` in-place history mutation / rebuilt-copy identity / non-callable / server-managed private path, conversation-lock rewind helper, hand-built approval items (`execute_approved_tools` non-function tool, missing tool, missing call id, invalid raw type) |
| tests/test_agent_runner_streamed.py | 11 | 2 | 0 | 9 | ported: streamed parallel / single cancelled-tool (failing-body analog, as the non-streamed port). Remaining: invalid/non-callable filter types, monkeypatched finalizer (3 params), awaiting `run_loop_task`, `agent.tools` mutation, session-save `CancelledError` / `BaseException`, private event queue |
| tests/test_run_impl_resume_paths.py | 8 | 0 | 0 | 8 | monkeypatched `turn_resolution` / `run_single_turn` / `resolve_interrupted_turn`, `CancelledError` callback |
| tests/test_hitl_error_scenarios.py | 7 | 0 | 0 | 7 | private `_resolve_interrupted_turn` with hand-built ProcessedResponse, dynamic needs_approval type, falsy items, same-named sibling / public-vs-execution agent object identity |
| tests/test_handoff_history_duplication.py | 10 | 0 | 0 | 10 | ownership by Python object/copy lineage (rebuilt container, deepcopy, reorder, pop+append of `result.input`), dataclass-field introspection, `model_construct(status=None)` helper; see Concerns |
| tests/test_run_config.py | 10 | 0 | 0 | 10 | dictionary RunConfig normalisation/unknown keys, typed-instance identity, non-string / str-subclass / async formatter types (the empty-string case is covered by `py_core2_runconfig_settings::run_config_rejects_empty_output_guardrail_blocked_message`), `PureWindowsPath` cwd, `Mock()` model_settings, package export |
| tests/test_result_cast.py | 7 | 0 | 0 | 7 | gc/weakref/`release_agents`/asdict/pydantic rebuild, unchecked default cast, frozen dataclass |
| tests/test_tool_context.py | 7 | 0 | 0 | 7 | identity hash, required-field errors, dict run_config normalisation, annotation-based context detection |
| tests/test_agent_tool_state.py | 6 | 0 | 0 | 6 | module-level registry keyed by call-object identity with monkeypatched globals and gc/weakref; Rust keeps nested results on the run context by call id, behaviour tested through agent-as-tool resumes |
| tests/test_agent_config.py | 7 | 0 | 0 | 7 | dynamic constructor kwargs / dict model settings typing |

Totals: 89 before, 2 ported, 0 covered, 87 remaining.

Note on `test_run_step_execution.py`: the ported second half belongs to a case the ledger already
counts as ported (first half in `py_core17_r17a_step_execution.rs`), so the ledger count does not
move.

## Concerns

- **Approval bound to a different payload runs the changed call (security-relevant, not my
  files).** `RunState::approve` with an item whose call id matches a pending call but whose
  arguments differ authorises the pending (changed) call: `bind_decision` in `src/run_state.rs`
  does `tool_invocations.entry(call_id).or_insert(record)`, keeping the pending record. Python
  raises `ModelBehaviorError` ("unique call ID") and never runs the tool
  (`test_execute_approved_tools_rejects_changed_pending_invocation`). Test:
  `tests/py_core19_r19a_runner_audit.rs::an_approval_for_another_payload_does_not_run_the_changed_call`
  (`#[ignore]` with reason). Owner of `src/run_state.rs` / `src/run_context.rs` to fix.
- Nested-history ownership in Rust is by value against `RunResult::input`; Python tracks index +
  object identity. Cases like popping the owned item and appending an equal copy diverge
  (Rust still claims it). Documented design from earlier rounds, kept as python-only.
- `tests/py_core4_tracerunner_disabled.rs::runner_tracing_opt_out_inside_caller_span` failed
  once in the first full run and passes alone (global tracing state shared across tests in the
  binary); not touched by this change.

## Suggested changelog paragraph

- Sandbox: `process_context` now rewrites the run's original input, as Python's per-turn
  `prepare_agent` does: the rewrite is kept in `RunResult::input`, `to_input_list()` and the
  saved state, runs once per turn, never sees items the run generated, and a plain-text input is
  left alone. A compaction item the model returns no longer truncates the next turn's input
  within the same run.
- Tools: a function-tool call still finalizing (output guardrails, extractor, end hooks) when a
  sibling's failure surfaces after the post-invoke wait keeps running detached on its own task
  instead of being dropped; a late failure is logged.
- Sandbox memory: a run resumed from its `RunState` with an SDK session records an empty
  `input` in its rollout segment; other runs record the caller's input rather than the
  rewritten `RunResult::input`.

Behaviour changes: a capability `process_context` no longer sees generated items; `RunResult::input`
of a sandbox run now carries the rewrite. No public signature changed.
Known gaps closed: both 0.19.0/0.21.0 items. Remaining: session-backed runs rewrite only the
model input (Python's `original_input` includes history there); a resumed text-input run is
processed (the string marker is not serialized).

## Read depth

All Python bodies of the uncited candidates in the 10 files were read except: in
`test_agent_runner.py` the blocked-snapshot / hash-impostor helpers, falsy/legacy, model_dump
and filter-type cases were judged from names plus the earlier reports' group reasons; in
`test_hitl_error_scenarios.py`, `test_run_config.py`, `test_result_cast.py`, `test_tool_context.py`
the obviously typed/dynamic cases were read by head and assertions. Ledger counts have no
per-case ids, so the mapping of uncited names to the remaining counts is approximate.
