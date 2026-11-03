# Core6 nativetools: native tool concurrency, input-guardrail race, schema-1.7 approval owner

Counting: pytest case = parametrize product per test function (72, 64 and 198, matching the ledger).
Only the cases this round touched change; the other buckets of the three files are the ones in
`py-test-parity-core6-resume.md` and `py-test-parity-core6-streamedleft.md`.

| file | cases | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| test_native_tool_failure_history.py | 72 | 38 | 0 | 34 | 0 |
| test_tool_batch_failure_history.py | 64 | 25 | 0 | 39 | 0 |
| test_run_impl_resume_paths.py | 198 | 142 | 10 | 24 | 22 |

Before this round: 2/0/20/50, 18/0/39/7 and 140/10/24/24. The 22 remaining not-ported cases of
`test_run_impl_resume_paths.py` are the compaction-session mock-client cases, not in this area.

## test_native_tool_failure_history.py (72)
Ported 38 in `tests/py_core6_nativetools_failure_history.rs` (6 tests) plus 2 already in
`py_core2_toolerrors_batch.rs::approved_native_output_survives_custom_data_failure`:
- finalization survives sibling failure: 12 (computer/custom/patch extractor, custom/shell/local_shell hook, x streaming)
- output kept after drain timeout: 1; run cancelled drops the finalization: 1 (`JoinHandle::abort`)
- input verdict during drain (pass / reject / error, x custom/function, x streaming): 12
- stops queued native invocations (custom/local_shell x reject x streaming): 8
- parent cancellation after function failure (abort / `cancel(Immediate)` on the stream): 2
- approved native output survives custom data failure: 2 (existing)

Python-only 34: tool raising `CancelledError` (`tool_cancel=True`, 16); guardrail verdict from a cancelled
child task (`cancelled_child`, 4); native invocation that turns its own cancellation into an error or
suppresses it (`cleanup_error=True` 8, converted cancellation 2, suppressed cancellation 2); an
`on_tool_end` hook that raises (2, Rust hooks return `()`). A dropped Rust future cannot run code.
Observation points that Python gets by monkeypatching `gather_with_cancel` are replaced by the failing
tool setting an event right before it returns its error, and cancellation by a drop-guard.

## test_tool_batch_failure_history.py: 7 newly ported (`tests/py_core6_nativetools_guardrail_race.rs`)
`streamed_partial_history_waits_for_input_verdict` pass/reject/error (3), the non-streamed three (3) and
`nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail` (1). The `cancel`,
`cancel_at_verdict` and `tool_cancel` verdicts stay python-only (asyncio task cancellation).
`py_core6_streamedleft_batch.rs` (rejection contract) stays green.

## test_run_impl_resume_paths.py: 2 newly ported (`tests/py_core6_nativetools_approvals.rs`)
Schema 1.7 `approve` / `always_approve` of the name-fallback test: an approval whose owner lost its
agent identity does not run the call for the same-named sibling that is current; the run asks again.
Verified to fail (`executed=["one"]`) with the fix disabled.

## Rust changes
- `src/run_internal/tools.rs`, `local_tools.rs`, `mod.rs` (`execute_side_effects_inner`): function tools and
  native tools (computer, custom, shell, apply patch, local shell) now run side by side (`run_tool_batch`;
  each native category serial, categories concurrent, port of `_execute_tool_plan`). A failure in either
  side stops the other: queued native calls never start, an invoking call is dropped, a call that already
  produced its output (function: after the body; native: output committed) finalizes under a bounded wait
  (`RunConfig::tool_post_invoke_wait`, default 100 ms) and is abandoned after it, keeping the output it
  committed. Native finalization order is now output commit, then custom data extractor, then end hooks
  (as Python); a failing extractor no longer runs the end hooks. The first selected failure is raised.
- Input-guardrail race: a pending parallel guardrail is polled with the turn's tool batch (guardrail first).
  A tripwire or error while no tool failure is selected cancels the tools and is the error; after a tool
  failure the verdict is awaited only when outputs were committed (else the guardrail is cancelled), a
  passing verdict admits the partial history, a rejecting or failing verdict drops it, and the raised error
  is always the tool error (`partial_history_rejected` on the run).
- `src/run_context`-level fix landed in the state layer: `src/run_state/wire.rs` stamps the owner identity on
  approval items owned by a duplicate-named current agent and `foreign_owner_call_ids` (new, derived,
  `#[serde(skip)]` field of `RunState`) lists parked approvals whose owner is another same-named agent
  under schema 1.7+; `src/run_internal/mod.rs` treats them as unbound so a decision on them authorizes
  nothing. `src/run_context.rs` itself did not need to change.

## Breaking changes
- `RunConfig` gains `tool_post_invoke_wait: Option<Duration>` (+ `with_tool_post_invoke_wait`); struct
  literals without `..Default::default()` need the field.
- `RunState` gains `foreign_owner_call_ids` (skipped on the wire; literals need `..Default::default()`).
- Behavior: native tools no longer wait for all function tools, and a rejecting guardrail verdict that
  arrives while tools run cancels them. Approval items of a duplicate-named current agent now serialize
  `agent.identity` (Python's shape).

## Deferred / known divergences
- A function tool abandoned after the wait keeps the output committed before its end hooks, but not custom
  data from its extractor (Python also commits before the extractor; Rust runs the extractor after the hooks
  for function tools).
- Owner identity stamping assumes parked approvals belong to the current agent when the names match.

## Follow-up check (current tree)
`py_core2_runner_session_input::a_replayed_call_stays_before_its_output` failed in the partial full-suite run
and passes on the current tree (30/30). It is a session de-duplication test that none of the
`run_internal/{tools,local_tools,mod}.rs` changes touch (they only change how tool batches run, not the items
that are saved), and it went green without any change from me, so the earlier failure was another agent's
in-flight edit. Binaries run on the current tree: `py_core2_toolerrors_*`, `py_core4_programmatic_*`,
`py_core6_resume_*` (including the compaction one, which compiles now), `py_core3_guardrails_*`,
`py_core6_nativetools_*`, `py_core6_streamedleft_*`, `local_tools_tests`, `py_core2_runner_session_input`:
39 binaries, 359 passed, 0 failed.
