# Python -> Rust test parity: core2 / scopedapprovals

Scope (relative to `openai-agents-python/tests/`): `test_function_tool_approval_arguments.py`,
`test_agent_scoped_approvals.py`, `test_hitl_error_scenarios.py`, `test_hitl_session_scenario.py`,
`test_resume_tool_use_behavior.py`.

## How tests were counted

`def test_` per file times the product of its `pytest.mark.parametrize` list lengths (done by
hand from each decorator). `test_hitl_error_scenarios.py`: `grep -c "def test_"` says 67, but two
hits are nested helpers (`async def test_tool()` inside two tests), so pytest collects **65
functions / 88 tests**, not the 90 in the earlier core-a triage.

## Per-file table

| Python file | #tests | ported | covered | python-only | not ported (bug deferred) | failures found |
| --- | --- | --- | --- | --- | --- | --- |
| test_function_tool_approval_arguments.py | 108 | 50 | 0 | 58 | 0 | 3 (fixed) |
| test_agent_scoped_approvals.py | 104 | 52 | 0 | 48 | 4 | 4 (2 fixed, 2 deferred) |
| test_hitl_error_scenarios.py | 88 | 59 | 3 | 16 | 10 | 9 (6 fixed, rest deferred) |
| test_hitl_session_scenario.py | 2 | 2 | 0 | 0 | 0 | 0 |
| test_resume_tool_use_behavior.py | 48 | 48 | 0 | 0 | 0 | 1 (fixed) |
| **total** | **350** | **211** | **3** | **122** | **14** | |

Rust test files (all new, rustfmt'd):

| file | fns | Python source |
| --- | --- | --- |
| `tests/py_core2_scopedapprovals_fn_args.rs` | 11 | function_tool_approval_arguments (runner/streamed) |
| `tests/py_core2_scopedapprovals_realtime.rs` | 2 | function_tool_approval_arguments (realtime mode) |
| `tests/py_core2_scopedapprovals_agent.rs` | 11 | agent_scoped_approvals |
| `tests/py_core2_scopedapprovals_hitl.rs` | 18 | hitl_error_scenarios (run/resume, nested agents, serialization) |
| `tests/py_core2_scopedapprovals_hitl_decisions.rs` | 21 | hitl_error_scenarios (rule vs decision races, checkpoints, MCP callbacks) |
| `tests/py_core2_scopedapprovals_session.rs` | 2 | hitl_session_scenario (local mock Conversations endpoint) |
| `tests/py_core2_scopedapprovals_resume_behavior.rs` | 1 (48 cases looped) | resume_tool_use_behavior |

Parametrizations are looped inside one Rust fn with the case in every assert message.

### test_function_tool_approval_arguments.py (108)

Ported (50): defaults_and_alias (6), benign_omission (2), manual_decision_roundtrip (8),
invalid_typed_input runner+streamed (12) + realtime (6), preparation_on_copied_tools (2; `copy.copy`
and `dataclasses.replace` = `Clone` / clone with a field replaced), parallel_calls (1),
realtime_prepared_approval_and_errors (3), changed_or_uninspectable (4 of 6: str->int, bool->int,
float 1.0, extra key), root_parameter_order (1), resume_selection (2, through the runner instead of
private `resolve_interrupted_turn`), cancelled_policy (1), input_guardrail_blocks_application_validation (2).

Python-only (58), pydantic semantics with no serde counterpart:
- nested_transformations (4), validator_exception (9), application_validation (20),
  realtime_application_validation (2), nested_input_aliases (2): pydantic field/model validators,
  `AliasChoices`, `default_factory`, `model_post_init`, `field_serializer`.
- plugin_validation x2 tests (6+2): pydantic plugin hooks.
- compiled_pattern / pattern_validation (6+4): `Field(pattern=re.compile(...))`.
- changed_or_uninspectable int->float `1` (1) and `tuple[int, ...]` (1): pydantic coercions; JSON `1`
  into `f64` is not a change in serde's number model, tuples do not exist in JSON.
- invalid_keyword_reconstruction (1): Python `**kwargs` parameters.

### test_agent_scoped_approvals.py (104)

Rust keys approvals by agent *name* (`ToolApprovalItem.agent: String`); Python by agent object
identity. Converters: Python `TypeAdapter` python-mode = `serde_json::Value` round trip, JSON mode
= string round trip (both ported); Temporal converter is python-only.

Ported (52): sticky_approval_stays_with_agent_across_handoffs (8; ported with distinct agent
names, Python uses two `same-name` agents), legacy_sticky_decisions (2),
function_sticky_rejection_vs_native (2), function_exact_approval_vs_native_sticky (1),
sandbox_operation_approval (1), context_converters_preserve_function_decisions (8),
mixed_hosted_decisions (4), runner_executes_converted_context_decisions (16),
unresolved_context_owners[missing] (1), reused_context_snapshots (1),
converted_context_retains_other_graph_decisions (8).

Python-only (48): name-only query/inspection API (1+4; Rust lookups always name the agent),
distinct same-named owners/capabilities via object identity (3+12+4 tied_duplicate), unresolved
owners target/source/clone/copy (4; `clone()`/`copy.copy` identity witnesses), gc/weakref
retention (2), Temporal converter variants (4+8+4+1+1).

Not ported, bug deferred (4): converted_context_preserves_approved_invocation (2) and
converted_context_does_not_repeat_completed_call (2) — see cross-area bug 1.

### test_hitl_error_scenarios.py (88)

Ported (59): resumed_hitl_executes_approved_tools (2), resuming_skips_non_hitl (2),
nested_agent_tool_resumes_after_rejection, changed_nested_parent (error half; see bug 4),
nested distinct / not-inherit (2), pending_stay_pending (2), no_duplicate_pending_shell,
route_local_shell, preserve_max_turns, current_turn, deserialize_tool_calls (2),
deserialize_mcp_tools (2), hosted_mcp_unknown_legacy_identity, shell_without_call_id,
tool_output_types, fails_closed_invalid_arguments (7), receives_valid_object_arguments,
agent_as_tool_nested_approvals, continuation_runs_outer_callbacks_once (2),
permanent_namespaced_new_call_id, resume_skips_checker, rechecks_rejection_after_checker,
prefers_decision_during_guardrail (2), execute_skips_checker, checkpoints_before_tool_use_behavior,
sticky_rejection_before_checker (4), collect_runs_skips_checker, apply_patch_concurrent_decision (2),
deferred_display_name, rebuild_object_pending_and_rejections, unmatched_with_function_runs,
skips_shell_with_existing_output, approved_shell_outputs_with_pending, executes_pending_computer,
checkpoints_computer_output, skips_computer_existing_output, rejected_shell_emit,
rejected_shell_not_duplicated, mcp_callback_processed, mcp_exact_siblings, mcp_changed_siblings.
Private-helper tests (`resolve_interrupted_turn`, `_select_function_tool_runs_for_resume`,
`_collect_runs_by_approval`, `ShellAction.execute`, ...) are driven through `Runner` + shared
`RunContext`/`RunState`.

Covered (3) by the ports above: executes_non_hitl_without_output and
skips_non_hitl_with_existing_output (`hitl.rs::resuming_skips_approvals_for_non_hitl_tools`: the
auto call runs exactly once across pause/resume), rebuild_function_runs_handles_pending_and_rejections
(`hitl_decisions.rs::resume_emits_rejections_and_keeps_pending_calls`).

Python-only (16): mcp_approvals_hashable (1, `set` of Python objects), needs_approval invalid type
x2 (2, `NeedsApproval` is a typed enum), persisted_item_counter streamed (1, private
`_current_turn_persisted_item_count`), falsy pending item x2 + rebuild falsy (3, `__bool__`
override / `is` identity), same_named_sibling (1), public-vs-execution agent outputs (1+2,
`set_public_agent` clone identity), approval_created_during_tool_execution (2, Python evaluates the
rule twice: planning + execution; Rust once), mcp_exact_retry (1, re-entering one private planner
step in the same context), mcp_callback_failure (1, Rust MCP callback is infallible), rebuild
deferred lookup_key (1, persisted `tool_lookup_key` tuple), local_mcp_recipient_binding (1,
rebuild-from-approvals path).

Not ported, bug deferred (10): rebuild_from_pending_approvals / object_approvals (2, bug 3),
rebuild_preserves_unmatched_pending_approvals (3, bug 2), changed_approval_while_policy_waits (3,
bug 1), changed_handoff_under_approved_id (1, bug 1), validates_changed_shell_before_sibling
(1, bug 1). The 3 nested-agent resume ports are counted as ported; they currently fail only
because of the in-flight edit in bug 5.

## Rust bugs fixed (src)

1. `src/run_internal/tools.rs` `execute_function_tool`: decision-first. Python consults the
   recorded decision before the `needs_approval` rule; Rust ran the rule first, so a resumed call
   re-ran the rule and a rejected call executed if the rule then said "no approval". Now: status ->
   rule only if undecided -> status re-read (decision recorded while the rule ran wins); a decision
   recorded while a rejecting pre-approval guardrail ran also wins.
2. `src/run_internal/tools.rs`: function approvals read under the invocation-scoped key
   (`approval_tool_key`) they are written with — a sticky rejection of a function tool named
   `apply_patch` was never found.
3. `src/tool.rs`: new `FunctionTool::evaluate_needs_approval` + private `approval_args_check` set by
   `FunctionTool::new`: a dynamic rule is consulted only when the arguments parse into the typed
   args and name exactly the declared parameters (no default fill-in, no extra key); otherwise
   approval is required without the rule. Port of `evaluate_function_tool_approval`'s
   prepared/unchanged check. Used by runner and `src/realtime/session.rs` (one-line switch).
4. `src/run_context.rs`: `ApprovalsState::function_tool_status` / `RunContext::function_tool_status`
   (pub(crate)): an ownerless (pre-1.18 `approvals` bucket) record contributes only exact-call
   decisions to function tools, never a sticky grant. Port of `_approval_records_for_tool`.
5. `src/run_internal/mod.rs`: `resumed_from_failure = state.pending_tool_processing` — outputs that
   completed before an approval pause now feed the tool-use behaviour on resume
   (StopOnFirstTool/StopAtTools/custom), as `resolve_interrupted_turn` does.
6. `src/run_internal/mod.rs` + `tools.rs` (`ToolEnv.pending_approvals`, new loop field
   `resumed_pending_approvals`): a call still pending in the resumed snapshot stays pending without
   re-running its rule (`_function_requires_approval` short-circuit).
7. `src/run_internal/tools.rs`: resuming a nested `Agent::as_tool` run skips the outer call's input
   guardrails and `on_tool_start` hooks (already ran when it first started).
8. `src/run_internal/local_tools.rs` `approval_gate`: decision re-read after the rule runs; the
   apply-patch per-operation rule loop stops once a decision lands.
9. `src/run_internal/local_tools.rs` + `mod.rs`: computer output is committed before a failing
   `custom_data_extractor` error propagates (`LocalOutcome::OutputThenError`), so a retry does not
   act again.
10. `src/run_internal/mod.rs`: exact same-id hosted MCP approval-request siblings invoke
    `on_approval_request` once and emit one response.

## Cross-area bugs deferred (not fixed)

1. **Invocation ledger lives in the run loop, not in `RunContext`** (`src/run_internal/mod.rs`
   `tool_invocations`, ~L2290 / `validate_response_invocations`). Python keeps it in
   `RunContextWrapper`, so an approval recorded on a context binds that exact invocation: a later
   run reusing the call id with changed args raises "reused a tool call ID", and a completed call
   is not re-run with a reused context. Rust runs the changed call / re-runs the completed one.
   Owner: the call-id-reuse agent (`test_tool_approval_call_id_reuse.py`). Minimal fix: record
   `(invocation_type, scope, fingerprint)` in `ApprovalRecord` on `approve_tool`/`reject_tool` and
   check it when status is read; mark completed call ids on the context.
2. Resuming with an agent that lacks the pending tool errors "Model reused a tool call ID" instead
   of keeping the approval pending (`rebuild_preserves_unmatched_pending_approvals`). Ledger/identity
   area (`invocation_identity_raw` falls back to a different identity when the tool is missing).
3. A snapshot with no parked model response does not rebuild function runs from pending approvals
   (`resume_tools` returns `RunAgain` when `model_responses` is empty). Design difference; port of
   Python's rebuild path is in run_state/run_internal ownership.
4. Changed parked call: Rust resolves the tool inventory (`is_enabled` callbacks) before refusing
   the snapshot; Python refuses first. Error asserted, callback-order half not.
5. **In-flight regression (not mine):** the new `mark_invocation_executed` /
   `invocation_executed_uncommitted` guard (`src/run_internal/tools.rs` ~L361, `mod.rs` ~L3122)
   marks a nested `Agent::as_tool` call executed when it interrupts, so resuming it fails "A tool
   call already executed, but its output was not committed". Breaks 3 of my hitl tests and
   `runner_tests::agents_as_tools_and_nested_approvals`. Fix: don't mark (or unmark) when the
   outcome is `ToolOutcome::Interruptions`.

## Breaking changes

None to public signatures. Behaviour: a recorded decision now governs function tools even when
`needs_approval` is false (Python parity); typed tools with a dynamic rule require approval for
arguments that omit a defaulted/`Option` field or add an undeclared key.

## Shared-file edits

None to `src/lib.rs` / `Cargo.toml`. Edited: `src/tool.rs`, `src/run_context.rs`,
`src/run_internal/{tools.rs,local_tools.rs,mod.rs}`, `src/realtime/session.rs` (one line).

## Gates

- My 7 binaries: 66 fns, all green (11 agent, 11 fn_args, 18 hitl, 21 hitl_decisions, 2 realtime,
  1 resume_behavior covering 48 cases, 2 session). The 3 nested-agent hitl fns were red for a while
  because of bug 5; they pass on the final run (the other agent fixed it), so bug 5 is resolved.
- `cargo test --all-features --lib`: 642 passed, 0 failed.
- Full `cargo test --workspace --all-features` mid-round: no failures attributable to my edits
  (only a link error in another agent's `py_core2_runconfig_cancel`; later `runner_tests` failures
  `structured_output_is_validated` / `tool_name_collisions_follow_policy` are unrelated in-flight work).
- clippy `-D warnings` on lib + my targets: clean for my files; lib currently fails only on other
  agents' in-flight code (`run_internal/tool_invocation.rs:328` nonminimal_bool,
  `run_state/sandbox_payload.rs:363` redundant closure).
- Ignored tests: none. Output dir deleted at the end.
