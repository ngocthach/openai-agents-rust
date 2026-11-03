# Python-only audit, round 17, id r17a

Files: `test_run_state.py` (88), `test_run_step_execution.py` (44), `test_agent_scoped_approvals.py` (36),
`test_native_tool_failure_history.py` (30), `test_start_hook_tool_concurrency.py` (35).
New tests: `tests/py_core17_r17a_step_execution.rs` (9 tests), `tests/py_core17_r17a_run_state.rs` (7 tests).
No Rust source change was needed: every ported case passed on the first correct premise.

| file | python_only before | ported now | covered now | python-only after | remaining groups (one-line reason) |
|---|---|---|---|---|---|
| test_run_state.py | 88 | 29 | 0 | 59 | pydantic/dataclass object introspection and hooks (context serializers, `get_interruptions` hardening, typed extras); falsy/None typing; Python-only merge of two private item lists; `_deserialize_processed_response` tool-object resolution; object-identity sidecars; monkeypatched registries; 2 weaker nested-state ambiguity ports left python-only (message differs) |
| test_run_step_execution.py | 44 | 12 | 7 | 25 | asyncio cancel / `BaseException` / parent cancel / shield / loop handlers / eager task factory; ContextVar; dict arguments, exception-object outputs, fallible `is_enabled` (typed out); private helpers; 1 detached-task divergence |
| test_agent_scoped_approvals.py | 36 | 0 | 0 | 36 | 23 Temporal vendor converter, 2 gc/weakref, ~4 no name-only query in Rust API, ABOUT 14 ARE A REAL DIVERGENCE (name-keyed owner/capability binding), NOT valid python-only |
| test_native_tool_failure_history.py | 30 | 0 | 0 | 30 | asyncio `CancelledError` tool, cancelled child task, cleanup that converts or refuses cancellation |
| test_start_hook_tool_concurrency.py | 35 | 0 | 0 | 35 | shared mutable `agent.tools` from hooks; Rust hooks get `&Agent`, mutation cannot be written |
| total | 233 | 41 | 7 | 185 | |

## test_run_step_execution.py (ported 12, covered 7)

Ported (`py_core17_r17a_step_execution.rs`):
- `surface_sleeping_post_invoke_failure_before_sibling_error`, `surface_post_invoke_failure_unblocked_during_settle_turns`:
  a sibling's output-guardrail tripwire that is still running (or released only after the other call failed) wins
  over the plain sibling error (streamed and non-streamed). 2 cases.
- `do_not_wait_for_cancelled_sibling_tool_before_raising`, `bound_cancelled_sibling_self_rescheduling_cleanup`,
  `preserve_triggering_error_over_cancelled_sibling_cleanup_error`: a hung sibling is dropped at once and cannot delay or
  replace the triggering error (one Rust test, 3 cases). Also asserts `do_not_wait_indefinitely...` surfaces its error
  within the wait bound (the "keeps running detached" half is not ported, see below).
- Hosted MCP identity cases, run end to end with the snapshot rewritten the way the Python test hand-builds the pending
  item: `manual_hosted_mcp_approval_uses_current_scoped_decision_after_identity_conflict`,
  `resolve_interrupted_turn_uses_current_scoped_hosted_mcp_decision` (stale pending for another server, same request id:
  the current sticky decision answers), `resolve_interrupted_turn_keeps_callback_owned_hosted_mcp_request_out_of_pending`,
  `manual_hosted_mcp_approval_reprompts_for_partial_pending_identity` (typed decision, no ledger, pending lost its
  server label: asks again), `hosted_mcp_approval_does_not_reuse_legacy_name_for_a_different_current_tool`. 5 cases.
- `execute_tools_uses_public_agent_for_hosted_mcp_interruptions`, `resolve_interrupted_turn_uses_public_agent_for_resumed_hosted_mcp_approvals`:
  interruption and resumed approval response belong to the run's agent, response keeps the programmatic caller. 2 cases.

Covered now (7; r15b kept them python-only while naming these end-to-end tests, which the rules do not accept as a reason; each cited test was read): `manual_hosted_mcp_approval_does_not_reuse_stale_pending_identity`
(`py_core2_approvals_reuse_b::sticky_mcp_approval_rejects_same_id_on_another_server`), `execute_tools_omits_completed_mcp_approval_request_replay` x2
(`runner_omits_completed_mcp_approval_request_replay`), `manual_hosted_mcp_approval_rejects_incomplete_exact_call_decision` and
`resolve_interrupted_turn_rejects_incomplete_pending_decision` (`py_core2_approvals_context::incomplete_hosted_mcp_uses_only_exact_call_decisions`),
`manual_hosted_mcp_approval_does_not_apply_legacy_exact_without_pending` (`hosted_mcp_legacy_bare_name_approval_does_not_grant_access`),
`incomplete_current_hosted_mcp_request_does_not_reuse_scoped_pending_identity` (`unbindable_current_mcp_request_requires_reapproval`).

Valid python-only after (25):
- asyncio `CancelledError` raised by a tool body, or `BaseException` subclasses (reason: asyncio cancel / Python exception hierarchy): `still_raise_when_sibling_cancelled`,
  `cancel_sibling_when_tool_raises_cancelled_error`, the five `use_*failure_error_function_for_*cancelled_tool` cases, `cancelled_function_tool_error_trace...`,
  `surface_hook_failure_over_sibling_cancellation`, `surface_output_guardrail_failure_over_sibling_cancellation`, `do_not_run_on_tool_end_for_cancelled_tool`,
  `mixed_tool_calls_preserve_shell_output_when_function_tool_cancelled`, `single_tool_call_still_raises_cancelled_error`, `still_raise_non_cancellation_base_exceptions`,
  `prioritize_fatal_base_exception_over_user_error`, `prioritize_tool_error_over_same_batch_cancelled_error`, `drain_completed_fatal_failures_before_raising`,
  `raise_late_fatal_sibling_exception_after_cancellation` x3.
- asyncio task and event-loop internals (parent cancel + shield, loop exception handler, eager task factory, detached background task): `do_not_wait_indefinitely...` (second half: Rust abandons a
  post-invoke sibling after the wait instead of detaching it; recorded divergence), `report_late_cleanup_exception_from_cancelled_sibling`,
  `parent_cancellation_does_not_wait_for_tool_cleanup`, `parent_cancellation_wins_when_shield_raises...`, `parent_cancellation_does_not_report_tool_failure_as_background_error`,
  `execute_tool_plan_parent_cancellation_*` x2, `execute_function_tool_calls_eager_task_factory...`.
- ContextVar propagation (`function_tool_preserves_contextvar...`): Python contextvars.
- Dynamically typed inputs the Rust types rule out: `execute_tools_allows_unhashable_tool_call_arguments` (dict arguments), `allow_exception_objects_as_tool_outputs`,
  `function_tool_enablement_error_cancels_sibling_checks_before_execution` (Rust `is_enabled` returns `bool`, cannot fail).
- Private helpers: `processed_response_reports_interruptions`, `execute_function_tool_calls_allows_non_agent_function_tool`.

## test_run_state.py (ported 29)

Method: every Python test name was matched against all Rust test files and sources; 132 names (185 parametrized cases) had no match. Their bodies were read in full.
The ledger credits 88 as python-only, so about 97 of the 185 were already counted ported/covered by earlier rounds under renamed Rust tests (r15a, r16b).
I verified 38 of them against current Rust tests (`py_core2_runstate_items.rs` for the `deserialize_items_*`, `ToolApprovalItem` naming/round trip, action-description
serialization, `from_json_agent_not_found`, last-processed-response and original-input cases; `py_core2_approvals_reuse_b::unbindable_legacy_mcp_approval_requires_current_reapproval`)
and did not count them again.

Ported (`py_core17_r17a_run_state.rs`):
- `approval_pipeline_models_can_apply_detached_decisions` (custom, shell x approve/reject): 4.
- `ambiguous_current_and_nested_approval_identity_fails_closed`, `native_current_and_nested_...` (shell), `ambiguous_exact_nested_...`,
  `unsafe_current_sibling_cannot_bypass_approval_ambiguity`: two pending approvals of one agent with the same
  invocation identity refuse a decision, function and shell, live and restored from JSON, nothing recorded. 12 cases. Mapping assumption: a Rust outer state keeps nested approvals in the
  same list, so "current vs nested" is "two list entries"; the "unsafe/uncopyable payload" aspect of some has no Rust form and only the ambiguity is asserted.
  `unsafe_exact_approval_across_nested_states_fails_closed` (2) is NOT counted: Python expects "cannot be safely distinguished" across two separate nested states, Rust emits the "multiple current pending approvals" message from one list.
- `completed_current_invocation_does_not_own_nested_approval`: a completed invocation with the identity of a pending approval does not block its decision, live and restored. 4.
- `uncopyable_noncanonical_approval_does_not_select_pending_owner`: an identity-less item takes no decision and never lands on a pending owner. 4 (current/nested collapse).
- `approval_takes_precedence_over_rejection_when_both_true`: 1.
- `current_schema_missing_ledger_rejects_malformed_current_authority`: the tool does not run for a replacement call without the canonical payload. 1.
- `nested_agent_tool_hitl_resume_survives_json_round_trip_after_gc` (approve/reject, with a conditional outer tool disabled after serialization, restored twice) and
  `multiple_nested_agent_tool_states_survive_multiple_dropped_functions` (two nested agent tools): 3.

Valid python-only after (59 by ledger count; my own rebuilt candidate count is 116 cases, so about 28 of those were credited earlier, ids are not recorded):
- pydantic/dataclass/introspection of context and payload objects: context serializer/deserializer/strict cases (9; Rust context is application JSON, strict handling is in `core_python_run_state.rs`),
  guardrail/tool-output normalization of datetime/dataclass/pydantic/bytes (`*_normalizes_non_json_*`, `serializes_output_containers_of_models`, `pydantic_tool_output...`,
  `non_utf8_bytes...`, `to_string_serializes_non_json_outputs`, `12062`, `12093`).
- `get_interruptions_*` snapshot hardening (about 45 cases): pydantic subclass/`__getattribute__`/storage descriptor/`model_serializer`/typed-extra hooks, deepcopy and alias, NaN/cycle values,
  uncopyable objects. Rust approvals are owned JSON values, none of these object states can be built.
- truthiness and typing: falsy trace state / processed response / hosted tool, tracker snapshot key filtering, `approve`/`reject`/`to_json` with `None` context or agent.
  Note: Rust `to_json` accepts a state whose `current_agent_name` is empty (Python raises "No current agent"); left as is because `RunState::default()` is serialized by existing callers.
- merge of `last_processed_response.new_items` into `generated_items` (4: anonymous tool_search snapshot, direct id/type attributes, tool call items, same call id): Rust keeps one item list.
- `_deserialize_processed_response` tool-object resolution (16): Rust snapshot is data only; tools are resolved from the stored model response at resume.
- generated/session item sidecar by object identity (3), reserved mount registration collision with monkeypatched imports (2), `_lookup_function_name` (5), private checkpoint identity (1),
  `_mark_tool_call_completed` private helper (1; Rust `completed_call_ids` only reads `call_id` by construction), invalid raw item fallback and missing-type item (2).

## test_agent_scoped_approvals.py (0 moved, 36 remain; all 104 bodies read)

- Temporal pydantic converter variants (23): vendor SDK (`temporalio.contrib.pydantic`); the JSON round trip they exercise is ported with the `python`/`json` converters.
- `approval_context_does_not_retain_discarded_agent_graphs` x2: gc/weakref.
- `name_only_query_does_not_choose_between_agent_owners`, `name_only_inspection_selects_a_unique_exact_call` (non-Temporal rows): Rust approval queries always name the agent, there is no name-only query.
- Same-named distinct agents, converted-context capability binding and unresolved owners (`converters_preserve_distinct_same_named_owners`, `converted_context_does_not_authorize_same_named_distinct_capability`,
  `runner_requires_reapproval_for_unresolved_context_owners` source/clone/copy/target): NOT a preamble-valid reason. A Rust restored context binds decisions by agent name and tool name, so a
  different tool or clone with the same names is authorized; Python binds to a live-object token. Recorded design divergence (see unresolved questions), not reversed.
  Exact row counts are approximate (ledger has no ids). These ~14 are counted in `remaining_python_only` for ledger arithmetic only.

## test_native_tool_failure_history.py (0 moved, 30 remain; all bodies read)

Rust already ports the sibling-failure matrix with the `tool_cancel=False`, no cancelled-child and no cleanup-error rows. Remaining rows are asyncio only: a tool raising
`CancelledError` (verdict matrix `tool_cancel=True`, 12), a guardrail whose child task was cancelled (`cancelled_child`, 8), a native invocation that converts its cancellation into
an error (`cleanup_error=True`, 8), `converted_native_cancellation_bounds_late_finalization` x2 and `pending_native_invocation_cleanup_is_bounded` x2 (a task that refuses cancellation).
A dropped Rust future cannot run cleanup, convert or refuse its own cancellation. The row total above (32) is 2 over the ledger's 30, so two rows were credited earlier.

## test_start_hook_tool_concurrency.py (0 moved, 35 remain; all bodies read)

Every case has a hook assign `agent.tools` (or `handoffs`) on an `Agent` shared by concurrent runs and expects "concurrent runs" rejection. Rust hooks, handoff callbacks and
`is_enabled` receive `&Agent`; a per-run replacement goes through `RunOptions::prepare_agent`, which returns a new `Arc<Agent>`. The shared mutation cannot be written (borrow checker), so there
is no behaviour to port. The 4 isolation guarantees stay ported in `py_core5_leftover1_start_hook_concurrency.rs`.

## Source bugs fixed / breaking changes

None.

## Concerns (decisions left to the owner)

1. Approval identity in a restored context (about 14 of the 36 cases of `test_agent_scoped_approvals.py`).
   - Original decision: decisions are keyed by agent name + tool name (documented in r15b and r16d as a design difference, kept).
   - Concern: a decision recorded for one tool is honoured for a different tool or an agent clone that has the same names (Python test: an approval for `low_privilege` must not authorize `privileged`,
     both named `operate`). Not covered by any preamble-valid reason.
   - Trade-off: Rust agents have no object identity that survives serialization, so closing the gap means adding a capability fingerprint (tool schema + description + origin) to each decision and refusing a
     mismatch on restore; that changes the serialized approvals shape and can re-prompt users for decisions that today carry over.
   - Options: (a) keep name keys and document the trust boundary (restored context is trusted); (b) add a tool fingerprint to sticky decisions and require re-approval on mismatch (schema bump);
     (c) fingerprint only when two agents or tools in the graph share a name (smallest change, covers the Python cases).
2. Post-invoke sibling work past the wait is abandoned (dropped) in Rust, detached in Python (`do_not_wait_indefinitely_for_sleeping_post_invoke_sibling`, second half). Keep, or spawn detached tasks?
3. `RunState::to_json` accepts an empty current agent name (Python raises "No current agent"). Refuse, or keep accepting `RunState::default()` that existing callers serialize?

## Read-depth caveats

- Read in full: all bodies of `test_agent_scoped_approvals.py`, `test_native_tool_failure_history.py`, `test_start_hook_tool_concurrency.py`.
- `test_run_step_execution.py`: bodies of every name-unmatched case read (lines 1-260, 780-3250, 3248-4247, 4540-4730); cases whose name matched a Rust test (about 55) were not re-read and are assumed ported.
- `test_run_state.py`: bodies of all 132 name-unmatched tests read; the 181 name-matched tests were not read (name match is a substring over Rust tests and comments, a comment-only mention would be a false match).
- Comment-only check: of the name-matched tests, those whose name appears only in comment lines of `tests/*.rs` (12 in run_state, 18 in run_step_execution, 11 of which are my own new ports) were located one by one in their dedicated test files (`py_core8_r8g8_tool_batch.rs`, `py_core10_r10a_item_skips.rs`, `py_core10_r10a_resume_dropped_tool.rs`, `py_core8_r8g7_run_state_misc.rs`, `py_core9_r9gaps_hosted_arguments.rs`, `py_core16_r16b_items_state.rs`); the remaining matches have the name in code. Their Python bodies were not read.
- Per-case ids are not in the ledger. The rebuilt candidate lists (run_state 185 cases, run_step_execution 53) are larger than the ledger counts (88, 44); the "after" numbers use ledger arithmetic
  (before - ported - covered), and my own count of valid python-only cases in run_state (116) is higher than 57, so some cases I list as python-only are probably already counted elsewhere.
- Parametrize multiplicities came from an AST count of the decorators, not from pytest collection (the venv was not readable).
