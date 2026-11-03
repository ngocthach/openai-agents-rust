# Py test parity core4: tool-name collision (`test_tool_name_collision_policy.py`)

Python file: `openai-agents-python/tests/test_tool_name_collision_policy.py` (1513 lines).
Rust files: `tests/py_core4_collision_resume.rs` (21 tests), `tests/py_core4_collision_nested.rs` (5 tests).

## Counting

- `grep -c "def test_"` = 28 functions.
- Parametrize expansions add 8 cases: `deserialize` x2, `override_all_tools` x2, `namespace` x `return_error_to_model` x4, `malformed_raw_item` x2 (L897), `malformed_raw_item` x2 (L1354), `streamed` x2. 28 + 8 = **36 cases**.

## Per-file table

| file | #tests | ported | covered | python-only | failures found | open |
|---|---|---|---|---|---|---|
| test_tool_name_collision_policy.py | 36 | 26 (26 green) | 4 | 6 | 14 (all fixed) | 0 |

- Ported (26): all green. Listed by Python test name in the test files.
- Covered (cited, `py_core3_shelltools_collision_policy.rs`, not edited): `resume_error_mode_rejects_current_collision_before_side_effects` (L79), `resume_rebinds_queued_handoff_to_current_warn_winner` (L259), `resume_rejects_missing_queued_handoff_before_side_effects` (L312; message differs: Rust "Model reused a tool call ID", Python "Tool route not found in agent agent", noted there), `cross_kind_duplicate_call_id_fails_before_execution` (L1304).
- Python-only (6 cases, 3 groups):
  1. Formatter cancellation, L416 and L1449 x2: Rust `tool_error_formatter` is a sync `Fn`; a run cannot be cancelled while a formatter is suspended on an await.
  2. Delegating-override variant, L680 `override_all_tools=True`: Rust `Agent` has no overridable `get_all_tools`. The `get_mcp_tools` variant is ported.
  3. In-flight mutation during an `await` in `get_all_tools`, L943 and L1029: Rust moves the `RunState` into the run, so no shared handle is reachable during the await. The static part of L1089 is ported.

## Rust fixes (src/)

All in `src/run_internal/mod.rs` unless noted.

1. **Resume not-found output** (`process_response`, `execute_side_effects_inner`). An approved call whose tool is gone got no `function_call_output` under `return_error_to_model`. Recorded in `ProcessedResponse::missing_approved`, emitted after the failure check so a retry re-classifies it. Test: `resume_preserves_model_order_for_function_outcomes`.
2. **MCP recipient on resume** (`process_response`). An approved MCP call resumed against a local tool with the same name ran the local tool. Now refused with the Python "different recipient binding" error. Test: `resume_warn_mode_rejects_mcp_approval_for_local_winner`.
3. **Nested replacement** (`process_response`, function and handoff branches). A queued agent-as-tool call whose nested run is interrupted cannot resume on a different tool or handoff. Refused with Python's "Cannot reconcile queued tool X with a new tool or handoff while its Agent.as_tool() run is interrupted" (`_reject_nested_replacement`). Test: `replacing_interrupted_agent_tool_fails_before_side_effects`.
4. **Persisted identity** (`process_response` not-found branch; `reconcile_resumed_approvals`). A resumed approval whose name or namespace differs from the queued call refuses with "Persisted tool identity ..." when the snapshot parked that call. Otherwise, a response-backed call the snapshot never parked holds the whole turn. Tests: `resume_rejects_conflicting_persisted_identity_before_sibling_effects`, `resume_rejects_legacy_approval_name_change_before_side_effects`, `resume_rejects_cross_kind_approval_identity_before_sibling_effects` (the cross-kind raw item is caught because the typed reader clears the call id), `resume_rejects_response_backed_approval_lookup_mismatch_before_effects`.
5. **Parent program caller** (`reconcile_resumed_approvals`). An approval whose `caller_id` names no parent program item in the response refuses with "does not match a parent program item". Test: `resume_rejects_forged_program_caller_against_parent_program`.
6. **Cleared response output, silent drop** (`reconcile_resumed_approvals`). An approved call the re-processed response no longer carries is refused with the "unique call ID" error when its tool still exists. When the tool is gone, it stays pending. Tests: `resume_rejects_function_approval_reclassified_as_handoff_live_state` / `_deserialized_state`, `reclassified_handoff_is_rejected_before_run_hook`, `resume_rejects_queued_handoff_reclassified_as_function`.
7. **Malformed approvals** (`reconcile_resumed_approvals`; `src/run_state/wire.rs` `interruption_from_wire`). An approval with no `arguments` key or no `call_id` (an `id` alone) is read with empty `arguments` or `call_id`. An unmatched malformed approval holds the whole resumed turn: nothing runs, every approval is returned as pending. Tests: `approved_malformed_approval_without_arguments_stays_pending`, `approved_malformed_approval_with_item_id_only_stays_pending`, `approved_malformed_queued_approval_with_changed_call_id_stays_pending`, `approved_malformed_queued_approval_without_type_stays_pending`.

## Test expression changes (disclosed)

- The four malformed-approval tests asserted the returned approval's call id (`"lookup_call"`). Rust's reader now clears that id on purpose, because it holds the malformed item. They now assert one pending approval named `lookup`, the closest Rust equivalent of Python's identity assertion (`lookup_approval in resumed_result.interruptions`).
- These were the only test edits. The coordinator's direction was to implement the checks. The assertions were not weakened for the other 22 tests.

## Gates

- `cargo test --all-features --test py_core4_collision_resume --test py_core4_collision_nested`: 26 passed, 0 failed.
- Regression: `cargo test --all-features` over the related runner, runstate, approval, handoff, MCP, resume and collision binaries (311 `test result: ok` lines, 0 failed lines). `py_core3_shelltools_collision_policy` also passes its `#[ignore]`d model-order case.
- `cargo clippy --all-features --test py_core4_collision_resume --test py_core4_collision_nested -- -D warnings`: clean.
- `cargo test --all-features --lib`: 651 passed, 0 failed.

## Shared-file edits

- `src/run_internal/mod.rs`: `ProcessedResponse::missing_approved` (field and literal), not-found branch in `process_response`, nested-replacement and MCP checks in `process_response`, new `reconcile_resumed_approvals` called from `resume_tools`, emission loop in `execute_side_effects_inner`.
- `src/run_state/wire.rs`: `interruption_from_wire` keeps a missing `arguments` or `call_id` empty. The generated-item reader keeps its `"{}"` default.
- Another agent edited `reconcile_resumed_approvals` in parallel to add server-managed program evidence. Their change is kept.
- No edits to `src/lib.rs`, `Cargo.toml`, or any `py_core3_*` file. `py_core3_shelltools_collision_policy.rs:200` still has `#[ignore]`. That file belongs to its owner, and the model-order case now passes when run with `--ignored`.
- No public API changes.

Status: DONE

## Follow-up: side effects of the resume-path changes (fixed)

Nine tests outside this file failed after the resume-path changes. Root causes, both in `reconcile_resumed_approvals` (`src/run_internal/mod.rs`), with one wire-level guard:

1. Hosted MCP approval requests (`mcp_approval_request`, `tool_origin` None) were treated as orphaned approved calls and re-pended. Fix: hosted approval requests are excluded from reconciliation, since they are answered by their own path.
2. Approvals that surface from an in-flight nested agent-as-tool run were treated as orphans of the outer response. Fix: while `ctx.nested_states()` is non-empty, absent approvals are left to the nested run. This is what fixed `nested_agent_tool_hitl_resume_with_duplicate_agent_names`. It was not a name-identity issue, because the check does not compare agent names.
3. `src/run_state/wire.rs` `interruption_from_wire`: the missing-`arguments` and missing-`call_id` clearing now skips `mcp_approval_request` items, which legitimately have no arguments and use `id` as their call id.
4. The malformed-approval hold only applies to function-origin approvals.

Results: `py_core2_approvals_reuse_b`, `py_core2_runner_routing`, `py_core2_runstate_resume`, `py_deferred_writes_e`, `py_core4_collision_resume`, `py_core4_collision_nested`: all exit 0. Lib: 651 passed. Clippy `-D warnings` on both collision targets: clean.
