# Python test parity, core round 2: approvals

Scope: `test_tool_approval_call_id_reuse.py`, `test_run_context_approvals.py`,
`test_run_internal_approvals.py`, `test_run_state_compatibility_corpus.py`,
`test_run_state_pending_input.py`, `test_run_state_agent_identity.py`, plus the 14 cases the
scoped-approvals agent deferred to this area (coordinator request).

Counting: no pytest in the reference venv, so cases = `def test_` x product of literal
`parametrize` list lengths (AST script), with the corpus's dynamic `SOURCES` parametrizations
expanded from `sources.json` (21 versions, 20 features). That gives 123 for the corpus, not the
84 in the task brief (the brief/core-a figure predates the corpus growing); both are noted.

## Per-file table

| Python file | cases | ported | covered | python-only | not ported | failures found |
|---|---|---|---|---|---|---|
| test_tool_approval_call_id_reuse.py | 130 | 116 | 0 | 12 | 2 | ~20 (see bugs 1-12) |
| test_run_context_approvals.py | 31 | 27 | 1 | 3 | 0 | hosted MCP identity model missing (bug 13) |
| test_run_internal_approvals.py | 7 | 4 | 0 | 3 | 0 | 0 |
| test_run_state_compatibility_corpus.py | 123 | 47 | 0 | 76 | 0 | value-disclosing agent error (bug 16) |
| test_run_state_pending_input.py | 54 | 0 | 12 | 0 | 42 | - |
| test_run_state_agent_identity.py | 18 | 0 | 15 | 3 | 0 | - |
| scoped-approvals deferrals (agent_scoped_approvals + hitl_error_scenarios) | 14 | 14 | 0 | 0 | 0 | bugs 14, 15, 17, 18 |
| **total** | **377** | **208** | **28** | **97** | **44** | |

Rust test files (all green): `tests/py_core2_approvals_context.rs` (25),
`tests/py_core2_approvals_reuse_a.rs` (30), `tests/py_core2_approvals_reuse_b.rs` (36),
`tests/py_core2_approvals_binding.rs` (8), `tests/py_core2_approvals_corpus.rs` (6). Many Rust
tests loop over Python's parametrizations (streamed x non-streamed, converters, schema versions).
Python corpus fixtures copied to `tests/fixtures/py_run_state_corpus/`.

### Classification notes
- call_id_reuse python-only (12): pydantic serializer warnings / `model_dump` signature probes
  (3); cancelling an async `tool_error_formatter` (Rust formatter is sync) (1);
  `bound_call_id_with_missing_arguments` mutates the private `_restored_unbound_approval_call_ids`
  from inside a tool and uses `model_construct` (1); five tests need a raising
  `on_llm_end`/`on_tool_end`/`on_tool_start`/`on_handoff` hook (Rust hooks return `()`;
  `failed_handoff_hook` was ported by failing the handoff callback instead) (5);
  `deferred_sticky_decision_preserves_mirrored_per_call_binding` (2) relies on the
  `tool_lookup_key` / `_allow_bare_name_alias` item fields Rust's `ToolApprovalItem` does not have.
- call_id_reuse not ported (2): `test_legacy_changed_pending_call_fails_before_approval_callback`
  drives private `_collect_runs_by_approval` with a hand-built mismatched pending item; the
  observable behaviour is covered in spirit by `legacy_schema_historical_sticky_call_id_is_not_reexecuted`
  (changed) but not ported 1:1.
- Unit tests of `get_approval_status(current_invocation=...)`,
  `collect_manual_mcp_approvals`, `process_hosted_mcp_approvals`, `_dedupe_*` are ported through
  `Runner` (same inputs, same errors/ordering).
- run_context_approvals python-only (3): non-string request id `123` (Rust ids are `String`),
  `provider_data` wrapper with integer id, deferred-top-level bare-name alias key
  (`deferred_top_level:get_weather`, no alias concept in Rust). Covered (1):
  `deferred_top_level_approval_does_not_alias_to_visible_bare_sibling` by
  `explicit_same_name_namespace_does_not_alias_to_bare_tool`.
- run_internal_approvals python-only (3): `_build_function_tool_call_for_approval_error` helper
  conversions of arbitrary dict/objects.
- corpus ported (47): supported-versions check (1), minimal rewrite idempotence (21), feature
  rewrite (20, parse + idempotent rewrite + durable topology, not Python's full
  subset-comparator), approval decisions control resumed runs (3), invalid fixtures (2 of 3 here
  + malformed agent). Python-only (76): `integration_tests._contract_state` comparator self-tests
  (8), monkeypatched-`from_json` immutability (4), sandbox credential fixtures (4; sandbox area,
  Docker driver options), and ~60 exception/traceback/redaction-of-Python-objects tests
  (ExceptionGroup, frames, slots, `__notes__`, hostile descriptors, cancellation).
- pending_input / agent_identity: not re-ported this round (context budget spent on the 130-case
  call-id file and the coordinator's 14 deferrals); counts are core-a's triage:
  pending_input 12 covered by `core_python_run_state.rs`, 42 **not ported**; agent_identity 3 in
  `py_core_a_run_loop.rs` + 12 in `core_python_run_state.rs` (15 covered), 3 sandbox-manifest
  python-only.

## Rust bugs fixed (src)

Call-id reuse / invocation ledger (`src/run_internal/mod.rs`, `tool_invocation.rs`, `tools.rs`,
`local_tools.rs`):
1. Native calls (shell, apply_patch, computer, local shell) are scoped by the tool they resolve
   to (`tool_invocation_identity_named`, `native_tool_name`): the same call id replayed onto a
   replacement tool is "reused a tool call ID".
2. Empty call id on a tool invocation fails with "Tool invocations require a non-empty string call
   ID before execution." before approval callbacks/formatters/handoffs (`lacks_call_id`).
3. Response validation before `on_llm_end` (`validate_response_invocations`): changed same-id
   siblings ("in one response"), uncanonical repeated ids, changed replay of a completed call
   ("reused a completed tool call ID"), ledger mismatch; `on_llm_end` is not called for them.
4. Exact replays of completed calls are dropped (not recorded, not executed, no
   `tool_called` event) and so is the reasoning item tied to them; the turn still runs again.
5. "Run history reused a tool call ID for different completed invocations" check.
6. apply_patch fallbacks: function calls named `apply_patch` now route to the apply_patch tool
   (`apply_patch_fallback_call`, `routes_to_apply_patch`), and fallbacks are identified as the
   `apply_patch_call` they run as.
7. "already executed" guard: a call whose user code started but whose output never committed is
   not run again (shared `RunContext` and the snapshot ledger's `executed` flag); nested
   agent-tool interruptions unmark it.
8. Nested `as_tool` runs get their own invocation lifecycle (`with_fresh_invocations`), and
   `resumed_pending_approvals` only holds the resumed agent's own call ids (outer/inner call-id
   collision).
9. Custom calls whose tool vanished on resume: undecided stay pending, rejected still emit their
   rejection output, approved fail "Tool X not found".
10. Hosted MCP requests that cannot be bound (missing arguments or server label) never reach
    `on_approval_request` and are always re-asked; legacy snapshots mark such pending items
    unbound (`reconstruct_legacy_pending_bindings`), and a missing `arguments` is no longer
    defaulted to `"{}"` for MCP requests (`run_state/wire.rs`).
11. Legacy (<1.15) snapshots rebuild completed ledger records for function/custom/MCP calls found
    in history (incl. original input), so historical replays are skipped / refused.
12. Deferred top-level lookup key (`namespace == name`) in the approval scope digest.

Approval identity model (`src/run_context.rs`, `run_state/wire.rs`):
13. Hosted MCP decisions keyed by typed identities (`server_tool`, `request`, `query`) instead of
    `agent + "label:tool"`: sticky decisions scoped by server label; exact name-only query mirror;
    legacy bare-name decisions only count per exact call; incomplete requests use exact-call
    decisions only; `try_approve_tool`/`try_reject_tool` (+ `ApprovalsState::try_*`) return
    Python's `UserError`/`ModelBehaviorError` without mutating state; sticky decisions replace
    the opposite decision wholesale (Python `_apply_approval_decision`); rejection message comes
    from the deciding record. Serialized as Python's `context.hosted_mcp_approvals`.
    New `RunContext::approval_status_for_item` / `ApprovalsState::status_for_item`.

Coordinator's deferred scoped-approval bugs:
14. (bug 1) Decisions bind the exact call: `approve_tool`/`reject_tool` and `RunState::approve`/
    `reject` record the invocation (`approval_binding`) in a context ledger / the state ledger;
    a run imports the context ledger, so a reused context or converted context refuses changed
    arguments and skips a completed call. New public `RunContext::export_tool_invocations` /
    `import_tool_invocations` (Python converters carry `_tool_invocations`). Local tools also
    refuse a decision recorded for changed content while their approval rule was waiting.
15. (bug 2) Resuming with an agent that no longer has a pending call's tool keeps the approval
    pending (`call_has_tool`, `unmatched_pending`).
16. Missing current agent error is value-free: "Run state agent not found in agent map"
    (was `Could not find agent '<name>' ...`, which leaked snapshot data).
17. (bug 3) A snapshot without its model response rebuilds the parked calls from its pending
    approvals (`resume_tools`).
18. (bug 4) An edited parked call is refused before tool-inventory `is_enabled` callbacks run
    (`validate_parked_response`).

## Shared-file edits
- `src/lib.rs`: `#[doc(hidden)] pub use run_internal::tool_invocation::tool_invocation_identity_for_tests;`
- Other agents' test files, assertions corrected to Python: `tests/py_core2_runstate_items.rs`
  `missing_agent_in_map_error` (Python's message), `tests/py_core2_scopedapprovals_hitl_decisions.rs`
  `resolved_function_decisions_skip_the_checker` (approves the exact calls, as Python does).
- No Cargo.toml edits.

## Breaking changes
- Hosted MCP approval keys changed (in-memory `ApprovalsState` keys for hosted MCP are reserved
  `\u{1}\u{2}...` keys; old `label:tool` decisions are still read).
- A decision now binds its exact call: approving call id X then seeing X with other arguments
  errors instead of running.
- Missing-agent error text changed.
- `RunContext` gained public `try_approve_tool`, `try_reject_tool`, `approval_status_for_item`,
  `export_tool_invocations`, `import_tool_invocations`.

## Not mine / observed in-flight failures (other areas)
- `py_core2_runstep_execution::execute_tools_runs_hosted_mcp_callback_when_present` ("caller
  programmatic, but the tool allows only direct") - tool_caller area.
- `core_python_run_state::python_state_re_serialized_by_rust_keeps_the_python_envelope`: only
  difference is the new `current_response_boundary_unproven` field (another agent's in-flight).
- `py_deferred_writes_e` checkpoint tests flapped while `failure_committed` was being added in
  tools.rs by another agent; green on the last run.

## Gates
- My 5 test binaries + `--lib` green (lib 642).
- `cargo clippy --all-features --lib --tests(mine) -- -D warnings` clean.
- Selected 70 approval/run-state/runner/hitl/deferred binaries run: only the in-flight failures
  above. A full `cargo test --workspace` was not possible (disk at 99% made linking fail).
- rustfmt on my test files only.

Status: DONE_WITH_CONCERNS
Summary: Ported 208 cases (+28 covered, 97 python-only) across the approvals files and the 14 scoped-approval deferrals; fixed 18 Rust divergences (call-id reuse ledger, hosted MCP identity model, decision binding, rebuild/unmatched resume).
Concerns/Blockers: test_run_state_pending_input.py (42 cases) not ported this round; corpus count is 123 vs 84 in the brief; full workspace test not run (disk); a few failures belong to other agents' in-flight edits.
