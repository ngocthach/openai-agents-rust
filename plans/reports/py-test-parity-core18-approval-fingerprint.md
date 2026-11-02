# Parity core18: function-tool approvals bound to a capability fingerprint

Closes the divergence recorded in `py-test-parity-core17-r17a.md` (unresolved question 1) and in the 0.19.0 known
gaps: a Rust `RunContext` keyed function-tool decisions by agent name + tool key only, so an approval for one tool
authorised a different same-named tool on a same-named agent once the context (or its exported approvals) was reused
or restored. Python binds to the owning agent object (`_FunctionToolApprovalKey` owner tokens in
`src/agents/run_context.py`).

## What changed

- `src/run_context.rs`
  - `function_tool_capability_fingerprint(agent_name, &FunctionTool)`: SHA-256 hex of canonical JSON (sorted keys,
    via the existing `tool_invocation` digest; `sha2` was already a dependency) of
    `{kind, agent, name, namespace, description, parameters, origin}`. Deterministic across processes; pinned by a test.
  - Bound records live under their own key `agent \u{1} tool_key \u{3} fingerprint`; unbound records keep the key and
    JSON shape they always had, so old `ApprovalsState` JSON deserializes unchanged and keeps authorising by name.
  - `ApprovalCapability { agent, tool_key, fingerprint }` and a per-context registry keyed by call id.
  - `approve_tool` / `reject_tool` (and `try_*`) bind automatically when the item's call id is registered for the same
    owner and tool key. Hosted MCP items are never bound.
  - Execution (`function_tool_status`): a decision bound to the running tool's fingerprint decides first; decisions
    bound to other fingerprints are ignored (treated as undecided, so the call re-prompts), for sticky and per-call
    decisions; unbound decisions follow the old path exactly.
  - `approval_status_for_item`: when the call has a registered capability, only the matching bound record or unbound
    records answer.
  - Name-only queries (`is_tool_approved`, `rejection_message`, `status`): an unbound record answers first (old
    behaviour); otherwise a bound record answers if it is the only one for that owner + tool, or the only one holding
    an exact decision for the call id (port of the name-only branch of `_approval_records_for_key`).
- `src/run_internal/tools.rs`: the runner computes the fingerprint from `env.agent.name` and the tool, passes it to
  every status lookup, and registers `call_id -> ApprovalCapability` (owner = `env.agent_key`) before emitting a
  function-tool approval interruption.
- `src/run_state.rs` / `src/run_state/wire.rs`: new field `RunState::approval_capabilities`; `approve` / `reject` /
  `try_approve` / `try_reject` bind through it; `forget_restored_function_decision` also forgets the bound key. In the
  envelope a bound record is written to `context.function_tool_approvals` like any agent-owned decision plus a
  `capability` field (Python reads entries by key and ignores it, binding the decision to its owner as it always
  does); the registry travels under the Rust-only `rust.approval_capabilities`. Unbound entries are byte-identical to
  before. Edge: two bound records of one owner and tool key (two capabilities) become two Python entries with the same
  owner and tool key, and so do an unbound (hand-built `approve_tool`) and a bound record of one owner and tool key; a Python reader keeps the last.
- `src/run_internal/mod.rs`, `src/result.rs`: `build_result` / `to_state` export the registry; resume imports it.
  Two resume paths answer a *rejected* parked call with a tool other than the one that asked (the stand-in for a
  call whose tool is gone, and an MCP call whose recipient changed). Those tools are flagged
  (`FunctionTool::resumes_registered_capability`, crate-private) so the call is matched against the capability the
  interruption registered and the rejection still answers the model, as Python does. Only reached after the decision
  was read as a rejection; an approval there is still refused as before.

## Public API additions

- `run_context::function_tool_capability_fingerprint`
- `run_context::ApprovalCapability`
- `RunContext::approve_tool_for_agent(&Agent<C>, &ToolApprovalItem, always) -> Result<()>` and
  `RunContext::reject_tool_for_agent(&Agent<C>, &ToolApprovalItem, always, message) -> Result<()>`: find the function
  tool whose qualified name is `item.tool_name` on that agent and bind to its fingerprint; a user error (nothing
  recorded) when there is no such tool or the item is a hosted MCP request. Available for `C: Send + Sync + 'static`.
- `RunContext::export_approval_capabilities` / `import_approval_capabilities` (import replaces entries with the same
  call id).
- `RunState::approval_capabilities` (public field, serde default).

`ToolApprovalItem` is unchanged.

## Breaking changes

- None at the type level for users who build `RunState` with `..Default::default()` (every literal in the repo
  does); a full struct literal would need the new field.
- Behavioural: an approval taken on a runner interruption now stops authorising if the tool's description, parameter
  schema, namespace or origin, or the agent's name, changes between the interruption and the resume. Unbound
  decisions (hand-built items through `approve_tool`, old snapshots, old `ApprovalsState` JSON) are unchanged.

## Not changed

Hosted MCP (still scoped by server label + tool name), shell, apply_patch, computer, local shell and custom tools keep
name-based decisions; `RealtimeSession` approvals and the sandbox apply-patch checker still use the name-based query.
None of the ported cases need them, and they have no `FunctionTool` to fingerprint.

## Existing tests touched

- `tests/py_core9_r9gaps_approval_identity.rs::a_snapshot_names_same_named_owners_and_accepts_name_only_owners`: it
  counted approval keys ending in `operate`; decisions taken on runner interruptions are now bound, so their key ends
  with the fingerprint. The filter now strips the `\u{3}<fingerprint>` suffix first; the expected count (1, both
  decisions folded onto the one name-only owner) is unchanged.

## Python cases ported (`tests/py_core18_approval_fingerprint.rs`)

Counts are per Python parametrization; only previously python-only rows are counted.

| Python test | Rows moved | Rust test |
|---|---|---|
| `test_converted_context_does_not_authorize_same_named_distinct_capability` (python, json) x always x streamed | 8 | `converted_context_does_not_authorize_same_named_distinct_capability` |
| `test_context_converters_preserve_distinct_same_named_owners` (python, json) | 2 | `context_converters_preserve_distinct_same_named_owners` |
| `test_name_only_query_does_not_choose_between_agent_owners` | 1 | `name_only_query_does_not_choose_between_agent_owners` |
| `test_name_only_inspection_selects_a_unique_exact_call` (none, python, json) | 3 | `name_only_inspection_selects_a_unique_exact_call` |
| total | 14 | |

Also added, not counted (already ported elsewhere or Rust-specific): `context_converters_preserve_function_decisions`
and `runner_executes_converted_context_decisions` (re-run through the bound API, streamed and non-streamed),
`runner_requires_reapproval_for_a_renamed_owner` (the `missing` row), `decision_after_state_round_trip_stays_bound_to_the_asking_tool`,
`bound_approvals_round_trip_through_json` (ApprovalsState and RunState JSON round trips, Python-shaped entries),
`pre_binding_approvals_json_still_authorizes_by_name` (backward compatibility, streamed and not),
`binding_requires_a_function_tool_of_the_agent`, `capability_fingerprint_is_stable_and_covers_the_visible_signature`.

Adaptations, applied in every ported case:

- Python's `low_privilege` / `privileged` (and the two same-named owners) differ only in their function body. A
  fingerprint cannot see a closure, so the Rust tools also differ in description.
- Python's hand-built items carry the agent object; Rust uses `approve_tool_for_agent` / `reject_tool_for_agent`.
- `name_only_query...`: Python's `get_approval_status` without an invocation returns `None` even for one owner; the
  Rust name-based `get_approval_status` is the same query as `is_tool_approved`, so that one assertion is dropped.
- `name_only_inspection...`: the "inspection is not an execution grant for the other owner" assertion uses an item
  that names an agent but carries no tool, so it is checked through the runner in the distinct-capability test instead.

## Remaining python-only (22 by ledger arithmetic, 36 - 14)

- Temporal pydantic converter rows (vendor `temporalio.contrib.pydantic`); the JSON round trip they exercise is ported
  with the `json` converter.
- `test_approval_context_does_not_retain_discarded_agent_graphs` x2: gc/weakref.
- `test_runner_requires_reapproval_for_unresolved_context_owners` `source` / `clone` / `copy` / `target` and
  `test_tied_duplicate_owner_restore_requires_reapproval`: the agents involved are content-identical (clones, shallow
  copies, identical closures), so only object identity tells them apart. A content fingerprint cannot, by design;
  these stay a documented divergence (a restored decision is honoured by an identical clone).

Count caveat: the ledger's 36 has no per-case ids. A per-parametrization recount of the candidates (24 Temporal rows,
2 gc, 4 unresolved-owner rows, 4 tied-duplicate rows, plus the 14 ported here) is 48, so some of those rows were
already counted elsewhere; the CSV uses ledger arithmetic.

## Verification

- `cargo test --workspace --all-features --no-fail-fast`: 7,858 passed, 0 failed.
- `cargo test --workspace`: 3,703 passed, 0 failed.
- `cargo clippy --workspace --all-features --all-targets -- -D warnings` and without `--all-features`: clean.
- `RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps`: clean.
- The first full run surfaced 6 regressions (rejected parked calls answered by a stand-in or a re-bound MCP tool, and
  Python-shaped owner entries missing from `function_tool_approvals`); all fixed in source as described above, with
  one test filter adapted to the new key form.

## Unresolved questions

- Should hand-built items on `approve_tool` also bind when the context knows the agent graph (e.g. a run that just
  finished)? Today only runner-raised interruptions and the explicit `*_for_agent` API bind.
