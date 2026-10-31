# Python test parity, core round 2: runstate (`test_run_state.py`)

## Counting
`ast` walk over `tests/test_run_state.py`: 312 `test_*` functions (the grep count of 313 includes a
nested `async def test_tool()` helper) and **469 cases** once `parametrize` is expanded (product of the
literal lists). The first-round report's 470 counted the nested helper.

| Python file | #tests (cases) | ported | covered | python-only | not ported | failures found |
|---|---|---|---|---|---|---|
| test_run_state.py | 469 | 293 | 10 | 165 | 1 | 25 Rust divergences (all fixed, below) |

By definition: ported 201, covered 9, python-only 101, not ported 1 (312 total). Of the ported cases,
9 definitions (about 12 cases) assert a documented deliberate Rust difference (see below).

Also ported for the runner agent (cases of `test_agent_runner_streamed.py`, counted in the runner report,
not above): `test_ambiguous_serialized_approval_state_fails_before_tool_execution` (10) and
`test_serialized_mixed_approval_guardrail_preserves_only_accepted_outputs` (16).

## Test files (all green)
- `tests/py_core2_runstate_envelope.rs` (45): envelope, schema-version errors and redaction, version gates,
  guardrail boundary (1.20), approval records.
- `tests/py_core2_runstate_approvals.rs` (26): ledger binding, unbound decisions, completed-invocation check,
  legacy 1.13/1.14 reconstruction, context override, approve/reject resumes. Expected outcomes were confirmed
  by running the same scenarios through the Python `Runner`.
- `tests/py_core2_runstate_items.rs` (47): item (de)serialization, input normalization, processed-response
  groups, handoff outputs, ToolApprovalItem names/namespace.
- `tests/py_core2_runstate_resume.rs` (23): `TestRunStateResumption`, nested agent-as-tool decisions.
- `tests/py_core2_runstate_sidecars.rs` (45): sandbox payload (sanitizer and envelope), nested-history digest
  and sidecars (digests hard-coded from Python), generated/session index map, fallible approvals, hosted MCP
  records, item metadata, `tool_input`, strict `local_shell_call_output`, programmatic resume (incl. hosted MCP),
  deliberate-difference tests (`rust_refuses_...`, `rust_keeps_...`).
- `tests/py_core2_runstate_boundary.rs` (5): the 26 runner-deferred cases, the legacy-1.16 refusal, and the
  nested-handoff list surviving a snapshot.

## Rust bugs fixed
Round 1 (15): Python's exact schema-version/parse errors with no payload echo; version gates (programmatic
items < 1.13, agent-owned approvals < 1.18, boundary < 1.20); `tool_output_guardrail_result_start` round trip,
validation and use as the resumed redaction window; wrong-type tool guardrail rows dropped;
`reasoning_item_id_policy` persisted and applied; `to_json_with_tracing_api_key`; ledger only read from 1.15,
null/ill-typed entries refused; completed ledger entries must match a restored call and output; legacy
(<1.15) pending approvals bound to the interruption; historical call ids without a ledger entry are unbound;
assistant input normalized on write; tool-use tracker filtered per entry; deferred top-level lookup key;
handoff-output agent fallback and derived `output`/`tool_name`.

This round (10):
16. `sandbox` payload: read and written as opaque JSON (`RunState::sandbox`), carried through a resumed run;
    `src/run_state/sandbox_payload.rs` ports `sanitize_run_state_sandbox_mount_authority` (envelope and manifest
    validation, per-mount field/URL/strategy/pattern/rclone sanitization, credential-file source blanking,
    redacted-authority marker). Errors are fixed strings that never name a value.
17. Nested-history ownership: `RunState::nested_history_owned_items` is written as `{index, digest, input_index}`
    with Python's digest (`json.dumps(sort_keys)`, `ensure_ascii`, assistant normalization) and read back with
    Python's validation; `generated_session_item_indexes` validated with the same warnings. The loop persists and
    hydrates the list, so a resumed run no longer repeats items a nested handoff already carried
    (`src/run_state/nested_history.rs`, `run_internal/mod.rs`).
18. `RunState::try_approve` / `try_reject` (Python's raising `approve`/`reject`): refuse a recognized call
    without an id, per-call decisions without an id, items with no canonical invocation identity, hosted MCP
    requests without request id / server label / tool name (hosted branch delegated to the approvals agent's
    `ApprovalsState::try_*`, function branch checked in `run_state.rs`).
19. `RunItem::tool_lookup_key` and `allow_bare_name_alias` survive a read and are written for approval items.
20. `tool_input` persisted at `context.tool_input` and restored when the resuming context carries none.
21. `local_shell_call_output` items validated strictly on read (non-empty string call id, string output).
22. Typed hosted MCP records are ignored below schema 1.14.
23. Programmatic resume: hosted MCP `mcp_list_tools` / `mcp_approval_request` / `mcp_call` now obey the server's
    `allowed_callers` (was unchecked) (`run_internal/mod.rs`).
24. "Current response boundary cannot be proven": the wire loader ports `_restore_current_response_item_identities`
    plus `_current_response_boundary` (ownership range vs generated items, interruption indexes, processed
    items, terminal-suffix aliasing in the session history through a valid index map, turn-one fallback) into
    `RunState::current_response_boundary_unproven`; resuming such a state under output guardrails fails with
    Python's message before any tool runs. An unproven state keeps that on the wire (unaliased indexes, no
    ownership), so a second round trip cannot make it look proven.
25. Items without an agent (no `agent`, `source_agent`, `target_agent`) are now an error instead of loading
    with an empty agent.

## Shared-file edits
- `src/run_state.rs`: re-exports `SUPPORTED_SCHEMA_VERSIONS`; new fields (below).
- `src/items.rs`: `ToolApprovalItem::{tool_namespace, qualified_name}`; `RunItem::{tool_lookup_key,
  allow_bare_name_alias}`.
- `src/run_internal/mod.rs` (small, re-read before each edit): reasoning-policy / sandbox / tool_input /
  nested-ownership hydrate and write, boundary refusal, hosted MCP `allowed_callers`, custom-tool description
  in the processed-response snapshot, session-history reasoning policy uses the resolved policy.
- `tests/runner_tests.rs`: `structured_output_is_validated` (see below); `tests/py_core2_runstep_execution.rs`: one config line (see below).
- `docs/project-changelog.md`: new "Unreleased (0.6.0)" section.
- No `src/lib.rs` / `Cargo.toml` edits.

## Breaking changes (0.6.0 tree)
- New public fields: `RunState::{reasoning_item_id_policy, tool_input, sandbox, nested_history_owned_items,
  current_response_boundary_unproven}`, `ProcessedResponseSnapshot::tool_output_guardrail_result_start`,
  `RunItem::{tool_lookup_key, allow_bare_name_alias}`. Struct literals need `..Default::default()` / `RunItem::new`.
- Error message texts for schema/ledger/parse failures are Python's.
- **Documented difference (decided by coordinator): `approve`/`reject` stay infallible.** Python's raise is
  available through `try_approve` / `try_reject`; the infallible forms record nothing for an unbindable item.
  `RunState::approve`/`reject` stay infallible and `try_approve`/`try_reject` return `Result`. Reason: about 180
  call sites in about 50 test files owned by other agents call `state.approve(..)` as a statement; changing the
  return type makes every one an `unused_must_use` warning, so `clippy --all-targets -D warnings` fails across
  the crate, and I may not edit those files. The approvals agent had meanwhile made the underlying
  `ApprovalsState::approve` record nothing for an unbindable item (same effect as Python's raise leaving
  `_approvals` empty), so no decision is silently misrecorded. Flipping the signature is mechanical (add `?` /
  `.unwrap()` at the call sites). Not planned: the coordinator decided to keep the signature.

## Deliberate differences from Python (also in `docs/project-changelog.md`)
Asserted by tests with `rust_refuses_...` / `rust_keeps_...` names in `py_core2_runstate_sidecars.rs`:
- an item without an agent, without a type, or with an unreadable raw item is an **error**, not a skip
  (Python's 6 skip cases: 4773, 4801, 10862, 10971, 11040, 11208): a skipped item is history the resumed run
  never sees.
- a `tool_call_output_item` whose raw item has no `type` is **kept** (Python's union adapter drops it): the
  payload is plain JSON and nothing is lost.
- an unknown mount-like entry in the sandbox payload is an error (also Python's behavior for unregistered types).

## Python-only (165 cases), grouped
- Pydantic/deepcopy snapshot internals of `get_interruptions` (copy hooks, descriptors, typed extras, cyclic
  data, non-finite floats, uncopyable payloads).
- Live-object identity: approvals owned by `id()`, current-vs-nested ambiguity, deep-copied or equal-but-distinct
  generated/session items, items skipped because their agent is not in the map.
- Context serializer/deserializer/pydantic/dataclass context metadata and `RunContextWrapper` rebinding.
- Private helpers called with live agents/tools: `_deserialize_processed_response`, `_lookup_function_name`,
  `_serialize_item` with mock classes, datetime/bytes/BaseModel output coercion, `_AgentToolResumeCheckpoint`,
  `_tool_invocation_status` and the private `get_approval_status(existing_pending=, current_invocation=)`
  probes (about 12 cases), registry monkeypatching for reserved mount collisions (2).
- 3 of the 7 `approval_decision_requires_canonical_invocation` cases (a `function_call`/`mcp_approval_request`
  without `arguments`, an unknown call with only an item id): `FunctionCall.arguments` is always present.

## Not ported (1 case)
- `test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval`: legacy (<1.15) hosted per-call
  decisions are not marked unbound on read in Rust (Python re-asks). Small; not closed this round.

## Scoped-approvals failure (`legacy_sticky_decisions_require_approval_after_restore`)
Not caused by my changes, and now green. A probe showed the policy of a resumed, undecided pending call was
re-run at every schema version (1.21, 1.17, 1.14) with no restored decision, so the stricter ledger read, the
1.18 approvals gate and the history-unbound ids were all uninvolved (it also reproduced with my history change
disabled). The approvals agent's later `run_context.rs` / `tools.rs` fix resolved it; the test passes.

## Late regressions caused by my changes, fixed
- History call ids are no longer marked unbound for hosted MCP requests (the runner writes their ledger entry
  when the approval is raised, so a missing entry meant nothing); this had broken
  `py_deferred_writes_e::a_checkpointed_mcp_approval_survives_callback_failure_and_json_retry`.
- Approval/handoff call items with a payload that is not a function call (`{"type": "invalid"}`) are kept with
  empty defaults instead of failing with "missing field `call_id`" (Python keeps the dict); this was the
  tool-errors agent's report on `tool_approval_item_with_an_unrecognized_raw_item_is_kept`.
- Interruptions now keep the item `id` and the lookup-key/alias metadata of the matching history item, so the
  serialized interruptions equal the serialized approval items (Python compares them); without it a re-read
  Python fixture lost its proven boundary (`core_python_run_state`).
- `tests/py_core2_runstep_execution.rs`: the hosted MCP helper now sets `allowed_callers` to
  `["direct", "programmatic"]`, because hosted MCP requests from a program obey their server's
  `allowed_callers` (Python's processing does; that Python test bypasses processing).

## `runner_tests::structured_output_is_validated`
Not my change. The invalid-JSON validation error is now data-redacted (`error.rs` `is_data_redacted`,
`agent_output.rs`), and a redacted error deliberately carries no `run_data` (`run_internal/mod.rs`, Python's
`DONT_LOG_MODEL_DATA` default). The test predates that and asserted `run_data().is_some()` under the default
policy. Fixed in the test: the default case asserts no run data and no payload in the message; with
`set_dont_log_model_data(false)` it asserts run data is attached.

## Gates
- New and existing runstate tests green (`py_core2_runstate_*`, `core_run_state`, `core_python_run_state`,
  `core_trace_resume`, `runner_tests`).
- clippy `-D warnings` is clean on the lib and my test targets plus `runner_tests` (the earlier
  `tool_invocation.rs:328` lint was fixed by its owner).
- rustfmt applied to `src/run_state*`, my tests and the new modules.

Status: DONE_WITH_CONCERNS
Summary: 469 cases: 293 ported (incl. deliberate-difference tests), 10 covered, 165 python-only, 1 not ported; sandbox payload, nested-history sidecars, item metadata, tool_input, programmatic resume and the output-guardrail boundary refusal are implemented, plus the runner's 26 deferred cases.
Concerns/Blockers: 1 legacy hosted-MCP case unported. `approve()`/`reject()` stay infallible by coordinator decision (documented difference; `try_approve`/`try_reject` are the fallible path).
