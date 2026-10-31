# Python test parity — core2 / runstep

Scope: `test_run_impl_resume_paths.py`, `test_run_step_execution.py`, `test_run_step_processing.py`,
`test_process_model_response.py`, `test_run_internal_items.py`, rest of `test_max_turns.py`.

Counting: AST walk of each file, `def test_*` × product of inline `@pytest.mark.parametrize`
list lengths (scratchpad `c2rs/count.py`). Matches task numbers: 198 / 111 / 17 / 29 / 54 / 52
(52 − 9 already in `tests/py_core_b_max_turns.rs` = 43).

Approach: Rust's step machinery (`process_response`, `execute_side_effects`, item hygiene) is
private and shaped differently from Python's free functions, so every port drives the same model
output through `Runner` / `RunState` and asserts the decision Python checks (items, tool runs and
args, next step = final output / run again / handoff / interruption, error type+message). No
`#[doc(hidden)]` seam was added.

## Per-file table

| Python file | #tests | ported | covered | python-only | not ported | failures found |
| --- | --- | --- | --- | --- | --- | --- |
| test_max_turns.py (rest) | 43 | 43 | 0 | 0 | 0 | 5 bugs (1-5) |
| test_run_internal_items.py | 54 | 37 | 0 | 7 | 10 | 3 bugs (6-8) |
| test_run_step_processing.py | 17 | 17 | 0 | 0 | 0 | 0 |
| test_process_model_response.py | 29 | 26 | 0 | 2 | 1 | 5 bugs (9-13) |
| test_run_step_execution.py | 111 | 51 | 0 | 46 | 14 | 6 bugs (14-19) |
| test_run_impl_resume_paths.py | 198 | 10 | 0 | 5 | 183 | 0 |
| **total** | **452** | **184** | **0** | **60** | **208** | 19 |

Rust files (all green, `--all-features`):
- `tests/py_core2_runstep_max_turns.rs` — 36 tests (43 Python incl. one 8-way matrix test)
- `tests/py_core2_runstep_items.rs` — 37
- `tests/py_core2_runstep_processing.rs` — 43 (17 + 26)
- `tests/py_core2_runstep_execution.rs` — 51
- `tests/py_core2_runstep_resume.rs` — 3 (10 Python incl. one 8-way matrix)

## PYTHON-ONLY groups
- items (7): non-dict input entries (`"raw-item"`), `_BrokenModelDump` / monkeypatched `json.dumps`
  fingerprint failures, `CountingSequence` `__getitem__` read counting, `extract_mcp_request_id`
  on objects with `provider_data` attributes / raising `__getattr__` (Rust items are `Value`).
- process_model_response (2): `FalsyShellTool` / `FalsyApplyPatchTool` overriding `__bool__`.
- run_step_execution (46): asyncio task semantics — `asyncio.CancelledError` raised by tools and
  sibling-cancellation drains, parent-task cancellation, `asyncio.shield`, eager task factory,
  `ContextVar` propagation, `BaseException` subclasses, monkeypatched `asyncio.wait` ordering,
  `_execute_tool_plan` category cancellation (5), plus internals with no Rust analogue: unhashable
  dict `arguments`, returning exception objects as outputs, `ProcessedResponse.has_interruptions`
  on a dataclass, direct `execute_function_tool_calls` with an unbound tool.
- resume_paths (5): tests that monkeypatch `run_single_turn` / `resolve_interrupted_turn` /
  `execute_final_output` / event-queue timing (`final_output_short_circuit`,
  `persistence_uses_saved_count`, `run_again_resets_persisted_count`, `stream_waits_for_event_consumption`×2).

## NOT PORTED (behavioural, left for a follow-up; target file in brackets)
- items (10) [`py_core2_runstep_items.rs`]: programmatic structured tool errors encoded as
  `{"error": …}` JSON (2) — Rust has no `function_tool_error_output` equivalent: a rejected/failed
  tool with `output_json_schema` called from a `program` caller still gets plain text
  (`src/run_internal/tools.rs` rejection/failure sites); nested-history owned-item refs
  (`filter_…`, `reconcile_…`×2, `normalize_resumed_input` clean nested, digest fixed-size,
  digest status-equivalence) (6); fingerprint ignores `_agents_tool_*` metadata (1; source fixed,
  no test); `coerce_tool_search_output_raw_item` legacy type (1).
- process_model_response (1): local MCP function tool title on the run item (needs FakeMCPServer).
- run_step_execution (14) [`py_core2_runstep_execution.rs`]: hosted-MCP approval identity/scope
  matrix (stale pending identity, legacy name, scoped decision after conflict, incomplete exact
  decisions, partial pending identity, `resolve_interrupted_turn` scoped/callback-owned/public
  agent, omits completed replay ×2, public agent for interruptions).
- resume_paths (183) [`py_core2_runstep_resume_b.rs`]: callback-retry history/order (8+4+4),
  failed-approval-callback stream deferral (9), redacted callback checkpoint (4), preapproval /
  cancelled-callback stream checkpoints (2), callback failure → session (4), resumed session append
  recovery / ambiguity / repeated failure / renewed interruption / compaction refresh
  (16+6+1+4+4+2+1+2+1), pending-write checkpoint validation (17+1+2+1), server-managed flag (3),
  legacy approval name fallback (12), handoff append recovery & fresh streamed handoff
  (16+1+1+1+4+2+1+4+12+1), terminal-marker family (16+2+2+1+1+1+1), legacy callback failure (4+4).
  Several of these overlap `tests/py_deferred_writes_*.rs` (another area) but were not
  individually verified as COVERED.

## Rust bugs fixed (src)
`src/run_internal/mod.rs`
1. Max-turns handler output skipped output guardrails and `on_agent_end`/`on_end` hooks; now
   validated, gated, a guardrail error still persists the output, a tripwire persists nothing.
2. Handler output validation: wrapped (list) output types rejected bare values; invalid output
   raised `ModelBehaviorError` instead of `UserError("Invalid run error handler final_output…")`.
3. Max-turns handler output was streamed before being persisted, and input was written in the same
   batch (a failing/cancelled final save lost the user input). Input is saved first; the message
   is published only after the session holds it; on save failure it is withdrawn from the result.
4. Refusal / invalid-final-output handlers did not append the synthesized message (and with
   `include_in_history=false` deleted the model's own message).
5. Resuming a `RunState` with `final_output` set was refused; Python resumes it (see Breaking).
9. Shell/apply_patch calls and shell outputs kept server `created_by`; now sanitized before being
   recorded/executed. Missing shell/computer/apply_patch/local-shell tools now raise Python's
   messages; a shell call with no shell tool now raises (was silently ignored).
10. `custom_tool_call` named `apply_patch*` with an ApplyPatchTool now runs as an apply_patch
    pseudo-call (`normalize_apply_patch_fallback_call`); unknown custom tools raise
    "Tool X not found in agent Y".
11. Function lookup: last duplicate wins, deferred top-level tools resolve via the synthetic
    `name.name` shape and yield the bare name to a visible tool, namespaced calls never route to a
    handoff, qualified-name collisions between namespaced and dotted tools raise `UserError`,
    not-found errors name `ns.tool`. Deferred tools no longer count in bare-name collision checks.
12. Hosted `mcp_call` items get description/title from `mcp_list_tools` metadata.
13. MCP approval responses copy the request's `caller`.
16. Hosted `shell_call_output` and server `tool_search_*` items count as tool activity (a turn
    holding only them runs again).

`src/run_internal/items.rs`
6. Program retention: a bare unanswered call no longer keeps its program active (only hosted
   items, tool outputs, pending hosted shells do); orphan program chains are dropped.

`src/run_internal/mod.rs` (`load_session_history`) / `src/run_internal/oai_conversation.rs`
7. Session history read now strips `_agents_tool_description`, `_agents_tool_title`,
   `created_by` and applies the reasoning-id policy (non-remote sessions); fingerprints ignore the
   `_agents_tool_*` keys.

`src/run_internal/tools.rs` / `src/run_internal/local_tools.rs`
14. Function-tool error spans now carry `{"tool_name", "error"}` data (fatal and non-fatal).
15. Rejected-tool span: data `{tool_name, "Tool execution for <id> was manually rejected by
    user."}`, message = rejection text only when sensitive data is traced, output set likewise.
17. Function spans use the qualified name (`shipping.get_shipping_eta`).
18. Concurrency cap: after a failure, draining finished siblings no longer starts queued calls.
19. Enablement is re-checked before a batch runs (`Tool X is currently disabled for agent Y.`);
    deferred tools called under their synthetic namespace see it in `ToolContext::tool_namespace`.
- `LocalCall::raw()` helper added (local_tools.rs).

## Breaking changes
- A finished `RunState` (with `final_output`) is now resumable instead of returning
  `UserError("This run state already finished…")` — Python parity.
- Error-handler outputs for structured agents: invalid value → `UserError` (was `ModelBehaviorError`).
- Shell call without any shell tool → `ModelBehaviorError` (was ignored).
- Function span names for namespaced tools are qualified.

## Cross-area failures observed (not caused by this area's edits; not fixed)
Seen in `cargo test --all-features --no-fail-fast` while ~12 agents edit concurrently:
- `src/run_internal/oai_conversation.rs` unit test `an_item_the_server_already_holds_is_not_resent`
  (tracker `primed_from_state` logic edited by another agent).
- `tests/runner_tests.rs::tool_name_collisions_follow_policy` asserts "collision" but the message
  is now Python's "Ambiguous tool routing configuration…" (test stale); `structured_output_is_validated`
  expects `run_data` on the validation error; `agents_as_tools_and_nested_approvals` fails with
  the ledger "already executed" error (tool_invocation.rs/wire.rs edited 11:42 by another agent).
- `tests/py_memory_session*.rs` repeating session callback grows the session (callback matching in
  `load_session_history` rewritten by another agent).
- Others failing in that run: `py_core2_approvals_reuse_a`, `py_core2_runconfig_lifecycle`,
  `py_core2_scopedapprovals_agent/hitl`, `py_deferred_writes_e`, `py_core2_toolerrors_batch`
  (hung once), `py_sandbox_a_runtime_4`, `sandbox_tests` — owners should re-check after their edits.
- Clippy `-D warnings` fails only in `src/run_internal/tool_invocation.rs:328` and
  `src/run_state/sandbox_payload.rs:363` (other agents); clean in my files.

## Shared-file edits
None (`src/lib.rs`, `Cargo.toml` untouched).

## Gates
- `cargo test --all-features` for my 5 binaries + `py_core_b_max_turns` + `py_core_a_run_loop`: all green
  (36+37+43+51+3+9+9).
- `--lib`: 641 pass / 1 fail (cross-area, above).
- rustfmt applied to my 5 test files only. Output dir deleted at the end.
