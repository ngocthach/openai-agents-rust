# Audit fixes — core area (handoffs, local tools, guardrail cancellation)

Date: 2026-10-30. Source of tasks: `plans/reports/independent-parity-audit-report.md`.
Python reference: `/Users/ngocthach/myGit/openai-agents-python/src/agents/`.

Every row re-verified against both codebases before changing anything.

## Closed rows

### Finding 1 — `Handoff.failure_error_function` (was a real gap)

Re-verified: `handoffs/__init__.py:211` + `run_internal/turn_resolution.py:605-632` vs Rust
`handoffs.rs` (no field) and `run_internal/mod.rs` (unconditional `Err(e)`).

- `src/handoffs.rs`: new `pub failure_error_function: Option<ToolErrorFn<C>>` on `Handoff<C>`
  (default `None` in `new` and `custom`, like Python — *not* the `FunctionTool` default
  formatter), added to the manual `Clone`, plus `Handoff::with_failure_error_function`.
- `src/run_internal/mod.rs::execute_handoffs`: on an invocation error with a formatter set the
  span records `"Handoff invocation failed"` (Python's message, empty data), a
  `RunItemKind::ToolCallOutput` carrying the formatted text is pushed with a `ToolOutput` stream
  event, and the step returns `NextStep::RunAgain`. The return sits before the transfer hooks,
  input filter and nesting block, matching Python's "a failed invocation is not a transfer".
  Without a formatter the old span error (`"Error invoking handoff"`) and `Err(e)` are kept
  byte-for-byte.
- Sync formatter only (`ToolErrorFn<C>`), matching the local `FunctionTool` convention; Python's
  async-formatter support is a Python-runtime idiom.

Tests (`tests/core_audit_fixes_handoff.rs`): a run recovers and the calling agent answers
(`handoff_call_item`, `tool_call_output_item`, `message_output_item`, `last_agent == "A"`, error
text reaches the next model call); no formatter still terminates the run; a custom formatter's
text is used and no `handoff_output_item` is emitted; the formatter receives the run context.

### Finding 4 — `custom_data_extractor` on `ComputerTool` / `ApplyPatchTool` / `CustomTool`

Re-verified: `tool.py:922`, `:1565`, `:1598`; contexts at `tool.py:129/149/169`; invoked in
`run_internal/tool_actions.py:189/835/1078`. Confirmed the set is exactly those three
(`ShellTool` has none) and that Python assigns the extractor result directly — it does **not**
merge `tool_context.custom_data` for these three, unlike the function-tool path.

- `src/tool/local.rs`: new `ComputerToolCustomDataContext<'a, C>`,
  `ApplyPatchToolCustomDataContext<'a, C>`, `CustomToolCustomDataContext<'a, C>` (field-for-field
  ports, borrowed like the existing `FunctionToolCustomDataContext`), each with a `Debug` impl
  and a matching `*CustomDataExtractor<C>` alias; `custom_data_extractor` field +
  `with_custom_data_extractor` builder on all three tools.
- `src/run_internal/local_tools.rs`: `LocalOutcome::Output` gained `custom_data: Option<Value>`;
  the computer, apply-patch and custom paths run their extractor after `raw_item` exists and pass
  the result through `normalize_custom_data` (so a non-object return is a `UserError`, as in
  Python). Shell / local-shell / rejection paths pass `None`.
  The apply-patch closure now also returns `status` so the extractor can see it.
- `src/run_internal/mod.rs`: the local-tool output item carries that `custom_data` instead of
  hardcoded `None`.

Tests (`tests/core_audit_fixes_tools.rs`): computer (sees `tool_call`, screenshot URL,
`raw_item`), apply-patch (operations + `completed`/`failed` status over two calls), custom tool
(`input`, `output`, `tool_context.tool_name`), empty object normalizes to `None`, non-object
return is a user error.

### Finding 5 — server-managed `nest_handoff_history` warning

Re-verified: `run_internal/turn_resolution.py:534-540` vs the silent `&& !self.server_managed()`
guard. `src/run_internal/mod.rs` now restructures the branch so a requested nesting under a
server-managed conversation logs Python's exact warning text (agent names interpolated) at
`log::warn!` on `logger::LOG_TARGET` and continues with delta-only input. Plain `log::warn!` is
used rather than `logger::log_model_action_warning` — that helper redacts its payload under the
model-data policy, and agent names are not model data. The existing input-filter `UserError`
still precedes it, matching Python's order.

Test: a server-managed run with `nest_handoff_history(true)` emits the warning and the second
model call carries no `<CONVERSATION HISTORY>` wrapper.

### Finding 9 — typed `ShellTool` environment (**was already closed; small addition only**)

The audit row is stale. `src/tool/local.rs` already had `ShellToolEnvironment`
(`Local`/`ContainerAuto`/`ContainerReference`, internally tagged, `to_value()`),
`ShellToolContainerNetworkPolicy::{Allowlist, Disabled}` with domain secrets,
`ShellToolContainerSkill`, `ShellToolLocalSkill`, `ShellTool::with_typed_environment`, a
wire-shape unit test, and `pub use local::*` in `src/tool.rs` makes all of it reachable.
`ShellTool::environment` stays a `Value` escape hatch.

Python's `ShellToolHostedEnvironment` (`tool.py:1401`) is a **union alias** of
`ContainerAutoEnvironment | ContainerReferenceEnvironment`, not a fifth shape. Rust equivalent
added instead of a duplicate enum:

- `ShellToolEnvironment::is_hosted()`
- `ShellTool::hosted_typed(ShellToolEnvironment)` — the typed counterpart of the existing
  `Value`-taking `hosted()`, rejecting `Local`.

Test: `the_typed_shell_environment_builds_a_hosted_tool`.

### Unverified item — parallel input-guardrail cancellation: **already equivalent, now proven**

Read both sides. Python (`run.py:1800-1870`) gathers the guardrail task with the model task and,
on `InputGuardrailTripwireTriggered`, cancels the model task when
`should_cancel_parallel_model_task_on_input_guardrail_trip()` is true
(`run_internal/agent_runner_helpers.py:172`).

Rust `run_internal/mod.rs::run_turn` drives both under `tokio::select!`; on a trip it returns
`(guard_results, None)` and the pinned `model_fut` is dropped at the end of the block, which
cancels the in-flight call. Verified there is no detached work that would survive the drop:
no `tokio::spawn` anywhere in `src/run_internal/` or `src/models/run_context.rs`. No code change
was needed.

Python's gate returns `False` only inside a Temporal workflow whose history predates the
cancellation patch (`temporalio.workflow.patched(...)`). That is a Python-only replay-determinism
concern with no Rust counterpart — Rust always cancels, matching the non-Temporal path.

New regression test with a `Model` whose future holds a `Drop` guard: the run returns
`InputGuardrailTripwireTriggered` promptly, the model call started once, never completed, and was
dropped. A second test pins the blocking-partition behavior (a `.blocking()` guardrail never
starts the model call).

## Files changed

- `src/handoffs.rs`
- `src/run_internal/mod.rs`
- `src/run_internal/local_tools.rs`
- `src/tool/local.rs`
- `tests/core_audit_fixes_handoff.rs` (new, 7 tests)
- `tests/core_audit_fixes_tools.rs` (new, 6 tests)

Not touched: `src/run_state*`, approvals code, `src/sandbox/**`, `src/lib.rs`, `Cargo.toml`.
**No shared-file edits were needed** — `src/tool.rs` already re-exports `local::*`, so the new
public types reach the crate root without a change there.

## Breaking changes (unreleased tree only)

- `Handoff<C>`, `ComputerTool<C>`, `ApplyPatchTool<C>`, `CustomTool<C>` each gained a public
  field. Struct-literal construction breaks, but every one of these types has a private field or
  private constructor path in practice and all in-repo literals are internal (verified by grep
  across `src/`, `tests/`, `examples/`), so no caller can have been literal-constructing them.
  Additive for builder users.
- `crate::run_internal::local_tools::LocalOutcome::Output` gained `custom_data` — `pub(crate)`,
  not public API.

## Behavior changes

- A handoff whose invocation fails **and** that has a `failure_error_function` no longer ends the
  run. Default (`None`) behavior is unchanged.
- A server-managed conversation that requests `nest_handoff_history` now logs a warning. The
  resulting model input is unchanged.

## Behavior notes (recorded so a future audit does not re-flag them)

- The three new extractors run after `end_hooks` (inside the spanned `run` block), while Python's
  `finalize_output` runs the extractor just before `on_tool_end`. Rust's existing function-tool
  path (`run_internal/tools.rs:451`) already extracts after the hooks, so this is consistent with
  local convention; the extracted data lands on the same item either way.
- The recovered handoff tool-output item is appended after `order_current_turn_tool_outputs`
  has already run for the turn. Python behaves the same way: the failure item goes onto
  `new_step_items` and is returned without a further re-order (ordering there is driven by
  `_commit_tool_output`, which the handoff path does not use). Same for the pre-existing
  "Multiple handoffs detected" items.
- Docs delta: none. `grep -rln "custom_data_extractor\|failure_error_function\|
  nest_handoff_history\|ShellToolEnvironment" docs/ README.md` returns nothing, so no page
  enumerates these contracts.
- Verified against `run_internal/tool_actions.py:835-845`: Python assigns the extractor result
  directly for the custom tool and does **not** merge `tool_context.custom_data` the way the
  function-tool path does, so the Rust custom-tool path deliberately does not merge either.

## Gates

- `cargo clippy --all-features --all-targets -- -D warnings` — clean.
- `cargo test --all-features --no-fail-fast` across the whole workspace: every test binary passes
  except `core_run_state` (1) and the three sandbox binaries (5) — all in other agents' in-flight
  files, none in this area. `--lib` (629), `core_audit_fixes_handoff` (7),
  `core_audit_fixes_tools` (6), `runner_tests`, `local_tools_tests`, `core_server_conversation`,
  `core_tool_handoff_rows`: all green.
- `rustfmt --edition 2024` applied to every file above.

## Notes for the controller

- `tests/core_run_state.rs::round_trips_through_the_envelope` fails on the current tree. It is
  pure `RunState` wire serialization (`sample_state()` → `to_json` → `from_json`), untouched by
  this area; `src/run_state.rs` and `src/run_state/wire.rs` are another agent's in-flight edits.
- The sandbox suites fail on the current tree too: `an_archive_cannot_write_through_a_symlink`
  (`sandbox_tests`), `daytona_round_trips_a_workspace_archive`,
  `blaxel_persists_and_hydrates_through_a_tar_file`,
  `cloudflare_runs_a_full_session_against_the_worker_protocol`,
  `modal_persists_a_tar_archive_by_default`. All in `src/sandbox/**` /
  `src/extensions/sandbox/**`, the other agent's in-flight archive-validation work. The shared
  tree also failed to compile twice mid-session from that area (`FileEntry::is_symlink`, then
  `SandboxArchiveLimits`/`join_sandbox_path`), so the gates above were run once it compiled.
- Python's `context_wrapper._mark_tool_invocation_executed(..., invocation_role="handoff")` has no
  Rust counterpart anywhere (grep: zero hits). Rust instead derives the already-executed set from
  generated-item call ids in `process_response`, so the recovered handoff output is correctly
  skipped on resume. Pre-existing design difference, outside this area's scope.

Status: DONE
Summary: Closed findings 1, 4 and 5; finding 9 was already closed (added only the `hosted_typed`
constructor and `is_hosted()` for Python's `ShellToolHostedEnvironment` union alias); proved the
parallel-guardrail cancellation row already equivalent via future-drop semantics and added a
regression test. 13 new tests; clippy `-D warnings` and the touched suites pass under
`--all-features`.
Concerns/Blockers: Six pre-existing failures remain on the shared tree, all outside this area and
all from other agents' in-flight work — `core_run_state::round_trips_through_the_envelope`
(`run_state` wire format) and five sandbox archive tests. The whole-suite gate should be repeated
once those land.
