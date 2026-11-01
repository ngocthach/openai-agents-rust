# Parity gaps closed: r16gaps

Status of the nine documented 0.17.0 divergences. Tests: `tests/py_core16_r16gaps_*.rs`. The CSV counts the Python
cases each item unblocks (the `test_agent_scoped_approvals.py` row is the two cases of item 1 that stay open; the
counts are per Python test function, not per parametrization).

| # | Item | Result |
|---|------|--------|
| 1 | Context bound to an agent by name; same-named distinct capability | NOT FEASIBLE (see below) |
| 2 | End-hook-failed call re-executed on a fresh run sharing the context | DONE |
| 3 | Streamed `ToolCalled` after `on_llm_end` | DONE |
| 4 | `CallbackSink` gets the bound session | DONE |
| 5 | `#[function_tool]` on methods / associated functions | DONE |
| 6 | `MCPToolCancellationError` | DONE |
| 7 | Realtime: aggregate interrupt failures; STT transcript before setup; connect-time tool list | DONE (connect-time tool implementations: partly, see below) |
| 8 | `apply_patch` approval checker stops after a decision lands | DONE |
| 9 | Vercel shutdown retries the stop while detaching mounts | DONE |

## Per item

1. Not feasible without a breaking change. Python binds a decision to the live `Agent` object through an owner token that
   `ToolApprovalItem(agent=Agent, ...)` carries. Rust's `ToolApprovalItem.agent` is the agent's name, and the approvals
   ledger is keyed by name, so a user-built item cannot say which capability it approved. Telling a same-named distinct
   agent apart needs a new owner field on that public struct (74 literal constructions in the repo alone) plus restore-time
   owner resolution and tie handling for serialized states. Not attempted; the divergence stays documented.
2. `run_turn` now re-arms the executed-but-uncommitted ledger for the tool outputs of a turn that fails (end hook,
   guardrail): a fresh run sharing the context that replays the same call id is refused with `already executed`, for
   function and computer calls. A successful run's call ids stay reusable. `src/run_internal/mod.rs`.
3. A streamed run emits the validated `tool_called` events before a failing `on_llm_end` hook ends the run, without
   adding the items to the checkpoint (the response is still resumed by processing it). Success-path ordering is
   unchanged (events still follow the hook there). `src/run_internal/mod.rs`.
4. Additive: `EventSink::bind(Arc<dyn SandboxSession>)` (defaulted), `CallbackSink::with_session` and
   `raising_with_session`; `InstrumentedSession::new` binds the sink (a `ChainedSink` forwards to its members). An unbound
   session-aware sink reports `CallbackSink requires a bound session` (as `event_sink_failed`). `src/sandbox/events.rs`.
5. `#[function_tool]` works inside `impl` blocks. A method with `&self`/`self` gets `<fn>_tool(&self)` binding a clone of
   the receiver (`Self: Clone + Send + Sync + 'static`); `&mut self` is rejected. An associated function must be marked
   `#[function_tool(associated)]`, because an attribute macro cannot see that a receiver-less function sits in an `impl`.
   The parameter struct is local to the constructor, so it cannot name the impl's generic parameters. `macros/src/lib.rs`.
6. New variant `AgentsError::MCPToolCancellationError { message, run_data }` (constructor `mcp_tool_cancelled`). Raised
   when the client cancels a tool call (rmcp `Cancelled` after the isolated-retry budget) or a custom `MCPServer` returns
   it; `invoke_mcp_tool` rebuilds the message as Python does (tool name and credential-free server name), logs nothing,
   passes it through unchanged, and the default failure policy reports the generic tool message without consulting
   `tool_error_formatter`. Rust has no task-cancel to observe, so the source is the error itself.
7. (a) The interrupt path reports `Multiple Realtime interrupt operations failed: truncate_audio=...; cancel_response=...`
   when both steps fail (Rust's emit cannot fail, so Python's `emit_audio_interrupted` entry has no counterpart).
   (b) A transcript the STT server sends before setup finishes is buffered and delivered once the session is up.
   (c) A tool list in the connect-time settings now also decides dispatch: a tool it no longer names is not callable.
   A spec carries no implementation, so a connect-time tool with an implementation still goes through
   `RealtimeRunConfig::initial_tools` / `initial_handoffs` (already present).
8. Additive: `ApprovalPolicy::DynamicForAgent` and `ApprovalPolicy::evaluate_for_agent`; the runner now passes the agent
   name to every local-tool policy. The sandbox `apply_patch` custom tool uses it and stops checking once a decision for
   the call exists, returning the checker's last answer as Python does. A plain `evaluate` shows a `DynamicForAgent` rule
   an empty agent name.
9. `shutdown` on a session with S3 mounts now unmounts each attached mount, stops the sandbox, retries the stop once
   when a failed unmount left it running, and reports the first failure; a failed mount transition stops the sandbox and a
   stop failure there replaces the mount error (Python raises it from the handler). A session whose mount transition
   failed refuses further commands (`mount_transition_failed`). `src/extensions/sandbox/vercel/mod.rs`.

## Breaking changes

- `AgentsError` has a new variant (exhaustive matches must add it); `ApprovalPolicy` has a new variant.
- `EventSink` has a new defaulted method; no implementor breaks.
- A failed turn now leaves its committed call ids as executed-but-uncommitted, so a fresh run sharing the context that
  replays them is refused (Python's behaviour).
- A failed Vercel mount transition makes the session refuse commands afterwards; its stop error replaces the mount error.

## New test files

`py_core16_r16gaps_mcp_cancellation`, `_vercel_shutdown`, `_callback_sink_session`, `_apply_patch_approval_stop`,
`_end_hook_fresh_run`, `_streamed_tool_called`, `_realtime_interrupt_errors`, `_stt_setup_transcript`,
`_realtime_dispatch_override`, `_function_tool_methods`. `py_core14_r14c_public_api_contract.rs` lists the new variant.

## Gates

- Full `cargo test --workspace --all-features --no-fail-fast`, skipping two tests that hang for reasons unrelated to this
  work (`a_clear_revokes_an_encrypted_deferred_compaction` in another agent's `py_core16_r16e_encrypt_interleave.rs`, and
  `live_docker_authority_preserves_resume_and_cleanup`, which needs a Docker daemon): 643 test binaries passed and 1
  failed. The failure, `vercel_lets_a_mounted_session_start_again_after_an_early_failure`, was caused by this work (a
  failed mount transition made the session refuse a retry even though no mount had attached). Fixed: the refusal now
  applies only once mounts attached, and the next `start` clears it. All Vercel binaries and the new ones pass again; the
  full run was not repeated after that one fix.
- Clippy `-D warnings` is clean with `--all-features --all-targets` and with default features.
- `core_run_state` failed in an earlier partial run because of another agent's in-flight edits and passed in the full run.

Status: DONE_WITH_CONCERNS (item 1 not feasible; two unrelated hanging tests skipped in the full run).
