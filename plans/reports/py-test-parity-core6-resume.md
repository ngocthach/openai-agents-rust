# Core6 resume parity: test_run_impl_resume_paths.py and test_native_tool_failure_history.py

Counting: pytest case = parametrize product per test function (198 and 72, matching the ledger).

| file | cases | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| test_run_impl_resume_paths.py | 198 | 159 | 10 | 24 | 5 |
| test_native_tool_failure_history.py | 72 | 2 | 0 | 20 | 50 |
| total | 270 | 161 | 10 | 44 | 55 |

## test_run_impl_resume_paths.py
Ported (159): session append recovery 16+6+1+4+1, pending-write wire validation 17+1+2, partial commit 1 (49);
terminal marker family 24; callback retry 26 (extractor-based, see below); legacy 1.16 callback failure 8;
resumed/fresh handoff append 23; legacy name fallback 10; compaction recovery 19 (replay 2, post-write failure 1, committed-append refresh 4, rolled-back replacement 12). Files: `tests/py_core6_resume_*.rs`
(helpers in `tests/py_core6_resume_support/mod.rs`).
Covered (10): tool-not-found matrix 8, each-output-once 1, no-duplicate-session-items 1, all in
`tests/py_core2_runstep_resume.rs` (`resumed_server_managed_run_sends_tool_not_found_output_matrix`,
`resumed_server_managed_run_sends_each_tool_output_once`, `resumed_approval_does_not_duplicate_session_items`).
Python-only (24): hooks/extractors that raise typed errors or CancelledError (`end_hook` variants 4, redacted nested-run callback 4, cancelled callback 1);
asyncio task cancel/block (append cancellation 2, cancelled compaction append 1, cancelled-compaction SQLite thread 4);
monkeypatched private run-loop functions 5; private `server_manages_conversation` kwarg spy 3.
Rust callbacks cannot raise, so the failing post-tool step is an extractor returning a non-object ("custom_data_extractor must return a JSON object").
Not ported (5): compaction cross-area gaps, `json_compaction_retry_does_not_promote_filtered_session_history` (1: the pending write
does not carry `compaction_model_exchange` and `match_model_exchange_suffix` in `src/memory/openai_responses_compaction_session.rs` is not wired, so
a whole-history compaction would take the hidden item) and `compaction_reload_preserves_session_retrieval_window` (2: compaction reads the
full store, ignoring `session_settings.limit`); legacy name fallback `1.7 approve`/`always_approve` (2, owned by another agent).
Compaction cases use a local mock `/v1/responses/compact` echoing its input (`mock_openai_with`); Python's raising decision hook is
modelled by a Session wrapper whose first completion notice fails.

## test_native_tool_failure_history.py
Unchanged: not attempted. All 50 remaining cases need native/local tools to run concurrently with the function batch,
be cancelled on a selected function failure, and keep finalization bounded. Rust runs `processed.local_calls`
sequentially after `execute_function_tools` returns (`src/run_internal/mod.rs`, `execute_side_effects_inner`), and the
function batch error returns before any native call starts. Design: run the local-call loop under a join with the batch,
sharing a failure signal that stops queued natives and cancels the in-flight invoke (not its finalization), plus a bounded
post-invoke wait; race the parallel input-guardrail verdict with tool execution. Not started.

## Rust bugs fixed
- Withheld-output marker: `RunState.withheld_output_call_ids` (Rust-only, under the envelope's `rust` key) records the outputs a failed attempt committed; a retry announces exactly those (not earlier-published siblings), orders them with its own outputs, and keeps results of a retried turn. Tool-output stream events are now held until the turn resolves (dropped on failure, released for retained partial outputs).
- Handoff turns report their response to the session before the pending-write record is dropped, so a failed compaction/rollback is retried alone (`append_acknowledged` retained); the notice's `has_local_tool_outputs` follows the written batch.
- `src/run_internal/session_write.rs` (new) + `mod.rs`: lost-ack append duplicated outputs on retry; now a digest-based pending write is recorded, settled at resume entry, and refused when ambiguous (wrong/missing session, changed tail, partial commit).
- `src/run_state/wire.rs`: pending-write and terminal-marker validation; pending writes written on a resumable step.
- `mod.rs`: terminal marker now set when a final/max-turns/rejected-handoff save fails; failed callbacks on unproven (legacy) boundaries close the checkpoint; withheld tool-output events under output guardrails (released on resolution, re-announced by the retry, dropped on failure); a crashing guardrail on a recovered turn no longer persists it; renewed-interruption append failure keeps its interruptions; public `run_data` omits the failed callback's guardrail results.
- `src/result.rs`, `src/run.rs`: failed streamed run `to_state()` returns its failure checkpoint.

## Deferred / known divergences
- Name fallback: Rust executes an identity-less approval under schema 1.7+ (Python refuses); approval matching in `src/run_context.rs`.
- Wire note: `before` digests are Rust-computed, so a Python-written pending write loaded in Rust reports "ambiguous" unless acknowledged.
- Breaking: failed streamed runs now answer `to_state()`; failed final saves now yield a non-resumable checkpoint.
- Hosted-MCP approval identity cases (14, `test_run_step_execution.py`) are outside these two files and untouched.
