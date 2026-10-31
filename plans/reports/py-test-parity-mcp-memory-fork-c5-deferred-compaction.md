# Fork C5: deferred_session_writes/test_compaction.py parity

Counting: def = AST `def test_` (12); expanded = parametrize product (39). Verified from runs:
`cargo test --all-features --test py_deferred_writes_compaction` = 3 passed, 6 ignored; with `--include-ignored` all 6 ignored fail, confirming each cross-area gap is real. Clippy clean for the file (no findings); rustfmt applied.

File: `tests/py_deferred_writes_compaction.rs`. No source changes; no Rust bugs fixable inside `src/memory/openai_responses_compaction_session.rs`.

Rust mapping: Python's `_CompactionRecordingSession` hooks map to `Session::on_run_completed`; `has_local_tool_outputs=true` = Python `deferred` entry, `false` = `run_compaction` entry.

## Per-test classification

| Python test | exp | class | Rust test / reason |
|---|---|---|---|
| the_compaction_deferral_reads_the_settling_batch_not_the_callers_input | 1 | PYTHON-ONLY | drives private `save_result_to_session` |
| the_settled_count_survives_the_compaction_deferral_branch | 1 | PYTHON-ONLY | private `save_result_to_session(settling_held_batch=True)` and its return count |
| a_partial_settle_on_a_compaction_session_still_fails_the_gated_resume_fast | 2 | CROSS-AREA (ported, ignored) | `a_partial_settle_fails_the_regated_resume_fast` |
| a_held_mcp_approval_pair_defers_compaction_when_it_settles | 1 | PYTHON-ONLY | private `save_result_to_session` |
| an_ordinary_mcp_approval_response_defers_compaction_too | 1 | PYTHON-ONLY | private `save_result_to_session` with an `MCPApprovalResponseItem` |
| the_final_sweep_settle_defers_compaction_and_counts_what_it_wrote | 1 | PYTHON-ONLY | private `save_final_turn_items_after_guardrails`, `_pending_session_write` |
| the_entry_settle_runs_the_compaction_bookkeeping | 1 | PYTHON-ONLY | private `resume_pending_session_write` and `_pending_session_write` dict |
| the_park_records_the_response_the_batch_belongs_to | 2 | PYTHON-ONLY | asserts `pending_session_write` key in the state JSON; Rust has no such object |
| the_entry_settle_defers_with_the_recorded_store | 1 | PYTHON-ONLY | same private resume function |
| approval_settlement_preserves_response_storage_mode | 18 | PORTED (end state) + CROSS-AREA (metadata, ignored) | green `approval_settlement_persists_the_pair_when_{the_run_continues,the_resume_is_terminal,the_resume_is_blocked}` (each loops streamed x 3 store pairs = 6) assert resume outcome, guardrail trip and the parked call/output pair; ignored `approval_settlement_keeps_the_storage_mode_when_*` assert the compaction decision's mode and settle-before-next-request |
| detached_repark_advances_response_storage_metadata | 4 | CROSS-AREA (ported, ignored) | `a_detached_repark_advances_the_response_storage_metadata` |
| partial_approval_repark_keeps_response_storage_setting | 6 | CROSS-AREA (ported, ignored) | `a_partial_approval_repark_keeps_the_response_storage_setting` |

Expanded totals (39): 9 python-only; 12 cross-area (ported, ignored: 2 + 4 + 6); 18 storage-mode cases ported with a green end-state half and an ignored cross-area metadata half. 9 + 12 + 18 = 39.

## Cross-area gaps (all in `src/run_internal/mod.rs`; not edited)

1. `notify_session_completed` (:1559, `store: self.current_agent.model_settings.store`) ignores the `RunConfig` `model_settings.store` override. Python passes the resolved settings' `store`. Evidence: first completion for a run configured `store=True` reports `store: None`. Minimal fix: use the resolved settings (`self.resolve_settings(..)`) for the notification.
2. An interrupted response is notified at interruption time as a non-deferred completion (`deferred=false`, e.g. `[resp_multi, false]` before any approval). Python defers the interruption-time write and settles it with the resumed batch as a `deferred` entry carrying the parked response's own `store`, before the next model request starts. Minimal fix: hold the notification with the parked batch (the same deferral C3 reports missing) and fire it with `has_local_tool_outputs=true` when the batch settles.
3. No guard refusing a re-gated resume after a partial settle persisted current-turn items (Python: `UserError "output guardrails after current-turn items"`). Minimal fix: track the settled-item count in the run state and refuse when output guardrails are re-enabled.

## Totals
12 def / 39 expanded: 3 green Rust tests (18 storage-mode end-state cases), 6 ignored cross-area tests, 9 expanded python-only cases.

Status: DONE_WITH_CONCERNS
Summary: Ported the 3 observable test families from test_compaction.py (3 green end-state tests, 6 ignored cross-area tests); the other 9 expanded tests drive private persistence functions and are Python-only.
Concerns/Blockers: Three run_internal gaps (store override ignored, interruption-time notify instead of deferral, no re-gated-resume guard) keep 6 tests ignored.
