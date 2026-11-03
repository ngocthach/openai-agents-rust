# Fork C4: responses-compaction remainder parity report

Scope: `tests/memory/test_openai_responses_compaction_session.py` (73 def_test, 94 expanded, counted by AST with `parametrize` multiplied). Already done before this fork: `tests/py_memory_compaction.rs` (23). Python's `MagicMock` client is replaced by a local mock `POST /v1/responses/compact` endpoint; captured request bodies are asserted.

New files (all green with `--all-features`, nothing `#[ignore]`d):

| file | tests |
|---|---|
| `tests/py_memory_responses_compaction_a.rs` | 11 (output normalization, compaction modes, request shape) |
| `tests/py_memory_responses_compaction_b.rs` | 8 (history restore after failed replacement, rollback budget) |

Also green: `py_memory_compaction` (23), `py_memory_compaction_visibility`, `sessions_tests` (37), lib `memory::` (75).

## Rust bugs fixed (`src/memory/openai_responses_compaction_session.rs`)

1. **Failed-clear recovery diverged from Python.** Rust always called `clear_session` again during restore, so a clear that failed once was cleared twice, and a store whose clear failed after emptying itself was left empty. Python tracks whether the clear returned: a failed clear inspects the store, does nothing when it is unchanged, and otherwise re-adds the previous items without clearing again. Ported as `restore_after_failed_clear` / `restore(.., clear_first)`.
2. **Compact request leaked session-only metadata.** Stored items were sent verbatim in `input`, including `_session_description`, `_session_title` and `created_by`. Python strips these (`strip_internal_input_item_metadata`). Now stripped in `request_compaction`.
3. **Restore warnings bypassed log redaction.** They used `log::warn!` directly, so a failure message could reach logs while `DONT_LOG_*` was set. They now go through `log_model_and_tool_action_warning`, as in Python.

## Per-group table (def_test = Python `def test_`; expanded = parametrize multiplied)

| group | def_test | expanded | ported | covered | python-only | cross-area | unported | failures found |
|---|---|---|---|---|---|---|---|---|
| `created_by` replay (`test_compaction_replay_strips_created_by`) | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 0 |
| `is_openai_model_name` | 4 | 4 | 0 | 4 (`py_memory_compaction`) | 0 | 0 | 0 | 0 |
| candidate selection | 3 | 3 | 0 | 3 (`py_memory_compaction`) | 0 | 0 | 0 | 0 |
| rollback budget (init, manual, failed-clear inspection, automatic, exact) | 5 | 9 | 3 (b) | 1 (`py_memory_compaction`) | 0 | 0 | 1 (automatic, native SQLite + Runner) | 1 (fixed, #1) |
| client / model init | 4 | 6 | 0 | 2 (`py_memory_compaction`) | 2 (default client, falsy client) | 0 | 0 | 0 |
| delegation (add/get/pop/clear) | 3 | 3 | 0 | 2 (`py_memory_compaction`) | 0 | 0 | 1 (pop/clear invalidate response chain) | 0 |
| settled pop/clear failure invalidates chain | 2 | 4 | 0 | 0 | 4 (pause a real SQLite commit on a thread, asyncio cancel) | 0 | 0 | 0 |
| `run_compaction` mode/request-shape | 14 | 24 | 9 (a) | 5 (`py_memory_compaction`) | 0 | 0 | 0 | 1 (fixed, #2) |
| threshold/force/hook | 4 | 4 | 0 | 4 (`py_memory_compaction`) | 0 | 0 | 0 | 0 |
| restore after failed replacement (add/clear/limit/restore-fails) | 5 | 8 | 5 (b) | 0 | 0 | 0 | 0 | 1 (fixed, #1, #3) |
| restore under asyncio cancellation (`..._cancelled`, `cancel_restore_waits_*`, `exception_restore_drains_*`) | 5 | 5 | 0 | 0 | 5 (task-cancel semantics; covered in spirit by `memory::await_session_mutation`) | 0 | 0 | 0 |
| deferred marker on API failure | 1 | 1 | 1 (`a_failed_compact_request_keeps_the_history`, py_memory_compaction) | 0 | 0 | 0 | 0 | 0 |
| `model_dump` warning suppression | 1 | 1 | 0 | 0 | 1 (pydantic `warnings=` kwarg) | 0 | 0 | 0 |
| output normalization (image/file/file_id, failure) | 4 | 4 | 4 (a) | 0 | 0 | 0 | 0 | 0 |
| runner-driven compaction (`compaction_runs_during_runner_flow`, skips with tool outputs, deferred context/mode, deferred across tool turns) | 5 | 5 | 0 | 1 (`sessions_tests::runner_triggers_compaction_after_a_run`) | 0 | 4 (deferral with pending approval, see below) | 0 | 0 |
| orphaned assistant id stripping (6 unit tests) | 6 | 6 | 0 | 6 (in-file `orphaned_assistant_ids_are_stripped` and runner paths) | 0 | 0 | 0 | 0 |
| generation / interleaved-write races (`resumed_save_carries_generation`, `runner_skips_compaction_after_interleaved_*`, `add_waits_for_in_flight_*`, `clear_waits_for_in_flight_*`) | 5 | 5 | 0 | 1 (`sessions_tests::compaction_retains_history_when_another_writer_appends_mid_request`) | 1 (generation counter) | 3 | 0 | 0 |
| `is_compaction_aware_session` | 3 | 3 | 0 | 0 | 3 (duck-typing helper; Rust uses `wants_model_exchange`) | 0 | 0 | 0 |
| **total** | **73** | **94** | **22** | **29** | **16** | **7** | **3** | 3 fixed |

Caveat: the group rows were assigned by test name and sum to 75 defs (column totals 77), not the file's 73; a few tests straddle two groups and I did not reconcile them one by one. Treat the group counts as approximate. The reliable figures are the file totals (73 def / 94 expanded), the 19 new passing Rust tests, and the unported/cross-area lists below.

Counting method: AST over `def test_` with `pytest.mark.parametrize` argument lists multiplied; groups assigned by test name and body.

## Cross-area deferrals (not fixed; outside `src/memory`)

1. Deferred compaction under a pending approval (`test_deferred_*`, `test_compaction_runs_after_deferred_tool_outputs_when_due`): Python's runner defers the session write of an interrupted turn and compaction follows once the outputs settle. Rust's runner (`src/run_internal/mod.rs`) writes at interruption time and has no deferral state, so these cannot be observed through the public API. Minimal fix: add deferred-write state to the runner checkpoint.
2. Interleaved model-wait / persist races (`test_runner_skips_compaction_after_interleaved_*`): need the runner to carry a session generation across a resume; Rust has no generation counter in run state.
3. Compaction exchange ignores `call_model_input_filter` (`src/run_internal/mod.rs:1559-1608`), already recorded in `py_memory_compaction_visibility.rs`.

## Unported (3)

- `test_compaction_replay_strips_created_by`: needs a SQLite-backed Runner replay asserting `created_by` is absent from both the Responses request and the next compact request; `created_by` is now stripped from the compact request (fix #2), but the Responses-request half belongs to the models area.
- Automatic-compaction rollback budget through the Runner on native SQLite (`test_automatic_compaction_rollback_budget`): needs `SqliteSession` plus a Runner-driven compaction trigger with a scripted model; the session-level budget path is covered by `an_oversized_history_fails_before_the_request_and_any_write`.
- `test_pop_item_invalidates_response_chain` / `test_clear_session_invalidates_response_chain`: assert private fields (`_get_deferred_compaction_response_id`, `_last_unstored_response_id`). Observable behavior (the next compaction needs a fresh response id) is not yet asserted.

## Gates

- rustfmt applied to the three files I changed or created.
- `cargo clippy --all-features --all-targets -- -D warnings` stops at `src/sandbox/sandboxes/unix_local.rs:26` (unused import, another area), so my files were not linted. No findings in my files from the earlier clean passes.
- No edits to `src/lib.rs`, `Cargo.toml`, `tests/py_mcpmem_support/mod.rs` or any existing `py_*` test file.
