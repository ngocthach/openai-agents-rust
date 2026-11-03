# Parity: tests/deferred_session_writes (fork C3)

Python source: `/Users/ngocthach/myGit/openai-agents-python/tests/deferred_session_writes/`.
`test_compaction.py` (39 expanded) is owned by another fork and excluded here.

Counting: `def_test` and `expanded` from an AST walk of each file (`def_test` x product of `@pytest.mark.parametrize` lengths). The walk undercounts one case where the parametrize list is not a literal: `test_a_failed_detached_completion_keeps_the_held_record` is 2 streamed x 2 agent factories = 4, not 2. Corrected totals: test_resume 40 (not 38), whole directory 160 (not 158), this fork's scope 121.

## Rust files (all green with `--all-features`, rustfmt applied)

| file | passing | ignored | content |
|---|---|---|---|
| `tests/py_deferred_writes_a.rs` | 10 | 1 | resume path |
| `tests/py_deferred_writes_b.rs` | 1 | 2 | recovery (atomic failure, lost-ack); handoff filter |
| `tests/py_deferred_writes_c.rs` | 1 | 0 | max-turns fallback |
| `tests/py_deferred_writes_d.rs` | 4 | 9 | all of test_filters.py |
| `tests/py_deferred_writes_e.rs` | 5 | 8 | resume tail, recovery tail |
| `tests/py_memory_deferred_writes.rs` (earlier, not edited) | 2 | 1 | initial end-state ports |

Method: Python's held-record internals (`pending_session_write`, `_pending_session_write`, `_PendingSessionWrite`, `_held_items_safe_to_settle`, `defer_interrupted_session_write`) do not exist in Rust. Each port asserts only what `Runner`, `RunState` and `Session` expose: the session's final content and the run outcome, for streamed and non-streamed runs where Python parametrizes both. Python's raising `custom_data_extractor` has no Rust equivalent (the extractor returns `Option<Value>`); the failure at the same boundary is an extractor that returns a non-object, which the runner rejects with "custom_data_extractor must return a JSON object or None".

## Per-file table (expanded counts, scope excludes test_compaction)

| file | def_test | expanded | ported (green) | covered | python-only | cross-area (ignored) | unported |
|---|---|---|---|---|---|---|---|
| test_resume.py | 20 | 40 | 29 | 0 | 3 | 8 | 0 |
| test_recovery.py | 8 | 29 | 12 | 1 | 0 | 16 | 0 |
| test_filters.py | 13 | 31 | 8 | 0 | 0 | 23 | 0 |
| test_max_turns.py | 5 | 9 | 4 | 0 | 5 | 0 | 0 |
| test_pending_write.py | 10 | 12 | 0 | 0 | 12 | 0 | 0 |
| **total** | **56** | **121** | **53** | **1** | **20** | **47** | **0** |

Check: 53 + 1 + 20 + 47 + 0 = 121. Nothing is left unported. The earlier 89 unported tests are now ported, ignored with a cited cause, or classified Python-only.

Classification notes:
- **Python-only (20)**: test_pending_write (all 12): pairing-guard predicate, registration policy, conversion policy, response-store and checkpoint-label assertions on `_PendingSessionWrite`. test_max_turns (5): they assert only that the held record is cleared or kept after a handler or finalization failure. test_resume (3): `_ContextRequiringSession` wrapper plumbing (2) and the held batch riding a non-streamed result into its checkpoint JSON (1).
- **Covered (1)**: `test_entry_settle_restores_the_conversations_sanitization` by `tests/py_memory_conversations.rs::conversation_writes_keep_ids_only_where_the_schema_requires_them` and `::the_runner_strips_message_ids_before_writing_to_the_conversation`.
- **Ported, outcome half only**: the tests that also assert held-record JSON (`test_a_detached_completion_clears_the_held_record`, `test_a_failed_final_settle_fails_closed*`, `test_a_failed_detached_completion_keeps_the_held_record`) are ported on their session and outcome assertions; the record assertions are Python-only.
- test_recovery ports: 8 expanded atomic-failure settle recoveries (green) plus 4 final-settle cases (green, covering the unguarded and guarded variants of `test_a_failed_final_settle_fails_closed*`, each streamed and non-streamed).

## Cross-area gaps (all in `src/run_internal`; each is an `#[ignore = "cross-area: ..."]` test)

The runner defers no interrupted session write today. Six distinct causes account for all 47 ignored expanded tests (by primary cause: 25 + 4 + 2 + 6 + 8 + 2). Some tests hit two causes; each is counted once, under the one that fails first.

1. **Interruption writes every generated item eagerly.** `NextStep::Interruption` calls `save_session()` (`src/run_internal/mod.rs:1205`, defined at `:1536`), so the parked call, assistant preamble and handoff call land in the session before any handoff filter, session callback or later resume can remove them. Python holds that response in `pending_session_write` and settles it later. This explains 25 ignored expanded tests: emptied-turn authority (2), the library filter keeping a secret out of history (2), the detached filtered handoff (1), unkeyed-sibling removal (4), the sibling completed before approval (2), equal preambles (6), detached terminal guardrail failure (2), the different-session refusal (2), the one-ordered-write assertion (2), and the JSON-retry filter authority (2). Minimal fix: when a `StopAtTools` gate is active, keep the interrupted response's items in a held record carried on `RunState` and settle them at the next resume (applying the handoff filter's verdict), instead of writing them in `save_session`.
2. **After-turn cancel is checked before the resumed turn runs.** The check is at the top of the run loop (`src/run_internal/mod.rs:1119`), before `resume_pending_tools` (`:1126`) executes approved tools. A cancel requested before streaming stops the resumed turn before the tool runs, so the checkpoint carries no output and a reattach completes with an unpaired call. Python's cancel waits for the turn. Minimal fix: honour `cancel_after_turn` after the resumed tool step completes. Primary cause of 4 ignored tests (carried pair x2, after-turn-cancel keeps batch, entry-settle drops filter-unpaired); it also affects the detached filtered handoff.
3. **A guardrail crash skips the session save.** In `finish_with_output`, a failing output guardrail returns at `src/run_internal/mod.rs:1316` (`let r = r?;`) before `save_session()` at `:1330`, so the executed output is not stored. Python's crash path saves the pair first (`test_resume.py:407`). Minimal fix: persist executed tool outputs on the guardrail error path. Affects 2 tests.
4. **A failing post-output step aborts the turn without keeping the executed output.** An extractor failure propagates from `merge_custom_data(..)?` (`src/run_internal/tools.rs:464`), and a failing tool-use-behavior decision propagates with `?` at `src/run_internal/mod.rs:2678`, both after the tools committed. No checkpoint keeps the executed outputs, so a retry runs the tool again. Python's retry skips the completed invocation. Verified: a faithful retry with the same failing extractor fails again, and the local tool ran twice in the MCP-approval case. This is a duplicated side effect on retry. Minimal fix: record executed outputs on `RunState` before the post-output step and have a retry skip completed invocations. Primary cause of 6 ignored tests (callback failure keeps output, callback order, MCP approval retry; each streamed and non-streamed); it also affects the JSON-retry filter authority tests.
5. **A lost-acknowledgement append is re-written on retry.** `save_session` tracks `session_saved_items` in process only (`src/run_internal/mod.rs:1536`-`1555`), so after a write that landed but reported failure, the retry from a checkpoint appends the approved output again and the session holds two outputs for one call. Python's retry does not re-append. Minimal fix: before a retry append, drop held items already present in the stored tail by `call_id`. Affects 8 expanded tests (the `failure="after"` half of the 16-way recovery case).
6. **A handoff filter that drops one output leaves its call stored.** Same root as cause 1: the call was written eagerly, so a filter that removes only its output cannot remove the call. Affects 2 tests (`a_filter_that_drops_one_output_takes_its_held_call_with_it`, streamed and non-streamed), plus the model-exchange ignore recorded in the memory-compaction report (the exchange is built from input before `call_model_input_filter`, `src/run_internal/mod.rs:1559` vs `:1820`).

## Things confirmed to already match Python

The max-turns fallback (handler runs with the approved pair already stored; its output enters history only when requested), partial-approval re-interruption keeping one canonical batch, gate-off re-interruption, reject persisting a paired rejection output, a new park during a detached resume joining the batch, `input_items`-only filters leaving history alone, additive filters keeping companions, and later-tripwire / detached-repark ordering.

## Source changes

None in this fork. `src/lib.rs`, `Cargo.toml`, `tests/py_mcpmem_support/mod.rs` and all earlier `py_*` files were left untouched.

## Gates

- `cargo test --all-features` for the six files above: green (23 passed, 21 ignored, 0 failed).
- rustfmt (`--edition 2024`) on `py_deferred_writes_{a,b,c,d,e}.rs`.
- Clippy `--all-features --all-targets -- -D warnings` is blocked by other areas' lint errors in lib code (`src/extensions/models/mod.rs`, `src/models/openai_responses_ws.rs`, `src/sandbox/sandboxes/unix_local.rs`), so these test files were not linted.
- `tests/py_memory_deferred_writes.rs` (not edited, per instruction) emits one `unused import: Session` warning; remove that import when that file is next touched.

## Unresolved

- Whether the six gaps are fixed in `src/run_internal` this round is outside this fork's ownership. When they land, the 21 ignored Rust tests (47 expanded) are the acceptance suite; remove each `#[ignore]` as its gap closes.
- Cause 1 is a design change (a held record on `RunState` plus a checkpoint field), not a point fix, and causes 2 to 5 depend on or interact with it.

Status: DONE_WITH_CONCERNS
Summary: All 121 expanded tests in scope are accounted for: 53 ported and green, 1 covered, 20 Python-only, 47 ignored against six confirmed `src/run_internal` gaps, 0 unported (test_compaction excluded; another fork owns it).
Concerns/Blockers: Six runner gaps need fixes outside this fork; cause 4 duplicates a tool's side effect on retry. Clippy could not run on these files because of other areas' lint errors.
