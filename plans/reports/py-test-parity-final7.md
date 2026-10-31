# Final 7 compaction-session cases

All seven remaining not-ported cases of `tests/memory/test_openai_responses_compaction_session.py` are now ported in
`tests/py_core6_final7_compaction.rs` (8 tests: 7 ports plus one positive control). No source change was needed.

| Python case | Rust test | Outcome |
|---|---|---|
| `test_rollback_does_not_revive_items_expired_during_compaction` x4 (budget None/2 x clear/add) | `rollback_does_not_revive_expired_items_{unbounded,budget}_{clear,add}` | ported |
| `test_resumed_save_carries_generation_into_compaction` | `resumed_save_carries_generation_into_compaction` | ported |
| `test_runner_skips_compaction_after_interleaved_model_wait` | `runner_skips_compaction_after_interleaved_model_wait` | ported |
| `test_runner_skips_compaction_after_interleaved_persist` | `runner_skips_compaction_after_interleaved_persist` | ported |

Nothing is python-only.

## How each scenario is driven

- Expiry: a real `EncryptedSession` (ttl 3s, real clock; Python monkeypatches the Fernet clock) over SQLite. Item 1 is written, 1.5s later item 2; the mock `responses/compact` endpoint sleeps 2.55s, so item 1 (4.05s old) expires during the request and item 2 stays live. A wrapper counts the mutation Python patches (`clear_session` or `add_items`).
- Interleaved model wait: run A's scripted model step blocks (`respond_async`) until run B has completed on the same session.
- Interleaved persist: a forwarding session wrapper pauses after run A's append returns, before `on_run_completed`; run B runs in that gap (Python pauses in `_add_items_with_generation`).
- Resumed save: a handoff turn's append is lost (wrapper fails it before writing), the checkpoint is resumed, and the resumed append pauses after landing while run B appends.
- Control: the same run with no second writer does issue one compaction request, so the skips are meaningful.

## Divergence (outcome identical)

- Rollback cases: Python replaces its pre-request snapshot, the replacement fails, and rollback must not revive the expired item (it raises `RuntimeError`). Rust re-reads the store after the request, sees the expiry as a change, and never attempts the replacement. The test asserts the same observable outcome: history is `[live]`, one compact request, the patched `clear_session`/`add_items` is never called, and `run_compaction` returns Ok (no error to propagate). The rollback-restore path itself stays covered by `py_memory_responses_compaction_b`.
- Generation cases: Rust has no mutation generation; the skip comes from matching the stored tail against the model exchange (plain list store, `previous_response_id` mode cannot replace a partial suffix). Outcome matches: zero compact requests, run A and run B items both stored.

## Bugs fixed / breaking changes

None. No source edits.

## Gates

- `cargo test --all-features --test py_core6_final7_compaction`: 8 passed.
- `py_core6_mcpmisc_compaction` (2) and every test binary mentioning compaction (suffix, visibility, session, resume, encrypt, sqlx, deferred writes, memory) plus `--lib compaction`: all pass.
- `cargo clippy --all-features --all-targets -- -D warnings`: clean. `rustfmt --edition 2024` applied.

Status: DONE
Summary: 7/7 remaining cases ported (plus a control); Rust matches Python's observable outcomes; no source changes.
Concerns: the expiry tests use the real clock (about 4s each, run in parallel) and carry about 0.4s of timing margin for the live item (2.55s of a 3s ttl); a very slow CI host could flake them. Rust does not raise on the expired-during-compaction case (it skips replacement silently with a warning).
