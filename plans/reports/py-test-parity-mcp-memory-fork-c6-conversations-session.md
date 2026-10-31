# Fork C6: conversations + test_session remainder

New files (rustfmt applied; run with `--all-features`):
- `tests/py_memory_conversations_remaining.rs`: 19 passed, 2 ignored.
- `tests/py_memory_session_remaining.rs`: 8 passed.

Verified from the actual run: 27 passed, 0 failed, 2 ignored. No source files changed, no existing `py_*` file or the support module touched.

Counting: def = AST `test_` functions, expanded = parametrize multiplied. Classes: PORTED-now (this fork), COVERED (earlier file, cited), PYTHON-ONLY, CROSS-AREA / divergence (ignored or unported).

## tests/memory/test_openai_conversations_session.py (52 def / 74 expanded)

| group (python tests) | def | exp | outcome |
|---|---|---|---|
| `start_openai_conversations_session` x4, constructor `init_*` x4 | 8 | 8 | PYTHON-ONLY: standalone start function, default-client lookup, constructor kwargs; Rust has `with_client` / `from_conversation_id` |
| `get_session_id` x3 (existing id, creates new, clear id) | 3 | 3 | COVERED: `py_memory_conversations.rs::add_items_creates_the_conversation_first`, `::clear_session_deletes_the_conversation_and_forgets_its_id`, `::add_items_posts_one_request_for_a_short_list` |
| zero limit, large limit | 2 | 2 | COVERED: `py_memory_conversations.rs::a_zero_limit_is_never_forwarded...`, `::a_large_limit_is_applied_after_the_providers_pagination` |
| add_items x4 (simple, creates id, empty x2) | 4 | 4 | COVERED: `py_memory_conversations.rs::add_items_posts_one_request...`, `::an_empty_add_items_*` |
| pop x2 (items, empty) | 2 | 2 | COVERED: `py_memory_conversations.rs::pop_item_*` |
| clear, uninitialized x2, failed delete, retry after failed delete | 5 | 5 | COVERED: `py_memory_conversations.rs::clear_session_*`, `::a_failed_delete_retains...` |
| clear cancellation x2 (settle delete, preserve replacement id) | 2 | 2 | PYTHON-ONLY: asyncio cancel-settle semantics and the settable `session_id` attribute. Rust drop semantics pinned by `a_dropped_writer_stops_sending_chunks_and_releases_the_lock` |
| clear + concurrent id lookup | 1 | 1 | PORTED: `a_concurrent_id_lookup_during_clear_does_not_clobber_the_new_id` |
| operations wait for complete append (add/pop/clear/read) | 1 | 4 | PORTED: `an_add_/a_pop_/a_clear_/a_read_waits_for_a_complete_append` |
| request sizes and order (0/20/21/41) | 1 | 4 | PORTED: `request_sizes_for_*` |
| later failure preserves prefix and stops | 1 | 1 | PORTED: `a_later_failure_keeps_the_prefix_and_a_follow_up_write_succeeds` |
| chunk lost response not retried | 1 | 4 | 2 PORTED: `a_lost_first/second_response_is_not_retried`; 2 PYTHON-ONLY (`configured_retries` variants: Rust `OpenAIClient` has no SDK retry layer, so nothing is retried either way) |
| chunk retry override keeps single-request retries | 1 | 1 | PYTHON-ONLY: `max_retries` / `with_options` do not exist in Rust |
| cancelled batch settles before queued writer | 1 | 2 | PYTHON-ONLY: asyncio shield/settle; Rust counterpart is `a_dropped_writer_stops_sending_chunks_and_releases_the_lock` |
| cancellation between chunks stops next request | 1 | 1 | PORTED (Rust-flavored, same test as above: dropped writer sends no further chunk) |
| single request cancellation does not wait for response | 1 | 1 | PORTED: `a_single_request_caller_does_not_wait_for_the_response` |
| runner batches input with lazy creation (run/streamed) | 1 | 2 | PORTED: one conversation create, input order, every batch <= 20, 42 items total. Exact `[20,20,1,1]` split is `#[ignore]` cross-area `the_runner_persists_input_before_the_reply` |
| runner resume does not replay partial repeated history | 1 | 8 | CROSS-AREA unported: needs `pending_session_write` and a run state that restores against an agent; neither exists in Rust (`src/run_internal`, `src/run_state*`) |
| runner resume periodic history / session atomicity | 1 | 5 | CROSS-AREA unported: same missing machinery (`Cannot reconcile the pending Session write` has no Rust counterpart) |
| runner integration basic | 1 | 1 | COVERED: `py_memory_conversations.rs::the_runner_stores_a_turn_in_the_conversation` |
| runner with conversation history | 1 | 1 | PORTED: `the_runner_sends_the_stored_conversation_history_to_the_model` |
| runner persists program item ids | 1 | 1 | CROSS-AREA unported: Rust has no `Program` / `ProgramOutput` item type (`src/items.rs`); the id-preservation rule itself is covered by `py_memory_conversations.rs::conversation_writes_keep_ids_only_where_the_schema_requires_them` |
| api failure: creation, add_items, clear | 3 | 3 | COVERED: `py_memory_conversations.rs::an_api_failure_while_creating...`, `::a_failed_chunk_keeps_the_prefix_and_stops`, `::a_failed_delete_retains...` |
| invalid item id in pop_item | 1 | 1 | DIVERGENCE, `#[ignore]`: `an_item_without_an_id_cannot_be_popped` (see below) |
| multiple sessions, concurrent first writes | 2 | 2 | COVERED: `py_memory_conversations.rs::separate_sessions_use_separate_conversations`, `::concurrent_first_writes_share_one_conversation` |
| lazy creation consistency | 1 | 1 | PORTED: `the_conversation_id_is_created_once_across_lookups` |
| concurrent first write recovers after creation failure | 1 | 1 | PORTED: `a_waiting_writer_recovers_when_the_first_creation_fails` (also earlier `py_memory_remaining_c.rs::a_failed_first_creation_is_retried_by_the_next_write`) |
| session settings: default / constructor / dict | 3 | 3 | 1 COVERED (`py_memory_conversations.rs::session_settings_default_to_none_and_are_configurable`); 2 PYTHON-ONLY (Python default is an empty `SessionSettings`, Rust `None`; dict coercion) |
| **total** | **52** | **74** | PORTED-now 19, COVERED 23, PYTHON-ONLY 17, divergence/cross-area 15 (13 resume + 1 program + 1 pop-without-id) |

74 = 19 + 23 + 17 + 15. The 2 ignored tests are the pop-without-id divergence and the exact-split cross-area test; the exact-split test is an extra assertion on top of the 2 PORTED runner-batches cases.

## tests/memory/test_session.py (40 def / 56 expanded)

| group | def | exp | outcome |
|---|---|---|---|
| `await_mutation` cancellation leaks a loop error; `post_commit_cancellation` | 2 | 4 | PYTHON-ONLY: asyncio cancellation of a task; a dropped Rust future cannot be re-awaited |
| close / worker-thread connection tests (`close_closes_worker_thread_connections`, `reaps_connections_from_exited_worker_threads`, `closes_retired_connections`, `keeps_connections_of_live_worker_threads`, `closed_rejects_empty_add_items`, `rollback_failure_evicts_connection`, `close_retries_quarantined_connection`) | 7 | 7 | PYTHON-ONLY: Python opens one sqlite connection per executor thread and tracks, reaps and quarantines them; Rust keeps one `Arc<Mutex<Connection>>` that closes on drop |
| file lock shared across instances | 1 | 1 | PYTHON-ONLY: process-local lock registry; Rust relies on SQLite file locking. Behaviour COVERED by `py_memory_session.rs::two_sessions_on_one_file_write_independently` |
| settings default / dict normalization / unknown dict field | 3 | 3 | PYTHON-ONLY: dict coercion and an empty default object; Rust has typed `SessionSettings` and `session_settings()` is `None` by default |
| basic functionality (run / run_sync / run_streamed) | 1 | 3 | COVERED: `py_memory_session.rs::session_memory_basic_functionality`, `py_memory_remaining_c.rs::session_memory_round_trips_through_run_sync` / `_run_streamed` |
| explicit instance, disabled, different sessions (each x3) | 3 | 9 | PORTED: `explicit_session_instance_keeps_history_for_every_runner_method`, `disabled_memory_sends_only_the_new_message_for_every_runner_method`, `different_sessions_stay_separate_for_every_runner_method` |
| appends list input, callback prepared input, repeating callback (each x3) | 3 | 9 | PORTED: `list_input_is_appended_to_history_for_every_runner_method`, `session_input_callback_prepares_input_for_every_runner_method`, `a_repeating_callback_does_not_grow_the_session_for_every_runner_method` |
| settings constructor | 1 | 1 | PORTED: `session_settings_constructor_stores_the_limit` |
| runner session settings override | 1 | 1 | COVERED: `py_memory_session.rs::runner_session_settings_override_the_sessions_own_limit`; also run for all three methods by `the_runner_settings_override_wins_for_every_runner_method` |
| direct round trip, pop (x2), pop corrupt (x2), limit, limit corrupt (x2), unicode, special chars, concurrent writes, failed add / clear / pop releases lock, add_items exception in stream, uses settings limit, explicit overrides, resolve | 17 | 17 | COVERED: `py_memory_session.rs` (`sqlite_session_round_trips_and_clears`, `sqlite_session_pop_item`, `pop_item_only_affects_its_own_session`, `pop_item_skips_a_corrupt_newest_row`, `pop_item_returns_none_after_dropping_only_corrupt_rows`, `get_items_honours_an_explicit_limit`, `get_items_limit_counts_valid_items_past_corrupt_newest_rows`, `session_settings_limit_also_counts_valid_items`, `sqlite_session_stores_unicode_content`, `sqlite_session_stores_special_characters_and_sql_keywords`, `sqlite_session_survives_concurrent_writes`, `a_failed_mutation_leaves_the_session_usable_and_unlocked`, `a_failed_clear_session_leaves_the_write_lock_free`, `a_failed_pop_item_leaves_the_write_lock_free`, `a_failing_session_write_propagates_through_the_stream`, `get_items_uses_the_sessions_own_limit_and_an_explicit_one_wins`, `session_settings_resolve_prefers_the_override`) |
| **total** | **40** | **56** | PORTED-now 19, COVERED 22, PYTHON-ONLY 15 |

56 = 19 + 22 + 15. Note: Rust's `Runner::run_sync` refuses to run inside a runtime, so the `RunSync` variants run on a plain OS thread (`std::thread::spawn` inside `spawn_blocking`); a blocking-pool thread still carries the runtime context.

## Rust bugs fixed
None. No source file in `src/memory/**` was changed by this fork.

## Divergences recorded (not fixed here)
1. **pop without id** (`src/memory/openai_conversations_session.rs`, `pop_item`): Python raises `KeyError('id')` and leaves the item stored; Rust returns the item and skips the delete, so it reports an item as popped while it is still in the conversation. Minimal fix: return an error when the newest item has no `id`. It was not applied because `tests/py_memory_conversations.rs::an_item_without_an_id_is_returned_without_a_delete` asserts the old behaviour and that file is read-only for this fork; the fix needs that test updated in the same change. The ignored test `an_item_without_an_id_cannot_be_popped` asserts Python's behaviour.
2. **Id handling location**: Python sanitizes ids in the run pipeline (`session_persistence`) and returns raw server items from `get_items`; Rust sanitizes on `add_items` and strips `id` / `status` / `created_by` on reads. Equivalent through the runner, different for direct session use, so ported tests compare content and server-assigned ids, not client ids.

## Cross-area deferrals (do not edit here; core agent owns `src/run_internal`)
1. `src/run_internal/mod.rs` `save_session` (~:1536): input and model output are appended in one call (`[20,20,2]` for 41 inputs + 1 reply); Python appends the input first and the reply afterwards (`[20,20,1,1]`), so a failing model call still leaves the input stored. Ignored test: `the_runner_persists_input_before_the_reply`.
2. No `pending_session_write` and no run state restore against an agent, so the partial-append resume tests (`test_runner_resume_does_not_replay_partial_repeated_history`, 8 expanded; `test_runner_resume_handles_periodic_history_for_session_atomicity`, 5 expanded) cannot be written. Python expectation: after a failed or partial append the resume either completes without replaying written items or fails with `Cannot reconcile the pending Session write`.
3. `src/items.rs`: no `Program` / `ProgramOutput` output items, so `test_runner_persists_program_item_ids` is unportable until they exist.

## Gates
- Tests: 27 passed, 0 failed, 2 ignored across the two files.
- rustfmt: applied to both files.
- Clippy: not run clean on these files. `cargo clippy --all-features --test py_memory_conversations_remaining --test py_memory_session_remaining -- -D warnings` stops in the lib on other areas' errors (`src/run_internal/agent_identity.rs:106`, `src/run_internal/mod.rs:2292,2657`, `src/stream_events.rs:48`, `src/sandbox/sandboxes/unix_local.rs:894,920`).

Unresolved: apply the pop-without-id fix together with an update to `an_item_without_an_id_is_returned_without_a_delete`?

Status: DONE_WITH_CONCERNS
