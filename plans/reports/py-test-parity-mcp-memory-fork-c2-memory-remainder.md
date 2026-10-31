# Fork C2: memory-core remainder parity report

Scope: unported tests in the memory-core Python files. Already done by the parent and earlier forks (not redone): `test_session.py` core, `test_session_limit.py`, `test_session_helpers.py` (partial), `test_openai_conversations_session.py` (partial), `test_openai_responses_compaction_session.py` (partial), one scenario of `test_compaction_model_visibility.py`, `tests/utils/test_simple_session.py`, `tests/utils/test_pretty_print_and_items.py` (partial).

Counting: "def" = Python `def test_` functions (AST-checked; `test_session_context_wrapper.py` has 12 real tests, the 13th grep hit is a nested helper `test_tool`). "Expanded" = def count with `parametrize` cases multiplied, as in the fork-A/B/D/E tables. Per-file `ported`/`covered`/`python-only`/`cross-area`/`unported` are def-level. Python parametrize expansion is not re-derived per case for files I did not port; those rows show the file total only.

## New Rust test files (all green with `--all-features`)

| file | tests | green |
|---|---|---|
| `tests/py_memory_remaining_a.rs` | 10 | 10 |
| `tests/py_memory_remaining_b.rs` | 5 | 5 |
| `tests/py_memory_remaining_c.rs` | 5 | 5 |

No source files were changed in this fork, so there are no Rust bug fixes. No `#[ignore]` was added: no test here needs a real network, credentials, docker or python.

## Per-file table

| file | def | expanded | ported | covered | python-only | cross-area | unported | failures found |
|---|---|---|---|---|---|---|---|---|
| `memory/test_session_persistence_sanitize.py` | 9 | 27 | 8 (26 expanded) | 0 | 1 | 0 | 0 | 0 |
| `memory/test_compaction_suffix.py` | 11 | 23 | 3 | 0 | 4 | 4 | 0 | 0 |
| `memory/test_session_context_wrapper.py` | 12 | 16 | 0 | 0 | 2 | 10 | 0 | 0 |
| `memory/test_sqlalchemy_compaction.py` | 9 | 15 | 1 | 0 | 2 | 2 | 4 | 0 |
| `memory/test_compaction_model_visibility.py` | 20 | 40 approx. | 1 (1 of 3 variants passes; 2 are cross-area `#[ignore]`d in `py_memory_compaction_visibility.rs`) | 0 | 0 | 19 | 0 | 0 |
| `memory/test_openai_responses_compaction_session.py` | 73 | 94 | 0 new (parent's `py_memory_compaction.rs` covers ~25) | 0 | see below | see below | ~48 | 0 |
| `memory/test_openai_conversations_session.py` | 52 | 74 | 2 new (+1 page-order check) | 0 | 13 | 0 | ~17 | 0 |
| `memory/test_session.py` | 40 | 56 | 3 runner-variant ports (run_sync, run_streamed) | 18 (basic/runner/settings, see `py_memory_session.rs`) | 3 | 0 | 16 | 0 |
| `utils/test_json.py` | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| `extensions/memory/test_async_sqlite_session.py` | 40 | 45 | 3 | 18 | 19 | 0 | 0 | 0 |
| `extensions/memory/test_memory_imports.py` | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 |

### Classification detail

**`test_session_persistence_sanitize.py`** (ported through `OpenAIConversationsSession` writes, where the sanitizer is observable)
- Ported: `preserves_ids_required_by_openai_conversation_items` (13 cases), `preserves_file_search_call_payload_id`, `preserves_program_payload_id`, `preserves_program_output_payload_id`, `strips_optional_or_policy_controlled_ids` (7 cases), `preserves_reasoning_id_for_openai_conversations`, `preserves_reasoning_encrypted_content`, `always_strips_provider_data`.
- Python-only: `passes_through_non_dict_items` (asserts identity of an arbitrary Python object; Rust `InputItem` is always an object).

**`test_compaction_suffix.py`** (SQLite backend; the TTL clock is not injectable in Rust, so TTL cases are excluded)
- Ported: `test_suffix_transaction_rolls_back_failed_insert`, `test_new_backend_write_invalidates_snapshot` (the mock writes the concurrent row directly into the SQLite file during the compaction request), `test_partial_previous_response_mode_preserves_history` (ported with `SqlxSession`, the second backend in the Python fixture).
- Python-only: `test_cancelled_suffix_transaction_settles_before_newer_append` (asyncio task cancellation). The invariant is covered by `memory::await_session_mutation` in `src/memory/mod.rs`.
- Python-only (clock): `test_expired_prefix_does_not_block_full_live_window`, `test_repeated_ttl_bursts_do_not_accumulate_expired_rows`, `test_prefix_cleanup_stops_before_retained_history`. These monkeypatch `cryptography.fernet.time.time`. Rust has no clock hook, so these would need one; the expiry behavior is covered by `sessions_tests::encrypted_session_skips_expired_items` with a real 1-second TTL.
- Cross-area: `test_failed_encrypted_suffix_transaction_preserves_original_tokens`, `test_unverifiable_rows_inside_snapshot_are_retained` (both wrap `EncryptedSession` around the compaction session; Rust supports only the reverse order, see fork D's report), `test_partial_compaction_does_not_split_tool_call_and_output`, `test_partial_compaction_preserves_reasoning_and_its_follower` (both use `call_model_input_filter`; see deferral 1).

**`test_session_context_wrapper.py`** (Python passes a `wrapper=RunContextWrapper` keyword to sessions that accept it, detected by signature inspection)
- Cross-area (10 tests): `runner_passes_same_wrapper_to_context_aware_session`, `runner_preserves_legacy_session_call_shapes`, `runner_does_not_treat_legacy_kwargs_as_wrapper_opt_in`, `runner_preserves_legacy_calls_when_signature_inspection_fails`, `runner_does_not_partially_enable_context_aware_session`, `retry_rewind_uses_same_context_scope_for_reads_pops_and_cleanup`, `retry_rewind_restores_partial_pops_in_the_same_context_scope`, `input_guardrail_persists_in_the_context_scope`, `resumed_run_persists_in_the_context_scope`, `compaction_session_keeps_context_aware_underlying_on_legacy_scope`. Rust's `Session` trait has no wrapper parameter. Minimal fix: optional context-aware methods on the trait with default bodies, plus the run pipeline passing the wrapper (`src/run_internal/**`).
- Python-only (2): `test_session_wrapper_method_requires_named_wrapper_parameter`, `test_session_wrapper_opt_in_requires_all_history_operations` (both test `inspect.signature`).

**`test_sqlalchemy_compaction.py`** (mapped to `SqlxSession` on SQLite; no SQLAlchemy in Rust)
- Ported: `test_other_instance_mutation_invalidates_suffix` (append, pop and clear; the other instance mutates from the mock's handler on a thread with its own runtime). Only the file backend is run; the Python fixture also runs an in-memory engine.
- Cross-area (2): `test_encrypted_prefix_cleanup_is_atomic`, `test_encrypted_unknown_row_is_retained` (encrypt-over-compaction, see above).
- Python-only (2): `test_cancelled_replacement_settles_before_other_instance_append` (asyncio cancellation), `test_compaction_with_engine_managed_sqlite_begin` (SQLAlchemy engine events).
- Unported (4, not yet ported, no Rust blocker found): `test_limited_sqlalchemy_compacts_visible_suffix`, `test_replacement_uses_timestamp_order_not_id_order`, `test_failed_replacement_preserves_history_during_other_instance_access`, `test_parent_constraint_rejection_propagates`.

**`test_compaction_model_visibility.py`**
- One of three variants of `test_automatic_compaction_retains_hidden_tool_output_until_full_replay` is green (`session` filter). The `model` and `redact` variants are cross-area `#[ignore]`d (deferral 1).
- Cross-area (19): every other test. They need `call_model_input_filter`, handoffs, pending approvals, hosted MCP approvals, reasoning-id policy, or encrypt-over-compaction (deferrals 1 and 3).

**`test_openai_responses_compaction_session.py`** (73 defs; this fork did not port the bulk of the remainder)
- Remaining tests are not ported in this fork. Rough breakdown: about 15 are `run_compaction` cancel or restore tests (`test_run_compaction_restores_history_when_*_is_cancelled`, `test_cancel_restore_waits_*`, `test_exception_restore_drains_*`, and similar). These depend on asyncio cancellation and are python-only. About 10 are runner-driven deferral or resume tests (`test_deferred_*`, `test_runner_skips_compaction_after_interleaved_*`, `test_resumed_save_carries_generation_*`), which are cross-area. About 20 are message-normalization and mock-call-sequence tests, which are unported.
- Unported, to do: the normalization family (`test_run_compaction_normalizes_compacted_user_image_messages`, `..._file_messages`, `..._file_id_inputs_and_preserves_metadata`, `..._preserves_history_when_output_normalization_fails`) and `test_run_compaction_suppresses_model_dump_warnings`. These are pure compaction-output transforms and are the best next ports.

**`test_openai_conversations_session.py`**
- Ported here: `concurrent_first_write_recovers_after_creation_failure` (a failed lazy creation leaves no id; the next write creates one), `operations_wait_for_complete_append` (8 concurrent appends, every write delivered once), page order of `get_items` ("oldest first, no `id`").
- Python-only (13): the cancellation tests (`test_clear_session_cancellation_*`, `test_cancelled_batch_settles_before_queued_writer`, `test_cancellation_between_chunks_stops_next_request`, `test_single_request_cancellation_does_not_wait_for_response`), the default-client fallback tests (`test_start_*`, `test_init_with_*_fallback`), and the Python-client-object tests.
- Cross-area: the `test_runner_resume_*` pair (runner resume and periodic-history atomicity) and `test_runner_persists_program_item_ids`.
- Unported (about 17): the `chunk_*` retry tests (need the Rust client's retry-policy behavior checked first), `test_runner_with_conversation_history`, `test_invalid_item_id_in_pop_item`, `test_api_failure_during_*` (partly covered by the parent's file), and the `test_session_settings_*` variants already covered elsewhere.

**`test_session.py`**
- Ported here: `session_memory_round_trips_through_run_sync` and `session_memory_round_trips_through_run_streamed`, the `run_sync` and `run_streamed` variants of the basic-functionality test. The `run` variant is in `py_memory_session.rs`.
- Covered (18): the runner and settings tests in `py_memory_session.rs` (run variant), `py_memory_limits_helpers.rs` (limit tests), and the SQLite pop, get, limit, unicode and special-character tests in `py_memory_session.rs`.
- Python-only (3): `test_await_mutation_cancellation_hides_later_failure_without_loop_error` (asyncio loop exception handler), and the two worker-thread connection tests (`test_sqlite_session_reaps_connections_from_exited_worker_threads`, `test_sqlite_session_closes_retired_connections_before_the_next_allocation`), which test the thread-local connection pool of the Python `SQLiteSession`. Rust's `SqliteSession` shares one `Mutex<Connection>` and has no such pool.
- Unported (16): the remaining runner-method parametrizations of `test_session.py` that the port above did not reach (`test_session_memory_different_sessions_*`, `*_explicit_instance_*`, `*_disabled_*`, `*_appends_list_input_*`, and the callback variants for `run_sync` and `run_streamed`). Their `run` variant is ported in `py_memory_session.rs`, so the gap is only the two extra runner methods.

**`utils/test_json.py`**: python-only (2). Tests `_to_dump_compatible` on Python iterables and `MappingProxyType`. Rust types are concrete, so there is no equivalent input space.

**`extensions/memory/test_async_sqlite_session.py`**
- Covered (18): the basic, pop, corrupt-row, limit, settings, unicode, runner-integration, isolation, empty-pop and limit-over-available cases (`py_memory_session.rs`), and the rollback cases for add, clear and pop (`a_failed_mutation_*`, `a_failed_clear_session_*`, `a_failed_pop_item_*`).
- Ported here (3): `sqlite_add_of_an_empty_list_stores_nothing`, `sqlite_order_survives_writes_within_one_timestamp`, `sqlite_pop_returns_the_latest_of_a_same_timestamp_burst`.
- Python-only (19): the close and closed-state tests (no `close()` in Rust; drop closes), the connection-level WAL and init-retry tests (aiosqlite internals), the quarantine and eviction tests for rollback and close failures (Rust has no equivalent connection pool), and the asyncio cancellation tests.

**`extensions/memory/test_memory_imports.py`**: python-only (2). Checks Python import paths and optional-extra error messages.

## Cross-area deferrals (for the run-pipeline and memory owners)

1. **Compaction exchange ignores `call_model_input_filter`.** `src/run_internal/mod.rs:1559-1608` (`notify_session_completed`) builds `model_exchange` from `model_prefix`, which is the input before the filter runs (the filter is applied at `src/run_internal/mod.rs:1820`). Python records the filtered input. Minimal fix: record the filtered `input` in the exchange. This blocks 2 ignored tests in `py_memory_compaction_visibility.rs` and the 2 cross-area suffix tests.
2. **Session run-context wrapper.** Python's duck-typed `wrapper=RunContextWrapper` keyword is not supported. Minimal fix: optional context-aware trait methods on `Session` with default bodies, and the run pipeline passing the wrapper when present. Blocks 10 tests in `test_session_context_wrapper.py`.
3. **Encrypt-over-compaction order.** `EncryptedSession` cannot wrap an `OpenAIResponsesCompactionSession`, which is the outer layer in Python. Blocks 4 cross-area tests (`test_encrypted_*` in `test_sqlalchemy_compaction.py` and `test_compaction_suffix.py`). Fork D reports the same gap.
4. **Pending approval defers automatic compaction.** `test_pending_approval_defers_automatic_compaction` and `test_pending_hosted_mcp_approval_defers_automatic_compaction` need the runner to skip compaction while an approval is pending. Blocks 2 visibility tests.
5. **Run-pipeline deferrals already reported by earlier forks** (deferred interrupted writes, `_PendingSessionWrite`) are not repeated here.

## Totals

- New Rust test functions: 20 (10 + 5 + 5), all green.
- Python defs classified in this fork's scope: 9 + 11 + 12 + 9 + 20 + 73 + 52 + 40 + 2 + 40 + 2 = 270 defs (file-level rows above).
- Ported with new Rust tests in this fork: 8 sanitize, 2 suffix, 1 sqlx compaction, 1 previous-response mode, 3 async-sqlite, 2 runner variants, 2 conversations (plus 1 page-order check), totalling 19 defs. The per-file table gives the exact split.
- Not ported in this fork, with reasons above: about 48 responses-compaction tests (mostly cancel-restore python-only and runner cross-area; about 20 normalization tests are the main gap), about 17 conversations tests, about 16 `test_session.py` runner variants, and the 4 SQLAlchemy compaction tests listed above.

## Gates

- `cargo test --all-features` for `py_memory_remaining_a`, `py_memory_remaining_b`, `py_memory_remaining_c`: all green.
- `rustfmt --edition 2024` applied to the three new files.
- `cargo clippy --all-features --all-targets -- -D warnings` could not run past the lib: `src/extensions/models/mod.rs:1016` (another area) fails first, so the new test files were not linted by clippy.
- No changes to `src/lib.rs`, `Cargo.toml`, `tests/py_mcpmem_support/mod.rs`, or any existing `py_*` test file.

Status: DONE_WITH_CONCERNS
Summary: 20 new Rust tests across three files, all green. Ported 19 Python defs (sanitize, compaction-suffix subset, sqlx compaction, async-sqlite subset, runner-method variants, conversations subset). Classified the rest, with the largest unported block in `test_openai_responses_compaction_session.py` (about 20 normalization tests still to port).
Concerns/Blockers: (a) `test_openai_responses_compaction_session.py` is mostly unported. The normalization family is the best next step. (b) Clippy cannot pass until the lib lint in `src/extensions/models/mod.rs:1016` is fixed by its owner. (c) Cross-area deferrals 1 to 4 above need the run-pipeline owner.
