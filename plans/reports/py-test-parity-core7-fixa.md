# Core7 fix round, area A (fixa): entries, run state, items, sqlalchemy twins

## Cases ported (new tests)
- `tests/py_core7_a_entries_swaps.rs` (5 tests): test_entries.py pins_parent_directories_during_open, fallback_rejects_swapped_parent_directory, revalidates_swapped_paths_during_open (file swapped for a link), a file swapped for a FIFO (path_changed_during_copy, no blocking), and the source-root swap pair. The swap runs from the session's `mkdir`, which the copy calls between listing a directory and opening what it listed. The root-swap case swaps at the destination `mkdir`, which precedes validation, so it asserts the validation refusal (child `src`); the root open itself is also `O_NOFOLLOW|O_DIRECTORY`, but there is no await between validation and that open to hook.
- `tests/py_core7_a_runstate_corpus.rs` (5): the unused fixture `security/v1_13_legacy_mount_credentials.json` (all three sentinels removed, redaction marker set, driver options gone, rewrite idempotent); 100,000-deep JSON (no overflow, no payload echo); non-object JSON; value-free errors for the context-type and completed-call-id carriers over `from_json` and `from_json_str`; `pending_input` that is not a list.
- `tests/py_core7_a_items_null_text.rs` (1): null `text` in an output_text part.
- `tests/py_core7_a_sqlx_compaction_memory.rs` (8): in-memory twins of test_sqlalchemy_compaction.py on a single-connection `sqlite::memory:` pool through `SqlxSession::with_pool` (7 mirrors of the file twin, plus the memory half of other_instance_mutation_invalidates_suffix, 3 operations).

## Source bugs fixed
- `src/sandbox/entries.rs`: `copy_host_dir` (unix) now walks with directory descriptors (new `pinned` module over `rustix`): root opened `O_NOFOLLOW|O_DIRECTORY`, each subdirectory opened relative to its pinned parent, each file opened relative to the descriptor of the directory it was listed in with `O_NOFOLLOW|O_NONBLOCK` and a regular-file check. A parent or source root swapped for a link after validation now fails instead of being followed (it was leaf-only before). Errors: `LocalDirReadError` with `reason` `symlink_not_supported` or `path_changed_during_copy` and `child`. Non-unix keeps the path-based copy.
- `src/run_state/wire.rs`: `pending_input` that is present but not a list (including null) is refused with `Run state pending_input must be a list` (was silently read as empty).
- `Cargo.toml`: new unix-only optional dependency `rustix` (fs, std), enabled by the `sandbox` feature. The crate forbids unsafe, so descriptor-relative opens needed a safe wrapper. Already in the lockfile transitively; `Cargo.lock` gained the edge.

## Breaking changes
- `RunState::from_json` rejects a non-list `pending_input` (previously accepted). `LocalDir` copy of a tree whose parent is swapped now errors (previously could read outside the tree). No public signature changed.

## Remaining python-only, one reason per group
- test_entries (3): `falls_back_when_safe_dir_fd_open_unavailable` (toggles Python's private dir_fd flags), `copies_to_a_posix_destination_from_a_windows_host` (PureWindowsPath), plus the root-open window (see above).
- test_run_state_compatibility_corpus (68): comparator self-tests and monkeypatched `from_json`/`to_json` (12), exception/traceback/slots/`__notes__`/ExceptionGroup/cancellation/hostile-descriptor redaction of Python objects (about 45), pydantic-style foreign mappings and context-wrapper subclasses (6). The `agent_name`/`agent_identity` carriers resolve at resume and are covered by `invalid_run_state_fixtures_fail_without_disclosing_values`.
- test_items_helpers (4): weakref/gc retention (2), pydantic rejection of a null refusal at construction, `model_construct` partial objects (typed Rust parts cannot be built invalid).
- test_sqlalchemy_compaction (7): cancelled replacement and failed replacement while another instance reads pause a SQLAlchemy `AsyncSession.execute` mid-transaction, and engine-managed `BEGIN` is a SQLAlchemy event; sqlx has no such seam. The 7 are the three seam tests on both halves plus the encrypted-outer rejected-insert variant.
- test_sqlalchemy_session (13): dialect quirks (rowcount, RETURNING), asyncio cancellation, cross-event-loop, engine property, SQLite config registry gc.
- test_manifest (11), test_manifest_config (18): pydantic `model_dump`/`model_validate` of manifests and dict-typed manifests; Rust `Manifest` has no serde surface (design decision, see concerns).
- test_run_state (165), pending_input (11), hook_identity (2), agent_identity (3), pydantic_context (8): reasons as recorded in py-test-parity-core2-runstate.md and core3-runstatectx.md (live-object identity, private helpers, deepcopy hooks, Python class names, pydantic context); re-read and kept.

## Not in scope but seen
- `tests/test_run_state.py::test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval` is still not ported: reading a pre-1.15 snapshot should mark hosted per-call decisions unbound. The fix is in run-context approval code outside this area.

## Counts
Python-only cases moved to ported: 24 (entries 4; run-state corpus 8: security fixture 1, deep JSON 1, value-free restoration 4, pending_input 2; items 1; sqlalchemy compaction memory twins 11). Rust tests added: 19 across 4 files. Remaining python-only in the audited area files: entries 3, corpus 68, items 4, sqlalchemy compaction 7 (3 seam tests on both halves plus one outer-reject variant), the rest unchanged (see CSV).

In the group list above, read the corpus remainder as 68 and the sqlalchemy compaction remainder as 7.
