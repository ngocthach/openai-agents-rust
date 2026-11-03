# Core7 fixc: sandbox providers and memory python-only audit

## Cases ported (moved python-only -> ported): 9 (+1 runtime row)

Vercel provider contract (3), `tests/py_core6_modalvercel_vercel_contract.rs`:
- `a_bare_legacy_id_resumes_without_creating_a_sandbox` (`test_installed_provider_resumes_legacy_id_without_name`)
- `resume_does_not_adopt_a_replacement_session` (`..._does_not_adopt_replacement_session`)
- `cleanup_stops_only_the_owned_session` (`..._cleanup_preserves_concurrent_replacement`)

Memory (3), `tests/py_core7_c_memory_settings.rs`: dictionary normalization, unknown field rejection, non-numeric limit
(`test_session.py::test_session_settings_constructor_normalizes_dictionary`, `..._rejects_unknown_dictionary_fields`,
`test_session_helpers.py::test_settings_normalization_and_limit_precedence` dictionary half).

Cloudflare (2), `tests/py_core6_cfdaytona_cloudflare.rs`: the two shutdown-failure cases now assert Python's swallow
(they asserted an error before, i.e. the old divergence). Daytona shutdown swallow: `tests/py_core7_c_shutdown_swallow.rs` (1).
Live-session concurrency limits (`test_session_manager_passes_concurrency_limits_from_run_config[live_session]`, plus
create and resume): `tests/py_core7_c_concurrency_limits.rs` (1 test, 3 sources).

## Source fixes

- `src/extensions/sandbox/vercel/mod.rs`: `split_provider_id` accepts a bare id (session id = name), as Python resumes a
  state with an id and no name. An empty identity is still `SessionStateInvalid`.
- `src/memory/mod.rs`: `SessionSettings` is `deny_unknown_fields` (Python raises `Unknown session settings`).
  Behaviour change: deserializing settings JSON with extra keys now fails.
- `src/sandbox/session.rs`, `runtime.rs`, `events.rs`: new defaulted trait method `SandboxSession::set_concurrency_limits`
  (mirrors `_set_concurrency_limits`), called for created, resumed and injected live sessions; the live-session manifest
  update now copies `local_dir` additions with the run config's `local_dir_files` limit. Public API addition only.
- `src/extensions/sandbox/cloudflare/sandbox.rs`: `shutdown` swallows a failed or unreachable DELETE (debug log, no provider
  text) unless a terminal mount transition is pending, as Python's `_shutdown_backend`. Behaviour change.
- `src/extensions/sandbox/daytona.rs`, `modal/sandbox.rs`: `shutdown` swallows provider errors (Python: `except Exception: pass`).
  Behaviour change: `delete`/`shutdown` no longer returns an error for a rejected stop or terminate.

## Known differences closed

- tar `--no-wildcards`: Python passes it only in Vercel (`vercel/sandbox.py:1390`) and the fingerprint helper; Rust does the
  same for both. Modal, E2B, Daytona, Blaxel and the shared helper pass no such flag in Python either. No difference remains.
- live-session `concurrency_limits`: closed as above.

## Remaining python-only, one reason per group

- test_session.py (13): `await_mutation`/post-commit cancellation (4, asyncio task cancel; a dropped Rust future cannot be re-awaited);
  SQLite worker-thread connection tracking, close, reap, quarantine, rollback eviction (7) and `closed_rejects_empty_add_items`
  (`SqliteSession` has no `close()`: one connection closes on drop); file lock across instances (1, SQLite file locking covers it, see
  `two_sessions_on_one_file_write_independently`). Negative limits (-1) are unrepresentable: `limit` is `Option<usize>`.
- test_session_helpers.py (2): cancellation of a pending mutation (asyncio); `test_independently_cancelled_operation_propagates`.
- Cloudflare (9), Daytona (6), Modal (41), Blaxel, E2B: SDK object duck typing, import guards, pydantic defaults, asyncio shielding,
  vendor handle internals, `caplog` assertions. E2B kill failure: Python logs then raises `CancelledError`; Rust returns the error
  (both surface the failure).
- Not touched (other owners or runner internals): compaction deferral tests that drive private `save_result_to_session`,
  context-wrapper sessions (no run-context wrapper in Rust `Session`), sqlalchemy twins.

## Gates

- Binaries: all `py_sandbox*`, `py_core6*`, `py_memory*`, `sessions_tests`, `sandbox*`, `py_core7_c*`, lib (653) run; see summary.
- clippy: `--all-features --lib` plus my test binaries and `--all-targets` default features clean. All-features `--all-targets`
  fails only on `tests/py_core7_a_entries_swaps.rs` (another agent, unused `symlink`).
- rustfmt on my files.
