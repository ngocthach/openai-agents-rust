# Python->Rust test parity, round 6: Cloudflare, E2B, Daytona, rclone sandbox providers

Slice: `tests/extensions/sandbox/test_cloudflare.py`, `test_e2b.py`, `test_daytona.py`, `test_rclone.py`.

## Counting method

An `ast`-style regex walk of each file (script: scratchpad `gen_cfdaytona_csv.py`) counts every
`def test_*` once and multiplies by the length of its `@pytest.mark.parametrize` list. That gives
263 expanded cases, which is the unit used by the ledger (96 / 80 / 72 / 15). The task brief's
80 / 76 / 63 / 6 are the un-expanded function counts (225). Every row below adds up to the real
case count; each case is in exactly one bucket.

| file | cases | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| test_cloudflare.py | 96 | 84 | 1 | 11 | 0 |
| test_e2b.py | 80 | 49 | 8 | 23 | 0 |
| test_daytona.py | 72 | 62 | 4 | 6 | 0 |
| test_rclone.py | 15 | 15 | 0 | 0 | 0 |
| total | 263 | 210 | 13 | 40 | 0 |

Ledger before this round: 35+35+34+15 = 119 ported, 28 python-only, 116 not ported/unclassified.

"Ported" includes cases already ported in `py_sandbox_{cloudflare,e2b,daytona,rclone_mounts}.rs`
that I re-read and confirmed against the Python assertions (the old report's counts were
approximate). Where the Rust assertion is deliberately weaker than the Python one it is called out
below.

## New tests

- `tests/py_core6_cfdaytona_cloudflare.rs`: 50 tests (the SSE decoder cases live in
  `tests/py_core6_cfdaytona_cloudflare_sse.rs`, 6 tests, to keep each file under the size guide). Mount config (S3/R2/GCS/session token/native
  GCS auth/registry round-trip), client timeouts, protected state and resume refusal, resume order
  (`running`, `unmount`, `hydrate`, `mount`), fingerprint fast path, terminal reapply failure,
  persist/hydrate unmount-remount order, secret redaction, streamed payload decode/truncation,
  persist excludes and skip paths, 503 retry limit, unsafe tar, full PTY set over a hand-written
  websocket worker (including the 64-session prune and a ready timeout in virtual time).
- `tests/py_core6_cfdaytona_e2b.rs`: 24 tests. Resume lifecycle, persist excludes, native snapshot
  timeout and tar fallback, exec error classification by HTTP status, PTY (tty, non-tty, exit
  wake-up, running session cleanup, stop, startup failures, not-found).
- `tests/py_core6_cfdaytona_daytona.rs`: 17 tests. Status table, exec budgets (session create and
  bounded cleanup), resume state, persist excludes and stopped-sandbox error, mount strategy
  lifecycle (preflights, credential refusal, deactivate/teardown), logged commands, PTY websocket
  (start/write/exit, stop, startup failure, 404/429 classification).
- rclone: all 15 cases already had 1:1 Rust tests (`rclone_arch_maps_upstream_linux_archives`
  loops all 10 machines); verified, nothing added.

## Rust bugs fixed (all in my area, no `src/sandbox/**` edits)

Cloudflare (`src/extensions/sandbox/cloudflare/sandbox.rs`, `mounts.rs`):
1. `read_file` turned a response body that failed part-way (timeout, reset) into an empty file
   (`bytes().await.unwrap_or_default()`). It is now an `ExecTransportError`.
2. A persist ignored paths registered with `register_persist_workspace_skip_path` (the session had
   no skip registry). The session now owns one and the `excludes` query includes it.
3. A resume of a sandbox the worker still serves re-attached buckets without detaching them, so a
   hydrate would have written through the live mount. `start` now detaches each non-native-detach
   mount first (order `unmount`, `hydrate`, `mount`), and skips the detach when the workspace
   fingerprint still matches the snapshot. A restore that fails for a reason other than a mount
   (archive, hydrate) puts the detached buckets back and leaves the session usable; a failing
   re-attach is the terminal case in item 4.
4. A mount that fails while a resumed workspace is re-attached left a half-mounted sandbox
   reusable. It is now deleted (`DELETE /v1/sandbox/{id}`), the session refuses further work with
   "sandbox session is unavailable after an ambiguous mount transition", and a delete that itself
   fails surfaces as `WorkspaceStopError` with `backend`, `reason=terminal_delete_failed` and
   `http_status`. The reply body is never read.
5. `CloudflareBucketMountStrategy` did not report `carries_credentials`, so a state written from a
   manifest with bucket credentials still carried the sandbox id. It now drops it like every other
   credentialed mount.
6. `system_state_preserved_on_start` was never true on resume, so accounts were re-provisioned over
   a surviving sandbox. It is true when the worker still serves the sandbox.
7. Cloudflare sessions never recorded or consumed the snapshot fingerprint, so the resume fast path
   never applied. `record_snapshot_fingerprint`, `resume_snapshot_fingerprint` and the state field
   are wired (fast path only for a preserved sandbox).

E2B (`src/extensions/sandbox/e2b/mod.rs`):
8. Persist ignored registered skip paths (no registry); the tar command now excludes them.
9. A native snapshot cannot omit a path, so a registered skip path now falls back to the tar
   archive (Python behaviour); previously the path leaked into the snapshot.
10. The native snapshot call had no deadline; it is now bounded by `timeouts.snapshot_tar` and fails
    as `SnapshotPersistError` (`native_snapshot_failed`, not retryable).
11. `connect` on resume sent `{}`; it now carries the configured `timeout`, so reconnecting extends
    the sandbox lifetime like Python's `connect(timeout=...)`.
12. `system_state_preserved_on_start` was never true on resume; it is true when the sandbox was
    reattached.

Daytona (`src/extensions/sandbox/daytona.rs`):
13. Persist ignored registered skip paths; now excluded.
14. A stopped sandbox during persist produced an anonymous retryable-looking transport error. It is
    now `SnapshotPersistError`, not retryable, with `backend=daytona`, `reason=sandbox_not_running`.
15. A PTY websocket upgrade answered with an HTTP error lost its status. It now reports
    `http_status` and the documented retryability (404 false, 429 true).
16. `system_state_preserved_on_start` was never true on resume; true when the sandbox was found.

## Breaking / behaviour changes

- Cloudflare `resume` now refuses (`MountConfigInvalid`, "protected bucket configuration cannot be
  resumed") any state or manifest that holds bucket credentials, before asking the worker anything.
  This is Python's rule (`CloudflareSandboxClient.resume`); Rust previously resumed such sandboxes.
- Cloudflare `shutdown` failures now carry `backend` and `http_status` context.
- E2B native-snapshot persist with registered skip paths returns a tar archive, not a snapshot
  reference.
- E2B `persist_workspace` native snapshot failure code is `SnapshotPersistError` (it was a transport
  error). Rust-wide `Snapshot*` codes are kept in place of Python's `WorkspaceArchive*` (known,
  documented deviation).

- Cloudflare, E2B and Daytona now run the in-sandbox resolver (extra `exec` round trips) before
  every read, write, mkdir, rm, ls and mv; a sandbox that cannot run the helper fails those calls.
  Python does the same.

## Deferred / cross-area findings (not edited)

- Nothing is blocked on `src/sandbox/**`. One observation for its owner: core `mkdir` (default)
  does not use the resolver even when `resolves_paths_remotely()` is true (only `ls`, `rm`, `rm_as`
  and `mv_via_exec` do); Cloudflare overrides `mkdir` itself, but another backend that turns the hook
  on would need the same, or a one-line change in `session.rs::mkdir`.
- `src/sandbox/session.rs::with_ephemeral_mounts_removed` re-attaches every detached mount after a
  failing operation. A caller must therefore keep the mount reapply (`apply_ephemeral`) outside the
  detach region if a failing reapply must not trigger a second attach; Cloudflare's resume does.

## Not ported (0)

The 12 cases first left open are closed in the second pass:

- Workspace escape (10: Cloudflare 3, E2B 3, Daytona 4): `read_file`, `read_bounded`, `write_file`,
  `mkdir` and, through `resolves_paths_remotely() == true`, the default `rm`/`ls`/`mv` of all three
  providers (and the providers' own `rm`/`ls`/`mv` overrides in E2B and Daytona) now run the shared
  `SandboxSession::validate_remote_path_access` first. A symlink leaving the root is
  `InvalidManifestPath`; one landing on a read-only grant is `WorkspaceArchiveWriteError`; neither
  reaches a file request. Cloudflare gained a `mkdir` that resolves first (the core default only
  normalized). Daytona's `mkdir_as` checks the requested user's access (core
  `check_mkdir_access`) and then uses the toolbox files API. Tests: `py_core6_cfdaytona_paths.rs`
  (rm/ls/mv escape and ordinary-path success, read, bounded read, write and mkdir for all three
  providers, plus `daytona_mkdir_as_user_checks_permissions_then_uses_the_files_api`).
- `e2b start_installs_runtime_helpers` is ported as the lazy install: Rust installs the resolver at
  the first path check, not in `start`; the test asserts probe, install, resolve in that order.
- Resume fingerprint gate (2): E2B and Daytona now record and consume the snapshot fingerprint
  exactly like Cloudflare (`record_snapshot_fingerprint`, `resume_snapshot_fingerprint`, the state
  field), only for a reattached sandbox. Tests: restore on mismatch, skip on match, state round
  trip (`py_core6_cfdaytona_paths.rs`).

Test mocks that play the sandbox now answer the resolver: `tests/py_sandbox_support/mod.rs`
gained `Resolver`, `resolver_reply`, `resolver_result`, `start_resolving`, `start_resolving_links`
(a resolver that follows registered symlinks) and `command_argv`; the provider test files use them,
and `py_sandbox_b_bounded_read{,_providers}.rs`, `sandbox_e2b_daytona_vercel_tests.rs` and
`sandbox_modal_cloudflare_blaxel_tests.rs` answer the resolver in their own servers. The one
assertion that indexed process starts (`e2b_start_prepares_workspace_root_for_command_cwd`)
filters out the resolver's own probe.

## Python-only (40), one reason per group

- Re-export / import-path / SDK class resolution (E2B 4, Daytona 2).
- pydantic defaults and positional dataclass construction (E2B 2).
- `caplog` / `DONT_LOG_TOOL_DATA` log assertions and swallow-and-log shutdown semantics: Rust
  `shutdown` returns the error instead of logging it (Cloudflare 4, E2B 4 kill-failure cases).
  Adapted tests assert the security property instead (no provider body or secret in the error).
- asyncio task cancellation and settle shielding (Cloudflare 3, E2B 1).
- Python `workspace_root_ready` readiness bookkeeping and the `test -d` probe (Cloudflare 2 rows of
  the preserved-state parametrization, E2B 6): Rust `start` always creates the root through the
  filesystem API and `exec` always uses the manifest root.
- Vendor SDK waiter, handle and timeout-class internals (E2B 7, Daytona 4): `wait`/`kill` handle
  fakes, pending-callback appends, httpcore exception module names, indeterminate waits.
- Private hooks (`_runtime_helpers`, `_before_shutdown`) (Cloudflare 2).

## Covered by existing tests (13)

- Core mount transitions (E2B 3, Daytona 2): `py_sandbox_b_mount_lifecycle.rs`
  (`mounts_are_restored_in_reverse_order`, `a_failed_detach_restores_the_detached_mounts_then_shuts_down`,
  `a_restore_error_wins_and_keeps_the_operation_error`) with the providers' mount strategies
  reporting `supports_native_snapshot_detach() == false`.
- Resume clearing of nested mounts (E2B 1, Daytona 1):
  `py_sandbox_a_runtime_2.rs::clearing_the_workspace_on_resume_preserves_a_nested_mount`.
- PTY pruning (E2B 1): `py_sandbox_b_pty.rs::pruning_prefers_exited_sessions`.
- E2B `exec_uses_manifest_root...` (1): `py_sandbox_e2b.rs::e2b_exec_uses_the_manifest_root_and_session_environment`;
  `shutdown_logs_pause_failure_and_falls_back_to_kill` x2:
  `e2b_shutdown_falls_back_to_kill_when_the_pause_fails` (log half is python-only).
- Daytona `has_command_not_found` (1): the `pkg_install_*` tests in `py_sandbox_daytona.rs`.
- Cloudflare `ambiguous_mount_terminal_delete_failure_is_observable` (1):
  `resume_reapply_surfaces_terminal_delete_failure`.

## Known weaknesses of ported cases

- Cloudflare `resume_sets_preserved_state_from_running`: only the preserved flag is asserted; the
  two `workspace_root_ready=true` rows are python-only.
- E2B native-snapshot hydrate (`py_sandbox_e2b.rs`) asserts the template and the released sandbox
  but not the lifecycle/MCP body Python checks; it goes through the same `create_sandbox` path the
  create tests already pin.
- Daytona status table: 404 on a file read maps to `WorkspaceReadNotFound` (no retryability), so
  that row asserts the error code instead of `retryable == false`.
- E2B PTY tests send the command as the process (`Start` with `pty` and `stdin`) instead of typing
  it into a shell as Python's `pty.create` + `send_stdin` does; the wire differs, the observable
  output, session id and exit code do not.

- E2B `pty_start_maps_httpcore_read_timeout_to_timeout_error` is python-only while its exec twin is
  ported: Rust's exec has a deadline that maps a stalled stream to `ExecTimeout`, whereas opening a
  PTY has no read timeout (a stalled stream just yields a live session), so there is nothing to map.
- E2B `pty_start_cleans_up_partially_created_session_on_failure` is ported as "a failed startup
  registers no session". Python creates the pty, then fails in `send_stdin`, and checks the handle
  was killed; in Rust the command is the process, so there is no half-created session to kill.
- Daytona's two budget tests delay the mock reply with an async sleep (a blocked worker thread made
  the first version flaky).

## Files

- Source: `src/extensions/sandbox/cloudflare/sandbox.rs`, `src/extensions/sandbox/cloudflare/mounts.rs`,
  `src/extensions/sandbox/e2b/mod.rs`, `src/extensions/sandbox/daytona.rs`.
- Tests: `tests/py_core6_cfdaytona_cloudflare.rs`, `tests/py_core6_cfdaytona_cloudflare_sse.rs`,
  `tests/py_core6_cfdaytona_e2b.rs`, `tests/py_core6_cfdaytona_daytona.rs`,
  `tests/py_core6_cfdaytona_paths.rs` (108 new tests).
- Shared-file edits (`src/sandbox/**`, `lib.rs`, `Cargo.toml`): none.
- CSV: `plans/reports/py-test-parity-core6-cfdaytona.csv`.

## Gates

- `cargo test --all-features` over every binary that touches these providers: py_core6_cfdaytona_*
  (cloudflare 50, sse 6, e2b 24, daytona 17, paths 11), py_sandbox_cloudflare 24, py_sandbox_e2b 43,
  py_sandbox_daytona 41, py_sandbox_rclone_mounts 8, py_sandbox_b_bounded_read 9,
  py_sandbox_b_bounded_read_providers 11, py_sandbox_b_compat 8, py_sandbox_b_compat_providers 19,
  py_sandbox_b_credentials 6, sandbox_e2b_daytona_vercel_tests 41,
  sandbox_modal_cloudflare_blaxel_tests 37, sandbox_tests 65: all pass. The other `py_sandbox_b_*`
  binaries use core fake sessions and none of these providers, so they cannot be affected. One
  Modal PTY test failed once while the machine was loaded and passed on rerun (not touched code).
- `cargo clippy --all-features` on the lib and the touched test binaries: no warning in any file I
  touched. `-D warnings` on the lib fails in other agents' files
  (`src/run_internal/local_tools.rs`, `src/run_internal/tools.rs`, `src/strict_schema.rs`,
  `src/realtime/debug.rs`), so the test targets could only be linted without `-D warnings`.
- `rustfmt --edition 2024` on every file I edited.
- The cargo output dir was cleaned and removed at the end.
