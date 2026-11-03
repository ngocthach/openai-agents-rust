# Python test parity, core round 6: Blaxel

Slice: `tests/extensions/sandbox/test_blaxel.py` and `test_blaxel_mounts.py`.

## Counting

`test_blaxel.py` has 227 `def test_` functions. Parametrized rows expand it to 251 collected
cases (percent-encoding 6, PTY output decoding 4, write_stdin 2, unsafe-mount start 2, retry-code
table 10, credential registration 2, drive detach log 2, unmount log policy 4; 227 + 24 = 251,
which matches the ledger). `test_blaxel_mounts.py` has 2. Every parametrized row is counted.
Counts were taken by reading each test body and the Rust test that asserts the same input and
output; nothing was classified from earlier reports.

| file | cases | ported | covered | python-only | not ported |
| --- | --- | --- | --- | --- | --- |
| tests/extensions/sandbox/test_blaxel.py | 251 | 188 | 23 | 40 | 0 |
| tests/extensions/sandbox/test_blaxel_mounts.py | 2 | 0 | 2 | 0 | 0 |
| total | 253 | 188 | 25 | 40 | 0 |

Per class (ported / covered / python-only):

| class | cases | P | C | O |
| --- | --- | --- | --- | --- |
| module re-export test | 1 | 0 | 0 | 1 |
| TestBlaxelSandboxSession | 45 | 31 | 10 | 4 |
| TestBlaxelSandboxClient | 8 | 4 | 3 | 1 |
| TestHelpers | 23 | 23 | 0 | 0 |
| TestImportGuards | 4 | 0 | 0 | 4 |
| TestValidateTarBytes | 6 | 6 | 0 | 0 |
| TestWorkspacePersistence | 4 | 3 | 0 | 1 |
| TestWorkspaceHydration | 7 | 5 | 0 | 2 |
| TestBlaxelSandboxClientExtra | 8 | 4 | 2 | 2 |
| TestBlaxelTimeouts | 4 | 2 | 0 | 2 |
| TestBlaxelSandboxSessionState | 2 | 1 | 0 | 1 |
| TestBlaxelSandboxClientOptions | 2 | 1 | 0 | 1 |
| TestTarExcludeArgs | 2 | 1 | 1 | 0 |
| TestStartLifecycle | 3 | 3 | 0 | 0 |
| TestPtyExec | 24 | 18 | 3 | 3 |
| TestPersistWithMounts | 3 | 3 | 0 | 0 |
| TestImportBlaxelSdkActual | 2 | 0 | 0 | 2 |
| TestValidateTarBytesExtra | 2 | 2 | 0 | 0 |
| TestTarExcludeArgsWithSkipPaths | 2 | 2 | 0 | 0 |
| TestTerminatePtyEntryErrors | 2 | 0 | 0 | 2 |
| TestCollectPtyOutputEdgeCases | 2 | 1 | 1 | 0 |
| TestActualImportSuccess | 2 | 0 | 0 | 2 |
| TestCleanupPaths | 2 | 2 | 0 | 0 |
| TestClientBranchCoverage | 3 | 3 | 0 | 0 |
| TestFinalCoverageGaps | 10 | 5 | 3 | 2 |
| TestMountsModule | 39 | 37 | 0 | 2 |
| TestSdkExceptionMapping | 17 | 15 | 0 | 2 |
| TestCoerceExecTimeout | 4 | 3 | 0 | 1 |
| TestDriveMounts | 10 | 7 | 0 | 3 |
| TestUnmountBucketLogging | 4 | 4 | 0 | 0 |
| TestFakeFs | 2 | 0 | 0 | 2 |
| TestShutdownLogging | 2 | 2 | 0 | 0 |
| total | 251 | 188 | 23 | 40 |

## Rust tests added or changed

- `tests/py_core6_blaxel_session.rs` (52): exec and structured error mapping, symlink and read-only
  grant guards, files, lifecycle, preview ports, client create/resume/delete, terminal URL,
  logging.
- `tests/py_core6_blaxel_archive.rs` (15): tar validation, persist, hydrate, skip paths, persist
  around mounts.
- `tests/py_core6_blaxel_pty.rs` (17): a real WebSocket mock for the terminal.
- `tests/py_core6_blaxel_mounts.rs` (33): bucket and drive mounts, unmount fallbacks, log policy.
- `tests/py_core6_blaxel_support/mod.rs`: the shared mock control plane and sandbox API, including
  an emulation of the in-sandbox `resolve-workspace-path` helper (the same role as Python's
  `_fake_helper_exec_result`).
- Edited existing tests only where the new behaviour changes their premise:
  `tests/py_sandbox_blaxel.rs` (the delete failure is now suppressed, and the mock answers the
  path resolver) and `tests/py_sandbox_b_bounded_read_providers.rs` (its Blaxel mock answers the
  path resolver).

### Covered (existing tests, read and verified)

- `py_sandbox_blaxel.rs`: `blaxel_exec_preserves_split_stdout_and_stderr`,
  `blaxel_exec_reports_a_nonzero_exit_code`, `blaxel_mkdir_uses_the_filesystem_api`,
  `blaxel_read_404_is_a_not_found`, `blaxel_running_probes_the_filesystem_api`,
  `blaxel_shutdown_deletes_the_sandbox_once`, `blaxel_shutdown_pause_on_exit_keeps_the_sandbox`,
  `blaxel_create_sends_the_full_config`, `blaxel_resume_reconnects_or_creates`,
  `blaxel_resume_rejects_a_foreign_backend_state`.
- `sandbox_modal_cloudflare_blaxel_tests.rs::blaxel_persists_and_hydrates_through_a_tar_file`
  (default tar excludes).
- `py_sandbox_b_pty.rs`: `pruning_prefers_exited_sessions`, `collect_drains_chunks_added_when_done`,
  `collect_returns_output_queued_before_the_window_ends`.
- `py_sandbox_rclone_mounts.rs`: `s3_mount_options_are_shell_quoted`,
  `gcs_mount_prefix_is_shell_quoted` (both `test_blaxel_mounts.py` cases).

### Python-only (one reason per group)

- Import guards, SDK/aiohttp import checks, package re-export: Python optional-dependency
  machinery (9: TestImportGuards 4, TestImportBlaxelSdkActual 2, TestActualImportSuccess 2,
  re-export 1), plus `test_import_sandbox_api_error*` (2).
- SDK object duck typing: `read` returning `str`, error-string / `.status` / args-dict sniffing of
  SDK exceptions that has no HTTP equivalent (`read_not_found_via_error_string`),
  `write` / `hydrate` payload type checks (`write_invalid_payload_type`, `hydrate_invalid_payload_type`,
  `hydrate_str_payload_encoded`, `persist_read_returns_str`), missing `drives` API (2),
  `sandbox_url` optional on the state model (`supports_pty_without_url`), non-Blaxel `delete()`
  `TypeError`, `close()` and `async with` on the client (3).
- Pydantic and dataclass behaviour: frozen models, `ge=1` validation, state defaults (4).
- asyncio cancellation shielding of awaited cleanup (2) and PTY entry internals that Rust has no
  counterpart for (`terminate_entry_all_none`, `terminate_ws_close_error`,
  `terminate_reader_already_done`, `terminate_pty_outer_exception`,
  `prune_returns_none_when_no_pid`).
- Negative `float` timeout (`Duration` is unsigned), fake-test-double tests (`TestFakeFs`, 2), and
  `build_docker_volume_driver_config` (no Rust hook for these strategies, 2).
- `test_assert_blaxel_session_wrong_type`: the Rust bucket strategy is session-agnostic (it only
  runs shell commands); the drive strategy's own backend check is tested.

## Rust bugs fixed (all under `src/extensions/sandbox/blaxel/`)

1. `sandbox.rs` `shutdown`: a failing sandbox delete is logged (tool-data policy) and suppressed, as
   Python's `shutdown` does. It used to raise `WorkspaceStopError`, so `client.delete` failed too.
2. `sandbox.rs` `read_file`, `read_bounded_unchecked`, `write_file`, `mkdir`: now run
   `validate_remote_path_access` (the in-sandbox resolver). A workspace symlink to an ungranted path
   was readable, and a symlink to a read-only extra path grant was writable, because only the
   lexical check ran. Now the first is `InvalidManifestPath` with `rel`, `reason`, `resolved_path`
   context and the second `WorkspaceArchiveWriteError` with `path`, `reason`, `grant_path`,
   `resolved_path`, matching Python.
3. Error taxonomy aligned with Python: read failures are `WorkspaceArchiveReadError` (transient
   statuses retryable), writes, mkdir (`reason=mkdir_failed`) and removes are
   `WorkspaceArchiveWriteError`. `WorkspaceWriteTypeError` was being used for I/O failures.
4. `persist_workspace`: a failing `tar` is `WorkspaceArchiveReadError` with `reason=tar_failed`,
   `output`, `retryable=false`; exec and read failures are wrapped as archive read errors; the
   scratch file is always removed.
5. `hydrate_workspace`: unsafe or invalid archive is `WorkspaceArchiveWriteError`
   `reason=unsafe_or_invalid_tar` before anything is uploaded; extract failure is `reason=
   tar_extract_failed`; upload and mkdir failures are archive write errors; cleanup is best effort.
6. `exec`: a non-success status is `HTTP <status>: <message>` with `backend`, `http_status`,
   `provider_error`, `provider_error_code` context and `retryable` from the structured body or
   Blaxel's documented code table (transient statuses retryable); 408 and 504 are `ExecTimeout`.
   A timeout while reading the response body used to surface as "malformed body"
   (`ExecTransportError`); it is now `ExecTimeout`. `ExecTimeout` carries `timeout_s`. A zero
   timeout is coerced to 1 ms, and the body timeout is whole seconds rounded up, as Python sends.
7. `start`: validates mount credential boundaries before any sandbox request (an unacknowledged
   credential mount used to reach the sandbox first), and a failing up-front root mkdir is only
   logged.
8. Skip paths: `BlaxelSandboxSession` now owns a persist-skip registry. Previously
   `register_persist_workspace_skip_path` was a silent no-op, so a registered path (including a mount
   credential file) was still archived.
9. `mounts.rs` bucket activation: the credential file is `.openai-agents-s3fs-passwd-*` or
   `.openai-agents-gcs-creds-*`; its path is registered as a skip path before it is written; a failing
   `chmod` aborts and removes the file; removal is an `exec rm -f` whose failure is an error that
   outranks the mount error ("failed to remove mount credential file"); it was ignored before, leaving
   credentials in the workspace. Mount failure text is `<tool> mount failed ...`.
10. `mounts.rs` unmount: logs follow Python (debug, debug, warning) and drop the mount path when
    tool-data logging is off (the single warning used to log the path unconditionally). Drive detach
    failures are logged, not raised.
11. `pty.rs`: the connect-timeout error carries `timeout_s`. `sandbox.rs` `running` logs a failed
    health check at debug.

## Cross-area notes

- No edits to `src/sandbox/**`.
- `src/extensions/sandbox/cloudflare/sandbox.rs:429` and `:487` (`mount_transition_terminal`,
  `ensure_usable` never used) and `src/mcp/util.rs:15`, `src/run_internal/session_write.rs:119`,
  `src/realtime/debug.rs:29` produce dead-code or unused warnings. They belong to other agents.

## Breaking changes

- `SandboxSession::shutdown` and `SandboxClient::delete` no longer return a failed sandbox delete.
- Blaxel error codes changed as in items 3 to 5: read failures were `ExecTransportError`, persist
  failures `SnapshotPersistError`, hydrate failures `SnapshotRestoreError`, filesystem write
  failures `WorkspaceWriteTypeError`. Credential cleanup failure is `MountConfigInvalid`.
- Drive `deactivate` no longer returns an unmount error.
- Each read, write and mkdir issues two extra `POST /process` calls (helper presence check, then the
  resolver), as Python does.
- Process-body `timeout` is whole seconds rounded up (was exact milliseconds).

## Follow-up: rm, ls and mv resolve paths in the sandbox

Python's base class runs `_validate_remote_path_access` for `rm`, `ls` and `mv`; the Rust Blaxel
session only checked lexically, so a workspace symlink leading outside the root could be listed,
removed or moved through. Fixed:

- `src/sandbox/session.rs` (shared, minimal): new hook `SandboxSession::resolves_paths_remotely()`
  (default `false`). When true, the default `rm`, `rm_as`, `ls` and `mv_via_exec` run
  `validate_remote_path_access` first; `rm` already did so when a read-only grant exists.
  `src/sandbox/events.rs` forwards the hook through the tracing wrapper.
- Why a hook and not an unconditional change: the resolver costs two extra `exec` round trips per
  call and the local/docker backends check through the real filesystem or scripted sessions that
  expect exactly one command, so an unconditional default would change every provider. Backends that
  do not opt in behave exactly as before; only Blaxel opts in.
- `blaxel/sandbox.rs`: `rm` (including the recursive-ancestor check), `ls` resolve first; the
  hook returns `true`, which also covers `mv`.
- Tests (`py_core6_blaxel_session.rs`): `rm_ls_and_mv_refuse_a_workspace_symlink_that_leaves_the_root`
  (rm and recursive rm, ls, mv with either end through the link: `InvalidManifestPath`, no
  filesystem or `mv` request issued), `rm_ls_and_mv_succeed_for_ordinary_paths_after_resolving`, and
  `other_backends_keep_the_unresolved_default` (a scripted session makes exactly one command for
  each of rm, ls and mv and reports the hook as false).

Python case totals are unchanged (253 cases: 188 ported, 25 covered, 40 python-only, 0 not ported).
