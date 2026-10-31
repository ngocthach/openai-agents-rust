# Python→Rust test parity — sandbox-a / docker

Subagent `docker`. Python files: `tests/sandbox/test_docker.py`, `test_docker_removal.py`,
`test_docker_removal_worker.py`, `test_docker_removal_client.py`.
Counting: `grep -c '^\s*\(async \)\?def test_'` per file (function counts; parametrize expansions not
multiplied). Rust tests: `tests/py_sandbox_a_docker_{1..6}.rs` (19 + 35 + 10 + 8 + 14 + 3 = 89 tests, grouped from the ported Python tests, plus 6 tests for runtime-helper requests that have no single Python test).

## 1. Per-file table

| file | #tests | ported | covered | python-only | open (not ported) | failures found |
| --- | --- | --- | --- | --- | --- | --- |
| test_docker.py | 108 | 84 | 0 | 24 | 0 | 14 Rust bugs, section 2 |
| test_docker_removal.py | 35 | 0 | 0 | 35 | 0 | – |
| test_docker_removal_worker.py | 23 | 0 | 0 | 23 | 0 | – |
| test_docker_removal_client.py | 4 | 0 | 0 | 4 | 0 | – |

84 + 24 + 0 = 108. The open bucket is empty. Several ported tests port one Python test group as one
Rust test (e.g. the five `prunes_*` persist tests are one `persist_excludes_ephemeral_entries_and_every_mount_path`).

### test_docker.py — python-only (24), one reason per group
- Deferred-cleanup asyncio tasks (`_cleanup_tasks`, `_wait_for_cleanup_tasks`, `_after_stop`): 857, 883, 909, 950.
- docker-py exec socket / length-framed stdin / spool (`_start_exec_socket`, `_stream_into_exec`,
  `_measure_stream`; `head -c` framing exists because docker-py cannot half-close a TLS stdin; the CLI
  transport does not frame): 992, 1076, 1115, 1155, 1204, 1223. The transport properties that remain
  (exit code, separate streams, stdin EOF, payload larger than pipe buffers) are asserted by
  `src/sandbox/sandboxes/docker.rs::tests::{a_large_stdin_payload_survives_a_child_that_floods_stdout_first,
  stdout_and_stderr_stay_separate_and_the_exit_code_survives, stdin_reaches_eof_so_a_draining_child_can_finish}`.
- `READ_PATH_PROBE_V3` probe internals (Rust reads with `cat`, no probe): 1556, 1570.
- docker-py `images.pull` repo/tag parsing: 1892.
- Pydantic `RunState` JSON schema of the Python state object: 2008.
- docker-py container lookup (`containers.get`) failure paths (no lookup step on the CLI path): 2971.
- Python `create` also starts the container, so start/post-start failure cleanup belongs to `create`;
  Rust `create` only creates (start is a separate call, cleaned by `shutdown`, tested in
  `delete_removes_the_container_and_only_the_volumes_it_generated`): 3030, 3112.
- asyncio cancellation, and the `_wrap_session` wrapper: 3406, 3484.
- Python `Mount` subclass trust list (`_mount_class_is_trusted`): 3635.
- docker-py `exec_run(workdir=...)` omission until `_workspace_root_ready`, and the `test -d` probe
  that recovers it (Rust always passes `-w <root>`; `docker create -w` makes the directory): 4017, 4785.
- docker-py PTY socket internals (closed-socket errors, blocking `exec_start` timeout): 4971, 5081.

## 2. Rust bugs fixed (all in `src/sandbox/sandboxes/docker.rs`)

1. **No injectable daemon seam.** Added `DockerRunner` (+ `DockerCliRunner` default, `with_runner` on client
   and session, `program()` for the PTY path). Every `docker` call now goes through it.
2. **`resolve_exposed_port` always refused** (trait default). Now reads `NetworkSettings.Ports` from
   `docker container inspect`, `HostIp` defaults to `127.0.0.1`, same `detail` codes as Python.
3. **`shutdown` stopped at the first failure.** It now removes the container, then attempts every
   volume, and reports the first failure (Python `delete` semantics). A failed `docker rm` no longer
   leaks the named volumes.
4. **`read_file` error code** for an unreadable path was `WorkspaceWriteTypeError`; Python raises
   `WorkspaceArchiveReadError`.
5. **PTY ignored the requested user.** `pty_argv` never passed `-u`. Fixed.
6. **`sudo` wrapper inside `docker exec -u`.** `exec_argv`/`pty_argv` used `request.resolved_argv()`,
   which prepends `sudo -u <user> --`, so Docker ran `sudo` as the user. Python uses the native
   `docker exec -u` and no `sudo` (`test_docker_exec_uses_native_docker_user_without_sudo`). Fixed.
7. **Resume replacement ignored the persisted identity.** A replacement container was created from the
   client's current options, dropping the persisted labels, published ports and network mode. Now
   rebuilt from the state (Python `test_docker_resume_forwards_persisted_labels_when_recreating_container`).
8. **Mount-path identity used a non-lexical `..`** (`normalize_segments` keeps `..`), so a root spelled
   `/mnt/sibling/..` did not collide with a grant at `/mnt/shared-data`. Uses `normpath` now.
9. **No protected-mount redaction.** Added `PROTECTED_MOUNT_ERROR_MESSAGE`,
   `manifest_has_protected_mounts`, `redact_for_protected_mounts`; applied to create, resume
   replacement, delete and persist. Python message is reproduced verbatim; code and op survive,
   message/context/cause do not. "Protected" = a strategy that exposes a credential in the sandbox, or
   a Docker volume given a secret-looking `volume-opt` (my definition; Python's is broader, see 5).
10. **Failed create leaked named volumes.** A create under a freshly generated session identity now
    removes the volumes it generated; a replacement that reuses a persisted identity does not (it
    would delete the paused session's storage).
11. **File read/write followed symlinks out of the workspace.** `read_file`/`write_file` only checked the
    path lexically. They now resolve in the container through the existing
    `RESOLVE_WORKSPACE_PATH_HELPER` (installed once per session), refusing an escape (exit 111 →
    `InvalidManifestPath`, "must not escape root") and a write under a read-only grant (114 →
    `WorkspaceArchiveWriteError` with `path`/`reason`/`grant_path`/`resolved_path`). `mkdir`/`rm`/`ls`/`mv`
    still use the lexical check (section 5).

12. **Leaf symlinks resolved to their target.** The resolver returned the realpath, so `rm` of a link
    would remove the target. It now returns the requested path (Python `_validate_remote_path_access`).
    `ls`, `mkdir`, `rm` (access mode 2 for recursive) and `mv` all go through the resolver now.
13. **Mount at the workspace root was traversed by persist.** The capture is now `tar --no-recursion`
    when a mount sits at the root.
14. **Protected-mount definition widened** to Python's `_manifest_has_configured_mount_authority`
    shape: a strategy that exposes a credential, a strategy the SDK does not ship (may hide one), or a
    Docker volume handed a credential-looking `volume-opt`.
15. Requests from the runtime helper: `persist_skip_registry` (runtime-registered skip paths are kept and
    excluded from the archive), `create_with_snapshot` (honors the snapshot the runner passes),
    `system_state_preserved_on_start` (true only for a reattached container), and `sandbox.stop` /
    `sandbox.shutdown` spans on the session with `delete` performing stop then shutdown (a failed
    stop is logged and shutdown still runs).

## 3. Cross-area bugs deferred (files I do not own)

- `src/sandbox/mount_entries.rs` S3 Files error text is "`s3_files` cannot be attached with the `rclone`
  Docker volume driver" where Python says "invalid Docker volume driver", and the refusal happens at
  `Mount::validate` (which `create` runs before any `docker` call) rather than at construction. Behavior
  is equivalent; no edit made.
- `src/sandbox/state.rs`: persisted-state redaction keys off `credential_authority()`, which is `None`
  for a credentialed Docker-volume mount (by design: no acknowledgement needed). Python redacts
  provider identity for those too. The docker resume tests use a MountScoped strategy for that reason.
  Minimal fix if parity is wanted: add a `carries_credentials()` notion beside `credential_authority`.

## 4. `#[ignore]`d tests

None. No test needs a live daemon: every Docker call goes through `DockerRunner`, and the PTY tests
spawn a local shell script (passed by path through `DockerRunner::program`, never via `PATH`).

## 5. Previously open items, now closed

1333 -> `py_sandbox_a_docker_5::persist_does_not_traverse_a_mount_at_the_workspace_root` (bug 13).
1400 -> `persist_excludes_a_runtime_registered_skip_path`. 1585/1655/1686/1748 -> docker_2 symlink tests.
1727 -> `rm_unlinks_a_safe_leaf_symlink_instead_of_its_target`. 3224/4333/4427/4286 -> docker_5 resume tests.
3546 -> `clearing_the_workspace_on_resume_preserves_a_nested_volume_mount`. 3658 ->
`an_s3_files_mount_cannot_use_a_docker_volume_driver`. The two `//` grant-target cases -> docker_1.

Limits worth knowing: the in-container resolver is exercised against a fake helper (it answers with the
candidate, or exit 111/114); the real shell script is not run here. The PTY tests spawn a local script and
take up to 3 s per test.

## 6. Shared-file edits needed from the controller

None.

## 7. Verification

- `c.sh test --all-features --test py_sandbox_a_docker_{1..6}`: all pass.
- Regression: `sandbox_core_parity_tests`, `sandbox_docker_exec_transport`, and `docker.rs` unit tests pass.
- `rustfmt --edition 2024` run on my files. Clippy (`--all-features --all-targets -D warnings`) reports
  nothing in my files; it fails in other agents' test files (`py_mcp_http_transport`,
  `py_sandbox_a_unix_local_1/2`, `py_tracing_core`).


`system_state_preserved_on_start` mirrors Python's `_set_start_state_preserved(reused_existing_container)`
(docker.py:1833). Tests: `system_state_is_preserved_only_when_the_container_was_reused` and
`resume_provisions_accounts_only_into_a_replacement_container` (docker_5): a reused container runs zero
`useradd`, a replacement runs one. The `sandbox.stop`/`sandbox.shutdown` sites in `docker.rs` use
`sandbox_span_result` and compile; `py_sandbox_a_docker_6` passes against them.
