# Round 13, r13c: the Docker removal service

The earlier decision not to build the removal service (r10c, r8g10) is reopened because the owner asked for the
whole Python SDK. Python's `DockerRemovalService` (`docker_removal.py`, `_docker_removal_worker.py`) and its four
test files were read in full. A safe Rust equivalent is feasible, so it is built, opt-in, with the three test
files ported.

| file | python_only before | ported | covered | python-only after | reason for the rest |
|---|---|---|---|---|---|
| tests/sandbox/test_docker_removal.py | 64 | 62 | 0 | 2 | see "Remaining" |
| tests/sandbox/test_docker_removal_client.py | 14 | 14 | 0 | 0 | |
| tests/sandbox/test_docker_removal_worker.py | 58 | 56 | 0 | 2 | see "Remaining" |

Totals: 132 ported, 0 covered, 4 remain python-only (136 before). `_docker_removal_helpers.py` has no test cases;
its recording container, worker and service fixtures are re-created in the Rust tests.

## Decision: feasible, behind an explicit opt-in

- Feature `docker-removal` (never default; `docker-removal = ["sandbox", "rustix/process", "rustix/thread"]`), in
  the docs.rs feature list. Without it nothing changes in the crate.
- The library never escalates. `DockerRemovalService::new` fails closed unless the caller already runs as root on
  Linux, and the worker is a host executable the caller runs with that privilege, exactly Python's design. The
  module docs say so.
- The worker never goes through `docker exec` (Python's docstring forbids it too: "not binaries supplied by the
  container"). The only exec is the workspace bootstrap `mkdir -p` with the image's default user, as in Python.
  The task text suggested `docker exec`/mount namespace; the namespace entry (`setns`, `fchdir`, `chroot`) is what
  is ported.
- `#![forbid(unsafe_code)]` stays: everything uses safe `rustix` calls (`openat` with `O_PATH`, `statat`,
  `unlinkat`, `Dir`, `thread::move_into_link_name_space`, `process::chroot`/`fchdir`/`setrlimit`,
  `thread::set_thread_uid/gid/groups`).

## What was built

- `src/sandbox/sandboxes/docker_removal.rs`: `DockerRemovalService` (+ `DockerRemovalOptions`), the
  `RemovalWorker`/`WorkerLauncher`/`RemovalHost` seams, `ProcessLauncher`. Same protocol as Python: container
  eligibility (`_state`: Docker 26+, builtin seccomp, runc, no privileged/caps/pid/userns/runtime/SecurityOpt,
  read-only rprivate bind mounts only, complete id, `/proc/<pid>/cgroup` on this host, identity uid/gid maps),
  pause (a caller-owned pause is left alone, an incarnation change aborts and leaves it paused), `bind` with the
  backslash / host-mount-outside-workspace / distinct-roots checks, `inspect` -> live-root and `validate_recursive_remove`
  authorization -> `remove` as the requested user (`user`, else the container's configured user, else `0`),
  capacity slots rejected before pausing (an uncertain outcome keeps its slot), uncertain transport leaves the
  container paused, release only after confirmed container removal, `close` attempts every worker. Error reasons
  are Python's: `docker_removal_capacity`, `docker_removal_canonical_path`, `docker_removal_bound_root`,
  `docker_removal_failed` (+ `worker_reason`, `errno`).
- `src/sandbox/sandboxes/docker_removal_worker.rs` (+ `src/bin/docker-removal-worker.rs`): the worker logic over a
  `WorkerFs` trait (real `HostFs`, or doubles): `bind_paths`, `Bindings::validate`, `selected_path`, `user_ids`
  with bounded account parsing, the descriptor-relative `remove` (two descriptors at most, deeper than `PATH_MAX`,
  streamed names, entry budget), `enter_container`, `run_remove_child`, the JSON-line `serve` loop, `worker_main`.
- `src/sandbox/sandboxes/docker.rs`: `DockerSandboxClient::with_removal_service` and
  `DockerSandboxSession::with_removal_service`. The client adopts the service's daemon connection and refuses to
  create or resume if the runner was replaced afterwards (`Use the Docker client's live removal service
  connection`). With a service: `create` starts the container and binds (with workspace bootstrap) before
  returning, removing the container on failure; a replacement made by `resume` does the same; reattaching asserts
  the live binding; recursive `rm`/`rm_as` go through the service, serialized with `stop`/`shutdown`;
  `validate_manifest_application` and `start` assert the binding; `shutdown` releases the authority only once the
  container is confirmed gone. A private `RemovalAuthority` trait keeps the client free of feature-gated code.

## Deviations from Python (all documented in the module docs)

1. **No `fork`.** Python forks the removal child from the long-lived worker. Rust forbids `unsafe`, and a process
   inside the container's chroot cannot start a host executable. So `remove` is a sibling process,
   `docker-removal-worker remove <pid>`, started by the service inside the same pause window. It enters the
   container, sets `RLIMIT_CPU`, resolves the user from the container's account files as root, drops
   groups/gid/uid, removes the original (alias-preserving) path and writes one JSON outcome. The wire shapes the
   service exchanges with a worker (`bind`, `inspect`, `remove` and their responses) are unchanged; only the
   transport splits.
2. **Worker executable.** Python runs `sys.executable -I -S <script>`. Rust runs the `docker-removal-worker`
   binary: next to the running executable by default, or `DockerRemovalOptions::with_worker_program`. It runs with
   `env_clear()` plus `LC_ALL=C.UTF-8`, its own process group, stderr null.
3. **Daemon access.** Python builds a `DockerClient` on a unix socket. Rust uses the `docker` CLI through
   `DockerRunner`, so there is no `socket_path` option (the CLI honours `DOCKER_HOST`); the local-host guarantee
   is the cgroup check, as in Python.
4. **Limits validated first.** `DockerRemovalService::new` checks the limits before root/worker lookup (Python
   checks root first); tests rely on it.
5. **Cancellation.** The service runs each operation on its own task, so dropping the caller's future never
   leaves a container paused (Python reaches the same outcome with `to_thread` shielding).
6. **`docker create -w` is omitted with a service.** The daemon would create a missing workdir as root, which
   the image user cannot write; Python never passes `working_dir` at create. Every exec names its workdir.
7. **Errors.** `SandboxError` with codes `SandboxConfigInvalid` (Python `ValueError`), `WorkspaceStartError`
   (`RuntimeError`) and `WorkspaceArchiveWriteError` (+ context reasons) replace Python's exception classes.
8. Python's `workspace_root_ready` and state-field restoration assertions in the client tests have no Rust
   counterpart: the Rust `resume` takes the state by reference and the session reads readiness from itself.

## Tests

- `tests/py_core13_r13c_worker.rs` (24): worker doubles and real directories through `HostFs`. No root needed.
- `tests/py_core13_r13c_service.rs` (34): fake daemon (inspect/pause/unpause/start/rm/exec), fake `/proc`, recording
  worker that asserts the container is paused whenever it is spoken to, gates for blocked removals.
- `tests/py_core13_r13c_client.rs` (11): client lifecycle on the real filesystem binding, and the real worker
  transport against shell scripts standing in for the executable (JSON-line protocol, child request, environment
  cleared, EOF => uncertain, child death => `RuntimeError`, structured child failures reaching the session error).
- The Linux-only code (`LinuxHost`, `worker_main`) cannot be compiled by the crate's test run on this macOS host;
  it was type-checked for `x86_64-unknown-linux-gnu` in a scratch crate copy of the worker module (safe rustix,
  `forbid(unsafe_code)` on). The crate itself cannot be cross-checked here: no C cross toolchain for
  `aws-lc-sys`.

## Remaining (valid python-only), one reason per group

- test_docker_removal.py (2):
  - `test_repeated_cancellation_waits_for_actual_host_completion[base_exception]`: Python `BaseException` raised
    from a worker thread; the success and exception variants are ported as "an aborted caller still completes the
    removal and thaws the container". A panic in a worker transport is a bug, not an outcome.
  - `test_worker_close_reaps_and_closes_output_after_broken_input_pipe`: asserts the order and exception choice
    of Python's `close()` over `SimpleNamespace` stdin/wait/stdout mocks. Rust's `close` shuts stdin, reaps the
    child and drops stdout by ownership; "close an exited worker, twice" is tested.
- test_docker_removal_worker.py (2):
  - `test_bound_descriptors_are_closed_after_partial_or_normal_lifetime[register]`: `ExitStack.callback`
    registration failure; Rust has no separate registration step (an `OwnedFd` is closed by drop, which the
    `open`, `fstat` and `none` variants check).
  - `test_removal_child_always_exits_without_resuming_parent[KeyboardInterrupt]`: Python signal/`os._exit`
    behaviour of a forked child; the other five variants are ported against `run_remove_child`.

Notes on partial ports: `test_pause_cleanup_preserves_operation_failure` checks which error wins and that the
container stays paused, not the `__context__`/`__cause__` identities; `test_live_manifest_update_preserves_removal_authority`
checks the authority half through `validate_manifest_application` (the capability/`apply_entry_batch` half is the
runtime's, covered elsewhere); the denied variant uses a policy denial inside the pause.

## Source fixes and changes

- `src/sandbox/sandboxes/docker.rs`: described above. Without a service, behaviour is unchanged (existing Docker
  test binaries re-run).
- `Cargo.toml`: feature `docker-removal`, `[[bin]] docker-removal-worker` (`required-features`), docs.rs list.
- Docs: `docs/system-architecture.md`, `docs/project-changelog.md` (Unreleased).

Breaking changes: none (new feature, new public types; `DockerSandboxClient`'s private struct gained a field).

## Unresolved questions

- Python's `socket_path` for a non-default daemon is not offered (see deviation 3). If wanted, the PTY path of
  `DockerRunner` needs a host-argument hook first.
- The worker cannot be exercised against a real container here (needs root on a Linux Docker 26+ host). A smoke
  test on such a host is the remaining verification.
- Whether the default worker lookup should also search `PATH` for installed binaries.

Status: DONE_WITH_CONCERNS
Summary: Docker removal service, worker binary and client wiring implemented behind `docker-removal`; 132 of 136 python-only cases ported, 4 remain with reasons.
Concerns: the privileged paths (namespace entry, credential drop) are type-checked for Linux but not run; no real-container smoke test was possible.
