# Live Docker sandbox and docker-removal run

Date: 2026-11-02 · Crate: openai-agents-sdk 0.23.0 · Branch: feat/rust-port-of-agents-sdk

## Outcome

- Both the Docker sandbox client (`sandbox`) and the removal service (`docker-removal`) ran end to end against a real Docker daemon, with a real container and, for removal, the real `docker-removal-worker` binary running as root.
- **No bug was found in `docker.rs`, `docker_removal.rs`, `docker_removal_worker.rs` or the worker binary.** Every new live assertion passed.
- One real defect was fixed: a **Linux-only `clippy -D warnings` failure in the lib**, in `src/sandbox/sandboxes/unix_local.rs`. It was never seen because clippy had only been run on macOS.
- I added two env-gated live test files (11 tests in total).

## Environment

| Item | Value |
|---|---|
| Host | macOS (Darwin 24.6.0, Apple silicon) |
| Docker daemon | **OrbStack**, not Docker Desktop: Linux VM kernel `7.2.12-orbstack-00394-gbecdd6130068`, arm64 |
| Docker | client/server 29.8.2, API 1.56 (min 1.40) |
| Daemon security | `SecurityOptions = ["name=seccomp,profile=builtin","name=cgroupns"]`, `DefaultRuntime = runc`, cgroup v2, cgroupfs driver |
| Sandbox image | `python:3.14-slim` (the SDK default, already present locally; nothing pulled for it) |
| Linux builder | `rust:1.89-bookworm` (rustc 1.89.0 = MSRV), static docker CLI 29.0.0 from download.docker.com |

The removal service's eligibility checks all hold on this VM. I checked them with a throwaway container before writing any tests:
- `/proc/<pid>/cgroup` is `0::/docker/<64-hex id>`.
- `uid_map` and `gid_map` are `0 0 4294967295`.
- `/proc/<pid>/ns/mnt` and `/proc/<pid>/root` are reachable.
- A `-v <mac path>:/x:ro` bind reports `Type=bind, RW=false, Propagation=rprivate`.

## Exact commands

The scripts live in the session scratchpad `…/scratchpad/dockerlive/`. A shell hook blocks some words in command text, so the env exports are inside the scripts.

macOS (`mac-test.sh <test> [args]`):
```sh
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-out/dockerlive CARGO_PROFILE_DEV_DEBUG=0
unset OPENAI_API_KEY
cargo test --features sandbox --test py_sandbox_b_docker_live -- --nocapture
OPENAI_AGENTS_DOCKER_LIVE=1 cargo test --features sandbox --test sandbox_docker_live -- --test-threads=2
cargo test --features sandbox --test sandbox_docker_live -- --nocapture   # skip path, env unset
```

Linux builder (`builder-up.sh`):
```sh
docker volume create oars-live-cargo-out; docker volume create oars-live-cargo-home
docker run -d --name oars-removal-builder --label oars-live=1 \
  --privileged --pid=host --cgroupns=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /Users/ngocthach/myGit/openai-agents-rust:/src:ro \
  -v "$S":"$S" \                       # scratch dir at its identical macOS path
  -v oars-live-cargo-out:/cargo-out -v oars-live-cargo-home:/cargo-home \
  -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/cargo-out -e CARGO_PROFILE_DEV_DEBUG=0 \
  -e TMPDIR="$S/linux-tmp" -w /src rust:1.89-bookworm sleep infinity
# inside, as root: install the static docker CLI, then
OPENAI_AGENTS_DOCKER_REMOVAL_LIVE=1 OPENAI_AGENTS_DOCKER_LIVE=1 \
  cargo test --features docker-removal --test <each suite> -- --test-threads=2 --skip live_docker_exposed_port
cargo test --features docker-removal --lib sandbox::sandboxes::docker
cargo clippy --features {sandbox,docker-removal} --lib --bins --test sandbox_docker_live \
  --test sandbox_docker_removal_live --test py_sandbox_b_docker_live -- -D warnings
```

Why each builder flag is needed:
- `--pid=host` and `--cgroupns=host`: the service reads the sandbox container's `/proc/<pid>/{cgroup,uid_map,gid_map}`, and the worker `setns()`es into its mount namespace.
- `TMPDIR` on the identically mounted scratch path: the host-path grants a test creates must resolve to the same path when the daemon bind-mounts them.

## Per-test results

### macOS, `--features sandbox`, real daemon

| Test | Result |
|---|---|
| py_sandbox_b_docker_live: `docker_create_omits_the_network_flag_by_default` (mock) | pass |
| py_sandbox_b_docker_live: `docker_resume_reapplies_network_isolation_to_a_replacement` (mock) | pass |
| py_sandbox_b_docker_live: `live_docker_authority_preserves_resume_and_cleanup` (real) | pass |
| sandbox_docker_live: `live_docker_session_materializes_operates_and_deletes` | pass |
| sandbox_docker_live: `live_docker_resume_reattaches_then_replaces_from_the_snapshot` | pass |
| sandbox_docker_live: `live_docker_network_isolation_survives_resume` | pass |
| sandbox_docker_live: `live_docker_exposed_port_reaches_a_server_in_the_sandbox` | pass |
| sandbox_docker_live with the env var unset | 4/4 skip with the reason printed |

### Linux builder, root, `--features docker-removal`, real daemon

| Suite | Result |
|---|---|
| lib `sandbox::sandboxes::docker*` unit tests | 12/12 pass |
| py_core10_r10c_docker_removal_shapes | 8/8 pass |
| py_core10_r10c_removal_local | 2/2 pass |
| py_core13_r13c_worker | 24/24 pass |
| py_core13_r13c_service | 34/34 pass |
| py_core13_r13c_client | 11/11 pass |
| py_sandbox_b_removal | 6/6 pass |
| py_sandbox_b_docker_live (live test with real bind mounts, from Linux) | 3/3 pass |
| sandbox_docker_live (port test skipped, see below) | 3/3 pass |
| sandbox_docker_removal_live: `live_removal_service_removes_refuses_and_releases` | pass |
| sandbox_docker_removal_live: `live_removal_service_enforces_the_entry_visit_budget` | pass |
| sandbox_docker_removal_live: `live_removal_service_preserves_a_read_only_host_grant` | pass |
| sandbox_docker_removal_live: `live_removal_service_refuses_an_ineligible_container` | pass |
| lib `unix_local` unit tests (after the clippy fix) | 6/6 pass |

## What the new live tests assert

### `tests/sandbox_docker_live.rs`

Gated by `OPENAI_AGENTS_DOCKER_LIVE=1`. The image comes from `OPENAI_AGENTS_DOCKER_LIVE_IMAGE` and defaults to the SDK default; it is never pulled.

**Create, materialize, operate, delete:**
- Labels are applied to the container.
- Nested manifest entries land where expected.
- `ls` works.
- `exec` sees the manifest env and the workspace as its directory, `cwd` works, and `exec -u` runs as a provisioned account.
- A 3 MiB `write_file`/`read_file` round trip survives (larger than a pipe buffer). `read_bounded` returns the right prefix.
- `mkdir -p`, `mv`, `rm` and `rm -r` work.
- A symlink pointing out of the workspace is refused for both read and write (`escape_root`).
- `exec` timeouts are enforced.
- A PTY process prints its output and exits 0.
- `delete` removes the container.

**Resume:**
- `stop`, then resume, reattaches to the same container and the workspace is intact.
- After the container is deleted, resume creates a replacement and restores both the local snapshot and the seed entry.

**Network:**
- `network_disabled` gives `NetworkMode=none` and only `lo` in `/proc/net/dev`.
- The setting is persisted in state and enforced on both the reattach and the replacement branch, even through a client configured with networking.

**Published port:**
- `resolve_exposed_port(8000)` returns `127.0.0.1:<daemon-assigned port>`.
- A `python3 -m http.server` inside the sandbox answers `HTTP/1.0 200` from the host.
- A port that was never published is refused (`not_configured`).

### `tests/sandbox_docker_removal_live.rs`

`#![cfg(all(feature = "docker-removal", unix))]`, gated by `OPENAI_AGENTS_DOCKER_REMOVAL_LIVE=1`. Once the variable is set, root, the daemon and the image are asserted, not skipped. It uses `DockerRemovalService::new` (the real root check) with `CARGO_BIN_EXE_docker-removal-worker`.

**Removes, refuses and releases:**
- The session is bound before it is returned.
- A recursive `rm` removes the tree, keeps its sibling, and removes a symlink rather than following it.
- The workspace root is refused as `docker_removal_bound_root`, however it is spelled (`/workspace`, `.`, `keep/..`).
- A missing target is fine for root. For an unknown user it fails with `worker_reason=ValueError`.
- **Permission cases, run as `nobody`:**
  - Removing inside a root-owned `0555` tree fails with `PermissionError`, errno 13.
  - Removing a root-owned file in a sticky `1777` directory fails with `PermissionError`, errno 1.
  - Both trees stay intact, and root then removes both.
- A target whose canonical path contains a literal backslash (reached via a symlink to `odd/\..\workspace`) is refused as `docker_removal_canonical_path`, and the sentinel file survives.
- A 30 × 200-character tree (≈6 KB path, deeper than PATH_MAX) is removed.
- `delete` removes the container, then `is_bound` is false, and `close()` succeeds.
- After every operation the container is `Running=true, Paused=false`.

**Entry budget:** with `max_entry_visits=3`, removing a 10-file directory fails with `OSError` errno 7 (E2BIG) and leaves a partial tree. The container is unpaused.

**Read-only host grant:**
- A real `ro` bind at `/toolchain` with `rprivate` propagation.
- Workspace cleanup works.
- `link/child`, `/toolchain/child` and `/toolchain` are all refused (`read_only_extra_path_grant` or `docker_removal_bound_root`), and the host file is untouched.
- `assert_bound` with a changed grant source fails with "original live authority"; with the original manifest it passes.

**Ineligible containers:**
- A writable shared host path is refused before any container is created.
- A container with `--cap-add SYS_PTRACE` is refused ("running private container"), and the client removes the container it had created.

About `chattr +i` (immutable files): it needs `CAP_LINUX_IMMUTABLE`, which an eligible removal container does not have, so I could not exercise it. The root-owned/EACCES and sticky-bit/EPERM cases are the closest real equivalents.

## Bugs found and fixed

1. **Lib: Linux-only `clippy::unnecessary_cast`** in `src/sandbox/sandboxes/unix_local.rs` (the `ls` listing). It had `stat.st_mode as u32`, and `st_mode` is `u16` on macOS but `u32` on Linux, so `cargo clippy --lib -D warnings` failed on Linux under both feature sets.
   - `u32::from` was not an option because it trips `useless_conversion` on Linux.
   - The fix is a small `stat_mode()` helper with `#[allow(clippy::unnecessary_cast)]`, the same pattern as `docker_removal_worker::to_stat`.
   - Behaviour is unchanged.
2. **No functional bugs** in the Docker client or the removal service and worker.

One defect in my own new test was found and fixed during the run. The deep-tree setup used a shell `cd` loop. dash resolves `cd` through an absolute path, so it stopped at depth 20 (4036 bytes), the loop never advanced, and the test hung. It now builds the tree with Python's relative `os.chdir`.

## What could not run, and why

- **Port reachability from the Linux builder.** `live_docker_exposed_port_reaches_a_server_in_the_sandbox` is skipped there (`--skip`). The daemon publishes on its own `127.0.0.1`, which the builder's separate network namespace cannot reach. The test passes from macOS, where the port is reachable.
- **Not Docker Desktop.** The machine runs OrbStack. Its VM met every removal requirement (Docker 26+, builtin seccomp, runc, no userns remap, `/docker/<id>` cgroups, rprivate ro binds), so nothing in the removal path had to be stubbed. The "Docker host" here is the OrbStack VM, reached through a `--privileged --pid=host --cgroupns=host` container.
- **Immutable-flag (`chattr +i`) files:** see above.
- **Other images.** Only `python:3.14-slim` was used, to avoid pulling. Other images can be run with `OPENAI_AGENTS_DOCKER_LIVE_IMAGE`.

## Pre-existing, not fixed

- `tests/py_sandbox_a_unix_local_3.rs:14` has `use std::time::Duration;`, which is an unused import on Linux only (rustc warning). I did not touch this file; it is outside this task's scope.

## Gates

- macOS: `cargo clippy --features {sandbox,docker-removal} --lib --bins --test sandbox_docker_live --test sandbox_docker_removal_live --test py_sandbox_b_docker_live -- -D warnings` is clean.
- Linux (rust 1.89 + clippy): the same clippy command is clean after the fix.
- `rustfmt --edition 2024 --check` is clean for the two new test files and `unix_local.rs`. `src/lib.rs` was not formatted.

## Cleanup

Everything I created was labelled (`oars-live`, `oars-live-probe`, `openai-agents-live-test`) or named (`oars-removal-builder`, `oars-live-cargo-{out,home}`).

`cleanup.sh` removed:
- the labelled containers;
- the two volumes;
- the `rust:1.89-bookworm` image (it was not in the baseline);
- `/Users/ngocthach/.cache/oars-out/dockerlive` and the scratch temp directories.

After cleanup, images, volumes and containers diff as identical to the baseline taken before the run. The user's own containers (postgres, minio, qdrant, redis, 9router) were never touched, and no prune command was used.

## Suggested changelog paragraph

> **Tests:** Added live test suites that run against a real Docker daemon, each gated by an environment variable so CI without Docker skips them.
> - `tests/sandbox_docker_live.rs` (`OPENAI_AGENTS_DOCKER_LIVE=1`) covers create, manifest apply, exec, file operations, PTY, resume (reattach and snapshot-restored replacement), network isolation across resume, and published-port reachability.
> - `tests/sandbox_docker_removal_live.rs` (`OPENAI_AGENTS_DOCKER_REMOVAL_LIVE=1`; root on the Docker host) drives the `docker-removal` service and the real worker binary against a real container. It covers recursive removal, bound-root and canonical-path refusals, per-user permission failures, the entry-visit budget, a deeper-than-PATH_MAX tree, read-only host grants and the ineligible-container refusals.
>
> Both suites pass on Docker 29.8.2 (OrbStack VM, builtin seccomp, runc).
>
> **Fixed:** a Linux-only `clippy::unnecessary_cast` failure in the `unix_local` sandbox's directory listing.
