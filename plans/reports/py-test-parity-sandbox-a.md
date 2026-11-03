# Python to Rust test parity: area sandbox-a (combined)

Status: DONE_WITH_CONCERNS

Scope: 25 Python files in `/Users/ngocthach/myGit/openai-agents-python/tests/sandbox/`, 943 `def test_`
(counted with `grep -c '^\s*\(async \)\?def test_'` per file; parametrize expansions are not multiplied).
Sub-reports, one per slice: `py-test-parity-sandbox-a-{runtime,mount-security,docker,entries,memory-snapshot,unix-local}.md`.
The controller (this file) owns the archive slice: test_tar_utils, test_workspace_paths, test_extract.

## 1. Per-file table

| python file(s) | #tests | ported | covered | python-only | owner |
|---|---|---|---|---|---|
| test_workspace_paths.py | 43 | 37 | 0 | 6 | controller |
| test_tar_utils.py | 43 | 40 | 0 | 3 | controller |
| test_extract.py | 32 | 29 | 0 | 3 | controller |
| test_docker.py | 108 | 84 | 0 | 24 | docker |
| test_docker_removal.py, _worker.py, _client.py | 62 | 0 | 0 | 62 | docker |
| test_entries, test_mounts, test_manifest, _application, _config, test_materialization, test_dependencies | 137 | 112 | 0 | 25 | entries |
| test_memory, test_snapshot, test_snapshot_defaults, test_session_state_roundtrip | 128 | 87 | 0 | 41 | memory-snapshot |
| test_mount_security.py | 134 | 45 | 0 | 89 | mount-security |
| test_runtime.py (lines 1-4000 by runtime, 4000-end by memory-snapshot) | 165 | 96 | 22 | 48 | runtime |
| test_unix_local, _file_io, _user_file_io, _special_files, _create | 91 | 55 | 0 | 36 | unix-local |
| **total** | **943** | **585** | **22** | **337** | |

585 + 22 + 337 = 944, one more than 943. The runtime split by line range counted one test that straddles
line 4000 in both halves (84 + 82 = 166 against a real 165). Nothing is unclassified.

Rust tests: 36 test binaries, `tests/py_sandbox_a_*.rs`, **666 tests, 666 pass, 0 fail, 0 `#[ignore]`**
(final run, `--all-features`). One Rust test often ports a group of Python tests that differ only in their
parameters, which is why 666 is larger than a one-to-one count would suggest for some slices and smaller
for others.

PYTHON-ONLY groups, one reason each (per-test lists are in the sub-reports):

- test_docker_removal*: a root, host-side, Linux-only removal service the Rust port deliberately omits
  (recorded in `plans/reports/sandbox-port-report.md`; accepted by the controller).
- test_mount_security: defences against hostile pydantic subclasses (`__deepcopy__`, `model_serializer`,
  `__traceback__`, raw-state `model_validate`). Rust closes these by construction: mounts are traits whose
  identity is a `&'static str`, and `SandboxSessionState` has no manifest field.
- docker-py internals (exec socket, length framing, spool), asyncio cleanup tasks, pydantic `RunState`
  schema, `monkeypatch` on `Path.exists`/`os.open`/`tempfile`, `pathlib` type identity, Python file-object
  seekability, Windows-native-host cases the Python tests skip themselves.

## 2. Rust bugs fixed (by file)

Controller:
- `workspace_paths.rs`: `..` is resolved lexically and only an escape is rejected (`pkg/../x` resolved to
  an error before); Python's exact `manifest path must be relative / must not escape root` messages with
  `rel`/`reason` context; Windows-absolute checks in `model_resource_path`; `sandbox.cwd` messages;
  grant `path` normalization (`/srv/../etc` to `/etc`, `//server/share` kept, `///x` collapsed) while
  `host_path` still rejects `..` and UNC; drive-qualified `host_path` accepted; `anchor` keeps the model's
  spelling; read-only-write error text and context; new `normalize_host_path` (symlink-resolving) and
  `sandbox_path_grant_host_path`.
- `archive.rs`: new `strip_tar_member_prefix` (Python's staged-copy rewrite: drops FIFOs/devices, rebases
  simple absolute symlink targets only when the archive establishes every directory and a regular leaf);
  new `TarValidationOptions::{reject_rel_paths, reject_symlink_rel_paths}`.
- `session.rs`: the default `write_new_file` no longer probes with a read, so create over an existing name
  overwrites as Python's base class does; new `SandboxClient::configured_snapshot`.
- `runtime.rs`: snapshot precedence is run-config, then client-configured, then SDK default. Python has no
  client-level snapshot (it only takes `create(snapshot=...)`); an explicit run-config snapshot wins over
  both, matching Python. `unix_local`/`docker` `create_with_snapshot` honor the passed spec.
- `pty.rs`: a signal-terminated child reports `-signum` (Python's `exit_code == -signum`). The crate forbids
  `unsafe`, so the number is recovered from the status name (`Interrupt`, `Interrupt: 2`, `Signal 40`),
  with macOS/Linux numbering for SIGBUS/SIGUSR1/2. Unit test added.
- `errors.rs`: `SandboxError::exec_nonzero` message and context now match Python's `ExecNonZeroError`.
- `docker.rs`: one type fix to let the crate build (the docker helper owns the rest).

Helpers (full lists in their reports): entries (insertion-ordered `Manifest` entries, host-source
confinement by default, `gather_in_order`, `Environment::resolve` concurrency, account provisioning, git
acquisition, `describe()` tree format); memory-snapshot (state grant/port handling, default snapshot dir 0700,
memory artifact naming `{rollout_id}_{slug}.md`); mount-security (rclone/mountpoint injection hardening,
`manifest_digest` folds in mount topology, credential-file checks, `carries_credentials` so credentialed
Docker-volume mounts are redacted from persisted state); runtime (injected-session start/validation,
rollout id grouping, `sandbox.*` spans with error attributes, restore-time account reprovisioning, active
agent memory manager); docker (`DockerRunner` seam, port resolution, shutdown settles every volume,
`WorkspaceArchiveReadError`, resume keeps the persisted identity, in-container symlink resolution,
protected-mount redaction, volume leak on failed create); unix-local (macOS `sandbox-exec` confinement,
`mv` via `rename(2)`, FIFO/special-file refusal, host_path grant refusal as Python does, double-unmount fix,
exclusive-create flush, and re-check-after-open for the path-swap race).

## 3. Breaking changes

- `manifest_digest` now covers mount topology: a session state persisted by an older build whose manifest
  has configured mounts fails `verify_manifest` once.
- `Manifest::describe()` renders Python's tree layout (`/workspace` with no trailing slash).
- `local_file`/`local_dir` sources are confined to the working directory unless granted (Python's default).
- `SandboxError::exec_nonzero` message is now the command's own output, not `Command ... exited with code`.
- Default `write_new_file` overwrites an existing file; backends with a no-clobber primitive still refuse.
- `SandboxPathGrant::new` normalizes `..` in the sandbox-side path instead of rejecting it.
- `unix_local` refuses grants with a `host_path`, and manifest users, as Python does.
- Memory rollout summaries are written as `rollout_summaries/{rollout_id}_{slug}.md`.
- New public API: `strip_tar_member_prefix`, `TarValidationOptions::{reject_rel_paths,
  reject_symlink_rel_paths}`, `WorkspacePathPolicy::normalize_host_path`, `sandbox_path_grant_host_path`,
  `SandboxClient::configured_snapshot`, `Manifest::entries_in_order`, `MountStrategy::carries_credentials`,
  `runtime::sandbox_span_result`. `Manifest::provision_accounts` is now public.

## 4. Open items, none of them in files this area owns

- `tests/py_sandbox_b_helper_startup.rs::runner_snapshot_does_not_execute_workspace_startup_files`
  (sandbox-b's test) fails on the approved leg on macOS. The approved command writes outside the
  workspace, which Python's `_darwin_exec_profile` denies without a path grant (reproduced with the real
  `sandbox-exec`). Fix is in the test: give it an extra path grant for the outside directory (Python
  compatible) or skip that leg on macOS. Not widened in the profile, to keep Python's confinement.
- `src/models/openai_client.rs:698` `models::openai_client::tests::error_events_become_errors` fails
  (another area's).
- Clippy `-D warnings` fails in other areas' files: `src/run_internal/agent_identity.rs:106`
  (type_complexity), `src/run_internal/mod.rs:2415` and `:2783` (needless_borrow),
  `src/stream_events.rs:48` (large_enum_variant), `tests/py_mcp_util_schemas.rs:237` (`.err().expect()`).
  With those three lint kinds allowed, the whole crate lints clean in this area's files
  (`--all-targets`, one finding in `py_sandbox_a_docker_3.rs` was fixed).
- Core files needed by the runtime slice: blocking input guardrails run after the session is created
  (`src/run_internal/mod.rs:1842`), and streamed memory follows the active agent only if `StreamCompletion`
  (`src/result.rs:250`, built at `:374`) gains a final-agent name.
- `Dir` children (`EntryKind::Dir`) are still key-ordered because the type is public; top-level entries
  keep insertion order.
- Path-swap race: `read_file`, `write_file`, `write_new_file` re-check the opened handle (dev+inode and
  authorization) before truncating. A descriptor-relative `openat` walk is impossible under
  `#![forbid(unsafe_code)]` with `libc` dev-only. Residual window, documented in the unix-local report:
  an A to B to A swap is not seen, a planted in-workspace hardlink is indistinguishable, and `mkdir`, `ls`,
  `rm`, `mv`, `same_file` are still path-based after validation.
- Two Python checks have no Rust counterpart: the root-mismatch check on mount rebind (the root is excluded
  from the digest by design) and the backend-id check (Rust strategies do not declare one).

## 5. Regression fixed after the full-workspace run

`py_sandbox_a_runtime_6::an_unsafe_mount_is_refused_before_snapshot_work` failed in the shared-tree run
(`WorkspaceArchiveReadError` where `MountConfigInvalid` is expected). Cause: `persist_session_snapshot` had
no mount-safety check of its own. The refusal used to come out of a later teardown path, and that path now
wraps its error with the archive-read code. Python's `stop()` validates every mount's credential boundary
before any snapshot work, so the check is now explicit: `Manifest::validate_mount_credential_boundaries`
runs first in `persist_session_snapshot` (`src/sandbox/manifest.rs`, `src/sandbox/session.rs`), before the
noop-snapshot early return, as in Python. sandbox-b's 20 test binaries, `sandbox_tests`,
`sandbox_core_parity_tests` and the provider suites stay green.

The same re-run exposed a separate problem in `py_sandbox_a_runtime_tracing`: two tests that drive a whole
`SandboxRunner` run on a `current_thread` runtime overflowed the 2 MB test-thread stack (SIGABRT). It is
not caused by the check above: it overflowed with that call disabled, and passes on an 8 MB stack. The
runner future has grown through core-runner edits elsewhere. The two tests now run on a thread with a 32 MB
stack (`on_a_big_stack` in the test file). The underlying growth in the runner future is not addressed
and is worth boxing in `src/run_internal` if it keeps growing.

Final whole-area run (`--all-features`, lib `sandbox::` plus every `py_sandbox_a_*`, `py_sandbox_b_*`,
`py_sandbox_rclone_mounts` and `sandbox_*` binary): 1805 passed, 0 failed, 0 ignored.

## 6. Process notes

- `Cargo.toml` and `src/lib.rs`: no net edits. A `libc` dependency was added and reverted in this round.
- Existing tests edited because a decision required it: `tests/sandbox_tests.rs` (memory artifact name,
  host trees staged under cwd for default confinement, describe format), `src/sandbox/capabilities/skills.rs`
  and `src/sandbox/manifest.rs` unit tests (staging under cwd).
- One read-only `git diff --stat` was run by mistake despite the no-git rule; it changed nothing.
- Build dir `/Users/ngocthach/.cache/oars-build/tp-sbxA` deleted at the end of this round.

Status: DONE_WITH_CONCERNS
Summary: All 943 Python tests are classified (585 ported, 22 covered, 337 python-only) with 666 Rust tests
passing and none ignored; the deferred items from the earlier round are fixed in source.
Concerns/Blockers: one sandbox-b test needs a path grant on macOS, other areas' files fail clippy and one
lib test, the path-swap race has a documented residual window, and the breaking changes in section 3 need
release notes.
