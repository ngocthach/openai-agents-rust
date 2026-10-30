# Sandbox core — final parity pass

Closes the remaining "partially closed" and "Deferred" rows of
`plans/reports/sandbox-port-report.md`, plus the `sandbox/sandboxes/docker.py` row of
`plans/reports/parity-audit-261030-0505-python-to-rust-report.md`.

Area owned: `src/sandbox/**`, `tests/sandbox_tests.rs`, new `tests/sandbox_core_*.rs`.
Python reference: `/Users/ngocthach/myGit/openai-agents-python/src/agents/sandbox/**`.

## Closed rows

### `sandboxes/docker.py`

| Python | Rust |
| --- | --- |
| `DockerSandboxClientOptions.labels` | `DockerSandboxClientOptions::{labels, with_label, with_labels}` → `docker create --label k=v` |
| `exposed_ports` + `_validate_docker_network_configuration` | `exposed_ports`, `with_exposed_port(s)`, `DockerSandboxClientOptions::validate()`; `network_disabled` + any port = `SandboxConfigInvalid`. Ports publish as `-p 127.0.0.1::<port>/tcp`, as Python does |
| `_validate_docker_path_grants` | `validate_docker_path_grants(&Manifest)`: SYS_ADMIN/FUSE + read-only host bind refusal, duplicate targets, grant-inside-root, grant-vs-volume conflict, leading-slash alias collapse (`docker_mount_path_identity`) |
| `_docker_volume_name` / `_docker_volume_names_for_manifest` | `mounts::docker_volume_name(session_id, mount_path)` (`sandbox_{session}_{sha256[:12]}_{sanitized}`) and `sandboxes::docker::docker_volume_names_for_manifest`. Volume mounts now emit `src=<name>`; `shutdown` removes them (`docker volume rm`), which previously leaked. A replacement container built during a resume keeps the paused session's id, so it attaches — and later removes — the volumes the original created; only a state whose mount authority was rebound gets a fresh id, which is the case where fresh storage is the point |
| `_assert_existing_container_network_configuration_matches` | `assert_container_network_matches(&Value, network_disabled)` |
| `_assert_existing_container_labels_match` | `assert_container_labels_match(&Value, &labels)` |
| `_assert_existing_container_path_grants_match` | `assert_container_path_grants_match(&Value, &Manifest)` |
| `container = None if requires_fresh_resource` | The container id is only *required* on the reattach branch. A state sanitized by `sanitized_for_persistence` carries none by design, and must still resume into a replacement; requiring it up front would have made a credentialed Docker state unresumable. Covered by `a_sanitized_docker_state_still_resumes_into_a_replacement` |

All three resume assertions are pure functions over the `docker container inspect` JSON, so they
are tested against fixtures with no daemon. `DockerSandboxClient::resume` now reads the container
attributes, runs all three, and forces a fresh container when the state's mount authority was
rebound. The create-time identity (`network_disabled`, `labels`, SDK session id) travels in the
new `SandboxSessionState::provider_metadata`.

**Not applicable:** the raw-socket exec transport (`_LENGTH_FRAMED_STDIN_SCRIPT`,
`_DockerExecSocket`). It exists because `docker-py` cannot stream stdin reliably over a TLS
`DOCKER_HOST`; this port drives `docker exec -i`, where the CLI owns the stdin framing. The
length-framed `head -c` preflight guards against a `head` that cannot do `-c` — a hazard that only
arises when the SDK writes the pipeline itself.

### `session/snapshot_lifecycle.py` — the fingerprint fast path (was *Deferred*)

The report deferred this as "needs an in-sandbox runtime helper, and a `sha256sum` BSD images do
not ship". Re-verified against the source: the helper script itself falls back
`sha256sum → shasum → openssl`, so it is portable, and the helper mechanism is the point of the
row rather than an obstacle to it.

- New `src/sandbox/runtime_helpers.rs`: `RuntimeHelperScript` with `from_content`,
  `install_command()`, `present_command()`, and the two scripts as **byte-identical** copies in
  `src/sandbox/helpers/{workspace-fingerprint,resolve-workspace-path}.sh`. The install path
  embeds the script's own sha256 prefix, so a unit test asserting
  `/tmp/openai-agents/bin/workspace-fingerprint-fe85e2af8427` also asserts the copies have not
  drifted from Python's.
- `SandboxSession::ensure_runtime_helper_installed` (provided; works for every backend).
- `SandboxSessionState::{snapshot_fingerprint, snapshot_fingerprint_version}` +
  `SnapshotFingerprint`, `SNAPSHOT_FINGERPRINT_VERSION{,_V1}`. Both backends return `None` from
  `resume_snapshot_fingerprint` while the workspace is not fully materialized, which is Python's
  `_can_reuse_preserved_workspace_on_resume` gate — and is what makes a sanitized state, whose
  `workspace_root_ready` is deliberately cleared, fall back to a full restore.
- `session::{compute_and_cache_snapshot_fingerprint, read_cached_snapshot_fingerprint,
  parse_snapshot_fingerprint_record, live_workspace_matches_snapshot_on_resume,
  snapshot_fingerprint_cache_path, clear_workspace_root_on_resume,
  workspace_resume_mount_skip_relpaths}`.
- `persist_session_snapshot` fingerprints before archiving and clears the record if the archive
  fails. `restore_session_snapshot` — which **every** backend including the six cloud providers
  already calls — skips the restore when the live workspace still matches, and otherwise does
  Python's *pruned clear* first (removing stale files the archive does not contain, while leaving
  ephemeral mount directories alone, since under them is live remote storage).

In shared-graph mode `SandboxRunner::prepare` also unions **every** graph agent's `run_as`
account into the manifest the session is created with. Python never needs this, because each of
its agents brings its own session; here a handoff target with its own `run_as` would otherwise
run `sudo -u <account that was never created>`.

### `runtime_session_manager.py` — per-agent sessions (was *Deferred*)

Added as an opt-in mode; the committed one-session-per-graph design stays the default.

- `SandboxSessionMode::{SharedGraph (default), PerAgent}` + `SandboxRunConfig::session_mode`.
- `SandboxAgent::{graph_agent_names, graph_agent, prepare_graph_with_bindings}`.
- `PreparedSandboxRun::agent_guards`, `SandboxRunOutcome::sandbox_states_by_agent` (Python's
  `sessions_by_agent`), released/suspended together with the starting agent's session on both the
  buffered and the streamed path.
- Each node materializes **its own** `resolved_manifest` and runs as its own `run_as` user.

Differences, deliberate: Rust already rejects duplicate agent names in a graph, so the agent name
*is* the resume key and Python's `_ensure_resume_key` / `_stable_resume_key_for_agent` /
`_assign_resumed_agent_key` / `_used_resume_keys` disambiguation has nothing to disambiguate.
Sessions are created eagerly at `prepare` rather than lazily per turn; `PerAgent` requires a
`client` (one injected session cannot be split) and says so.

### `runtime_session_manager.py` — live-session manifest updates (was *Deferred*)

`SandboxRunner::prepare` now reconciles an injected session. The capabilities process the
manifest the **session** was started with (Python's base, not the agent's default), and
`apply_live_session_manifest` either confirms it is unchanged or applies Python's rule set
verbatim: host-grant topology change, `root`/`environment`/`users`/`groups` change while running,
removed entries, retyped entries, changed or added mount entries are each refused with their
reason; plain added or changed entries are materialized into the live workspace. The recursive
`Dir` diff is `_diff_live_session_entries`. Entries are compared by structural signature, because
`Entry` holds `Arc<dyn MountStrategy>` and cannot be `PartialEq`.

The session keeps its original manifest object: nothing the update may add carries a path grant
or a mount, so no consumer of the stored manifest sees a different answer afterwards.

### `SandboxConcurrencyLimits::manifest_entries` (was accepted but not applied)

`Manifest::apply` now batches entries exactly as `_apply_entry_batch` does: an entry joins the
current batch unless it is a `Mount` or its path overlaps a queued one (`paths_overlap` —
equal, parent or child), in which case the batch is flushed first and a mount is applied alone.
A batch runs `manifest_entry_width()` entries at a time and its results are collected in batch
order, so the `MaterializationResult` is deterministic. Added
`SandboxConcurrencyLimits::manifest_entry_width` and `SandboxRunConfig::concurrency_limits`
(Python's run-config field), which overrides the manifest's.

### Read-only path grants on raw writes

`SandboxSession` gained `workspace_path_policy()` and `normalize_path_for_write()` (provided).
`normalize_path` now delegates to [`WorkspacePathPolicy`], and the ad-hoc `granted_path` helper is
gone. Every write path the core owns goes through the write form: the trait's `mkdir`/`rm`/`mv`
and `write_file_as`/`mkdir_as`/`rm_as`/`mv_as` (so **all** backends inherit the check on those),
`DockerSandboxSession::write_file`, `UnixLocalSandboxSession::host_path_for(for_write)`, the
editor and the filesystem capability. A recursive `rm` also runs `validate_recursive_remove`,
which refuses deleting a tree containing a read-only grant.

The six provider `write_file` implementations live outside this area — see **Cross-area requests**.

### `workspace_paths.py` — `WorkspacePathPolicy`

Ported with `new` / `with_grants` / `for_manifest`, `sandbox_root`,
`root_is_existing_host_path`, `absolute_workspace_path`, `relative_path`,
`normalize_sandbox_path(path, for_write)`, `validate_recursive_remove`,
`extra_path_grant_rules`, `matching_grant`. Longest-grant-wins precedence, workspace root beats
any grant that also covers it, and the read-only refusal carries Python's
`reason = read_only_extra_path_grant` + `grant_path` context.

### `util/retry.py` — wired into a call site

Python's only core-sandbox `retry_async` is on `DockerSandboxSession.persist_workspace`
(`docker.py:1444`). The Rust equivalent is the trait's default `persist_workspace`, which now
retries through `retry_async` with `is_transient_sandbox_error` (new): an explicit
`SandboxError::retryable` wins, otherwise only `ExecTransportError` retries. Losing a suspend's
workspace to a dropped transport is exactly the failure worth repeating.

### `apply_patch.py` — `normalize_operation` and the pluggable format

`SandboxEditor::normalize_operation` returns an operation whose `path`/`move_to` are in the
policy's canonical form. `PatchFormat` (trait) + `V4aFormat` (default) replace the hard-wired
`apply_diff` call, installed with `SandboxEditor::with_patch_format`.

### `base_sandbox_session.py` — the access-probe scripts

The four scripts are byte-identical copies in `src/sandbox/helpers/{read-path-probe,
write-access-check,mkdir-access-check,rm-access-check}.sh`, exposed as consts, with
`ACCESS_PROBE_TIMEOUT` (10 s) and the provided methods `probe_read_path`, `check_write_access`,
`check_mkdir_access`, `check_rm_access`. They are wired into the trait's default `mkdir` and
`rm` and into `write_file_as`, which run them on the **failure** path — where Python runs them
*before* every operation. The caller gets the same typed `WorkspaceArchiveWriteError` naming the
reason (missing parent, unwritable parent, wrong type) instead of a bare non-zero exit, and the
happy path keeps one round trip instead of two.

### `entries/artifacts.py` — `user=` and the symlink-safe copy

`Entry::apply_as(session, dest, user)` routes every write through the session's `*_as` methods,
including the recursive `Dir` walk and the `local_dir` batch copy; metadata (`chmod`/`chgrp`) is
skipped when a user is given, as Python does, because it would run as the session owner and fight
the ownership the write just set.

The fd-walk row: `read_host_file` now opens with `O_NOFOLLOW`. The directory scan already
classified entries without following links; the residual race was a regular file being swapped
for a symlink between the scan and the read, and that now fails with `ELOOP` instead of copying
the link target into the sandbox. No new dependency: `tokio::fs::OpenOptions::custom_flags`.

### `sandbox_session_state.py`

- `redacted_host_path_grant_paths` (Python's `REDACTED_HOST_PATH_GRANT_PATHS_KEY`).
- `rebind_path_grants(&trusted_manifest)` — refuses by name when the trusted manifest no longer
  supplies a grant the live manifest had, and rebinds the mount authority at the same time.
- `assert_path_grants_rebound()` — the provenance flags are now **enforced**, not just recorded.
- `sanitized_for_persistence()` (`_sanitize_persisted_provider_identity`): a state whose live
  manifest held mount credentials persists no `provider_id` and a `uuid5`-derived session id, and
  loses `workspace_root_ready`. `to_json_string` applies it, so the sanitization cannot be
  forgotten at a call site.
- Both backends' `resume` call `rebind_path_grants` before anything reattaches.

### `session/{sinks,events}.py`

- `HttpProxySink`: POSTs each event as JSON with configurable headers and a 5 s default timeout,
  falling back to a spool file when the collector is unreachable.
- `WorkspaceJsonlSink`: appends to a JSONL file inside the workspace, with `{session_id}`
  templating, `flush_every`, a `max_bytes` budget that stops the sink rather than truncating, and
  Python's forced flush points (`persist_workspace` start, `stop`, `shutdown` start).
- `SandboxSessionEvent::trace_id` + `current_trace_id()`: every event is stamped with the SDK
  trace it ran inside, which is the key that joins a sandbox event stream to the run's model and
  tool spans.

### `session/dependencies.py` — the `Dependencies` container (was *Deferred*)

Ported as `src/sandbox/dependencies.rs`: string-keyed `Arc<dyn Any + Send + Sync>` with
`bind_value` / `rebind_value` / `bind_factory{cache, closer}` / `get` / `get_as::<T>` / `require`
/ `contains` / `clone_template` / `close`. Owned results close newest-first; a closed container
refuses to build anything new; the cache lock is held across the factory so two concurrent
resolutions cannot both build an owned result. `SandboxSession::dependencies()` and
`Snapshot::{persist_with, restore_with}` (both additive defaults) thread it to the snapshot, which
is the one load-bearing use in Python. The existing typed `register_remote_snapshot_client`
registry stays as the ergonomic path.

### `sandboxes/docker_removal.py` (was *Deferred*)

The privileged in-container worker does not apply: a Docker workspace lives inside the container
and goes with `docker rm`, and a bind mount's host source belongs to the caller. The one real gap
the report named is now closed: Unix-local `shutdown` with `run_as` hit `EACCES` on a `0700`
directory created by `sudo -u agent` and left the temp root behind. It now falls back to
`remove_workspace_as_manifest_users()` — the trusted manifest names every account the sandbox ran
as, so those files are unlinked by the accounts that own them, then the root is removed as the
session owner. This gets the same outcome as Python's worker without `nsenter` or root.

### `runtime.py` — `apply_result_metadata` (was *Deferred*)

Python attaches three private attributes to a `RunResult`. The Rust equivalents, now complete:

| Python | Rust |
| --- | --- |
| `result._sandbox_session` | `SandboxStreamedRun::session()` (new), `PreparedSandboxRun::session()` |
| `result._sandbox_resume_state` | `SandboxRunOutcome::sandbox_state` + `sandbox_states_by_agent`, `SandboxStreamedRun::sandbox_state()` |
| `result._sandbox_cleanup` | the `StreamCompletion` hook installed by `run_streamed_with_options`, which fires the cleanup itself instead of leaving it to the caller |

### `runtime_session_manager.py` — `_manifest_with_run_as_user`

`SandboxAgent::resolved_manifest` now adds the `run_as` user to `Manifest::users` when it is not
already declared, so `provision_accounts` creates the account before the first command needs it.

One consequence is worth stating, because it is a **behaviour change on an existing path**: an
*injected* session plus an agent with `run_as` whose account the session's manifest never
declared is now refused. `useradd` on a session whose processes are already running changes the
accounts under them, so Python refuses it too (`_process_live_session_manifest` adds the user,
then `_validate_running_live_session_manifest_update` sees `users` differ). Before this pass the
Rust path accepted it and then ran commands as an account that did not exist. The error now names
`with_run_as` and says to declare the account on the manifest the session was created with.
Covered by `an_injected_session_refuses_an_agent_whose_run_as_account_it_never_declared`.

### Core hooks requested by the cloud-provider area

Section 9 of `plans/reports/sandbox-providers-final-parity-report.md` asked for four core
additions. All four are in, all additive — the provider sessions keep compiling unchanged and can
adopt them when convenient.

1. **`ErrorCode::ExposedPortUnavailable`** (`"exposed_port_unavailable"`). Port-resolution
   failures no longer have to borrow `SandboxConfigInvalid`.
2. **`ExposedPortEndpoint` in `crate::sandbox::types`**, with `new` / `with_query` / `url_for`
   (default-port elision, IPv6 bracketing, signed-query append), plus `ExposedPortUnavailable`
   and `exposed_port_unavailable(...)` building Python's error shape (`op =
   resolve_exposed_port`, `reason = not_configured | backend_unavailable`, `backend`, `port`,
   `exposed_ports`; `not_configured` is explicitly non-retryable). `SandboxSession` gained
   `exposed_ports()` and `resolve_exposed_port(port)` as **default** trait methods — the default
   refuses with the right reason for the port. `src/extensions/sandbox/ports.rs` can now become a
   `pub use crate::sandbox::types::{...}`; that file is not mine to edit.
3. **`SandboxSession::read_bounded(path, max_bytes)`** (default: read then truncate, with a
   positive-`max_bytes` check). The doc states the invariant Cloudflare needs — a result of
   exactly `max_bytes` is not EOF — and names
   `context["reason"] = "bounded_read_wire_limit"` as the convention for a transport that
   refuses the request outright. `WorkspaceJsonlSink::flush_buffer` now uses it: with a
   `max_bytes` budget it reads `max_bytes + 1` (one byte over is enough to prove the budget was
   exceeded without pulling in the rest of a workload-written log) and stops the sink on a
   `bounded_read_wire_limit`, exactly as Python's sink does.
4. **`src/sandbox/http.rs` sets `retryable` per status.** `transport()` is retryable (the request
   never reached the provider); `http_error()` sets `retryable = status_is_retryable(status)` and
   records `context["http_status"]`; a success status with an unparseable body is explicitly
   non-retryable. `status_is_retryable` is 429 plus `TRANSIENT_HTTP_STATUS_CODES` plus other 5xx,
   excluding 501. The four exec-driven provider retries read this flag, so a 400 or a 401 now
   fails once instead of three times.

`DockerSandboxSession` overrides `exposed_ports()`, so a declared-but-unresolvable Docker port
reports `backend_unavailable` rather than `not_configured`. Docker itself still cannot resolve a
host port without `docker port`, which is a separate row.

Also confirmed for the coordinator: `sandbox::mounts::tests::builds_docker_arguments` passes —
the volume arguments now carry `src=<named volume>`, and the test asserts both the unnamed and
the session-named form plus the path-hash disambiguation.

## Remaining, with the inherent reason

| Row | Why it stays open |
| --- | --- |
| `unix_local.py`: `_PTY_CHILD_SIGNAL_DEFAULTS` / `_restore_pty_child_signal_defaults` | Python resets `SIGINT`/`SIGQUIT` to `SIG_DFL` in a `preexec_fn`, between fork and exec. `portable-pty`'s `CommandBuilder` exposes no hook there, and no dependency adds one — the fork happens inside the crate. It is also largely moot in Rust: `execve` already resets every *handled* signal to default, and only `SIG_IGN` is inherited, which this SDK never sets. An embedding application that ignores `SIGINT` process-wide would pass that on to PTY children; that is the residual difference, stated rather than papered over. |
| `sandboxes/docker.py`: raw-socket exec transport | Not applicable. See the Docker section above. |
| `sandbox_client.py`: polymorphic client-options registry | Rust client options are concrete types the caller constructs; there is no untrusted-dict deserialization path for them to survive. Unchanged from the previous report. |

## Cross-area requests (files I do not own)

The read-only-grant check on a **raw** `write_file` is inherited by every backend for
`mkdir`/`rm`/`mv` and the `*_as` family, but `write_file` is implemented per backend. The six
provider sessions each need one word changed — `normalize_path` → `normalize_path_for_write` — in
their `write_file`:

- `src/extensions/sandbox/daytona.rs:1267`
- `src/extensions/sandbox/vercel/mod.rs:1412`
- `src/extensions/sandbox/cloudflare/sandbox.rs:840`
- `src/extensions/sandbox/blaxel/sandbox.rs:1225`
- `src/extensions/sandbox/modal/sandbox.rs:1025`
- `src/extensions/sandbox/e2b/mod.rs:1412`

Python's other `retry_async` call sites are all provider-side and belong to that area:
`blaxel/sandbox.py:664`, `modal/sandbox.py:1811`, `vercel/sandbox.py:150` and `:1525`,
`daytona/sandbox.py:1033`, `cloudflare/sandbox.py:1445` and `:1488`, `e2b/sandbox.py:1340`.
`src/sandbox/retry.rs` now exports `is_transient_sandbox_error` for them.

## Shared-file edits

- `Cargo.toml`: `uuid` features `["v4"]` → `["v4", "v5"]`. Needed for
  `_sanitize_persisted_provider_identity`, which derives the replacement session id with
  `uuid5(NAMESPACE_URL, …)` exactly as Python does. No new dependency.
- `src/lib.rs`: **not touched**.

## Breaking changes

- `MountStrategy` gained `docker_create_args_for_session` — a **default** method, so existing
  implementations keep compiling. Docker volume mounts now carry `src=<named volume>`, where they
  previously created an anonymous volume per container; a caller that located the old anonymous
  volume by inspection will not find it by that route.
- `Snapshot` gained `persist_with` / `restore_with`, both defaults.
- `SandboxSession` gained defaults only: `normalize_path_for_write`, `workspace_path_policy`,
  `ensure_runtime_helper_installed`, `resume_snapshot_fingerprint`,
  `record_snapshot_fingerprint`, `computes_snapshot_fingerprint`, `dependencies`,
  `probe_read_path`, `check_write_access`, `check_mkdir_access`, `check_rm_access`,
  `run_access_check`, `persist_workspace_once`, `exposed_ports`, `resolve_exposed_port`,
  `read_bounded`.
- `ErrorCode` gained `ExposedPortUnavailable`, so an exhaustive match on it needs a new arm.
- `src/sandbox/http.rs` now sets `retryable` on every error it builds. A caller that treated
  `retryable == None` as "retry anyway" will stop retrying 4xx responses — which is the point.
- `SandboxSessionState` gained fields (`redacted_host_path_grant_paths`, `snapshot_fingerprint`,
  `snapshot_fingerprint_version`, `provider_metadata`), so a struct literal of it needs updating;
  the builders are unchanged and older persisted JSON still parses.
- `SandboxRunConfig` gained `session_mode` and `concurrency_limits`; `PreparedSandboxRun` gained
  `agent_guards`; `SandboxRunOutcome` gained `sandbox_states_by_agent`. Struct literals of these
  need updating.
- `SandboxEditor` gained a private `patch_format` field (built by `new()`).
- `DockerSandboxClientOptions` gained `exposed_ports` and `labels`, so a struct literal needs
  updating; `new()` and the builders are unchanged.
- `Entry::apply` is unchanged; `apply_as` is new.
- A raw `write_file` / `mkdir` / `rm` / `mv` into a read-only path grant now fails where it
  previously succeeded. That is the point of the row.
- Docker sessions now remove their named volumes on `shutdown`.
- An **injected** session plus an agent with `run_as` whose account the session's manifest never
  declared is now refused instead of silently running `sudo -u <missing account>`. This matches
  Python; the account belongs on the manifest the session was created with.

## Verification

| Check | Result |
| --- | --- |
| `cargo test --all-features --lib sandbox::` | 288 passed |
| `cargo test --all-features --test sandbox_tests` | 65 passed (unchanged suite) |
| `cargo test --all-features --test sandbox_core_parity_tests` | 28 passed (new suite, see below) |
| `cargo test --all-features --test sandbox_e2b_daytona_vercel_tests` | 39 passed — the six provider backends inherit the changed trait defaults |
| `cargo test --all-features --test sandbox_modal_cloudflare_blaxel_tests` | 33 passed |
| `cargo test --all-features --doc sandbox` | 13 passed |
| `cargo clippy --all-features --all-targets -- -D warnings` | clean, whole workspace |
| `rustfmt --edition 2024` on every file touched | applied; `src/lib.rs` never formatted |

New tests in `tests/sandbox_core_parity_tests.rs` cover: the exposed-ports/network conflict, the
SYS_ADMIN + read-only-bind refusal, grant-outside-root and duplicate-target validation, volume
naming and hash disambiguation, all three resume assertions against `docker container inspect`
fixtures (no daemon needed), longest-grant-wins and the read-only refusal in
`WorkspacePathPolicy`, raw `write_file`/`mkdir`/`rm`/`mv` refusal under a read-only grant with
*no* backend call made, fingerprint record parsing, helper install-once behaviour, the end-to-end
fingerprint fast path against a real Unix-local session, grant redaction/rebinding, the
credential-bearing state losing its provider identity, `provider_metadata` round-tripping with
older JSON, and parent-before-child entry batching with a mount settling alone. The in-module
suites add the `Dependencies` lifecycle, the runtime-helper install paths (which double as
byte-identity assertions against the Python scripts), the volume-name arguments, and the HTTP
status retryability table. Also covered: the exposed-port refusal and its non-retryable
`not_configured` reason, `ExposedPortEndpoint::url_for` (default-port elision, IPv6 bracketing,
signed query), bounded reads with a zero budget and a missing file, `Entry::apply_as` routing
every byte through `exec` as the named user with no `chmod`/`chgrp` pass, and the injected
session refusing an undeclared `run_as` account, and a sanitized credentialed Docker state
resuming without tripping on its deliberately absent container id (which runs against a real
daemon when one is present, and removes the replacement container afterwards).

Because the changed trait defaults (`normalize_path` via the policy, the write-form
`mkdir`/`rm`/`mv`, `restore_session_snapshot`'s pruned clear, the `persist_workspace` retry) are
inherited by all six cloud backends, both provider integration suites were run as well.

**Threat model, stated rather than fixed.** The helper install root is `/tmp/openai-agents/bin/`
*inside the sandbox*, and `present_command`'s `test -x` trusts whatever already sits at a
hash-predictable path. On Docker and the hosted providers that `/tmp` belongs to the container.
On Unix-local it is the **host's** `/tmp`, so another local user could pre-seed the path. Python
has the identical design, and the Unix-local backend is already documented as providing no
OS-level confinement, so this changes no boundary that existed before.

### Known limitation

`DockerSandboxSession` can report that a declared port is `backend_unavailable`, but cannot
resolve it: the host port the daemon assigned needs a `docker port` lookup, which is a separate
row and was not part of this pass.

### A note on two transient failures

Mid-pass, `resumes_a_suspended_sandbox_from_persisted_state` and
`resumes_a_run_that_paused_inside_a_handoff_destination` failed with "Model reused a tool call ID
for a different invocation", raised by `validate_restored_tool_invocations` in
`src/run_internal/mod.rs` — another area's in-flight `_tool_invocation.py` port, which for a
while rejected the replay of an approved call across a resume. Both pass again on the current
tree; recorded here because it is exactly the failure mode an approval-resume regression would
produce, and the sandbox suite is where it surfaced first.

Status: DONE
Summary: Every partial and deferred sandbox-core row is closed except three that are inherently
not expressible (PTY `preexec_fn` signal defaults, the raw-socket Docker exec transport, the
untrusted-dict client-options registry), each with its reason. The four core hooks the
cloud-provider area asked for are in, all additive. One shared-file edit (`uuid` gains the `v5`
feature); `src/lib.rs` untouched. Gates: clippy `--all-features --all-targets -D warnings` clean
workspace-wide; 288 lib + 65 existing + 28 new + 39 + 33 provider integration + 13 doc tests
pass; both provider integration suites re-run after every shared-default change.
Concerns/Blockers: the read-only grant check on a raw `write_file` is complete for the two core
backends and for every backend's `mkdir`/`rm`/`mv`/`*_as`; the six provider `write_file`s need the
one-word change listed under Cross-area requests. `src/extensions/sandbox/ports.rs` should
collapse into a re-export of the core `ExposedPortEndpoint` now that it exists, and the six
provider sessions should override `resolve_exposed_port` rather than offering an inherent method
of the same name — both are that area's call.
