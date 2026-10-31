# Python to Rust test parity, round 6: Modal and Vercel sandbox providers

Slice: `tests/extensions/sandbox/test_modal.py`, `test_vercel.py`, `test_vercel_provider_contract.py`.

## How cases were counted

A case is one pytest-collected item: each `test_*` function, multiplied by the length of its
`@pytest.mark.parametrize` lists (Python `ast` walk, `literal_eval` of the argument list). That is
110 + 104 + 12 = 226 test functions, expanding to 119 + 130 + 29 = 278 collected cases, which is
the unit the whole-suite ledger uses. A parametrized function whose variants land in different
buckets is split (the delayed-exit PTY test and the Vercel contract wire test).

| file | cases | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| test_modal.py | 119 | 65 | 13 | 41 | 0 |
| test_vercel.py | 130 | 54 | 29 | 47 | 0 |
| test_vercel_provider_contract.py | 29 | 17 | 4 | 8 | 0 |
| **total** | **278** | **136** | **46** | **96** | **0** |

The per-file CSV is `py-test-parity-core6-modalvercel.csv`. Every case of the three files is in
exactly one bucket; earlier rounds had left most of them unclassified. Earlier ports were
re-verified by reading the cited Rust tests before being counted as covered.

## New Rust tests (111, all green)

| file | tests | covers |
|---|---|---|
| `tests/py_core6_modalvercel_modal_persist.rs` | 33 | tar persist excludes, filesystem/directory snapshot stash and restore, tar fallback, restore preserves ports/resources, resume, bucket mount config, strategy builders, durable-mount detach around a directory snapshot, resume fingerprint fast path |
| `tests/py_core6_modalvercel_modal_pty_hydrate.rs` | 13 | PTY start/write/drain/delayed tail/failure, start and exec failures, tar hydration (protected members, control, large archive), helper chunking and retry classification |
| `tests/py_core6_modalvercel_modal_paths.rs` | 13 | in-sandbox path resolver, read classification, resume with mounts, strategy registry round trip, protected-mount redaction, rebound authority, escaping mount path |
| `tests/py_core6_modalvercel_vercel_mounts.rs` | 29 | S3 mount creation checks, lifecycle ordering, persist/restore around a mount, redaction, native-snapshot refusal, exec timeout, mounts attached last, resume fingerprint fast path |
| `tests/py_core6_modalvercel_vercel_session.rs` | 17 | start roots, path policy via the resolver, resume statuses, archives, write retry/validation, native snapshot restore |
| `tests/py_core6_modalvercel_vercel_contract.rs` | 6 | wire options, snapshot lifetime, runtime selection, unknown runtime, failed delete retryable, pending reconnect |

Fixtures: `tests/fixtures/fake_modal_helper_scripted.py` (rule-driven fake of the helper protocol,
with stateful sandbox ids, stopped sandboxes and four PTY modes) and
`tests/fixtures/modal_helper_driver.py` (runs the real bundled `helper.py` against fake SDK objects
for the 8 MiB upload chunking, `cat` failure classification and the retryable classification).
Both need only `python3`; the Modal SDK is not required.

## Rust bugs fixed (all in src/extensions/sandbox/modal/** and vercel/**)

1. `modal/sandbox.rs`: the tar archive ignored paths registered with
   `register_persist_workspace_skip_path`, and the Modal and Vercel sessions did not keep a skip
   registry at all (`persist_skip_registry` was the trait default, so registrations were
   forgotten). Both sessions now hold the registry and exclude `./<path>`.
2. `modal/sandbox.rs`: a `snapshot_filesystem` image captured the ephemeral entries and the staged
   rclone configuration (a mount credential). It now sets those paths aside and restores them. A
   filesystem image includes `/tmp`, so unlike the directory snapshot (which stages its tar under
   `/tmp`, outside the captured root) the set-aside archive is read from the command's output into
   this process, the paths are removed, and the archive is written back only after the image exists
   (`with_excluded_paths_held_in_memory`; same shape as Python's `tar cf -` / `tar xf -`).
3. `modal/sandbox.rs`: a failed set-aside left a partially removed workspace. It now attempts a
   restore before returning the error.
4. `modal/sandbox.rs`: a manifest with a Modal bucket mount (not detachable in place) persisted as
   a native image. It now falls back to tar, as Python's `_native_snapshot_requires_tar_fallback`.
5. `modal/sandbox.rs` + `modal/helper.py`: a helper error carried no retry classification.
   `helper.py` now adds `retryable` (from `_modal_retryability`) to its error response and
   `ModalHelper::call` forwards it, so a permanent Modal error is not retried by the tar persist loop.
6. `modal/sandbox.rs` and `vercel/mod.rs`: `hydrate_workspace` accepted archive members under a path
   the manifest owns as ephemeral (including a mount path). Validation now passes
   `reject_rel_paths = manifest.ephemeral_persistence_paths()` before anything is uploaded.
7. `modal/sandbox.rs` and `vercel/mod.rs`: `read_file` and `write_file` (and Modal's `read_bounded`)
   checked the path lexically only. They now go through `validate_remote_path_access` (the
   in-sandbox resolver), so a symlink out of the workspace or into a read-only grant is refused, as
   Blaxel already did. Vercel's `read_bounded` stays lexical: the existing bounded-read contract
   tests (`py_sandbox_b_bounded_read*.rs`) drive it with mocks that cannot answer a resolver
   command, and they are not mine to edit.
8. `modal/mounts.rs`: a GCS mount using native auth (service-account file, token) became an
   unsigned Modal bucket. `from_configured_mount` now refuses it
   ("gcs modal cloud bucket mounts require access_id and secret_access_key").
9. `modal/mounts.rs` + `modal/sandbox.rs`: a Modal bucket mount carrying inline credentials or a
   named secret now reports `carries_credentials`, which activates the shared protected-mount
   handling: the persisted state drops the sandbox identity, persist/hydrate/shutdown errors carry
   no provider text, and a state whose authority was rebound creates a fresh sandbox on resume.
10. `modal/sandbox.rs`: `create` and `persist_workspace` use `try_mount_targets`, so a mount path
    that climbs out of the workspace is an error instead of a silently dropped mount.
11. `vercel/mod.rs`: `create` now validates the S3 mount layout before a sandbox exists (mount at
    the workspace root, outside it, overlapping another mount or a manifest entry, and the
    strategy's own `validate`).
12. `vercel/mounts.rs`: an `endpoint_url` carrying a query, fragment or userinfo (a signed URL) is
    refused; it would have been passed in the mount command's argument vector.
13. `vercel/mod.rs`: a session with a credentialed mount reports exec, read, write, port, persist
    and stop failures without provider text (reusing `redact_for_protected_mounts`).
14. `vercel/mod.rs`: the tar archive command now passes `--no-wildcards`, so an entry named
    `cache[1]` excludes itself rather than a sibling `cache1`.
15. `vercel/mod.rs`: `resume` replaced a `pending` sandbox immediately, losing its workspace. It now
    waits for it, and replaces it only if it never reaches `running` (as Python does).
16. `vercel/mod.rs`: `terminate` recorded the session as stopped even when the stop request failed,
    so a second `shutdown` was a no-op. A failed stop now stays retryable.
17. `vercel/mod.rs`: a mounted session refuses to hydrate a native snapshot, which would replace the
    sandbox and drop the mounts.
18. `vercel/mod.rs` + `src/sandbox/manifest.rs` (`Manifest::split_mounts`): `start` now applies the
    manifest without the S3 mounts first, then attaches the mounts, on a fresh start, on a resume
    with no snapshot, and on a resume that restores one (the restore steps are replayed locally
    rather than through `restore_session_snapshot`, whose last step applies ephemeral entries in
    manifest order). An entry declared after a mount, or reaching the mount path through an alias,
    is now written before the mount exists, as in Python.
19. `vercel/mod.rs` and `modal/sandbox.rs`: the resume snapshot-fingerprint fast path. A reattached
    sandbox is marked state-preserved and carries the fingerprint from the persisted state;
    `resume_snapshot_fingerprint`, `record_snapshot_fingerprint` and `system_state_preserved_on_start`
    are implemented and `session_state` carries the fingerprint. A workspace that still hashes to
    the stored value skips the archive or image restore; a replacement sandbox never does.
20. `modal/sandbox.rs`: Python's Modal-owned mount detach exists, and is now ported. Around a
    directory snapshot, durable (non-ephemeral) mounts at or under the root, other than Modal bucket
    mounts, are detached (`teardown_for_snapshot`), the image is taken, and they are attached again
    (`with_ephemeral_mounts_removed`); after `mount_image` on restore they are attached again.

## Breaking or behavioural changes

- `src/sandbox/manifest.rs` gained one additive public method, `Manifest::split_mounts`. It is the
  only shared-file edit of this round (`session.rs` and `events.rs` untouched).
- A Vercel manifest's S3 mounts are always attached after every other entry, not in manifest
  position.

- Each Modal/Vercel `read_file` and `write_file` now costs the resolver's extra commands (a
  presence check, an install on first use, and the resolver call), i.e. two or three more helper
  spawns or REST calls per file operation. This mirrors Python.

- `ModalCloudBucketMountConfig::from_configured_mount` (and `ModalCloudBucketMountStrategy::for_mount`)
  now return `MountConfigInvalid` for a GCS mount without HMAC credentials. A named-secret GCS bucket
  is `ModalCloudBucketMountConfig::gcs(..).with_secret_name(..)`.
- Modal sessions whose manifest has a credentialed bucket mount now produce a state with
  `mount_authority_redacted`, so a bare resume needs the runner's rebind (the shared contract).
- Modal sessions with a bucket mount persist as tar even when `SnapshotFilesystem` or
  `SnapshotDirectory` is configured.
- `VercelSandboxClient::create` can now fail earlier for a bad mount layout, and
  `VercelCloudBucketMountStrategy::validate` rejects signed endpoint URLs.
- `helper.py` error responses may now carry a `retryable` boolean (additive).

## Covered cases and the Rust test that asserts each

Modal (13):
- `create_passes_manifest_environment`: `py_sandbox_modal::modal_create_passes_manifest_environment`
- `create_passes_idle_timeout`: `py_sandbox_modal::modal_create_passes_timeouts`
- `create_passes_resources` (2): `py_sandbox_modal::modal_create_passes_resources` (scalar) and
  `modal/sandbox.rs::serializes_a_cpu_and_memory_range_as_a_pair` (request/limit pair)
- `create_sets_default_cmd_...`, `create_can_opt_out_of_default_cmd`:
  `py_sandbox_modal::modal_create_sets_a_default_cmd_unless_opted_out`
- `create_uses_custom_image_builder_version`: `py_sandbox_modal::modal_create_uses_a_custom_image_builder_version`
- `create_passes_named_modal_secret_for_cloud_bucket_mount`:
  `py_sandbox_modal::modal_create_passes_cloud_bucket_mounts_and_named_secrets`
- `create_exposes_declared_ports`: `py_sandbox_modal::modal_create_exposes_declared_ports`
- `resolve_exposed_port_reads_tunnel_metadata`:
  `sandbox_modal_cloudflare_blaxel_tests::modal_resolves_an_exposed_port_through_a_tunnel`
- `stop_is_persistence_only_and_shutdown_terminates`:
  `py_sandbox_modal::modal_stop_is_persistence_only_and_shutdown_terminates`
- `clear_workspace_root_on_resume_preserves_nested_cloud_bucket_mounts`:
  `py_sandbox_a_runtime_2::clearing_the_workspace_on_resume_preserves_a_nested_mount`
- `snapshot_directory_teardown_failure_restores_partial_cleanup`: the generic half (a failed detach
  restores the mounts already detached, skips the operation, then shuts the session down) is
  `py_sandbox_b_mount_lifecycle::a_failed_detach_restores_the_detached_mounts_then_shuts_down`;
  the Modal-specific `terminate`/`sandbox_id is None` assertions are Python state fields.

Vercel (29):
- `s3_dynamic_mount_is_rejected_before_materialization`:
  `sandbox_e2b_daytona_vercel_tests::vercel_refuses_to_attach_a_mount_after_the_sandbox_started`
- `s3_mount_upgrades_mountpoint_below_minimum`:
  `sandbox_e2b_daytona_vercel_tests::vercel_installs_mountpoint_when_the_image_ships_an_old_one`
- `s3_mount_failure_redacts_full_activation_traceback`, `..._discards_transformed_stderr` (2):
  `sandbox_e2b_daytona_vercel_tests::vercel_redacts_a_failing_mount_command_and_stops_the_sandbox`
- `supports_pty_is_disabled_...`: `py_sandbox_vercel::vercel_supports_pty_is_disabled`
- `create_passes_provider_options`: `py_sandbox_vercel::vercel_create_passes_provider_options`
- `create_retries_transient_transport_error`, `create_does_not_retry_non_transient_error` (2):
  `py_sandbox_vercel::vercel_create_retries_a_transient_error_but_not_a_bad_request`
- `exec_read_write_and_port_resolution`: `py_sandbox_vercel::vercel_exec_read_write_and_port_resolution`
- `exec_marks_typed_not_found_non_retryable`, `exec_marks_typed_rate_limit_retryable` (2):
  `py_sandbox_vercel::vercel_exec_classifies_not_found_and_rate_limit`
- `retryability_status_table` (12): `py_sandbox_vercel::vercel_retryability_status_table` (the same 12 statuses)
- `read_and_write_reject_paths_outside_workspace_root`:
  `py_sandbox_vercel::vercel_read_and_write_reject_paths_outside_the_workspace_root`
- `rejects_sandbox_local_user_arguments`: `py_sandbox_vercel::vercel_rejects_a_sandbox_user` (exec;
  Rust `read_file`/`write_file` take no user)
- `resume_reconnects_existing_running_sandbox`:
  `py_sandbox_vercel::vercel_resume_reconnects_an_existing_running_sandbox`
- `serialized_session_state_omits_token_and_resume_uses_live_client_token`:
  `py_sandbox_vercel::vercel_serialized_state_omits_the_token`
- `tar_persist_raises_archive_error_on_nonzero_exec`:
  `py_sandbox_vercel::vercel_tar_persist_raises_archive_error_on_nonzero_exec`
- `snapshot_mode_resume_uses_native_snapshot_reference`:
  `sandbox_e2b_daytona_vercel_tests::vercel_persists_and_restores_a_native_snapshot`

Vercel provider contract (4, the `released_policy_and_state` parametrization):
`vercel/mod.rs::serializes_network_policies` and `::serializes_per_domain_network_policy_rules`
(unit tests).

## Python-only groups (one reason each)

- Re-exports and import hooks (`test_*_package_re_exports_*`).
- Pydantic state and the Python `from_state` / constructor / `trusted_s3_mounts` API: the Rust session is built by the client from a manifest it owns, so there is no
  half-trusted construction path. Traceback and hostile-exception-descriptor redaction inspects
  Python frames and exception slots.
- `SandboxRuntimeSessionManager` / live manifest delta processing (the five `injected_session` /
  live-manifest Vercel tests).
- asyncio cancellation and task settlement (the Modal directory persist/hydrate and PTY
  cancellation tests, the Vercel mount/entry/shutdown cancellation tests); Rust cancellation is
  future drop.
- asyncio locks and `aclose`/`shutdown`/`stop` sequencing with a lazily recreated sandbox: the Rust Vercel and Modal sessions are bound to one provider session, and
  `stop`/`shutdown` do not recreate it.
- SDK call timeouts and handles (`_call_modal`, `Sandbox.from_id`, `wait_for_status`, stdin buffer
  limits, fallback-read polling): the helper process owns the SDK, and the pieces with
  host-observable behaviour are ported through the helper protocol and driver.
- Environment and kwargs handed to the SDK (`MODAL_IMAGE_BUILDER_VERSION`, token/project/team
  kwargs).
- `examples/sandbox/extensions/modal_runner.py` (4 cases).
- Python shapes with no Rust twin: `PureWindowsPath` handling, legacy `sandbox_id`-only Vercel
  state, `interactive` state flag, Python's in-memory empty-backup branch.

## Not ported

None. The four cases left open in the first pass are now ported: the resume fingerprint fast path
(`resume_skips_the_restore_when_the_workspace_fingerprint_matches` in the Modal and Vercel
files), the two durable-mount detach cases (`directory_restore_reactivates_durable_workspace_mounts`,
`directory_persist_detaches_only_durable_workspace_mounts`), and mounts-last ordering
(`entries_materialize_before_the_mount_and_a_populated_mountpoint_fails`,
`a_resumed_start_attaches_the_mount_after_the_restored_entries`).

## Cross-area findings (not edited)

- `tests/py_sandbox_b_bounded_read.rs` (E2B and Daytona cases, 5 tests) failed during my final run:
  those providers' `read_bounded` is being changed by another agent and the mocks in that file do
  not answer the new commands. Not my files; Vercel and Modal pass there.

- PTY `collect` race (`src/sandbox/pty.rs`): fixed by the coordinator.
- `src/sandbox/http.rs:34` 408/425: rejected by the coordinator (the shared set matches Python's
  `{500,502,503,504}`); Vercel keeps its own table.
- Modal's tar command and the shared `workspace_archive_excludes` consumers pass excludes without
  `--no-wildcards`: a documented low-impact difference. Vercel passes it.
- `tests/sandbox_modal_cloudflare_blaxel_tests.rs::modal_starts_a_pty_session_through_the_helper`
  (an earlier-round test) passes alone but fails under heavy machine load: its 400 ms PTY window can
  elapse before the helper's Python interpreter starts. I did not touch it.
- Other agents' failures observed mid-run and not mine: `src/strict_schema.rs:340` clippy
  `nonminimal_bool`, and transient lib compile errors in `src/run_internal/*` while another agent edited.

## Gates

- New binaries: 111 tests, all pass, repeated runs stable (PTY tests poll instead of sleeping on a fixed window).
- Tool-use note: one read-only `git status` was run by mistake (inside a combined rustfmt command);
  no other git command was used.
- Second pass: the sandbox provider binaries (Blaxel, Cloudflare, Daytona, E2B, Modal, Vercel, rclone, `py_sandbox_b_*` provider/mount/bounded-read, `sandbox_tests`) plus the rest of the suite that the run pulled in: 406 test binaries, 0 failures.
- Existing: `py_sandbox_modal` 12, `py_sandbox_vercel` 16, `sandbox_e2b_daytona_vercel_tests` 41,
  `sandbox_modal_cloudflare_blaxel_tests` 37, `py_sandbox_b_mount_lifecycle` 5, vercel/modal lib unit tests 36: all pass.
- `cargo clippy --all-features --all-targets` (re-run after the last test edits): no diagnostics in `tests/py_core6_modalvercel_*` or
  `src/extensions/sandbox/{modal,vercel}`. The `-D warnings` run stops on other agents' files (`src/strict_schema.rs:343`, `src/run_internal/tools.rs:757`, `src/run_internal/local_tools.rs:1063`, `src/realtime/debug.rs:29`); none is in my targets.
- rustfmt run on every file touched in my area.
- `src/sandbox/**`: only the additive `Manifest::split_mounts` in `manifest.rs`.

Status: DONE
