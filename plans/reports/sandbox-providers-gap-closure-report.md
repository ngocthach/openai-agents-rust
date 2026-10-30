# Cloud sandbox providers: closing the remaining parity gaps

Closes the rows at lines 60, 119, 120 and 121 of
`plans/reports/parity-audit-261030-0505-python-to-rust-report.md`, against the Python sources in
`/Users/ngocthach/myGit/openai-agents-python/src/agents/extensions/sandbox/`.

Every row was re-verified against the current tree before anything changed, because the audit
predates the mount entry layer (`src/sandbox/mount_entries.rs`, `src/sandbox/mount_patterns.rs`)
that another agent has since landed and wired into each provider. No stream edited
`src/sandbox/**`.

## Row 119 — per-session `rclone` pattern (E2B)

**Verified open.** `E2BCloudBucketMountStrategy::fuse_pattern()` was synchronous and appended only
`--allow-other`, so `--uid`/`--gid` never reached a mount.

`rclone mount` runs as root while the sandbox runs commands as another user, so without the
ownership flags the mounted tree is attached but not usable by the agent.

Ported into `src/extensions/sandbox/e2b/rclone.rs`, which is where this crate keeps the port of
Python's shared `extensions/sandbox/_rclone.py`:

- `rclone_pattern_for_session(session, pattern)` — Python's `rclone_pattern_for_session`. A
  non-FUSE pattern is returned untouched and costs no round trip.
- `default_user_ids(session)` — Python's `_default_user_ids`: `sh -lc "id -u; id -g"` with a 30 s
  timeout. A non-zero exit, fewer than two lines, or a non-numeric line yields no flags rather
  than an error, exactly as Python leaves it. A transport failure still propagates.
- `append_option(args, option, values)` — Python's `_append_option`. The "do not duplicate an
  option the caller set" rule now covers `--uid` and `--gid`, not just `--allow-other`.

In `src/extensions/sandbox/e2b/mod.rs`, `delegate()` now takes the resolved pattern and the static
`fuse_pattern()` is gone. `activate` and `restore_after_snapshot` resolve the pattern against the
live session; `deactivate` passes the declared pattern, mirroring Python's `_delegate_for_session`
/ `_delegate` split (`e2b/mounts.py` L103, L145 vs L115, L125). Teardown only needs the remote
name, so it does not pay for the `id` round trip.

Tests: four unit tests in `e2b/rclone.rs` driven by the core `ScriptedSandboxSession` — the ids
reach a FUSE mount; a pinned `--uid` is kept and only the missing options are appended; a failed,
short or non-numeric `id` leaves the mount without ownership flags; an NFS pattern is untouched
and issues no exec. `e2b_activates_an_rclone_cloud_bucket_mount` now has the sandbox report
`1000`/`1001` and asserts both flags reach the mount command line.

**No breaking change**: both reshaped methods were private.

## Row 60 — Vercel `mounts.py`

Ported into a new `src/extensions/sandbox/vercel/mounts.rs`, which matches both the Python layout
and the other Rust providers. `vercel/mod.rs` re-exports it, so
`extensions::sandbox::vercel::VercelCloudBucketMountStrategy` still resolves.

**Verified already closed, not reworked:**

- The access-key/secret pairing and blank-value checks of `_validate_s3_mount` already live in
  `S3Mount::validate` (`src/sandbox/mount_entries.rs`), which
  `VercelCloudBucketMountStrategy::for_mount` calls, so a lone access key never reached
  `--no-sign-request` through that path. The same checks were added to the strategy's own
  `validate` so a hand-built strategy is covered too.
- Credentials were already out of argv. Python's Vercel path passes `env=` plus `sudo=True`
  straight to the provider SDK (`mounts.py` L150-155) and never uses the generic 0600 env-file
  layer; the Rust port does the same through `ExecRequest::with_envs`.

**Closed:** the `rpm --query` version probe with the 1.21.0 floor, `dnf install -y
--setopt=gpgcheck=1 fuse mount-s3` under sudo with a post-install re-verify; the empty-directory
check (`find -mindepth 1 -maxdepth 1 -print -quit`); the canonical-path check, run twice as Python
does; `id -u`/`id -g` to `--uid`/`--gid`; `--allow-overwrite`/`--allow-delete` tied to writability
rather than authentication; the `findmnt --output SOURCE` probe with the "refusing to manage an
unexpected filesystem" refusal; a strict `umount` under sudo that reports an ambiguous state
instead of ending in `|| true`; the fixed redacted message for a failing credentialed mount; and
`sudo` through `ExecRequest::with_user(root)`.

Escalation is deliberately selective: mount, umount and `dnf` run as root, while the `id`, `rpm`,
`find`, `findmnt` and `realpath` probes do not — running `id` as root would report root's ids and
defeat `--uid`.

**`MountpointMountPattern` was deliberately not reused** for the argv and env-file machinery, for
two checkable reasons: Python's own Vercel module does not use the generic pattern either (it has
its own `_mount_args`), and the generic argv is incompatible — bucket and mount path last, no
`--allow-other`, no `--uid`/`--gid`, and a `sh -lc` command line the Vercel REST transport does
not need because it carries `env` out of band. `redact_sensitive_values` from that module **is**
reused, so every helper command's stderr is redacted against the mount's own credentials.

The runtime-trust half of the row (`_runtime_s3_mount_activation_allowed`,
`_runtime_trusted_s3_mount`, active/detached tracking) is now session-side state in
`vercel/mod.rs`: a start gate, active and detached sets, a trusted-target check against
`manifest.mount_targets()`, and `fail_s3_mount_transition`, which stops the sandbox. That the
start gate closes no legitimate path was checked: `restore_session_snapshot` and
`Manifest::apply`/`apply_ephemeral` are reachable only from each backend's `start()`.

Twelve new Vercel-only tests in `tests/sandbox_e2b_daytona_vercel_tests.rs` dispatch the mock on
the command body, covering the writable and read-only argv (the missing `--allow-delete` is the
named regression), the anonymous mount, the install path, an unsupported major version, a
symlinked path, a populated directory, the redacted failure with the sandbox stop, late-activation
refusal, an untracked-path no-op, the foreign-filesystem refusal, and a detach/restore round trip.

## Row 120 — Vercel snapshot magic, runtime images, network policy

All three verified open and closed in `vercel/mod.rs`.

- **Snapshot magic.** Vercel now has its own encoder and decoder with `UC_VERCEL_SNAPSHOT_V1\n`,
  matching Python's `_VERCEL_SNAPSHOT_MAGIC` (`sandbox.py` L88), so a reference round-trips with a
  Python-written one. The decode is strict, with no fallback to E2B's magic, as Python's is.
  E2B's encoder was not touched.
- **Runtime images.** `RUNTIME_IMAGES` and `resolve_runtime_image` mirror Python's
  `_RUNTIME_IMAGES` (`_provider.py` L36-41) entry for entry — `node22`/`node24`/`node26` and
  `python3.13` — defaulting to `node24` and failing on an unknown alias. I re-checked this table
  and the magic string against the Python source myself; both match exactly.
- **Network policy.** `NetworkTransformer`, `NetworkPolicyRule` and a new
  `NetworkPolicy::CustomRules` variant give the per-domain form
  `dict[str, list[NetworkPolicyRule]]` with `NetworkTransformer.headers`. The existing
  list-shaped `Custom` variant is unchanged.

Python sends `image=` to the provider SDK and never a `runtime` field (`_provider.py` L114-124),
which is why the create request now carries `image` instead of `runtime`.

## Row 121 — Modal resources and the Blaxel drive mount entry

**Verified open**, except that `BlaxelDriveMountConfig` and `BlaxelDriveMountStrategy` already
existed.

### Modal: the snapshot restore timeout

The audit said restore reused the capture timeout. It did not: restore reached `create_session` →
`create_request`, which emits `create_timeout` from `sandbox_create_timeout`, and that defaults to
`None`, so **restore was effectively unbounded**. Python ignores `sandbox_create_timeout_s` on
restore and wraps all of `_run_restore` in
`wait_for(timeout=state.snapshot_filesystem_restore_timeout_s)` (`modal/sandbox.py` L463, L1954),
default 60 s.

`ModalSandboxClientOptions` gains `snapshot_filesystem_restore_timeout` (default 60 s) and
`with_snapshot_filesystem_restore_timeout`. `restore_from_image` clones the options with
`sandbox_create_timeout` set to it, so the bound reaches the helper through the same mechanism
exec timeouts already use.

### Modal: `(request, limit)` cpu and memory

`cpu_limit: Option<f64>` and `memory_limit_mb: Option<u32>` plus `with_cpu_range(request, limit)`
and `with_memory_mb_range(request, limit)`. `create_request` emits a scalar when there is no limit
and a two-element array when there is.

The wire shape was checked against Modal's own source rather than assumed: in
`modal-labs/modal-client`, `py/modal/_resources.py:convert_fn_config_to_resources_config` tests
`isinstance(cpu, tuple)` / `isinstance(memory, tuple)`. A JSON array arrives in Python as a
**list**, matches neither branch, and **silently zeroes both the request and the limit** with no
error. `src/extensions/sandbox/modal/helper.py` therefore gained `_resource_pair()`, which
converts a two-element list back to a tuple before `Sandbox.create`. A test execs `HELPER_SOURCE`
under a non-`__main__` module name and asserts the conversion, so the helper is covered without
calling Modal.

`ModalImageSelector` and `ModalSandboxSelector` stay unported, and the reasoning was re-confirmed:
the port drives Modal through a stateless subprocess that addresses everything by Modal id over
one JSON request/response, so a live `modal.Image`/`modal.Sandbox` object cannot cross that
boundary. `with_image_id` already covers the id half.

### Blaxel: the drive mount entry

`BlaxelDriveMount { drive_name, drive_path, read_only }` now implements the core `ConfiguredMount`
trait with `mount_type()` returning `"blaxel_drive_mount"`, alongside `MOUNT_TYPE_BLAXEL_DRIVE`,
`BlaxelDriveMountConfig::from_configured_mount`, `BlaxelDriveMountStrategy::for_mount` and
`BlaxelDriveMount::mount()`. It follows the same `for_mount` shape the Cloudflare, Modal and
Vercel strategies already use. The existing strategy and config are untouched. A drive attached as
a manifest entry produces a byte-identical `/drives/mount` request body, which a test asserts.

Two deliberate unifications with the core model, both documented on the type: Python's
`drive_mount_path` has no new field because core `Mount::with_mount_path` already does the job
(and reaches the manifest as well as the drive API call), and Python's separate `drive_read_only`
is unified with `Mount::read_only` so the request body's `readOnly` and the SDK flag cannot
disagree.

## Breaking changes

From the Vercel rows:

1. `--allow-delete`/`--allow-overwrite` follow writability, not authentication. A read-only
   authenticated mount no longer gets `--allow-delete`. This is the bug the audit named.
2. The snapshot magic is `UC_VERCEL_SNAPSHOT_V1\n`; an archive written by the previous Rust Vercel
   build no longer decodes. Strict, with no E2B fallback, matching Python.
3. An unknown runtime alias is `SandboxConfigInvalid` instead of being forwarded to the API.
4. `runtime: None` now sends `image: vercel/sandbox/node:al-24` instead of nothing. The API's
   universal image does not ship the mount tools, so this is also what makes a default-runtime
   sandbox mountable.
5. The create request sends `image`; the `runtime` branch is gone.
6. `NetworkPolicy` gained a `CustomRules` variant, so exhaustive matches on it break.
7. `VercelCloudBucketMountStrategy::validate` rejects a lone access key, a lone session token and
   blank credentials.
8. `activate` is refused outside `start()`, `deactivate` is a no-op on an untracked path,
   `umount` is strict, and any mount-transition failure stops the sandbox.
9. A failing credentialed mount reports a fixed message instead of the provider's stderr.

From the Modal row:

10. `ModalSandboxClientOptions` gained three public fields on a struct that is not
    `#[non_exhaustive]`, which breaks struct-literal construction. Nothing in `src/`, `tests/` or
    `examples/` constructs it outside `new()`. No existing builder signature changed;
    `with_cpu`/`with_memory_mb` now clear the paired limit, which is unobservable to callers that
    predate the limit fields.

Row 119 introduced no breaking change.

## Deferred

- **Python's `_s3_mounts_started`** ("starting the same mounted session again is not supported")
  needs session-lifecycle state beyond this round's scope.
- **Vercel `ephemeral` in `_validate_s3_mount`** is structurally satisfied rather than checked:
  `src/sandbox/entries.rs` forces `ephemeral = true` for every mount entry, and the Rust `Mount`
  has no `ephemeral` field a strategy could read.
- **Modal's restore timeout on `_restore_snapshot_directory_image`** (`sandbox.py` L1982) has no
  Rust counterpart to bound, because `ModalWorkspacePersistence` has no `SnapshotDirectory`
  variant. That is a separate parity row.
- **`adopt_recorded_root`** is still imported from the E2B module by Vercel. It is a shared
  root-adoption helper that Daytona imports too, not the snapshot encoder, so copying it would
  duplicate three callers' logic.
- **The nine Vercel breaking changes are not in the repo's 0.3.0 breaking-changes doc.** `docs/`
  was outside this round's file ownership, so the coordinator should fold them in.

## Observations outside the assigned rows

- **Daytona adds `--allow-other` where Python does not.** Python's `daytona/mounts.py` passes
  `self.pattern` to its delegate unchanged (L194) and never calls `rclone_pattern_for_session`;
  the Rust `DaytonaCloudBucketMountStrategy::fuse_pattern()` appends `--allow-other`. The audit
  recorded Daytona as at parity, and dropping the flag would likely leave a Daytona FUSE mount
  unusable by the command user, so this was left alone. It is a behavioral question for the
  coordinator, not an obvious bug.
- **The `CustomRules` wire shape is reconstructed, not read off a serializer.** The Rust types
  mirror Python's `_network_policy.py` models exactly — which is the parity contract — but how the
  vendor SDK serializes them onto the REST body could not be read (the provider `.venv` is
  hook-blocked, and the v3 schema types `networkPolicy.allow` as untyped). The pre-existing
  list-shaped `Custom` form is unchanged, so nothing regresses.

## Verification

Run with `CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud1`,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_INCREMENTAL=0`, after all three streams had landed:

- `cargo clippy --all-features --all-targets -- -D warnings` — clean.
- `cargo test --all-features` — green, no failures: 535 lib tests, 33 in
  `sandbox_e2b_daytona_vercel_tests`, 25 in `sandbox_modal_cloudflare_blaxel_tests`, the doctests
  and every other suite.
- `rustfmt --edition 2024 --check` — clean on all nine changed files. `src/lib.rs` was never
  formatted or edited.

No network, no real credentials, no git commands.

## Files changed

| File | Row |
| --- | --- |
| `src/extensions/sandbox/e2b/rclone.rs` | 119 |
| `src/extensions/sandbox/e2b/mod.rs` | 119 |
| `src/extensions/sandbox/vercel/mounts.rs` (new) | 60 |
| `src/extensions/sandbox/vercel/mod.rs` | 60, 120 |
| `src/extensions/sandbox/modal/sandbox.rs` | 121 |
| `src/extensions/sandbox/modal/helper.py` | 121 |
| `src/extensions/sandbox/blaxel/mounts.rs` | 121 |
| `src/extensions/sandbox/blaxel/mod.rs` | 121 |
| `tests/sandbox_e2b_daytona_vercel_tests.rs` | 60, 119 |
| `tests/sandbox_modal_cloudflare_blaxel_tests.rs` | 121 |
