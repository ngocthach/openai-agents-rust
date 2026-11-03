# Sandbox archive parity — audit findings 2, 3, 7

Date: 2026-10-30. Source: `plans/reports/independent-parity-audit-report.md` findings 2 (no `extract()`, no
in-process member validation), 3 (`SandboxArchiveLimits`), 7 (git helpers). Re-verified against Python
(`sandbox/util/tar_utils.py`, `sandbox/session/archive_ops.py`, `sandbox/session/archive_extraction.py`,
`sandbox/session/base_sandbox_session.py:262,1338`, `run_config.py:48-50,192-222,254`,
`sandbox/runtime_session_manager.py:299-432`, `sandbox/util/github.py`) before any change — all three rows were
genuinely open.

## Closed rows

| Row | Rust |
|---|---|
| `safe_tar_member_rel_path`, `validate_tarfile`, `validate_tar_bytes`, `should_skip_tar_member`, `safe_extract_tarfile`, `UnsafeTarMemberError` | `src/sandbox/archive.rs`: `safe_tar_member_rel_path`, `validate_tar_members`, `validate_tar_bytes`, `should_skip_tar_member`, `extract_tar_to`, `unsafe_tar_member` → `SandboxError{code: workspace_archive_write_error, context: member/reason}` |
| `safe_zip_member_rel_path`, `validate_zipfile`, `UnsafeZipMemberError`, `WorkspaceArchiveExtractor` | `safe_zip_member_rel_path`, `validate_zip_bytes`, `validated_zip_entries`, `unsafe_zip_member` |
| `ArchiveResourceLimitError` | `archive_limit_exceeded` → same code with `reason`/`limit`/`actual`/`member` context |
| `BaseSandboxSession.extract(path, data, compression_scheme)` | `SandboxSession::extract` + `extract_with_limits` default methods (`src/sandbox/session.rs`), `CompressionScheme::{Tar,Zip}` with suffix inference, archive written at `path` then members restored into its parent through `mkdir`/`write_file`, with the `ls`-based pre-existing-symlink parent guard and per-directory cache |
| `SandboxArchiveLimits` + `run_config.py` validation + defaults | `src/sandbox/types.rs`: `SandboxArchiveLimits` (`max_input_bytes`/`max_extracted_bytes`/`max_members`, each `Option`), `validate()` (zero → `sandbox_config_invalid`, "archive_limits.X must be at least 1"), `DEFAULT_MAX_ARCHIVE_{INPUT_BYTES,EXTRACTED_BYTES,MEMBERS}` = 1 GiB / 4 GiB / 100_000 |
| `SandboxRunConfig.archive_limits` + `_set_archive_limits` | `SandboxRunConfig::archive_limits` field + `archive_limits()` builder (`src/sandbox/runtime.rs`); validated in `SandboxRunner::prepare` and pushed with `SandboxSession::set_archive_limits` on all three session paths (injected, create/resume, per-agent). Stored by `unix_local` in a `Mutex`; `archive_limits()`/`set_archive_limits()` default to no-op on other backends, as `record_snapshot_fingerprint` already does |
| `clone_repo`, `ensure_git_available` | `src/sandbox/util.rs`, driven through `SandboxSession::exec`: `command -v git` probe → `git_missing_in_image`; shallow `--depth 1 --no-tags --branch` then the `--no-checkout` + `checkout <ref>` fallback for a sha → `git_clone_error` |

## Validation is now in process, on every backend

- `unix_local::hydrate_workspace` no longer pipes bytes into the host `tar`. `extract_tar_gz` and its
  `tar -xzf -` child are deleted; the archive is validated and unpacked by `archive::extract_tar_to` on a
  blocking thread (directories and files first, symlinks last, `O_NOFOLLOW` + `create_new`, Python's
  `_restored_regular_file_mode` bits, `_ensure_no_symlink_parents` before every leaf).
- Every remote backend validates the bytes on the host before they are uploaded or unpacked remotely:
  `sandboxes/docker.rs` (one line), `extensions/sandbox/{e2b,modal,daytona,vercel,blaxel,cloudflare}` — each
  after the provider-native snapshot-reference branch, which carries no tar.
- `restore_session_snapshot` calls the new `SandboxSession::validate_restorable_archive` *before*
  `clear_workspace_root_on_resume`, so a workspace is never emptied for an archive that cannot replace it
  (Python's `_restore_snapshot_into_workspace_on_resume`). Overridden by `unix_local` and by all six extension
  providers (e2b, modal and vercel skip the check for their provider-native snapshot references, which are not
  tars). **Docker does not override it** — `sandboxes/docker.rs` is owned by another agent this round and the
  override is more than the one line that was allowed — so a docker resume still clears the workspace before
  its hydrate validates. One method to add, listed under follow-ups.
- Refused: absolute member paths, `..` traversal, Windows drive/separator paths, hardlinks, device/FIFO/socket
  members, duplicate paths, members descending through an archive symlink or a non-directory, members written
  through a pre-existing symlink in the destination, and — for `extract()` — symlink members outright, plus the
  three resource ceilings.

## Deliberate deviation from Python — the controller owns this call

`TarValidationOptions::hydrate()` keeps `allow_external_symlink_targets = true`. **Python passes `False` at all
seven of its hydrate call sites** (`sandbox/sandboxes/unix_local.py:1504` and every provider in
`extensions/sandbox/*/sandbox.py`), so Python refuses a symlink member whose target is absolute or climbs out
of the archive — including a virtualenv's `bin/python -> /usr/bin/python3`. Python does not round-trip such a
workspace either; supporting it is a decision this port made, documented in `unix_local.rs`'s
`rebase_absolute_symlinks`. Two reasons the strict flag was not adopted:

1. It would reverse that documented decision for outward links.
2. It would additionally break *in-root* absolute links, which Python handles by rewriting them at capture
   (`strip_tar_member_prefix`) while this port rewrites them after extraction — so a Rust archive still carries
   them absolute and strict validation would refuse them before the rebase can run.

What the controller should weigh: on `unix_local` containment is enforced on use — `host_path_for`
canonicalizes against the root before every operation, verified by `rejects_path_traversal_and_symlink_escapes`
— so a restored outward link is not a read/write channel. On docker and the remote providers the equivalent
guarantee rests on their path-probe scripts, which this round did **not** verify. If the stricter posture is
wanted, `TarValidationOptions::hydrate_strict()` is the exact Python policy and switching
`validate_hydrate_archive` to it is a one-line change that makes all seven call sites strict.

Workspace hydration applies no resource ceilings, matching Python: its `validate_tar_bytes` and
`safe_extract_tarfile` call sites pass no limits, and `SandboxArchiveLimits` reaches only `extract()`.

Other deviations: only `tar` and `tar.gz` are read in process (Python's `r:*` also accepts bz2/xz; those are
refused with `invalid tar stream` reason "unsupported archive compression"). `validate_tar_members` omits
Python's unused `reject_rel_paths`/`reject_symlink_rel_paths` parameters (no caller in either SDK).

## Breaking changes

- `FileEntry` gains `is_symlink: bool` (Python's `EntryKind.SYMLINK`); struct-literal construction needs the new
  field. `parse_ls_la` fills it from the `l` mode character; the four provider listings set `false` except e2b,
  which maps `FILE_TYPE_SYMLINK`. Required by the `extract()` parent-symlink guard.
- `SandboxRunConfig` gains `archive_limits` (struct-literal construction).
- `unix_local` hydrate now fails with `workspace_archive_write_error` (+ `member`/`reason` context) instead of
  `snapshot_restore_error` for a bad member, and fails *before* writing anything, where the host `tar` used to
  part-extract and then fail.
- `SandboxSession` gains five default methods (`extract`, `extract_with_limits`, `archive_limits`,
  `set_archive_limits`, `validate_restorable_archive`); no implementor is forced to change.

## Shared / out-of-ownership edits (each minimal, all listed)

- `Cargo.toml`: `sandbox` feature + `dep:tar`, `dep:flate2`, `dep:zip`; three optional deps
  (`tar` without `xattr`, `flate2` on miniz_oxide, `zip` with `deflate-flate2` only — all pure Rust, no new
  `-sys` crate: verified with `cargo tree`); dev-dependency `zip` for the test fixtures. `Cargo.lock` updated.
- `src/sandbox/mod.rs`: `pub mod archive;`, one `pub use archive::{…}` block, `SandboxArchiveLimits` +
  `DEFAULT_MAX_ARCHIVE_*` added to the `types` re-export, and the stale "not ported: archive extraction limits"
  doc line corrected.
- `src/sandbox/sandboxes/docker.rs`: one line, `validate_hydrate_archive(archive)?` at the top of
  `hydrate_workspace` (file owned by another agent this round).
- `src/sandbox/events.rs`: `InstrumentedSession` delegates `validate_restorable_archive`, `archive_limits` and
  `set_archive_limits` — without it a wrapped session silently drops the runner's limits.
- `src/extensions/sandbox/{e2b,modal,daytona,vercel,blaxel,cloudflare}`: one validation line each, plus
  `is_symlink` in four `FileEntry` literals.
- `src/sandbox/entries.rs`: the mount-ephemeral fix above (owned, listed here for completeness).
- `tests/sandbox_tests.rs`: `an_archive_cannot_write_through_a_symlink` now asserts the refusal reason and that
  *nothing* was extracted (it previously asserted the partially extracted link survived).
- `tests/sandbox_e2b_daytona_vercel_tests.rs`, `tests/sandbox_modal_cloudflare_blaxel_tests.rs`: four probe
  archives were opaque strings (`b"payload"`, `[0x1f,0x8b]`); they are now well-formed tars (local `probe_tar`
  helper / 1024 zero bytes), and the modal payload assertion compares the base64 length.

## Tests

`tests/sandbox_archive_tests.rs` (12): traversal, absolute path, hardlink, char device, path through an archive
symlink, duplicate path, member/reason context, pre-existing symlink parent (inside and outside the root),
happy-path hydrate (modes, directories, relative and outward links), `extract()` for tar and zip, zip-slip,
symlink member refused by `extract`, unknown/explicit compression scheme, all three limits plus the no-limit
case, zero-ceiling validation, a gzipped `persist_workspace` archive round-tripping through validation, and the
two `clone_repo`/`ensure_git_available` paths through `ScriptedSandboxSession`. Unit tests in `archive.rs` cover
the member-path policy, `should_skip_tar_member`, the strict symlink-target policy, the file-mode policy and
scheme inference.

Gates on the shared tree after all edits: `cargo clippy --all-features --all-targets -- -D warnings` clean;
`cargo test --all-features` = 1170 passed, 0 failed, 1 pre-existing ignored, across 40 targets. The five
failing tar/archive tests and the non-compiling `sandbox_archive_tests` target the coordinator saw were this
work mid-flight; all are green now (four provider probe archives had to become well-formed tars, and
`an_archive_cannot_write_through_a_symlink` now asserts pre-extraction refusal). Own files formatted with
`rustfmt --edition 2024`.

## Extra row (coordinator request)

`Entry::with_kind` forced `ephemeral = true` for every mount entry, so a `Mount` built with
`with_ephemeral(false)` produced an entry the snapshot and resume paths still treated as ephemeral. The entry
now takes the mount's own flag, as Python's `Mount.ephemeral` (`entries/mounts/base.py:386`, default `true`)
does. Test: `a_mount_entry_follows_the_mounts_own_ephemeral_flag` in `src/sandbox/entries.rs` — a durable mount
is absent from `ephemeral_mount_targets()`/`ephemeral_entry_paths()`, a default mount is still in both.

## Open / follow-ups

- `ScriptedSandboxSession` (`src/sandbox/testing.rs`, not owned this round) has no `Extract` method in
  `SandboxMethod`, so an `extract()` call against a script goes through the default implementation and consumes
  `write_file`/`mkdir`/`ls` steps. Adding an `Extract` variant would mirror Python's scripted surface.
- `SandboxOp` has no `Extract` variant (`errors.rs` not owned); archive errors are reported under
  `SandboxOp::Write`.
- Providers that could report link metadata in their listings (daytona, blaxel) still return
  `is_symlink: false`; the `extract()` parent guard is therefore advisory on those backends.
- `sandboxes/docker.rs` needs the three-line `validate_restorable_archive` override the other backends got, or
  a docker resume keeps clearing the workspace before the archive is validated.
- `validated_tar_entries`/`validated_zip_entries` buffer every member payload in memory, where Python streams
  member by member into the session. With `archive_limits` unset (the default on both sides) a decompression
  bomb is fully materialized on the host before the first write; declared sizes are checked on the header pass
  first, so configured limits do catch it. Streaming the payloads would close the gap.

Status: DONE_WITH_CONCERNS
Summary: Added `src/sandbox/archive.rs` with Python-equivalent in-process tar/zip member validation, a safe
extractor and resource limits; routed `hydrate_workspace` on every backend (local and remote) and the new
`SandboxSession::extract` through it; added `SandboxArchiveLimits` + `SandboxRunConfig::archive_limits` wired
through `set_archive_limits`; ported `clone_repo`/`ensure_git_available`; 12 new integration tests plus unit
tests, clippy and the full suite green.
Concerns/Blockers: (1) One policy call the controller owns — all seven Rust hydrate sites accept symlink
members whose target is absolute or leaves the archive, where all seven Python sites refuse them; the reasoning
and the one-line switch to `hydrate_strict()` are in "Deliberate deviation from Python" above. (2)
`sandboxes/docker.rs` still lacks the `validate_restorable_archive` override every other backend got, because
that file belongs to another agent this round. (3) `FileEntry::is_symlink` and `SandboxRunConfig::archive_limits`
are additive but breaking for struct-literal construction, and one-line edits land in files owned by other
agents (docker, events, six extension providers, three test files).
