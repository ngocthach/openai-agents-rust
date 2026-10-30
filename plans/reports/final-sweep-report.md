# Final sweep — symlink hydrate policy, docker resume guard, log target sweep

Date: 2026-10-30. Build dir `oars-build/sweep4` (3.0 GB). No git commands run.

## 1. Symlink policy in archive hydrate — strict, as Python

### Verification against Python (before any change)

- `sandbox/util/tar_utils.py:_validate_symlink_target` refuses a symlink member whose target is
  absolute (`absolute symlink target not allowed: …`) or whose lexical resolution climbs out of
  the archive (`symlink target escapes archive root: …`).
- `allow_external_symlink_targets=False` is passed at **every** hydrate/pre-restore site:
  `sandboxes/unix_local.py:1479` (restore pre-check) and `:1507` (hydrate), `sandboxes/docker.py:1502`,
  and `extensions/sandbox/{e2b:1619, modal:2062, daytona:1171, vercel:1451,1496, blaxel:754,
  cloudflare:1504}`. The permissive default of `validate_tarfile` has no hydrate caller.
- Venv case: Python does **not** round-trip `bin/python -> /usr/bin/python3`. `_rebase_symlink_target`
  (unix_local.py:190-215) only replaces a prefix that matches `roots`; an outward target is returned
  unchanged, and strict hydrate then refuses it. `tar_utils._simple_rebase_components` says the same
  for docker. So the port's "allow outward links" decision was a deviation, not parity.
- Python rewrites *in-root* absolute targets at **capture**: `unix_local._archive_workspace` (rebase +
  `_symlink_target_stays_under` containment proof) and `docker.persist_workspace` via
  `strip_tar_member_prefix(relativize_symlinks_under=root)`. The six remote providers capture with a
  plain remote `tar` and do **not** rebase — in Python an in-root absolute link breaks their round
  trip too.

### Changes (Python-equivalent combination)

| Area | Change |
|---|---|
| `src/sandbox/archive.rs` | `TarValidationOptions::hydrate()` is now `allow_external_symlink_targets: false`; `hydrate_strict()` removed (it was the same thing). This flips all eight hydrate sites at once, since each goes through `hydrate()` / `validate_hydrate_archive`. |
| `src/sandbox/archive.rs` | New `rebase_archive_symlink_targets(raw, roots)` — port of `_rebase_symlink_target` + `_symlink_target_stays_under` (40-hop budget, archive-established components, relative-link hops only), plus a tar re-emitter that rewrites only the link headers it proved (archive returned unchanged when nothing qualifies). Gzip in → gzip out. |
| `src/sandbox/session.rs` | Default `persist_workspace_once` body extracted into `pub async fn capture_workspace_tar(session)` so a backend can capture identically and post-process. |
| `src/sandbox/sandboxes/unix_local.rs` | New `persist_workspace_once` override: capture, then rebase against `roots = (root, canonicalize(root))` (Python's `(root, root.resolve())`, which is what makes macOS `/var` vs `/private/var` links work). Restore-time `rebase_absolute_symlinks` + `rebase_symlink_target` **deleted** — under strict hydrate an absolute in-root target can never reach extraction, so it was dead and misdocumented the policy. |
| `src/sandbox/sandboxes/docker.rs` | Same capture-side rebase against the container root, so a docker resume is not a regression under the strict policy (matches Python's `strip_tar_member_prefix` rebase). |
| `src/sandbox/mod.rs` | Re-exports `rebase_archive_symlink_targets` and `capture_workspace_tar`. |

Deviation worth recording: docker now uses the unix_local proof (`_symlink_target_stays_under`)
rather than Python's slightly stricter docker-only pair (`_simple_rebase_components` +
`_archive_establishes`, which additionally refuses `..`-bearing targets and directory leaves). The
unix_local proof is the one that resolves symlink hops, so every target it accepts is still provably
contained; porting two proofs for one behavior was not worth the duplication.

### Tests

- `src/sandbox/sandboxes/unix_local.rs`: `persist_rebases_a_link_into_the_workspace_so_a_restore_follows_the_new_root`
  (real `tar` capture → archived linkname is `../lib/real.so` → hydrate into a *different* session
  root → link resolves to the archived file) and
  `a_link_out_of_the_workspace_stays_absolute_and_is_refused_on_restore` (the venv case: archive keeps
  `/usr/bin/env`, hydrate refuses it, nothing extracted).
- `tests/sandbox_archive_tests.rs`: two new refusal rows (`bin/python -> /usr/bin/python3`;
  `escape -> ../outside`) asserting the reason and that the workspace stays empty; the happy-path
  hydrate now carries an in-archive climbing link (`sub/up -> ../sub/notes.txt`) in place of the
  outward link it used to assert.
- `src/sandbox/archive.rs` units: `the_hydrate_policy_refuses_absolute_and_escaping_symlink_targets`
  (now including the accepted `../sibling` case) and
  `capture_rebases_only_provably_contained_in_root_targets`.
- `tests/sandbox_tests.rs::an_archive_cannot_write_through_a_symlink`: the probe archive's link points
  at an absolute path outside the workspace, so the member-list pass now refuses it for the target
  before the descends-through-symlink pass runs. Assertion updated (that second reason is still
  covered by the `path through a symlink` row in `sandbox_archive_tests.rs`).

## 2. Docker `validate_restorable_archive`

`src/sandbox/sandboxes/docker.rs` now overrides it with `validate_hydrate_archive`, so
`restore_session_snapshot` refuses a bad archive before `clear_workspace_root_on_resume` empties the
container workspace. Test: `a_resume_refuses_an_archive_that_cannot_replace_the_workspace`
(`DockerSandboxSession::new` needs no daemon) — a symlink member with an absolute target is refused
with the member/reason context, a plain file archive is accepted.

## 3. Log target sweep

81 call sites across 26 files gained `target: crate::logger::LOG_TARGET` (the remaining matches
already carried `target:` on the following line; the two hits in `logger.rs` are the doc-comment
anti-example and a comment). Messages, levels and arguments are unchanged, and every redacting
helper in `logger.rs` was already on `LOG_TARGET`, so redaction behavior is untouched.

Files: `util.rs`, `mcp/{handler,http,sse}.rs`, `memory/{dapr_session,openai_responses_compaction_session}.rs`,
`models/openai_chatcompletions.rs`, `realtime/{debug,openai_model}.rs`, `run_internal/{mod,local_tools}.rs`,
`sandbox/{events,pty,runtime,memory/manager,sandboxes/docker}.rs`, `tracing/processors.rs`,
`voice/{pipeline,result}.rs`, `extensions/{tool_output_trimmer, models/anthropic,
sandbox/{daytona, e2b/mod, blaxel/{mounts,pty}, cloudflare/pty}}.rs`.

No dotted child target was introduced, and this is a correction to the task's premise, verified
against Python:

- `tracing/logger.py` (`openai.agents.tracing`) is imported only by `tracing/{create,spans,scope,traces}.py`.
  The Rust files that log under `src/tracing/` are `processors.rs` and `provider.rs`, whose Python
  counterparts import `..logger` → **`openai.agents`**. So no site qualified for a child target, and
  the verbose backend's existing dotted-child support (`is_sdk_target`) is already in place and tested.
- **Divergence the controller should know about:** Python's sandbox and extension modules use
  `logging.getLogger(__name__)` → `agents.sandbox.*` / `agents.extensions.*`, and
  `memory/openai_responses_compaction_session.py` uses `openai-agents.openai.compaction`. None of those
  is a child of `openai.agents`, so Python's own `enable_verbose_stdout_logging()` does **not** show
  them. About 35 of the 81 swept sites are in those modules, so Rust now surfaces under verbose
  logging what Python keeps quiet. This was the explicit request; reverting it means putting those
  modules back on their module-path target.

Test: `tests/logger_verbose_stdout.rs::an_sdk_modules_own_records_reach_the_verbose_backend` re-runs
the binary with the backend installed, calls
`models::openai_chatcompletions::items_to_messages` with an unknown item type, and asserts
`DEBUG openai.agents: Skipping item type custom_backend_item not supported by Chat Completions` on
the child's stdout.

## Shared-file edits

None. `src/lib.rs` and `Cargo.toml` were not touched (no rustfmt on `src/lib.rs`). `src/sandbox/mod.rs`
gained two re-export names.

## Breaking changes

- `TarValidationOptions::hydrate()` now refuses absolute/escaping symlink targets, and
  `hydrate_strict()` is gone. A workspace snapshot holding a link to the host no longer restores —
  the Python behavior.
- `SandboxSession` gains a public free function neighbour `capture_workspace_tar`; no implementor
  must change.

## Gates

`cargo clippy --all-features --all-targets -- -D warnings` clean.
`cargo test --workspace --all-features --no-fail-fast`: 1175 passed, 0 failed, 1 pre-existing ignored.
Changed files formatted with `rustfmt --edition 2024`.

## Open

- The remote providers (e2b, modal, daytona, vercel, blaxel, cloudflare) still capture with a plain
  remote `tar` and do not rebase in-root absolute links, so such a workspace cannot round-trip there.
  That is exactly Python's behavior for those backends; closing it would be a deliberate improvement
  beyond parity.
- `rebase_archive_symlink_targets` drops PAX records other than path/linkpath when it has to rewrite
  an archive (only on archives that contain a rebasable link). Python's rewriter drops the same two
  and keeps the rest; in practice the capture `tar` emits GNU long names, which are preserved. The
  same clone-the-header approach would re-emit a wrong size for a member whose true size lives only
  in a PAX `size` record (>8 GiB), reachable only when such a file shares an archive with a rebasable
  link.
- `rebase_symlink_target` with a workspace root of `/` matches nothing, where Python's `rstrip("/")`
  would match everything. Neither is a usable configuration; recorded as a behavioral delta.
- `docs/project-changelog.md` was not touched: its 0.4.0 section is already released and no other
  agent this round opened an Unreleased section. The user-visible item for release notes is the
  hydrate flip (a snapshot holding a link to the host no longer restores).

Checked while reviewing: `InstrumentedSession::persist_workspace` (`src/sandbox/events.rs:1093`)
delegates to `self.inner.persist_workspace()`, so a wrapped session reaches the backend's rebasing
`persist_workspace_once` rather than the non-rebasing default.

Status: DONE_WITH_CONCERNS
Summary: Hydrate validation is now Python-exact (absolute and escaping symlink targets refused at all
eight sites), with the in-root rebase moved to capture on unix_local and docker so snapshots still
round-trip; docker gained the missing `validate_restorable_archive` guard; 81 log sites moved onto the
SDK log target with a test proving a module's own debug line reaches the verbose backend.
Concerns/Blockers: One premise correction, documented above — Python's sandbox/extension modules log
under `agents.*`, not `openai.agents`, so routing them to `LOG_TARGET` makes Rust more verbose than
Python by design, not by parity.
