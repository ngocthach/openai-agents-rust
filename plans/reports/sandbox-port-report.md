# Sandbox port report

Port of the Python `agents.sandbox` package to Rust, inside `openai-agents-sdk` under the
`sandbox` feature. Source of truth: `/Users/ngocthach/myGit/openai-agents-python/src/agents/sandbox/`
(~28K lines) plus `docs/sandbox_agents.md` and `docs/sandbox/*.md`.

Phase 1 ported the core (agent, manifest, capabilities, sessions, runner). Phase 2 completed
parity for everything phase 1 had listed as not ported, except the cloud providers under
`extensions/sandbox` (a separate task, which builds on the mount types here). Phase 3 closed the
remaining gaps with two authorized core changes (plus their `lib.rs` re-exports) and two
sandbox-side ones.

Files created (nothing outside this list and the phase-3 core changes was touched):

- `src/sandbox/mod.rs`, `types.rs`, `errors.rs`, `workspace_paths.rs`, `util.rs`
- `src/sandbox/entries.rs`, `manifest.rs`, `session.rs`, `editor.rs`, `agent.rs`, `runtime.rs`
- `src/sandbox/mounts.rs`, `remote_mount_policy.rs`, `pty.rs`, `snapshot.rs`, `state.rs`
- `src/sandbox/capabilities/{mod,shell,filesystem,compaction,pty_tools,memory,skills}.rs`
- `src/sandbox/memory/{mod,config,prompts,storage,rollouts,manager}.rs`
- `src/sandbox/sandboxes/{mod,unix_local,docker}.rs`
- `src/sandbox/instructions/prompt.md`, `src/sandbox/memory/prompts/*.md` (verbatim copies)
- `tests/sandbox_tests.rs`, `examples/sandbox_local.rs`

Core files changed in phase 3 (authorized, additive):

- `src/error.rs` — the `AgentsError::Sandbox { code, message }` variant plus `sandbox()` and
  `sandbox_code()`.
- `src/result.rs` — `StreamCompletion`, `StreamCompletionHook`,
  `RunResultStreaming::{with_on_complete, set_on_complete, finish}`.
- `src/lib.rs` — re-exports for the two new result types.

## Verification

| Check | Result |
| --- | --- |
| `cargo clippy --features sandbox --all-targets -- -D warnings` | no diagnostics in any file owned here (the only remaining ones are in the concurrent `src/extensions/sandbox/**` work) |
| `cargo clippy --all-targets` (default features) | no sandbox diagnostics |
| `cargo test --features sandbox --test sandbox_tests` | 45 passed |
| `cargo test --features sandbox --lib sandbox` | 152 passed (incl. the provider modules) |
| `cargo test --features sandbox --doc sandbox` | 7 passed |
| `cargo test --lib` (default features) | 106 passed — the core changes are feature-independent |
| `cargo test --features sandbox --test runner_tests` | 27 passed — streaming is unchanged for existing callers |
| `cargo test --features sandbox --test {local_tools,sessions,tracing}_tests` | 6 / 12 / 1 passed |
| `rustfmt --edition 2024` on every new file | applied |

The two Docker tests ran for real on this machine (`docker info` and
`docker image inspect alpine:latest` both succeeded): container create/start/exec/write/read/rm,
manifest materialization, `-v` bind mounts with `:ro` enforcement, and PTY commands through
`docker exec` all pass. On a machine without a daemon, or without the image already present
locally, they print a skip notice and return — no image is ever pulled, so the default test run
needs no network.

Phases 1 and 2 touched no core file; per-turn preparation uses the existing
`RunConfig::call_model_input_filter` hook. Phase 3 added the three authorized core changes listed
above. `src/run.rs`, `src/run_internal/**`, `src/models/**` and `src/retry.rs` are still
untouched.

The `shell_quote` conflict reported at the end of phase 2 is resolved: the whole
`cargo test --features sandbox --lib sandbox` suite, provider modules included, now passes.

## What is ported

**Agent, preparation and the handoff graph** (`sandbox_agent.py`, `runtime.py`,
`runtime_agent_preparation.py`)

- `SandboxAgent<C>`: an `Agent<C>` plus `default_manifest`, `base_instructions`, `capabilities`,
  `run_as` and `sandbox_handoffs`. `Clone`/`Debug` are hand-written.
- `prepare(&CapabilityBinding) -> Agent<C>` reproduces `prepare_sandbox_agent`:
  required-capability validation, capability tool collection, `sampling_params` deep-merged into
  `ModelSettings::extra_args`, and the composed instructions (base prompt → `# Agent instructions`
  → `# Sandbox capability instructions` → `# Sandbox remote mount policy` → `# Filesystem` plus the
  rendered workspace layout, in Python's order). Instructions are `Instructions::Dynamic`, so an
  agent configured with dynamic instructions still resolves them per turn.
- `prepare_graph` prepares the whole sandbox handoff graph against one session. Each node keeps
  its own capabilities and `run_as`; targets are wired in through `Handoff::custom` with a
  `OnceCell`, so cyclic graphs work. Duplicate agent names are rejected, because `RunState`
  identifies the resumed agent by name.

**Per-turn preparation** (`runtime.prepare_agent`)

- `SandboxRunner` installs the capability `process_context` chain as a
  `RunConfig::call_model_input_filter`, which runs on every turn and for every agent, looking the
  capability set up by agent name. A filter the caller already configured still runs first.
  Agents outside the graph pass through unchanged.

**Manifest, entries and mounts** (`manifest.py`, `manifest_render.py`, `entries/**`,
`remote_mount_policy.py`, the portable part of `_mount_security.py`)

- `Manifest`: `version`, `root`, `entries`, `environment`, `users`, `groups`,
  `remote_mount_command_allowlist` (Python's default list), the credential-exposure
  acknowledgement API, `iter_entries`, `mount_targets`, `ephemeral_paths`, `host_path_grants`,
  `describe`/`render` with Python's depth (3) and 5000-char budget.
- `Entry`/`EntryKind`: `File`, `Dir`, `LocalFile`, `LocalDir`, `GitRepo`, `Mount`, each with
  `description`, `permissions`, `group` and `ephemeral` (always true for mounts).
- `Mount` + the `MountStrategy` trait (`validate`, `credential_authority`,
  `supports_native_snapshot_detach`, `docker_create_args`, `host_path_grant`, `activate`,
  `deactivate`, `teardown_for_snapshot`, `restore_after_snapshot`) and the `MountPattern` trait
  (`apply`/`unapply` with a `MountPatternConfig`). Built-in strategies:
  - `BindMountStrategy` — `docker -v <src>:<dst>[:ro]` on Docker, a workspace link on Unix-local.
  - `DockerVolumeMountStrategy` — `--mount type=volume,volume-driver=…`.
  - `InContainerMountStrategy` — runs a `MountPattern` inside the sandbox; this is the extension
    point the cloud providers will implement (FUSE, `mount-s3`, `rclone`).
- `Manifest::apply` skips mounts the container runtime already attached (`attaches_mounts_at_create`),
  activates in-container strategies, and `apply_ephemeral` re-applies only the ephemeral entries
  after a resume. `unmount_all` detaches on stop.
- `build_remote_mount_policy_instructions` is a verbatim port of the policy block.

**Credential exposure.** `MountPatternConfig::config_text` is the only place a mount credential
lives; it is redacted in `Debug`, excluded from `without_config_text()` teardown copies, and never
serialized. `Manifest::authorize_mount_activation` refuses to activate an in-container mount with
`MountScoped` or `Broad` authority unless the application acknowledged that exact mount path, and
the error text names only the path and the authority level.

**PTY sessions** (`session/pty_types.py`, `session/pty_output.py`, the backend `pty_*` methods)

- `PtyManager` owns the pty, the output pump, process ids and pruning; a backend only describes
  how to spawn (`PtySpawn`). Built on `portable-pty` 0.9, no `unsafe`.
- Python's contract is reproduced: yield defaults of 10 000 ms / 250 ms clamped to 250–30 000 with
  a 5 000 ms floor for empty writes, process ids in `[1000, 100000)`, at most 64 live processes
  with the 8 most recent protected from pruning, UTF-8 carry across reads, 4-bytes-per-token
  truncation, and the `Chunk ID` / `Wall time` / `Process exited with code` / `Process running with
  session ID` / `Output:` response layout.
- `exec_command` and `write_stdin` are exposed by the shell capability whenever the session
  supports a PTY. `tty=false` still polls; a non-empty `write_stdin` against such a process returns
  Python's "stdin is not available for this process" message.
- Docker PTY commands run through `docker exec -i[-t]`. The host-side process is only a proxy, so
  the command records its container-side pid in a file and the session kills it inside the
  container on `stop`, `shutdown` and `pty_terminate_all` — otherwise container processes leak.

**Snapshots and session state** (`snapshot.py`, `snapshot_defaults.py`,
`session/sandbox_session_state.py`, `session/snapshot_lifecycle.py`) — see the design note below.

**Memory** (`memory/**`, `capabilities/memory.py`, the memory part of `config.py`)

- `MemoryLayoutConfig`, `MemoryReadConfig`, `MemoryGenerateConfig` (the models are `AgentModel`,
  so a test can inject a scripted model), `SandboxMemoryStorage`, the rollout payload builder with
  Python's include/exclude item-type sets, and `SandboxMemoryGenerationManager`.
- The four prompt templates are verbatim copies loaded with `include_str!`.
- The `Memory` capability reads `memories/memory_summary.md` out of the live workspace through the
  new `Capability::bound_instructions` hook and renders the read prompt; it declares
  `filesystem` + `shell` (or just `shell` without live updates) as required capabilities.
- Phase 1 is a plain `Agent` with a structured output type; phase 2 is a `SandboxAgent` bound to
  the same session with only filesystem + shell. Neither carries the memory capability, so a flush
  can never trigger another flush. Both nested runs inherit the outer run's `model_provider`.

**Skills** (`capabilities/skills.py`)

- `Skill` (`SKILL.md` body plus `scripts/`, `references/`, `assets/`), `Skills` with inline,
  entry-backed and lazy sources, the `.agents` discovery root, the `load_skill` tool for lazy
  sources, `LazySkillSource` + `LocalDirLazySkillSource` (which reads descriptions from the
  `SKILL.md` front matter and rejects paths escaping the source), and the skills instruction
  section.

**Sessions, clients and the runner**

- `SandboxSession`: lifecycle, `exec`, file I/O, `ls`/`mkdir`/`rm`/`mv`, `normalize_path`,
  `persist_workspace`/`hydrate_workspace`, the PTY trio, `session_state`, `snapshot`, and the
  mount hooks. `SandboxClient`: `create`, `delete`, `resume`.
- `SandboxRunConfig { client, session, session_state, manifest, cwd, rollout_id }` and
  `SandboxRunner::{prepare, run, run_with_options, run_to_outcome, run_streamed_with_options}`.

### Snapshot and session-state design

A paused run has two halves of state. `RunState` already serializes the agent loop; this port adds
the sandbox half.

- `Snapshot` trait with `LocalSnapshot` (atomic temp-file rename, ids restricted to a single path
  segment) and `NoopSnapshot`, plus the serializable `SnapshotSpec`, the platform default snapshot
  directory, and stale-snapshot cleanup.
- `SandboxSessionState` is a small JSON document the caller persists next to the `RunState`:
  backend, session id, workspace root, provider id, snapshot spec and id, exposed ports, and a
  SHA-256 **digest of the manifest topology**. It carries no manifest contents, no host paths, no
  mount credentials and no environment values.
- On resume the caller supplies the same trusted `Manifest` again, built by application code;
  `verify_manifest` checks the digest. A tampered state therefore cannot make the SDK read new
  host paths or mount new storage. This is Python's
  `sanitize_manifest_mount_authority` / `rebind_manifest_mount_authority` pair, without
  serializing the manifest itself.
- The digest hashes entry paths **relative to the root** and not the root itself, because a
  backend that owns its workspace replaces the root with a real temp directory at create time; the
  root travels in `workspace_root` instead.
- `may_remove_workspace_root()` treats a root as deletable only when the flag is set *and* the
  path still looks like an SDK-created temp workspace (`openai-agents-sandbox-` under
  `temp_dir()`), so a hand-edited state cannot turn teardown into `rm -rf` on a home directory.
- Lifecycle: a run that ends with approval interruptions is **suspended** — PTYs are terminated,
  the workspace is captured into the snapshot, and neither the workspace nor the container is
  released; `SandboxRunOutcome::sandbox_state` carries the state. A run that completes flushes
  memory and then deletes. Docker resume reattaches when `docker container inspect` finds the
  container, and otherwise creates a replacement and hydrates it from the snapshot.

### Path traversal and mount access

Four layers, all covered by tests:

1. Lexical: `resolve_workspace_path` rejects `..`, Windows-absolute paths and absolute paths
   outside `Manifest::root`.
2. Symlink: `UnixLocalSandboxSession::host_path_for` canonicalizes the deepest existing ancestor
   of **both** the candidate and the root, so a planted link (`ln -s /etc escaped`) cannot be read
   or written through, and a caller-supplied root behind a symlink still works.
3. Mount grants: a bind mount deliberately points outside the root, so its source is an explicit
   host path grant. Writes through a read-only grant are refused by the SDK file APIs (and by
   Docker, via `:ro`). On Unix-local a shell command the model starts can still write through the
   link, because that backend adds no OS-level confinement — documented on the strategy.
4. Manifest: entry keys, `Dir` child names, mount paths and skill names are all validated.

Snapshots exclude every ephemeral path (`tar --exclude`), so a mounted bucket is never archived.

## What is not ported, and why

| Area | Python source | Reason |
| --- | --- | --- |
| Cloud provider backends: Blaxel, Cloudflare, Daytona, E2B, Modal, Vercel | `extensions/sandbox/*` (~13.6K lines) | A separate task. They implement `SandboxClient`/`SandboxSession`, and their storage mounts implement `MountPattern` behind `InContainerMountStrategy`. |
| The concrete remote storage mounts (S3, S3 Files, GCS, Azure Blob, R2, Box) and the FUSE / `mountpoint` / `rclone` / `s3files` patterns | `entries/mounts/providers/**`, `entries/mounts/patterns.py` (~1.5K lines) | Each is provider-specific (credentials, CLI tools inside the image) and belongs with the provider above. The traits and the credential-exposure policy they plug into are ported. |
| The defence-in-depth half of `_mount_security.py`: untrusted-dict provenance checks, per-field authority classification, rclone argument sanitization, error redaction decorators | `_mount_security.py` (2284 lines) | Most of it guards Python's "manifest from an untrusted dict" path, which has no analogue here: a Rust manifest is built in typed application code. The parts that still apply — the credential-exposure boundary, strategy/mount validation, and keeping credentials out of `Debug`, errors and serialized state — are ported. |
| Exposed ports and `ExposedPortEndpoint` | `types.py`, `session/*` | Needs per-provider port mapping; the field is reserved in `SandboxSessionState`. |
| Archive extraction limits and tar utilities | `util/tar_utils.py`, `session/archive_*.py` | `persist_workspace`/`hydrate_workspace` shell out to `tar`; the SDK-level resource limits are a separate hardening concern. |
| Runtime helper scripts, `same_file`, `read_bounded` | `session/runtime_helpers.py`, `base_sandbox_session.py` | Peripheral; the capabilities and the file APIs do not need them. |
| Tracing spans `sandbox.prepare_agent` / `sandbox.cleanup` | `runtime.py` | `SandboxRunner` sits outside the run loop, so the spans would not nest correctly. |

### Fidelity decisions worth knowing

- **Tool surface.** Python's shell capability exposes `exec_command` (+ PTY-only `write_stdin`)
  and its apply_patch is a `CustomTool`. Per the phase-1 task this port uses the crate's native
  `ShellTool` and `ApplyPatchTool`, so the model sees `shell` and `apply_patch`. Phase 2 added
  Python's `exec_command` and `write_stdin` alongside `shell`, because `write_stdin` needs a
  session id that a `shell_call` cannot return. `Shell::with_pty(false)` turns the pair off, and
  the instruction fragment names exactly the tools that are exposed.
- **`Capability` is stateless.** The binding (session, `run_as`, scope) is passed to `tools()` and
  `bound_instructions()` per run, so Python's clone-and-bind (`Capability.clone`, `bind`,
  `bind_run_as`, `bind_workspace_scope`) is unnecessary.
- **`as_memory()` instead of downcasting.** The runner finds the memory manager through a trait
  method rather than `Any`.
- **`SandboxRunConfig::options`** has no equivalent: Rust clients take their options at
  construction (`DockerSandboxClientOptions`).
- **macOS `sandbox-exec`.** Python's Unix-local backend wraps commands in `sandbox-exec` on macOS.
  This port does not; the backend is documented as providing no OS-level confinement on any
  platform.
- **Memory flush timing.** The runner flushes only on a final close of a session it owns. A
  caller-supplied session is never closed by the runner, so the caller flushes it explicitly
  through `Memory::manager()`.
- **`Default` matches `new()`.** `UnixLocalSandboxClient::default()` inherits the host
  environment, `Filesystem::default()` includes `view_image`, and `Shell::default()` enables the
  PTY tools, so none of those `Default` impls are derived.

## Public API summary

```text
openai_agents_sdk::sandbox
  SandboxAgent<C>            new / from_agent / with_instructions / with_model(_instance) /
                             with_tool / with_default_manifest / with_base_instructions /
                             with_capabilities / with_capability / with_run_as /
                             with_sandbox_handoff / with_sandbox_handoff_to / map_agent /
                             resolved_manifest /
                             resolved_model_name / memory_capability / prepare_input /
                             prepare / prepare_graph
  PreparedSandboxGraph<C>    starting_agent / agents_by_name / process_context
  SandboxRunner              prepare / run / run_with_options / run_to_outcome /
                             run_streamed / run_streamed_with_options -> SandboxStreamedRun
  SandboxStreamedRun<C>      streaming / sandbox_state / take_guard / restore_guard / finish
  SandboxRunConfig           with_client / with_session / session_state / manifest / cwd /
                             rollout_id
  SandboxRunOutcome<C>       result / sandbox_state / memory_manager
  SandboxSessionGuard        session / owns_session / close / suspend
  Manifest                   with_root / with_entry / with_env / with_user / with_group /
                             with_remote_mount_command_allowlist /
                             with_in_container_mount_credential_exposure_acknowledged(_broad) /
                             validated_entries / iter_entries / mount_targets / ephemeral_paths /
                             host_path_grants / authorize_mount_activation / apply /
                             apply_ephemeral / unmount_all / describe / render
  Entry / EntryKind          file / text_file / dir / local_file / local_dir / git_repo / mount +
                             with_description / with_permissions / with_group / with_child /
                             with_ephemeral / with_git_host / with_git_subpath / as_mount
  Mount / MountStrategy / MountPattern / MountPatternConfig / MountCredentialAuthority /
  HostPathGrant / BindMountStrategy / DockerVolumeMountStrategy / InContainerMountStrategy
  SandboxSession (trait)     start / stop / shutdown / running / exec / read_file / write_file /
                             ls / mkdir / rm / mv / persist_workspace / hydrate_workspace /
                             pty_exec_start / pty_write_stdin / pty_terminate_all /
                             supports_pty / attaches_mounts_at_create /
                             authorize_mount_activation / normalize_path / session_state /
                             snapshot / describe / manifest / workspace_root
  SandboxClient (trait)      backend_id / create / delete / resume
  persist_session_snapshot / restore_session_snapshot
  Snapshot (trait) / LocalSnapshot / NoopSnapshot / SnapshotSpec / resolve_snapshot /
  default_local_snapshot_base_dir / resolve_default_local_snapshot_spec /
  cleanup_stale_default_local_snapshots
  SandboxSessionState        new / with_* (incl. with_rollout_id) / to_json_string /
                             from_json_str / verify_manifest / may_remove_workspace_root
                             (+ manifest_digest)
  SandboxHandoffTarget<C>    Agent / Named + name / agent
  PtyManager / PtyExecRequest / PtyWriteRequest / PtyExecUpdate
  ExecRequest / ExecShell / ExecResult / FileEntry / Permissions / User / Group / file_mode
  SandboxWorkspaceScope / SandboxError / ErrorCode / SandboxResult / SandboxEditor
  build_remote_mount_policy_instructions
  capabilities::{Capability<C>, CapabilityBinding, Capabilities, Shell, Filesystem, Compaction,
                 Memory, Skills, Skill, SkillMetadata, LazySkillSource,
                 LocalDirLazySkillSource, CompactionPolicy, context_window_for_model,
                 ViewImageArgs, ExecCommandArgs, WriteStdinArgs, LoadSkillArgs}
  memory::{MemoryLayoutConfig, MemoryReadConfig, MemoryGenerateConfig, SandboxMemoryStorage,
           SandboxMemoryGenerationManager, RolloutExtractionArtifacts, RolloutTerminalMetadata,
           TerminalState, PhaseTwoInputSelection, PhaseTwoSelectionItem}
  sandboxes::unix_local::{UnixLocalSandboxClient, UnixLocalSandboxSession}
  sandboxes::docker::{DockerSandboxClient, DockerSandboxClientOptions, DockerSandboxSession,
                      DEFAULT_PYTHON_SANDBOX_IMAGE}
  util::{base64_encode, image_media_type, deep_merge, truncate_text_by_tokens}
```

## Review fixes applied after the first phase-2 pass

A review of the finished port found five issues that the happy-path tests missed. All are fixed
and covered:

1. **Teardown never unmounted.** The runner tears down through `close()` → `delete()` →
   `shutdown()`, and only `stop()` called `unmount_all`. An in-container mount materializes a real
   filesystem inside the workspace root, so `remove_dir_all(root)` would have recursed into live
   remote storage. `shutdown` now detaches every mount first and aborts workspace removal when a
   detach fails, as Python's `UnixLocalSandboxClient.delete` does. Covered by
   `teardown_detaches_mounts_before_removing_the_workspace` and
   `a_failed_unmount_keeps_the_workspace`.
2. **The memory manager was per capability, not per session.** `flush` latches `stopped` forever,
   so a second run of the same `SandboxAgent` silently dropped its rollout, and concurrent runs
   shared one rollout map. `Memory` now holds configuration only; `SandboxRunner::prepare` builds
   one manager per session and exposes it on `PreparedSandboxRun` and `SandboxRunOutcome` so a
   caller-owned or suspended session can still be flushed. Covered by
   `one_agent_generates_memory_for_every_session_it_runs_in` and
   `a_suspended_run_hands_back_its_memory_manager`.
3. **Resuming inside a handoff target did not work.** `run_internal::find_agent` locates the
   resumed agent by walking `Handoff.agent`, which is `None` for a `Handoff::custom`. The graph is
   now prepared in post-order and targets are wired in by value with `Handoff::new`, so a run that
   pauses inside a handoff target resumes correctly. Cyclic graphs are no longer expressible and
   are rejected with the loop they close — a by-value builder cannot express `A -> B -> A` anyway,
   and the only workaround would trip the duplicate-name check. Covered by
   `resumes_a_run_that_paused_inside_a_handoff_destination` and
   `rejects_cyclic_sandbox_handoff_graphs`.
4. **Resuming with nothing restored assumed the old workspace survived.** True for a suspended
   Unix-local directory, false for a replacement container or an evicted temp root. Both backends
   now check whether the workspace still holds anything and materialize the full manifest when it
   does not, instead of handing the model an empty workspace.
5. **PTY tail loss and a lost wakeup.** `collect` treated child exit as "done" and slept 20 ms for
   the pump, which drops the tail under load. The pump thread now sets an EOF flag and a process
   counts as done only once it has exited *and* EOF has been seen; the `Notified` future is armed
   before the buffer and done checks, so a wakeup in that gap is no longer lost. Pruned and
   finished Docker PTY processes now also run their container-side `kill`, which previously leaked.

## Phase 3: the authorized core changes

1. **`AgentsError::Sandbox { code, message }`** (`src/error.rs`). Sandbox failures no longer
   flatten into `UserError` / `Other("[code] …")`: `From<SandboxError>` now produces this variant,
   `Display` renders `[exec_timeout] Command … timed out`, and `AgentsError::sandbox_code()`
   returns the stable code so callers can branch on the cause. The code is a plain `String`, so
   the variant needs no feature gate; with the `sandbox` feature on it is always one of
   [`sandbox::ErrorCode::as_str`]. Covered by `errors::tests::{converts_into_the_sandbox_error_variant,
   round_trips_the_code_through_the_agents_error}` and
   `sandbox_errors_carry_a_machine_readable_code`.
2. **An async completion hook on `RunResultStreaming`** (`src/result.rs`), purely additive:
   `with_on_complete` / `set_on_complete` install an
   `Arc<dyn Fn(StreamCompletion) -> BoxFuture<'static, ()>>` that fires once when the stream ends,
   and `finish()` drains the stream and fires it for a caller that stopped early. It fires on the
   error path too, *before* the error is yielded, because nothing obliges a consumer to poll again
   after one — the usual `while let Some(ev) = next().await { ev?; }` returns immediately, and
   without that ordering a failed streamed run would leave its workspace, container and PTYs
   behind. `a_streamed_run_that_fails_still_releases_the_sandbox` covers it, and was confirmed to
   fail against the pre-fix ordering.
   `StreamCompletion` carries an owned snapshot of the run (`succeeded`, `interruptions`, `input`,
   `new_items`, `final_output`) rather than the `RunResult`, which is not `Clone`, so the hook can
   outlive the stream. Existing callers are unaffected — `runner_tests` passes untouched.
   `SandboxRunner::run_streamed` / `run_streamed_with_options` now return a
   `SandboxStreamedRun<C>` whose session is recorded, flushed and released (or suspended)
   automatically when the stream finishes; `take_guard`/`restore_guard` hand the lifecycle back
   for manual control. A failed run is not recorded as a rollout segment, since an empty segment
   would trigger a phase-1 model call on nothing.
3. **Memory on the streaming path, and one rollout per suspend.** The streaming completion hook
   builds the rollout payload from the `StreamCompletion` snapshot and appends it before releasing
   the session, so streamed runs now contribute memory exactly like buffered ones. The rollout id
   is resolved once in `SandboxRunner::prepare` (config → resumed state → generated) and stored in
   `SandboxSessionState::rollout_id`, so the two halves of a paused run append to the same
   rollout. Covered by `a_streamed_run_that_pauses_suspends_the_sandbox` and
   `a_suspended_run_keeps_both_halves_in_one_rollout`.
4. **Cyclic sandbox handoff graphs are supported.** A `SandboxHandoffTarget` is either an owned
   `Agent` or a `Named` reference resolved against the rest of the graph, so `A -> B -> A` is
   expressible with `with_sandbox_handoff` plus `with_sandbox_handoff_to("A")` — a by-value
   builder alone cannot express a loop.

   The prepared graph is a **spanning tree with lazy back-edges**. `run_internal::find_agent`
   locates a resumed agent by walking `Handoff.agent`, so every node must be reachable through
   owning handoffs; meanwhile a cycle of strong `Arc<Agent>` handoffs would be a reference cycle
   that leaks the prepared agents and the `Arc<dyn SandboxSession>` their tools hold. So the edge
   that first discovers a node owns it (`Handoff::new`, strong — this is the spanning tree, and
   it reaches every node), and every cycle-closing edge uses `Handoff::custom` resolving a
   `Weak<Agent<C>>` from a `OnceLock` filled after preparation. The graph keeps strong references
   to all prepared agents in `prepared_by_name`, so back-edges always upgrade while the graph is
   alive and everything drops together when it is not. Named targets that are not in the graph are
   rejected. Covered by `supports_cyclic_sandbox_handoff_graphs` (a four-hop loop that then runs a
   shell call) and `rejects_named_handoffs_that_are_not_in_the_graph`.

## Requested core changes

All of them are now resolved. Phase 1 requested five: the controller applied two (the
`[[example]]` entry and `base64` under the `sandbox` feature — the hand-rolled encoder is gone),
phase 2 solved two inside `src/sandbox` (per-turn preparation through
`call_model_input_filter`, and resume through `SandboxSessionState`), and phase 3 landed the last
one (`AgentsError::Sandbox`) together with the streaming teardown hook. Nothing is outstanding.

## Known gaps and open questions

- **`PtyManager::start` / `write_stdin` return `(PtyExecUpdate, Vec<String>)`.** The second
  element holds the sandbox-side cleanup commands of processes that finished or were pruned; a
  backend whose PTY process only proxies remote work (Docker, and the cloud providers) must run
  them or it leaks remote processes. Provider backends that only need the update can discard it.
- **Memory generation runs inline on the closing path**, so a slow phase-2 run delays the
  caller's `run()` return. Python runs it on a background worker tied to the session; moving it to
  a detached task would need somewhere to await it before process exit.
- **A streamed run abandoned without draining never releases its session.** The hook fires when
  the stream ends — including on the error path — and from `RunResultStreaming::finish`, so a
  caller that cancels mid-stream should call `finish()`. `SandboxStreamedRun::take_guard` is the
  manual escape hatch.
- **`run_streamed_with_options` returns `SandboxStreamedRun<C>`**, not the
  `(RunResultStreaming<C>, SandboxSessionGuard)` tuple phase 2 committed. A deliberate breaking
  change to the sandbox API, so that the session teardown is automatic; the provider agent should
  know.
- **`rustfmt src/lib.rs` formats the whole crate.** rustfmt follows `mod` declarations, so an
  earlier write-mode run on `src/lib.rs` reformatted every reachable module in place, including
  `src/extensions/**`, `src/models/**` and `src/retry.rs`. The `lib.rs` change here is a
  three-line `pub use` formatted by hand; `error.rs` and `result.rs` have no child modules and are
  safe to format directly. Worth checking the other agents' files for unrequested churn.
