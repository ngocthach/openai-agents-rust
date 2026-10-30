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
| `cargo clippy --features sandbox,encrypt --all-targets -- -D warnings` | clean, whole workspace |
| `cargo clippy --all-targets` (default features) | no sandbox diagnostics |
| `cargo test --features sandbox,encrypt --test sandbox_tests` | 65 passed |
| `cargo test --features sandbox,encrypt --lib` | 424 passed (incl. every provider module) |
| `cargo test --features sandbox,encrypt --doc sandbox` | 12 passed |
| `cargo test --lib` (default features) | 150 passed — the core changes are feature-independent |
| `cargo test --features sandbox,encrypt --test runner_tests` | 27 passed — streaming is unchanged for existing callers |
| `cargo test --features sandbox,encrypt --test {local_tools,sessions,tracing}_tests` | 6 / 23 / 1 passed |
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

## Phase 4: the mount entry layer

`entries/mounts/{base,patterns,_redaction}.py` and `entries/mounts/providers/*` are now ported.
`entries/artifacts.py` needed nothing: `Dir`, `File`, `LocalFile`, `LocalDir` and `GitRepo` were
already covered by `Entry` in phase 1.

### What was added

- **`src/sandbox/mount_patterns.rs`** — the three mount tools the core was missing:
  `FuseMountPattern` (`blobfuse2`, including the YAML configuration generator and both cache
  components), `MountpointMountPattern` (`mount-s3`) and `S3FilesMountPattern`
  (`mount -t s3files`). Plus the `_redaction.py` helpers: `url_contains_inline_authority`,
  `inline_url_authority_values` and `redact_sensitive_values`.
- **`src/sandbox/mount_entries.rs`** — the `ConfiguredMount` trait (Python's `_ConfiguredMount`)
  and the six storage types: `S3Mount`, `S3FilesMount`, `GcsMount`, `R2Mount`, `AzureBlobMount`
  and `BoxMount`, each with its rclone section, its Docker volume driver options and its
  credential-authority classification. `ConfiguredMountStrategy` is the single strategy that
  attaches any of them, either in-container through a `BuiltinMountPattern` or through a Docker
  volume driver.

### Design notes

- **The configuration is derived per activation, never stored.** Python's
  `build_in_container_mount_config` is an `async` method on the mount entry, called by the
  strategy with the live session. Rust's `MountStrategy` holds its configuration, so a naive port
  would bake `read_only` and the session-derived rclone remote name in at declaration time;
  `Mount::with_read_only(false)` applied afterwards would then disagree with the mount command.
  `ConfiguredMountStrategy` instead calls `spec.build_in_container_config(session.id(), …,
  mount.read_only, …)` inside `activate`, `deactivate` and `docker_create_args`.
- **The rclone remote name needs no new machinery.** A provider renders its own
  `[remote_name]` header into `config_text`, which is exactly what the existing
  `RcloneMountPattern` expects: with no `config_file_path` it writes that text verbatim, and with
  one `supplement_rclone_config_text` drops the generated header and merges the body into the
  caller's matching section. Both paths already match Python.
- **`MountPatternConfig` gained `secrets`.** `mount-s3` needs three separate credential values,
  not one configuration document. They are redacted in `Debug`, dropped by
  `without_config_text()`, and `InContainerMountStrategy::credential_authority` now tests
  `has_credentials()` so a mountpoint mount cannot skip the exact-path acknowledgement.
- **Mount patterns are a closed enum.** `BuiltinMountPattern` matches Python, which also accepts
  only its four built-in pattern classes, and removes the need to downcast a `dyn MountPattern`.
- **Credential authority follows Python's capability table.** A broad field — an Azure managed
  identity, a GCS service-account *file*, a Box JWT configuration file, a caller-placed rclone
  configuration, a free-form `s3files` option list — reports `Broad`; a mount-scoped credential or
  a URL with inline authority reports `MountScoped`. A Docker volume mount reports `None`: the
  credential goes to the container runtime on the host, not into the sandbox.

### Security work beyond a literal translation

- **Line-break injection.** Every value rendered into an rclone INI section, a blobfuse YAML
  document or a shell export file is rejected if it contains `\n` or `\r`, and every `mount -o`
  value is additionally rejected if it contains `,`. A bucket prefix like `x\n[other]\nkey = …`
  would otherwise append a remote section the caller never wrote.
- **Docker `--mount` is CSV-parsed.** Verified on this machine against `alpine:latest`:
  `--mount 'type=volume,dst=/mnt/x,volume-driver=local,volume-opt=o=a,b=c'` is refused with
  `unknown option 'b' in 'b=c'`, while the same value RFC 4180-quoted gets past the flag parser
  and reaches the driver. A GCS `service_account_credentials` JSON value — commas and quotes —
  would therefore have been read as several fields, inventing or corrupting mount settings.
  `docker_volume_mount_args` now quotes each field, which also fixes the pre-existing
  `DockerVolumeMountStrategy` that shared the unquoted builder. Neither probe created a container
  or a volume.
- **Credentials never reach the host process table.** Following Python, `mount-s3` credentials go
  to a `0600` environment file that the mount command sources; they are not passed through
  `ExecRequest::env`, which on the Docker backend becomes `docker exec -e K=V` and is visible in
  `ps`. The same applies to an endpoint URL carrying inline authority, which is passed as
  `${OPENAI_AGENTS_MOUNT_ENDPOINT_URL}`.
- **Scratch directories are excluded from every archive.** `workspace_archive_excludes` now also
  excludes `.sandbox-blobfuse-config`, `.sandbox-blobfuse-cache` and `.sandbox-mountpoint-env`.
  Without that a snapshot would capture the staged credentials, and the blobfuse block cache
  defaults to 50 GB.
- **Credentials are removed with the mount.** Each pattern deletes the file it staged on
  `unapply`, and on a failed mount, so a credential never outlives the attempt that used it.
- **Every credential-bearing type hand-writes `Debug`** and prints `<redacted>`.

### Provider wiring

| Provider | How it reaches the new mounts |
| --- | --- |
| Daytona, E2B | `DaytonaCloudBucketMountStrategy::for_mount` / `E2BCloudBucketMountStrategy::for_mount`. These wrap the generic in-container rclone path, exactly as Python's `_delegate()` does, so they support all four rclone-backed storage types. The strategies gained an optional `spec` field; the pre-existing `new(remote_name, remote_path, config_text)` constructor still works. |
| Cloudflare | `CloudflareBucketMountConfig::from_configured_mount` / `CloudflareBucketMountStrategy::for_mount`, mirroring `_build_cloudflare_bucket_mount_config`: S3, R2 and HMAC-authenticated GCS, with the same rejections (no S3 session token, no native GCS credential, no half-supplied key pair). |
| Blaxel | `BlaxelCloudBucketMountConfig::from_configured_mount` / `BlaxelCloudBucketMountStrategy::for_mount`, mirroring `_build_mount_config`: `s3fs` for S3, R2 and HMAC GCS, `gcsfuse` for native GCS. |
| Modal | `ModalCloudBucketMountConfig::from_configured_mount` / `ModalCloudBucketMountStrategy::for_mount`, mirroring `_build_modal_cloud_bucket_mount_config`, including the `GOOGLE_ACCESS_KEY_ID` / `GOOGLE_ACCESS_KEY_SECRET` pair for HMAC GCS. |
| Vercel | `VercelCloudBucketMountStrategy::for_mount`, mirroring `_validate_s3_mount`: Vercel's images ship `mount-s3`, so only an `S3Mount` fits and anything else is refused. |

The provider files were touched only for that wiring, plus one expectation in
`blaxel/sandbox.rs` that asserts the exact `tar --exclude` list and had to learn the three new
scratch directories.

### Review fixes applied after the first phase-4 pass

- **The credential boundary was not enforced on this strategy.** `ConfiguredMountStrategy::activate`
  relied on `Manifest::apply_entries` calling `authorize_mount_activation` for it, which left the
  snapshot re-attach path (`persist_session_snapshot` → `restore_after_snapshot`) ungated: a
  snapshot could re-stage a credential at a path the manifest never acknowledged. The call is now
  at the top of `activate`, which `restore_after_snapshot` delegates to, matching what the Daytona,
  E2B and Vercel strategies already did. `a_configured_mount_cannot_activate_at_an_unacknowledged_path`
  covers both directions and shows the gate runs *before* the tool probe.
- **Three rows of Python's capability table had no equivalent.**
  - `enables_broad_credential_discovery`: `blobfuse2` falls back to the ambient managed identity
    and the `s3files` helper authenticates from the sandbox's AWS environment, so both reach
    authority the mount never names. Azure + `fuse` and every `S3FilesMount` now report `Broad`
    unconditionally, as Python requires a broad acknowledgement for them unconditionally.
  - `required_any_fields`: a Box mount with no token, access token, JWT credentials or
    configuration file falls into rclone's interactive OAuth flow, which cannot complete in a
    sandbox. It is now rejected.
  - `_invalid_in_container_authority_value_fields` and the credential-set requirements: a blank
    credential value, and a half-supplied key pair, are rejected on every mount type. `S3Mount`
    previously accepted `secret_access_key` with no `access_key_id` and silently mounted unsigned.
- **`docker_create_args` swallowed the real error.** It cannot return one, so a rejected driver
  value became `None`, Docker fell through to `activate`, and the caller was told the backend does
  not support docker-volume mounts. The driver options are now also built in `validate`, where the
  real error surfaces.
- **Pretty-printed JSON credentials were rejected.** A Google service-account key and a Box JWT
  configuration are downloaded as indented JSON, so the line-break check refused the normal way of
  supplying them. Those two fields are now compacted with `serde_json` first, which keeps the
  injection check (a value that is not JSON and has a line break is still refused).
- **`read_only` on the Cloudflare, Blaxel and Modal `for_mount` paths.** Those backends take
  `read_only` from their own configuration, not from the `Mount`, so `from_configured_mount`
  returns a read-only configuration and the caller chains `with_read_only(false)` to allow writes.
  Defaulting the other way would silently widen a mount; this direction fails safe. Documented on
  each constructor.
- **Handoff tool names.** The concurrent core work lowercased them
  (`transfer_to_Reviewer` → `transfer_to_reviewer`); the three scripted-model tests here were
  updated to match.

## Phase 5: closing the parity-audit rows

Working through the `src/agents/sandbox/**` rows of
`plans/reports/parity-audit-261030-0505-python-to-rust-report.md`.

### Closed

| Audit row | What was added |
| --- | --- |
| `errors.py`: 5 unmapped codes and the structured fields | `ErrorCode::{InvalidCompressionScheme, WorkspaceArchiveReadError, WorkspaceArchiveWriteError, LocalChecksumError, GitCopyError}`, plus `SandboxOp` (Python's `OpName`) and `SandboxError::{op, context, retryable}` with `with_op` / `with_context` / `with_retryable` / `caused_by`. `caused_by` inherits the cause's retry classification, as Python's `__post_init__` does. |
| `apply_patch.py`: no no-clobber on create (**partial**) | `SandboxEditor::create_file` now fails with "apply_patch cannot create X because it already exists" instead of silently overwriting. `apply_patch` has a separate update mode, so a create against an existing path is a mistake. |
| `view_image.py`: no size cap, extension-based MIME | `MAX_IMAGE_BYTES` (10 MB) is enforced, and `detect_image_media_type` sniffs PNG/JPEG/GIF/WebP/BMP/TIFF magic bytes and SVG content, refusing a file that is not an image rather than labelling it `image/png`. |
| `unix_local.py`: 4-entry environment allowlist (**partial**) | `HOST_ENVIRONMENT_ALLOWLIST` now carries Python's 22 entries, so TLS CA settings and the locale survive a non-inheriting session, and `HOME` is pinned to the workspace root as Python does. `with_host_environment_allowlist` overrides it. |
| `sandboxes/docker.py`: no FUSE device or `SYS_ADMIN` (**partial**) | `MountStrategy::container_privileges` reports what a mount needs, and `docker create` adds `--device /dev/fuse` / `--cap-add SYS_ADMIN` / `--security-opt apparmor:unconfined` accordingly. Without this an in-container FUSE or `s3files` mount could never work on Docker. |
| `types.py`: `Permissions.from_mode` | Added, round-tripping with `to_mode`. |
| `util/token_truncation.py`: only a tail truncate | The whole policy API: `TruncationPolicy`, `truncate_text`, `formatted_truncate_text`, `formatted_truncate_text_with_token_count`, and Python's head+tail middle elision with the `…N tokens truncated…` marker. The old `truncate_text_by_tokens` is kept as an alias. |
| `util/checksums.py` + `materialization.py` | New `src/sandbox/materialization.rs`: `MaterializedFile { path, sha256 }`, `MaterializationResult`, `sha256_hex`, streaming `sha256_file`. `Entry::apply` and `Manifest::apply` now return a `MaterializationResult` instead of a path list; every call site already discarded the value. |
| `util/retry.py` (**partial**) | New `src/sandbox/retry.rs`: `BackoffStrategy`, `RetryPolicy`, `retry_async`, `TRANSIENT_HTTP_STATUS_CODES`. Distinct from `src/retry.rs`, which retries the *model*. |
| `manifest.py`: ephemeral helpers and `extra_path_grants` | `ephemeral_entry_paths`, `ephemeral_mount_targets`, `ephemeral_persistence_paths`, and `Manifest::{extra_path_grants, with_extra_path_grant}`. |
| `manifest.py`: polymorphic environment | New `src/sandbox/environment.rs`: the `EnvValue` trait, `StrEnvValue`, `EnvEntry` (description / ephemeral / value) and `Environment`. `Manifest::with_env_entry` declares a variable fetched at session start; `resolve_environment` performs the lookups, and the Unix-local backend runs them once in `start()` before any command can observe the environment. |
| `workspace_paths.py`: model/display paths, `SandboxPathGrant` (**partial**) | `SandboxWorkspaceScope::{model_path, model_resource_path, display_path}` re-render tool results against a run `cwd`, and `SandboxPathGrant` carries an absolute out-of-workspace grant with its validators (no relative path, no UNC, no parent segments, never the filesystem root). |
| `snapshot.py`: `RemoteSnapshot` | `RemoteSnapshotClient` (upload / download / exists), `SnapshotSpec::Remote { client_key }`, and a process-wide client registry. The key, not the client, lives in the serialized state, which is what lets a resumed state in another process find its storage again. |
| `sandbox_session_state.py`: missing fields (**partial**) | `workspace_root_ready`, `path_grants_require_rebind`, `mount_authority_redacted`, `mount_authority_rebound`. `workspace_root_ready` is wired for real: a run interrupted mid-materialization no longer resumes against a half-populated workspace that merely *looks* preserved. |
| `testing/sandbox.py` | New `src/sandbox/testing.rs`: `ScriptedSandboxSession`, `SandboxStep`, `SandboxMethod`, `SandboxCall`, `scripted_sandbox_session`, with FIFO matching, matchers, responders, scripted errors and `assert_complete`. An unexpected call fails where it happens rather than returning a plausible default. |
| `base_sandbox_session.py`: `run_as` never reached file operations (**partial**) | `read_file_as` / `write_file_as` / `mkdir_as` / `rm_as` / `mv_as` are default trait methods that route through `exec` (which already applies the `sudo` wrapper), so every backend gets them. The filesystem capability and `SandboxEditor` now use the capability's `run_as`, so a patched file is owned by the user whose commands have to write it next. |
| `entries/artifacts.py`: GitRepo commit-ref fallback (**partial**) | A ref that looks like a commit sha is fetched, falling back to a named clone; a branch or tag is cloned with `--branch`, falling back to a fetch. A failed subpath materialization now distinguishes `GitSubpathError` from `GitCopyError`. |
| `session/mount_lifecycle.py`: ownership guard | `MountTransition` / `MountTransitionGuard` mark the task that is detaching mounts for a snapshot. `persist_session_snapshot` holds it, and the shell and filesystem tools call `ensure_mounts_settled()`, so a concurrent command is refused instead of reading the empty directory where a mounted bucket should be. |
| `sandboxes/unix_local.py`: symlink handling on hydration | `rebase_absolute_symlinks` mirrors Python's `_rebase_symlink_target`: an absolute target pointing inside the workspace is rewritten link-relative, so a snapshot restored under a different root still resolves, and the components after the root are kept verbatim because `..` after a link component is resolved against that link's target. Targets outside the workspace are left unchanged, as Python leaves them. |
| `capabilities/{filesystem,shell}.py`: `configure_tools` | `ConfiguredCapability<C>` wraps any capability and passes its tool list through a configurator, so an application can rename, narrow, wrap or drop a bundled tool without losing the manifest changes, instructions and sampling parameters the capability also contributes. |
| `runtime_session_manager.py`: concurrency limits (**partial**) | `SandboxConcurrencyLimits` with Python's defaults (4 / 4), validated, carried on the `Manifest`, and applied to `local_dir` file copies, which now run in bounded batches instead of one at a time. `manifest_entries` is accepted and validated but not yet applied: entries share parent directories and mounts must settle in order, so materializing them concurrently needs an ordering pass this phase did not do. |
| `session/{sinks,events,manager}.py` (**partial**) | New `src/sandbox/events.rs`: `SandboxSessionEvent` (start/finish pairs correlated by span id), `EventPayloadPolicy`, the `EventSink` trait with `CallbackSink` / `JsonlFileSink` / `ChainedSink`, `Instrumentation`, and `InstrumentedSession`, a drop-in `SandboxSession` wrapper that records every operation. Command output and file contents are withheld unless the policy asks for them, and bounded when it does. The HTTP proxy and in-workspace sinks are left to applications: they are a few lines of a user's own `EventSink`. |

### Review fixes applied after the first phase-5 pass

- **`write_file_as` would have broken `apply_patch` for ordinary files.** The whole base64 payload
  went in one `sh -lc` argument, and Linux caps a single argument at 128 KiB, so any file over
  about 96 KB would have failed with `E2BIG` — and `SandboxEditor` routes every write through it
  once `run_as` is set. It now creates the parent directory, appends 48 KiB chunks to a sibling
  staging file, and moves that into place, so a failed write also never leaves a half-written
  file where the old one was. Covered by a test that writes 1 MB and asserts no argument exceeds
  100 000 bytes.
- **`workspace_root_ready` would have changed resume for existing states.** It defaults to `false`
  under serde, and the schema version did not change, so a state written before the field existed
  would have taken the full re-materialize branch and overwritten the agent's edits from the first
  half of the run. It now defaults to `true`; states written by this version always carry the real
  value. Docker's `resume` seeds it too.
- **Three APIs recorded intent that nothing acted on.**
  - `with_env_entry` was resolved only by the Unix-local backend, so another backend would have
    dropped a declared secret silently. `SandboxSession::resolves_environment` now reports whether
    a backend performs the lookups, and materialization refuses when it does not. Docker performs
    them, through a `0600` host file passed as `--env-file` — not `docker exec -e`, which would
    put the value in the host's process table.
  - `with_extra_path_grant` is now honoured: `normalize_path` resolves a path under a grant
    instead of refusing it, and Docker bind-mounts a grant that names a host path.
  - The state provenance flags (`path_grants_require_rebind`, `mount_authority_redacted`,
    `mount_authority_rebound`) are **recorded, not enforced**. They describe what the live
    manifest held so a caller can tell what has to be re-supplied; nothing in the SDK branches on
    them yet.
- **`view_image` read the whole file before checking the cap**, so an oversized file was still
  pulled into memory — and over the wire on a hosted backend — before being refused. It checks the
  size first now.
- **`ScriptedSandboxSession` overrode `authorize_mount_activation`**, which would have hidden a
  missing credential acknowledgement in users' tests. Removed; the trait default enforces the
  manifest policy.
- **The PTY tools did not call `ensure_mounts_settled()`.** They do now, like the shell and
  filesystem tools.
- **A git clone recorded a `MaterializedFile` with an empty checksum.** It writes a tree, not a
  file, so it records nothing instead.

### Second review pass on phase 5

- **The first attempt at symlink handling deleted links a restored workspace needs.** It removed
  every link whose target left the workspace, which breaks a virtualenv's
  `bin/python -> /usr/bin/python3` on every restore. `unix_local.py` L190-215 settles it: Python
  *rebases* an absolute target that points under the workspace root and returns one pointing
  outside it unchanged. The Rust side now does the same. The containment it was meant to provide
  is already there: `host_path_for` canonicalizes both the candidate and the root before every
  file operation, and `tar` refuses to extract *through* a symlink — verified on this machine
  (`Cannot extract through symlink`, nothing written outside), which the end-to-end test now
  asserts.
- **The Docker environment file leaked the secret it exists to protect.** It was written `0644`
  and tightened afterwards, in a shared `/tmp`, under a name derived from the container id — a
  window, and a predictable target for a planted symlink. It is now created `0600` in one step
  under a random name. The file was also missing from the interactive path, so a secret would
  have been present for `shell` and absent for `exec_command`; `pty_argv` was extracted and now
  adds `--env-file` too, with a test asserting both argument vectors carry the file and neither
  carries the value.
- **`with_extra_path_grant` did not actually widen anything on Unix-local**, which consults mount
  grants only. `Manifest::host_path_grants()` now also yields the extra grants, so they reuse the
  read-only enforcement mounts already had. Writes into a read-only grant are refused in
  `SandboxEditor::resolve`, since every `apply_patch` operation is a write.
- **`InstrumentedSession` inherited the `*_as` trait defaults**, which run through the wrapper's
  own `exec` — so a run-as write would have recorded the file's base64 contents in the audit log,
  the one thing the payload policy promises never to happen. All five are delegated to the inner
  session now, recording the path and byte count only, which also preserves any native
  implementation a backend provides. Events gained a `ts` field.
- **The editor's delete and post-move cleanup still ran as the session owner.** They use the
  run-as forms now.
- **Unix-local exec applied the manifest's literals twice**, so a literal re-overrode a
  looked-up value of the same name — the opposite of the documented precedence. Only the
  request's own variables are applied on top of `exec_env` now.

### Read-only enforcement scope

A read-only [`SandboxPathGrant`] is enforced where a write can actually be attributed:
model-facing `apply_patch` operations, Unix-local file operations (through
`host_path_grants`), and Docker bind mounts (`:ro`). A raw `write_file` through another backend's
session API is **not** checked against the grant — the backend would have to enforce it, and the
provider backends are outside this phase's scope.

### What the partially closed rows still leave open

| Row | Remaining |
| --- | --- |
| `sandboxes/docker.py` | `DockerSandboxClientOptions.labels`, `network_mode` validated against `exposed_ports`, resume-time container validation (network, labels, path grants), the `SYS_ADMIN` and read-only-bind refusal in `_validate_docker_path_grants`, `_docker_volume_name`, and the raw-socket exec transport. |
| `util/retry.py` | The module is there; nothing calls `retry_async` yet. Wiring it needs the provider call sites, which are outside this phase. |
| `workspace_paths.py` | `WorkspacePathPolicy`. |
| `unix_local.py` | `_PTY_CHILD_SIGNAL_DEFAULTS` / `_restore_pty_child_signal_defaults`. |
| `apply_patch.py` | `normalize_operation` and the `PatchFormat` / `V4AFormat` pluggable format protocol. |
| `base_sandbox_session.py` | The access-probe scripts (`_READ_PATH_PROBE_SCRIPT` and friends) and their timeouts. |
| `entries/artifacts.py` | The `user=` materialization path for entries, and the fd-based symlink-safe local copy. |
| `sandbox_session_state.py` | `_sanitize_persisted_provider_identity` and `REDACTED_HOST_PATH_GRANT_PATHS_KEY`; the provenance flags are recorded but not enforced. |
| `session/{sinks,events,manager}.py` | The HTTP proxy and in-workspace sinks, and the trace-id correlation helpers. |
| `runtime_session_manager.py` | `manifest_entries` concurrency is validated but not applied. |

### Deferred, with reasons

| Audit row | Why not now |
| --- | --- |
| `session/snapshot_lifecycle.py`: the fingerprint fast path | It is computed by `WORKSPACE_FINGERPRINT_HELPER`, an in-sandbox runtime helper script. The helper scripts are an explicit non-goal of this port, and a Rust-side equivalent would need either a tar crate or a `sha256sum` that BSD images do not ship. `workspace_root_ready` closes the correctness half of the row; the optimization half is open. |
| `session/dependencies.py`: the DI container | Its one load-bearing use, `RemoteSnapshot`, is served by a typed client registry instead: `SnapshotSpec::Remote { client_key }` plus `register_remote_snapshot_client`. A general string-keyed container would add a second, weaker way to pass the same objects Rust already passes by type. |
| `_unix_local_file_ops.py`: the out-of-process fd-walk worker | Its `O_DIRECTORY\|O_NOFOLLOW` traversal defends against a symlink swapped in mid-walk. This port canonicalizes both the candidate and the root before every file operation, and now sanitizes symlinks on hydration, which covers the planted-link case. The residual gap is a true TOCTOU race against a concurrently running sandbox process, which is stated here rather than papered over. |
| `sandboxes/docker_removal.py` + `_docker_removal_worker.py` | Mostly does not apply, with one known gap. The service exists because Python removes host directories a container wrote into as root. Here, Docker workspaces live inside the container and go away with `docker rm`, and a Docker bind mount's host source belongs to the caller and is never removed. The remaining case is Unix-local with `run_as`: a `0700` directory created by `sudo -u agent` makes `remove_dir_all` fail with `EACCES`, so `shutdown` errors and the temporary root is left behind. That is a known gap, not a reason to port the privileged in-container worker. |
| `runtime_session_manager.py`: per-agent sessions | Reverses this port's committed design, in which the whole handoff graph binds to one session — which is what makes a handoff see the files the previous agent wrote. Changing it is a design decision for the owner of `SandboxRunner`, not a parity fix. The resume-key machinery exists only to serve per-agent sessions. |
| `runtime_session_manager.py`: live-session manifest updates | Adding entries to a running session needs the host-path-grant validation and the running-session entry rules that go with it; it is a feature, not a gap in an existing one. |
| `runtime.py`: `apply_result_metadata` | Ported as types rather than as private attributes on `RunResult`: `SandboxRunOutcome` carries the session state and the memory manager, and `SandboxStreamedRun` plus the phase-3 completion hook carry the deferred cleanup. Attaching them to the core `RunResult` would need a core change this phase was not authorized to make. |
| `sandbox_client.py`: the polymorphic client-options registry | Rust client options are concrete types the caller constructs; there is no untrusted-dict deserialization path for them to survive. |

### Breaking API changes in this phase

- `Entry::apply` and `Manifest::apply` / `apply_ephemeral` return `MaterializationResult` instead
  of `Vec<String>`. Every call site in the workspace already discarded the value.
- `ErrorCode` and `SnapshotSpec` gained variants; `SandboxError` gained the public fields `op`,
  `context` and `retryable`. Exhaustive matches and struct literals of these need updating.
- `Entry` gained `copy_concurrency`, and `Manifest` gained private fields, so struct literals of
  either need updating; the builders are unchanged.
- `HOME` is now the workspace root for Unix-local commands even when `inherit_host_environment`
  is on, which is the default. This matches Python, but a command relying on the host's
  `~/.cargo`, `~/.rustup` or `~/.gitconfig` will no longer find it.

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

## Status

### Phase 4 — the mount entry layer

```text
Status: DONE
Summary: Python's mount entry layer is ported — the FUSE, mountpoint and s3files patterns, the
redaction helpers, and the six configured mounts (S3, S3 Files, GCS, Azure Blob, R2, Box) behind
one `ConfiguredMountStrategy` — and wired into every provider backend that translates a mount.
Concerns/Blockers: none outstanding. Three findings from review were fixed here rather than
deferred: the credential boundary was not enforced on the snapshot re-attach path, three rows of
Python's capability table had no equivalent, and `docker_create_args` swallowed configuration
errors. Docker's CSV parsing of `--mount` was verified empirically on this machine.
```

### Phase 5 — the parity-audit rows

```text
Status: DONE_WITH_CONCERNS
Summary: Eleven `src/agents/sandbox/**` audit rows are fully closed and ten more partially. The
work covers the five missing error codes and the structured error fields, apply_patch no-clobber,
the view_image size cap and magic-byte sniffing, the 22-entry host environment allowlist, Docker
FUSE/SYS_ADMIN privileges, the truncation policy, checksums and materialization results, the
sandbox retry module, the polymorphic environment, model/display paths and path grants,
`RemoteSnapshot`, the scripted test session, `run_as` on file operations, the git commit-ref
fallback, the mount-transition ownership guard, symlink rebasing on hydration, the tool
configurator, concurrency limits, and the session event/sink subsystem.
Concerns/Blockers:
  - Deferred with reasons, each recorded in the report: the snapshot fingerprint fast path (needs
    the in-sandbox runtime helper this port excludes), the `Dependencies` DI container (superseded
    by a typed client registry for its one load-bearing use), the out-of-process fd-walk worker
    (covered by canonicalization; the residual TOCTOU race is stated), `docker_removal` (does not
    apply, with one known `EACCES` gap on Unix-local with `run_as`), per-agent sessions (reverses
    the committed one-session-per-graph design), live-session manifest updates, and
    `apply_result_metadata` (ported as types rather than private result attributes).
  - `SandboxConcurrencyLimits::manifest_entries` is accepted and validated but not applied:
    entries share parent directories and mounts must settle in order.
  - Read-only path grants are enforced for model-facing edits, Unix-local file operations and
    Docker bind mounts, but not for a raw `write_file` through another backend's session API.
  - Ten rows are only partially closed; the report lists exactly what each still leaves open.
    The ones most likely to matter are the remaining Docker client options and resume-time
    container validation, `WorkspacePathPolicy`, and `retry_async` not yet being wired into any
    provider call site.
  - Breaking API changes are listed above; the most likely to surprise is `HOME` now pointing at
    the workspace root for Unix-local commands.
```
