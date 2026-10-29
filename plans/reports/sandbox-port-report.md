# Sandbox port report

Port of the Python `agents.sandbox` package core to Rust, inside `openai-agents-sdk` under the
`sandbox` feature. Source of truth: `/Users/ngocthach/myGit/openai-agents-python/src/agents/sandbox/`
(~28K lines) plus `docs/sandbox_agents.md` and `docs/sandbox/*.md`.

Files created (nothing outside this list was touched):

- `src/sandbox/mod.rs`, `types.rs`, `errors.rs`, `workspace_paths.rs`, `util.rs`
- `src/sandbox/entries.rs`, `manifest.rs`, `session.rs`, `editor.rs`, `agent.rs`, `runtime.rs`
- `src/sandbox/capabilities/{mod,shell,filesystem,compaction}.rs`
- `src/sandbox/sandboxes/{mod,unix_local,docker}.rs`
- `src/sandbox/instructions/prompt.md` (verbatim copy of the Python base prompt)
- `tests/sandbox_tests.rs`, `examples/sandbox_local.rs`

## Verification

| Check | Result |
| --- | --- |
| `cargo clippy --features sandbox --all-targets -- -D warnings` | clean (exit 0) |
| `cargo clippy --all-targets` (default features) | no sandbox diagnostics |
| `cargo test --features sandbox --test sandbox_tests` | 15 passed |
| `cargo test --features sandbox --lib sandbox` | 29 passed |
| `cargo test --features sandbox --doc sandbox` | 1 passed |
| `rustfmt --edition 2024` on every new file | applied |

The Docker test ran for real on this machine (`docker info` and `docker image inspect alpine:latest`
both succeeded): container create/start/exec/write/read/rm and manifest materialization all pass.
On a machine without a daemon, or without the image already present locally, it prints a skip
notice and returns — no image is ever pulled, so the default test run needs no network.

## What is ported

**Agent and preparation** (`sandbox_agent.py`, `runtime.py`, `runtime_agent_preparation.py`)

- `SandboxAgent<C>`: an `Agent<C>` plus `default_manifest`, `base_instructions`, `capabilities`,
  `run_as`. `Clone`/`Debug` are hand-written.
- `SandboxAgent::prepare(&CapabilityBinding) -> Agent<C>` reproduces `prepare_sandbox_agent`:
  required-capability validation, capability tool collection, `sampling_params` deep-merged into
  `ModelSettings::extra_args`, and the composed instructions (base prompt →
  `# Agent instructions` → `# Sandbox capability instructions` → `# Filesystem` + rendered
  workspace layout). Instructions are built as `Instructions::Dynamic`, so an agent configured
  with dynamic instructions still resolves them per turn.
- `prepare_input` applies every capability's `process_context` to the run input.

**Manifest and entries** (`manifest.py`, `manifest_render.py`, `entries/artifacts.py`)

- `Manifest`: `version`, `root` (default `/workspace`), `entries`, `environment`, `users`,
  `groups`; builder methods; `validated_entries()`; `describe()`/`render()` with the Python depth
  (3) and 5000-char budget plus truncation marker.
- `Entry`/`EntryKind`: `File`, `Dir` (nested children), `LocalFile`, `LocalDir`, `GitRepo`
  (`git init` + `git fetch --depth 1` + `checkout FETCH_HEAD`, with optional `subpath`), each with
  `description`, `permissions` and `group`, and `chmod`/`chgrp` applied after materialization.
  Host symlinks inside a `LocalDir` are skipped rather than followed.
- `Manifest::apply(session)` provisions accounts best-effort (`groupadd`/`useradd`/`usermod`,
  ignoring failures as Python does), creates the root, then materializes each entry.

**Capabilities** (`capabilities/`)

- `Capability<C>` trait + `CapabilityBinding { session, run_as, scope }`. Unlike Python, Rust
  capabilities are stateless and receive the binding per run, so Python's clone-and-bind
  (`Capability.clone`, `bind`, `bind_run_as`, `bind_workspace_scope`) is unnecessary.
- `Shell` → one `crate::tool::ShellTool` whose executor runs each `shell_call` command through
  `SandboxSession::exec`, anchored at the run working directory and the `run_as` user. A command
  that exceeds the call's `timeout_ms` becomes `ShellCallOutcome::Timeout`; a non-zero exit becomes
  an ordinary `Exit { exit_code }`; only transport/path failures return `Err`. Contributes the
  shell instruction fragment.
- `Filesystem` → `crate::tool::ApplyPatchTool` backed by `SandboxEditor`, plus the `view_image`
  `FunctionTool` returning `ToolOutputPart::Image` with a `data:` URL. As in Python, the
  filesystem capability exposes exactly these two tools and no read/write/list tools.
- `Compaction` → the `context_management` sampling parameter with Python's static (240 000) and
  dynamic (0.9 × context window) policies and the model → context-window table, plus
  `process_context` truncation at the newest `compaction` item.
- `Capabilities::default_set::<C>()` = filesystem + shell + compaction, matching
  `Capabilities.default()`.

**Sessions and clients** (`session/`, `sandboxes/`)

- `SandboxSession` async trait: `start`/`stop`/`shutdown`/`running`, `exec`, `read_file`,
  `write_file`, `normalize_path`, and default `ls`/`mkdir`/`rm`/`mv`/`persist_workspace`
  implementations built on `exec` (as in `base_sandbox_session.py`). `ExecRequest` mirrors the
  Python `exec` signature (`shell: bool | list[str]` → `ExecShell::{Login,Plain,None,Custom}`,
  `timeout`, `user` → `sudo -u <name> --`, plus `cwd`).
- `SandboxClient` trait: `create`/`delete`.
- `UnixLocalSandboxClient`/`UnixLocalSandboxSession`: temp workspace under `std::env::temp_dir()`
  (canonicalized once, so macOS `/var` → `/private/var` does not break prefix checks), commands via
  `tokio::process`, manifest environment exported per command, optional host-environment
  filtering, workspace removed on shutdown when the session owns it. Like Python, a manifest left
  at the default root has its root replaced by the real temp directory, so the model sees that
  path.
- `DockerSandboxClient`/`DockerSandboxSession`: `docker create/start/exec/stop/rm` through the
  `docker` CLI (no new dependency; Python uses `docker-py`). Writes stream through
  `docker exec -i … sh -c 'mkdir -p "$(dirname "$1")" && cat > "$1"' sh <path>`, with the path as a
  positional argument so a path with shell metacharacters cannot inject a command.
  `is_available()`/`image_exists()` are exposed for test gating.
- `persist_workspace`/`hydrate_workspace` use gzipped tar, which both backends support.

**Runner integration** (`runtime.py`, `runtime_session_manager.py`, `RunConfig.sandbox`)

- `SandboxRunConfig { client, session, manifest, cwd }` — the Rust equivalent of Python's
  `SandboxRunConfig`. `src/run.rs` and `src/run_internal/**` are untouched.
- `SandboxRunner::{prepare, run, run_with_options, run_streamed_with_options}` plus
  `SandboxSessionGuard`. The guard tears down a client-created session in every exit path
  (including preparation failures) and never touches a caller-supplied session. Streaming returns
  `(RunResultStreaming<C>, SandboxSessionGuard)` because `RunResultStreaming` cannot carry async
  teardown and `result.rs` is out of scope.
- `cwd` validation mirrors `_validate_workspace_scope`: `test -d` and `test -x` as the configured
  sandbox user.

**Errors and paths** (`errors.py`, `workspace_paths.py`)

- `ErrorCode` (the subset reachable from the ported code) and `SandboxError { code, message,
  exit_code }`, with `From<SandboxError> for AgentsError` mapping path/config codes to `UserError`
  and the rest to `Other("[code] message")`.
- `resolve_workspace_path`, `normalize_sandbox_cwd`, `SandboxWorkspaceScope::anchor`, and the
  Windows-absolute-path rejection from `windows_absolute_path`.

### Path traversal protection

Three layers, all covered by tests:

1. Lexical: `resolve_workspace_path` rejects `..`, Windows absolute paths and absolute paths
   outside `Manifest::root`. Used by `normalize_path`, every default session file op, the editor
   and `view_image`.
2. Symlink: `UnixLocalSandboxSession::host_path` canonicalizes the deepest existing ancestor and
   re-checks the root prefix, so a symlink planted inside the workspace (e.g. `ln -s /etc escaped`)
   cannot be read or written through.
3. Manifest: entry keys and `Dir` child names are validated before anything is written.

Both sides of the prefix check are canonicalized, so a caller-supplied workspace root that itself
resolves through a symlink (`tempfile::tempdir()` under macOS `/var/folders/...`) still works —
covered by `materializes_into_a_caller_supplied_workspace_root`, which also asserts that `delete`
leaves a caller-owned root in place.

`SandboxEditor::delete_file` reads the target before removing it, so deleting a path that does not
exist is reported to the model as a failed `apply_patch_call_output` carrying
`apply_patch_file_not_found` instead of `rm -f` silently succeeding. It reads bytes rather than
text, so binary files stay deletable.

## What is not ported, and why

| Area | Python source | Reason |
| --- | --- | --- |
| Cloud provider backends: Blaxel, Cloudflare, Daytona, E2B, Modal, Vercel | `extensions/sandbox/*` (~13.6K lines) | Explicitly out of scope. Each needs its own HTTP/SDK client and credential handling. Add one by implementing `SandboxClient` + `SandboxSession`. |
| Remote mounts (S3, S3 Files, GCS, Azure Blob, R2, Box), mount strategies and patterns | `entries/mounts/**`, `remote_mount_policy.py`, `_mount_security.py` (~4K lines) | Depends on the cloud providers above and on a large credential-exposure policy engine; no core value without them. |
| PTY sessions: `write_stdin`, `tty=true`, `pty_exec_start` | `session/pty_output.py`, `unix_local.py`, `docker.py` | Needs `openpty`/raw fd handling (unsafe code, forbidden in this crate) or a Docker attach socket. `SandboxSession::supports_pty()` returns `false`, matching Python's gating of `write_stdin`. |
| Sandbox-backed memory (phase-one/phase-two generation, rollouts, storage) | `memory/**`, `capabilities/memory.py`, `config.py` memory types (~1.5K lines) | Requires nested runs driven from inside the run loop, which `src/run.rs` would have to expose. |
| Skills capability and lazy skill sources | `capabilities/skills.py` (984 lines) | Not in `Capabilities.default()`; large and self-contained. |
| Snapshot specs, session-state serialization, resume | `snapshot.py`, `session/sandbox_session_state.py`, `session/snapshot_lifecycle.py` | `RunState` cannot carry sandbox state without touching core files. `persist_workspace`/`hydrate_workspace` cover the useful half; resume is done by passing a live `SandboxRunConfig::session`. |
| Archive extraction limits, tar utilities, exposed ports, `same_file`, `read_bounded`, runtime helper scripts | `util/tar_utils.py`, `session/archive_*.py`, `types.ExposedPortEndpoint`, `session/runtime_helpers.py` | Peripheral to the capability/manifest/session core. |
| Token truncation of tool output | `util/token_truncation.py` | `ShellTool` already applies the call's `max_output_length` in `run_internal/local_tools.rs`. |
| `Permissions`/`FileMode` pydantic ergonomics (`owner_can`, etc.) | `types.py` | Replaced by plain fields plus `from_ls_field`/`octal_string`. |
| Tracing spans `sandbox.prepare_agent` / `sandbox.cleanup` | `runtime.py` | Would need `custom_span` around runner-internal phases; `SandboxRunner` is outside the run loop, so the spans would not nest correctly. |

### Fidelity decisions worth knowing

- **Tool surface.** Python's shell capability is an `exec_command` `FunctionTool` (+ PTY-only
  `write_stdin`) and its apply_patch is a `CustomTool`. Per the task, this port uses the crate's
  `ShellTool` and `ApplyPatchTool` instead, so the model sees the native `shell` and `apply_patch`
  tool types. The shell instruction fragment names `shell`, not `exec_command`.
- **Handoffs.** Python re-prepares every `SandboxAgent` it hands off to. `SandboxRunner` prepares
  only the starting agent; handoff targets must be prepared with `SandboxAgent::prepare` against
  the same session. Documented on `SandboxRunner`.
- **Approval resume.** A run that ends with interruptions tears down a client-created session, so
  resuming needs `SandboxRunConfig::session` (or `SandboxRunner::prepare`). Documented on
  `run_with_options`.
- **`SandboxRunConfig::options`** has no equivalent: Rust clients take their options at
  construction (`DockerSandboxClientOptions`), so there is nothing to coerce at run time.
- **macOS `sandbox-exec`.** Python's Unix-local backend wraps commands in `sandbox-exec` on
  macOS. This port does not; the backend is documented as providing no OS-level confinement on
  any platform.
- **`Default` matches `new()`.** `UnixLocalSandboxClient::default()` inherits the host environment
  and `Filesystem::default()` includes `view_image`, so neither `Default` impl is derived — a
  derived one would silently disagree with the Python defaults.

## Public API summary

```text
openai_agents_sdk::sandbox
  SandboxAgent<C>            new / from_agent / with_instructions / with_model(_instance) /
                             with_tool / with_default_manifest / with_base_instructions /
                             with_capabilities / with_capability / with_run_as / map_agent /
                             resolved_manifest / resolved_model_name / prepare_input / prepare
  DEFAULT_SANDBOX_INSTRUCTIONS
  SandboxRunner              prepare / run / run_with_options / run_streamed_with_options
  SandboxRunConfig           with_client / with_session / manifest / cwd
  SandboxSessionGuard        session / owns_session / close
  PreparedSandboxRun<C>      agent / guard / session
  Manifest                   new / with_root / with_entry / with_env / with_user / with_group /
                             validated_entries / apply / describe / render   (DEFAULT_MANIFEST_ROOT)
  Entry / EntryKind          file / text_file / dir / local_file / local_dir / git_repo +
                             with_description / with_permissions / with_group / with_child /
                             with_git_host / with_git_subpath
  SandboxSession (trait)     start / stop / shutdown / running / exec / read_file / write_file /
                             ls / mkdir / rm / mv / persist_workspace / hydrate_workspace /
                             normalize_path / describe / supports_pty / manifest / workspace_root
  SandboxClient (trait)      backend_id / create / delete
  ExecRequest / ExecShell / ExecResult / FileEntry / Permissions / User / Group / file_mode
  SandboxWorkspaceScope      root / from_cwd / cwd / anchor
  SandboxError / ErrorCode / SandboxResult
  SandboxEditor              (ApplyPatchEditor over a session)
  capabilities::{Capability<C>, CapabilityBinding, Capabilities, Shell, Filesystem, Compaction,
                 CompactionPolicy, context_window_for_model, ViewImageArgs}
  sandboxes::unix_local::{UnixLocalSandboxClient, UnixLocalSandboxSession}
  sandboxes::docker::{DockerSandboxClient, DockerSandboxClientOptions, DockerSandboxSession,
                      DEFAULT_PYTHON_SANDBOX_IMAGE}
  util::{base64_encode, image_media_type, deep_merge}
```

## Requested core changes (worked around, not applied)

1. **`AgentsError::Sandbox` variant.** `SandboxError` currently flattens into
   `AgentsError::UserError` / `AgentsError::Other("[code] message")`, so callers must parse the
   code out of a string. A `Sandbox { code: String, message: String }` variant (or a boxed
   `source`) would let them match on it. Worked around with `From<SandboxError> for AgentsError`
   inside `src/sandbox/errors.rs`.
2. **Async teardown for `RunResultStreaming`.** `result.rs` has no hook for releasing a resource
   when a stream finishes (Python attaches `result._sandbox_cleanup`). Worked around by returning
   `(RunResultStreaming<C>, SandboxSessionGuard)` and asking the caller to `close()` the guard.
3. **A per-turn agent-preparation hook in the run loop.** Python re-prepares every `SandboxAgent`
   reached by a handoff, and applies `process_context` to the input of every turn.
   `src/run_internal/**` exposes no such hook, so only the starting agent is prepared and
   `process_context` runs once, on the initial input.
4. **`base64` available under `sandbox`.** The crate's `base64` dependency is optional and gated to
   `encrypt`/`realtime`/`voice`, and the `sandbox` feature must add no dependencies. Worked around
   with a 20-line encoder in `src/sandbox/util.rs`, used only for `view_image` data URLs. Adding
   `sandbox = ["dep:base64"]` would let that go away.
5. **No `[[example]]` entry for `sandbox_local`.** Cargo.toml is out of scope, so the example
   carries a `#[cfg(feature = "sandbox")]` main plus a fallback main, and the test file starts with
   `#![cfg(feature = "sandbox")]`. An `[[example]] name = "sandbox_local", required-features =
   ["sandbox"]` entry (matching `mcp_stdio`) would be tidier.

## Open questions

- Should `SandboxRunner::run_with_options` keep a client-created session alive when the run ends
  with approval interruptions, instead of tearing it down? That would make HITL resume work out of
  the box but leaks a workspace whenever the caller never resumes.
- Is a PTY/`write_stdin` capability wanted? It needs either `unsafe` (`openpty`) — currently
  forbidden by `#![forbid(unsafe_code)]` — or a dependency such as `rustix`/`nix`.
