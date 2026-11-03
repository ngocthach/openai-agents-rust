# Python→Rust test parity — sandbox-a slice `runtime` (test_runtime.py lines 1-4000)

Status: this slice covers py lines 1-4000 only (the memory-snapshot helper owns 4000-end in
`tests/py_sandbox_a_runtime_{5,6,7}.rs`). Counted with
`awk 'NR<=4000 && /^(async )?def test_/' test_runtime.py | wc -l` = **84** test functions
(parametrize expansions noted per row).

## 1. Table

| file | #tests (1-4000) | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_runtime.py L1-4000 | 84 | 58 | 5 | 21 | 8 (see §2-3) |

58 + 5 + 21 = 84.

Rust test files (all mine): `tests/py_sandbox_a_runtime_1.rs` (19), `_2.rs` (9), `_3.rs` (7),
`_4.rs` (16), `py_sandbox_a_runtime_unix_local.rs` (15), `py_sandbox_a_runtime_tracing.rs` (1).

### PORTED (58, python test numbers by order in file)
- Preparation/instructions: 19,20,21,22,23,25,26,27,28,29,30,31,32 (`_1`), 44 (`_1`).
- Injected sessions / stream cleanup: 33,35,36,54,64,65 (`_1`,`_2`).
- Limits: 76,77 (create + injected; the `resume` source is not ported), 78,79,80 (`_2`).
- Clear-on-resume mount rows live in `_2` (python rows 4000+ in the other half, listed there).
- Teardown/cancel/guardrail/identity (`_4`): 34+37, 38+39 (pinned to current behavior, DEFERRED),
  40,41,42,43,66 (by agent name), 45,46,48 (per-agent mode), 62,63,67, 70, 82 (session_state source).
- Helper scripts and unix_local (`_unix_local`): 13,15,16,17,18 (absolute half),49,51,53,55,56,57,58,59*,60*,69.
  (* DEFERRED, pins current behavior.)
- Tracing (`_tracing`): 61 (enabled half and the `tracing_disabled` half; stop/shutdown spans are DEFERRED to backends).
- Remaining ported rows: 2,3 are NOT ported (python-only, below); 14 is python-only.

### COVERED (5)
- 50 `tests/sandbox_tests.rs::teardown_detaches_mounts_before_removing_the_workspace`
- 52 `tests/sandbox_tests.rs::a_failed_unmount_keeps_the_workspace` (the log-redaction half is python-only)
- 71 `tests/sandbox_tests.rs::resumes_a_run_that_paused_inside_a_handoff_destination`
- 83 `tests/sandbox_core_parity_tests.rs::persisted_state_drops_the_host_paths_and_demands_them_back`
- 84 `tests/sandbox_core_parity_tests.rs::a_credential_bearing_state_loses_its_provider_identity_when_persisted`

### PYTHON-ONLY (21), one reason per group
- 1: `__deepcopy__` hook on a custom pattern; Rust has no deepcopy protocol.
- 2-10: `SandboxSession.aclose` wrapper, pre-stop hooks, `asyncio.CancelledError` re-entrancy and
  exception-traceback redaction; Rust teardown is `SandboxSessionGuard::close` -> `client.delete`.
- 11, 12: wrapper-level persist/hydrate and path-check routing (`SandboxSession` proxy object);
  Rust validates credential authority at mount activation, not at persist.
- 14: `_validate_remote_path_access` internals (`empty_stdout` transport error); no Rust counterpart.
- 24: monkeypatches `importlib.resources.files`; Rust `include_str!`s the prompt at compile time.
- 47: asserts private `_nested_history_owned_session_item_refs` (the replay half is ported in 46).
- 68: `result.run_loop_task` asyncio task internals.
- 72-75: duplicate agent names get resume keys in Python; Rust rejects duplicate names
  (`tests/sandbox_tests.rs::rejects_duplicate_agent_names_in_the_handoff_graph`).
- 81: shape of `RunState._sandbox` payload.

## 2. Rust bugs fixed (files I own)
1. `src/sandbox/runtime.rs` `prepare`: a stopped injected session is started (a running one is left
   alone), as `ensure_session` does.
2. `src/sandbox/session.rs` `clear_workspace_root_on_resume`: a mount at the workspace root now stops the clear.
3. `runtime.rs`: rollout id follows Python's grouping (conversation id, SDK session id,
   `RunConfig.group_id`, else `run-<uuid>`); explicit `SandboxRunConfig::rollout_id` and a resumed
   state's rollout id still win.
4. `runtime.rs` + `session.rs`: runner-created sessions get the SDK default local snapshot when none
   is configured (new `SandboxRunConfig::snapshot`, new defaulted `SandboxClient::create_with_snapshot`).
5. `runtime.rs`: the `tracing_disabled` opt-out: with `RunConfig::tracing_disabled` the preparation and
   teardown spans are masked (disabled trace scope, as Python's cleanup does); guard carries the flag.
   `runtime::sandbox_span` is public for backends. `sandbox.prepare_agent`, `sandbox.create_session`, `sandbox.resume_session`,
   `sandbox.start`, `sandbox.cleanup`, `sandbox.cleanup_sessions` spans (gated like
   `_supports_trace_spans`; futures boxed).
6. `runtime.rs`: a changed host source on an injected session is refused (`manifest_topology_matches`
   now compares extra path grants).
7. `runtime.rs`: a mixed duplicate grant target is refused (`host_path_grant_topology` keeps every
   grant for a host-backed target, path-only ones included).
8. `runtime.rs`: an unsafe injected-session manifest (unacknowledged mount-scoped credentials) is
   refused before the session is probed or started (`validate_mount_credential_boundaries`).
9. `runtime.rs` `run_to_outcome`: the memory manager follows the agent active when the run ends, not only
   the starting agent (not applied to the streamed path: `StreamCompletion` carries no final agent;
   needs a field in `src/result.rs`).
10. `src/sandbox/agent.rs` `resolved_manifest`: a `run_as` user already in a manifest group is not added to `users`.
11. `session.rs`: the workspace fingerprint skips the runtime-registered persist skip paths too.
Flipped to Python behavior in `tests/py_sandbox_a_runtime_5.rs`: tests for 6, 7, 8 and 10 (formerly
DEFERRED asserts).

## 3. Cross-area bugs deferred (not my files; tests pin current behavior)
- **unix_local ignores the default snapshot**: no backend overrides `SandboxClient::create_with_snapshot`.
  Fix in `src/sandbox/sandboxes/unix_local.rs` `impl SandboxClient for UnixLocalSandboxClient`:
  `async fn create_with_snapshot(&self, manifest, snapshot) { ... .with_snapshot(self.snapshot.clone().or(Some(snapshot))) }`
  (reuse the body of `create`). Docker needs the same.
- **Persist skip-path registry**: it already exists on the trait (`register_persist_workspace_skip_path`,
  `persist_workspace_skip_paths`, `persist_skip_registry`, used by `capture_workspace_tar` and now the
  fingerprint), but no backend overrides `persist_skip_registry`, so registrations are forgotten.
  Fix: unix_local and docker sessions each hold `persist_skip: Mutex<BTreeSet<String>>` and return
  `Some(&self.persist_skip)` from `persist_skip_registry`; docker's own `persist_workspace` must exclude
  `persist_workspace_skip_paths()`. The docker helper (agent abaf26b85a1529b78) was told this API.
- **Restore does not reprovision accounts** (python `test_resume_reprovisions_manifest_accounts...`,
  `_6.rs::deferred_restore_does_not_reprovision_accounts`): needs `Manifest::provision_accounts`
  (`src/sandbox/manifest.rs:773`, currently private) made `pub`, plus a session flag for "system state
  preserved". Then `restore_session_snapshot` calls it between `hydrate_workspace` and `apply_ephemeral`.
  Not done: without the flag, hosted backends whose VM kept its users would fail `useradd`.
- `unix_local.rs` `rm`: `NotFound => return Ok(())` ignores `recursive`; Python raises
  `ExecNonZeroError` for a non-recursive missing `rm` (test 59).
- `unix_local.rs` mkdir/ls/rm symlink-escape messages say "escapes the workspace root"; Python says
  "must not escape root" (test 60). `ls` now rejects as Python does.
- unix_local temp workspace prefix is `openai-agents-sandbox-` (Python `sandbox-local-`); cosmetic (test 49).
- **Blocking input guardrail ordering** (tests 34, 37, 38, 39). Python runs blocking input guardrails
  before the sandbox session is created or touched. Rust prepares the sandbox in
  `SandboxRunner::run_to_outcome` / `run_streamed_with_options` (`src/sandbox/runtime.rs`, the
  `Self::prepare` call) before `Runner` runs them (`src/run_internal/mod.rs:1842`, the `!run_in_parallel`
  partition inside the first-turn setup). Minimal fix, core files: give `RunOptions` (`src/run.rs:27`) an
  optional async "prepare the starting agent" hook that `run_internal` awaits right after the blocking
  guardrails and before the first model call, and let `SandboxRunner` pass its preparation there instead of
  calling `prepare` up front. Until then my tests pin: tripwire raised, session created once, released once.
- **Streamed memory follows the active agent** (`runtime.rs` handles the non-streamed path only).
  `StreamCompletion` (`src/result.rs:250`) has no final agent. Minimal fix: add
  `pub last_agent_name: String` to the struct, fill it from `result.last_agent.name.clone()` at the
  construction at `src/result.rs:374`, and use it in the `run_streamed_with_options` completion hook in
  `runtime.rs` to pick the memory manager as `run_to_outcome` does.
- `sandbox.stop` / `sandbox.shutdown` spans: Rust's `SandboxClient::delete` performs stop and shutdown inside
  the backend, so the runner cannot span them. `runtime::sandbox_span` is public; a backend's `delete`
  wraps its own `session.stop()` / `session.shutdown()` calls in
  `sandbox_span("sandbox.stop", json!({}), ...)` / `sandbox_span("sandbox.shutdown", ...)`
  (`src/sandbox/sandboxes/unix_local.rs` and `docker.rs`, `impl SandboxClient::delete`).
- A completed run returns `sandbox_state = None`; Python's `RunState._sandbox` is still set after a
  completed run (test 33 tail).
- Python acknowledges mount credential exposure by relative path; Rust acknowledgements are exact
  absolute paths only (test 18 relative variant).
- Injected live session `concurrency_limits`: Rust applies limits through the manifest only (test 76).

## 4. `#[ignore]`d tests
None.

## 5. Shared-file / controller edits
None needed in `src/lib.rs` / `Cargo.toml`. File-name collision with the memory-snapshot helper
(its `_6`/`_7` overwrote mine; my edits briefly landed in its `_6.rs` and were reverted). Mine are `_1`-`_4`,
`_unix_local`, `_tracing`; I also edited four tests in `_5.rs` as the controller asked.

## 6. Gates
All of `py_sandbox_a_runtime_{1,2,3,4,5,unix_local,tracing}` pass; `cargo test --lib sandbox::` 299 passed.
Clippy (`--lib` plus my test targets, `-D warnings`) stops in the lib on files I do not own, so my test
targets were not reached: `src/run_internal/agent_identity.rs:106` (complex type),
`src/run_internal/mod.rs:2104,2469` (needless borrow), `src/stream_events.rs:48` (large enum variant),
`src/sandbox/sandboxes/unix_local.rs:827,853` (missing `Debug` on `ConfinementInputs`, `HostFacts`).
None are in `runtime.rs`, `agent.rs`, `session.rs`, `runtime_helpers.rs` or `testing.rs`. One earlier
clippy hit in my `_3.rs` (`.err().expect`) was fixed.

## 7. Latest round
- `tracing_disabled` opt-out implemented and tested (`py_sandbox_a_runtime_tracing.rs`, 2 tests pass).
  `sandbox.stop` / `sandbox.shutdown` need the backends' `delete` to wrap their calls (see §3).
- Restore-time account reprovisioning is done: `restore_session_snapshot` (`src/sandbox/session.rs`) now calls
  `Manifest::provision_accounts` between `hydrate_workspace` and `apply_ephemeral`, unless the new defaulted
  trait method `SandboxSession::system_state_preserved_on_start()` (default `false`) says the OS state survived.
  Flipped in `tests/py_sandbox_a_runtime_6.rs`: `restore_reprovisions_accounts_and_surfaces_a_failure`, and the
  preserved-state test sets the flag. **Docker must override `system_state_preserved_on_start` to "reused an existing
  container"** (Python `sandboxes/docker.py:1833`, `_set_start_state_preserved(reused_existing_container)`); until
  then a docker resume that reuses a container would re-run `useradd`.
- Clippy retried: the lib still fails only in files I do not own (`src/run_internal/agent_identity.rs:106`,
  `src/run_internal/mod.rs:2272,2637`, `src/stream_events.rs:48`, `src/sandbox/sandboxes/unix_local.rs:850,876`),
  so my test targets have not been linted yet.

## 8. Span error attributes, snapshot precedence
- New `runtime::sandbox_span_result(name, data, fut)` records a failure like Python's
  `_apply_trace_finish_data`: span data gains `error.type`, `error_type`, `error_code` and (when set)
  `error_retryable`; the span error is `{message: "SandboxError", data: {operation, error_code[, error_retryable]}}`.
  A success records no error. `sandbox.create_session`, `sandbox.resume_session` and `sandbox.start` use it, and
  the `sandbox.stop` / `sandbox.shutdown` call sites in `unix_local.rs` and `docker.rs` now call it too
  (two small `sandbox_span` -> `sandbox_span_result` replacements per file; `sandbox_span` stays for infallible
  wrappers). Python records the concrete exception class name; Rust records `SandboxError` (the only error
  type the sandbox raises) and carries the discriminating value in `error_code`.
- Tests: `tests/py_sandbox_a_runtime_tracing.rs::a_failing_sandbox_operation_records_error_attributes_on_its_span`,
  `tests/py_sandbox_a_unix_local_4.rs::a_failing_stop_records_error_attributes_on_its_span` (a mount whose detach
  fails fails `stop`).
- Snapshot precedence (coordinator's change, `SandboxClient::configured_snapshot` + `runtime.rs::snapshot_spec`):
  run-config snapshot, then client-configured snapshot, then the SDK default local snapshot.
- Restore-time account reprovisioning is wired and its test flipped (see §7).

## 9. Final gates
- Tests: all of `py_sandbox_a_runtime_{1,2,3,4,5,6,unix_local,tracing}` and `py_sandbox_a_unix_local_4` pass on the
  latest runs.
- Clippy (`--lib` plus my test targets, `-D warnings`): the lib still stops on four lints in files I do not own,
  so my test targets are not linted by cargo: `src/run_internal/agent_identity.rs:106` (very complex type; add a
  `type` alias), `src/run_internal/mod.rs:2415,2783` (needless borrow; drop the `&`),
  `src/stream_events.rs:48` (large enum variant; box the large variant). The unix_local `Debug` lints from earlier
  runs are fixed. None of the four are in my files.
