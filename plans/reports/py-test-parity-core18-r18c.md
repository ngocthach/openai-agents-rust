# Python to Rust parity, round 18 (r18c): per-turn sandbox preparation and five sandbox files

## Part 1: "Python re-prepares the sandbox each turn; Rust prepares once per run"

### What Python does each turn

`run.py` (non-streamed) and `run_internal/run_loop.py` (streamed) call
`SandboxRuntime.prepare_agent(current_agent, current_input=<the run's original input>, ...)` at the
top of every turn. For an agent whose session and prepared clone are cached, that call:

1. re-binds `run_as` on the cached capabilities and rebuilds the clone only if `run_as` changed;
2. runs `_validate_workspace_scope` (`test -d` / `test -x` on the configured `cwd`) again;
3. runs every capability's `process_context` over the run's **original input** and writes the
   result back into `original_input` / `run_state._original_input` / `streamed_result.input`;
4. reuses the cached execution agent, whose instructions are the `_instructions` closure from
   `build_sandbox_instructions`. That closure awaits `capability.instructions(manifest)` **on every
   render**, i.e. before every model call, against the manifest captured at preparation.
5. opens one `sandbox.prepare_agent` span per turn and sets the active memory capability.

### Observable differences found, and what changed

| # | Difference | Observable? | Action |
|---|---|---|---|
| 1 | Capability instruction fragments: Python renders them before every model call; Rust rendered them once at preparation and froze them into the prompt | Yes. `Memory` reads `memories/memory_summary.md` from the session in its instructions, so a summary written by a tool on turn 1 is in Python's turn-2 system prompt and was missing from Rust's. Any custom capability with dynamic instructions differs the same way | **Fixed** (`src/sandbox/agent.rs`): the fragments are rendered inside the prepared agent's `Instructions::Dynamic` closure on every call, against the prepare-time manifest and binding, after the agent's own instructions (Python's order). New `Capability::caches_instructions()` (default `false`); `Skills` returns `true`, matching Python's per-capability skills-metadata cache, so a skill directory a tool creates mid-run is not listed (as in Python). `ConfiguredCapability` forwards it |
| 2 | `process_context` order relative to `call_model_input_filter` | Yes: Python's filter sees the processed input; Rust ran the caller's filter first and the capability chain on its output | **Fixed** (`src/sandbox/runtime.rs::install_per_turn_preparation`): the chain runs first, the caller's filter runs on its result |
| 3 | `process_context` scope and persistence: Python applies it to the original input only and persists the rewrite (`RunResult.input` / `to_input_list()` carry it); Rust applies it to the whole model input of each call (original input plus generated items) and leaves the run history unchanged | Yes in principle (a capability that rewrites, e.g. `Compaction` dropping items before a compaction item, changes `to_input_list()` in Python). No Python test asserts it; grep of `tests/sandbox` for `result.input`, `_original_input`, `calls[1].input` found none tied to `process_context` | **Not changed** (it needs a per-turn input hook in `src/run_internal/mod.rs` plus Rust's nested-history ownership reconciliation). Documented on `install_per_turn_preparation` as the remaining difference |
| 4 | `_validate_workspace_scope` probes per turn | Only through an exec log or a tool deleting the `cwd` mid-run (Python fails the next turn; Rust keeps going) | Not changed; remaining gap |
| 5 | One `sandbox.prepare_agent` span per turn vs one per run | Trace shape only | Not changed; remaining gap |
| 6 | `run_as` rebind on a cached agent; lazy session creation for a handoff target | Not observable in Rust: agents are immutable `Arc`s, and Rust's per-agent mode creates every graph agent's session up front (an existing documented design choice, see the note on `a_resumed_handoff_builds_the_new_agent_workspace_from_its_own_manifest`) | None |

Suggested replacement for the 0.18.0 changelog gap sentence ("Python re-prepares the sandbox each turn where
Rust prepares once per run"), for the next release section: *capability instructions are now rendered before every
model call and `process_context` runs before `call_model_input_filter`, as in Python; Python still applies
`process_context` to the run's original input only and persists the rewrite into `RunResult.input`, re-checks the
`cwd` and opens a `sandbox.prepare_agent` span every turn, where Rust filters each call's model input once
preparation is done.* (I did not edit `docs/project-changelog.md`: the sentence sits in a released section and the
release notes are written by the coordinator.)

### Tests (`tests/py_core18_r18c_sandbox_per_turn.rs`, 4, all pass)

Python has no runner-level test that asserts a second turn's system prompt, so these pin the behaviour of Python's
`_instructions` closure, `Memory.instructions` and `Skills._skill_metadata`:

- `capability_instructions_are_rendered_for_every_model_call` (counter fragment: `=1` then `=2`; failed before the fix)
- `a_memory_summary_written_mid_run_reaches_the_next_turn_prompt` (failed before the fix)
- `the_skills_index_is_resolved_once_per_prepared_agent`
- `the_call_model_input_filter_sees_the_processed_context` (failed before the fix)

Existing per-turn ports still pass: `py_sandbox_a_runtime_3::capability_context_processing_runs_once_per_turn`,
`py_sandbox_a_runtime_4::a_context_rewrite_does_not_duplicate_handed_off_history`.

## Part 2: python-only cases re-read

| file | python_only before | ported | covered | python-only after | reasons for what remains |
|---|---|---|---|---|---|
| tests/sandbox/test_compatibility_guards.py | 27 | 0 | 2 | 25 | `__all__` export surfaces (1 + 6 providers); dataclass / pydantic constructor field order (1 + 1 + 7 options + 9 session states) — Python introspection; Rust's `SandboxSessionState` is its own schema (`$schemaVersion`, `backend`, `provider_metadata`), not the pydantic field list |
| tests/extensions/sandbox/test_modal.py | 24 | 2 | 0 | 22 | asyncio cancellation settling of directory-snapshot persist/hydrate (11); `examples/sandbox/extensions/modal_runner.py` (4); in-process SDK stream polling of the PTY fallback reads (4); `PureWindowsPath` host paths (1); package re-exports (1); reconnect bounded by `sandbox_create_timeout_s` followed by a poll (1: the Rust helper reconnects per operation under the default bound) |
| tests/test_scripted_sandbox.py | 24 | 1 | 1 | 22 | `hasattr`/`dir`/type-hint introspection (1); `io.BytesIO`/`io.StringIO`/callable-stream snapshots and stream-type `TypeError`s (2 + 1 + 1 + 1 + 1 + 1 + 4); configured `ExecResult` mutated after queueing (1, ownership); `deepcopy` failure (1); dict step-spec validation (8, the builder makes them unrepresentable) |
| tests/sandbox/test_docker.py | 20 | 0 | 0 | 20 | docker-py HTTP response / exec socket / length-framed stdin / spool objects (857, 992, 1076, 1115, 1155, 1204, 1223); asyncio deferred-cleanup tasks (883, 909, 950); docker-py `images.pull` repo/tag split (1892: Rust passes the reference to `docker create` verbatim); `exec_run(workdir=None)` until the root is ready and its direct-state recovery (4017, 4785: Rust creates the container with `-w <root>`, so the daemon creates the root and every exec names it); PTY socket `sendall` errors and blocking `exec_create`/`exec_start` (4971 x2, 5081 x2: Rust's PTY is a `docker exec` child under the shared PTY manager); `Mount` subclass trust list (3635: `Mount` is a closed enum); resume cancellation (3406) and a failure injected into `_wrap_session` (3484) |
| tests/sandbox/test_unix_local_user_file_io.py | 19 | 16 | 0 | 3 | `normalize_path` monkeypatched to a different answer (rechecks captured authority); asyncio cancellation of the worker thread; the worker's private `_TRAVERSE_FLAGS` constant |
| **total** | **114** | **19** | **3** | **92** | |

### Ported (19)

`tests/py_core18_r18c_user_file_io.rs` (8 tests, 16 cases): a recording session (one dispatch, as the user, paths
resolved, stderr kept) and a real Unix-local workspace behind a `sudo` stand-in on the manifest `PATH`:

- `test_user_write_uses_shared_traversal_and_preserves_input` -> `a_user_write_carries_binary_contents_exactly`
- `test_user_lookup_failure_stops_before_file_io` (write x2) and `test_user_leaf_permission_failure_does_not_write`
  -> `a_refused_user_write_reports_a_write_error_and_moves_nothing`
- `test_user_rename_runs_in_the_worker_relative_to_both_parents` -> `a_user_rename_is_one_dispatch_naming_both_paths`
- `test_user_rename_failure_keeps_the_error` -> `a_failed_user_rename_keeps_the_error`
- `test_user_same_file_asks_the_worker` (x3) -> `a_user_same_file_check_asks_the_sandbox_as_that_user` (hard link,
  distinct files, symlink with `follow_symlinks=false`)
- `test_user_lookup_failure_stops_before_file_io` (ls x2) and
  `test_missing_system_python_does_not_retry_with_application_runtime[ls]`
  -> `a_refused_user_listing_is_one_dispatch_and_keeps_the_stderr`
- `test_user_file_operation_does_not_use_application_runtime[ls]` -> `a_user_listing_runs_only_as_that_user_in_the_sandbox`
- `test_user_listing_preserves_metadata_and_names` (x3) -> `a_user_listing_preserves_names_and_kinds` (symlink,
  socket, directory; a name with a newline and spaces). Rust's `FileEntry` has no owner/group names.

`tests/py_core18_r18c_modal_timeouts.rs` (2 tests): `test_modal_ensure_sandbox_bounds_app_lookup` and
`test_modal_ensure_sandbox_bounds_image_id_lookup` -> `creating_a_sandbox_bounds_the_app_lookup`; the extra test
`reconnecting_polling_and_terminating_are_bounded` pins the shutdown bounds of
`test_modal_shutdown_rehydrates_sandbox_and_terminates_without_wait_kwarg` (not counted: that case was not in the
python-only bucket). Runs the real `helper.py` through `tests/fixtures/modal_helper_driver.py` (new `call_timeouts` case).

`tests/py_core18_r18c_scripted_sandbox.rs` (1 test): `test_scripted_sandbox_retains_step_after_matcher_exception`
-> `a_failing_matcher_leaves_its_step_for_a_retry` (a Rust matcher fails by panicking; the step stays and the
session answers the retry).

### Covered (3)

- `test_scripted_sandbox_snapshots_supported_stream_call_arguments`: the recorded write payload is the bytes sent
  (`py_core5_leftover1_scripted_sandbox::scripted_sandbox_supports_capability_method_inventory`) and a returned call
  history is a detached copy (`...::scripted_sandbox_records_fifo_calls`).
- `test_optional_sandbox_discriminator_type_strings_are_stable[UnixLocalSandboxClientOptions]` and
  `[DockerSandboxClientOptions]`: `py_sandbox_b_compat_providers::every_backend_client_options_round_trip_their_type`
  asserts `type` = `unix_local` / `docker`. (Inferred: these are the two python-only rows among the 31 name-matched
  cases; `py_sandbox_b_compat::snapshot_and_backend_discriminators_are_stable` only checks the session-state rows.)

## Source changes (all in my areas unless noted)

- `src/sandbox/agent.rs`: capability instruction fragments rendered per model call (Part 1, #1); `CachedFragments`.
- `src/sandbox/capabilities/mod.rs`: new trait method `Capability::caches_instructions` (default `false`), forwarded by
  `ConfiguredCapability`; `process_context` doc now says "before each model call".
- `src/sandbox/capabilities/skills.rs`: `caches_instructions() = true`; the runtime skills root is listed as the run's
  user (`ls_as(root, run_as)`), as Python's `_resolve_runtime_metadata` lists with `user=self.run_as` (it already read
  each `SKILL.md` as that user).
- `src/sandbox/runtime.rs`: `install_per_turn_preparation` runs `process_context` before the caller's filter; doc
  states the remaining difference.
- `src/sandbox/session.rs`: new `SandboxSession::ls_as(path, user)` (Python `ls(path, user=...)`): default runs
  `ls -la --` through `exec` as the user. `src/sandbox/events.rs` forwards it.
- `src/sandbox/sandboxes/unix_local.rs`: `ls_as` override: the user's own `ls` must succeed under `sudo -u`, then
  the entries are read natively (exact names), the pattern `mkdir_as`/`rm_as` already use.
- `src/extensions/sandbox/modal/helper.py`: `App.lookup` bounded at 10 s; `Sandbox.from_id`, `poll` and `terminate`
  bounded at 30 s (Python's `_call_modal` bounds). A hung control-plane call no longer holds the helper forever.
- `tests/fixtures/modal_helper_driver.py`: new `call_timeouts` case.
- `src/run_internal/**`: **no edits**.

Breaking changes: none for callers. `Capability` and `SandboxSession` gained default methods (additive). Behaviour:
an error from a capability's `instructions`/`bound_instructions` now surfaces at the model call that renders the
prompt instead of at preparation (as in Python), and those are re-run every turn (a `Memory` read per model call).

## Gates

- `cargo test --all-features --lib` plus every test binary whose name matches
  sandbox/docker/unix_local/modal/vercel/e2b/daytona/blaxel/cloudflare/skill/memory/compat/removal/mount/capabilit/runtime
  (183 binaries + lib): 3,041 passed, 1 failed. The failure is
  `py_core8_r8g9_sqlx_memory::in_memory_compaction_is_abandoned_when_another_instance_mutates_the_suffix`
  ("pool timed out while waiting for an open connection" under load; matched only by "memory", no sandbox code); it
  passes when re-run alone (4/4).
- New binaries: `py_core18_r18c_sandbox_per_turn` 4, `py_core18_r18c_user_file_io` 8, `py_core18_r18c_modal_timeouts`
  2, `py_core18_r18c_scripted_sandbox` 1: all pass.
- `cargo clippy -D warnings` on lib + the four new test files, `--all-features` and default features: clean.
- rustfmt (`--edition 2024`) on every touched Rust file: clean. `src/lib.rs` untouched.
- `CARGO_TARGET_DIR` (`~/.cache/oars-out/r18c`) deleted.

## Read-depth caveats

- Part 1: read `run.py` / `run_loop.py` prepare sites, `SandboxRuntime.prepare_agent`, `runtime_agent_preparation.py`
  and the capability `instructions` implementations in full; the session manager (`runtime_session_manager.py`) only
  around `acquire_agent` / `ensure_session` usage.
- Part 2: every unmatched test body of the five files was read, except the long modal cancellation group (11 tests,
  read at head and assertion level) and the modal PTY fallback group (4 tests, two read in full). The ledger's modal
  bucket is marked "approximate"; which 24 of the 42 unmatched-by-name modal cases are the python-only ones was
  reconstructed from earlier reports (r12a, r16c), so the per-group counts there are a best reconstruction.
- `test_docker_write_rejects_workspace_symlink_to_nested_read_only_extra_path_grant` is counted covered in the ledger,
  but the cited Rust test only exercises the non-nested grant; not in my bucket, not changed.
