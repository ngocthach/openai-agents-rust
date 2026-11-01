# Round 10, r10c: per-agent sandbox resume and the Docker removal group

| file | python_only before | ported | covered | python-only after | reason for the rest |
|---|---|---|---|---|---|
| tests/sandbox/test_runtime.py | 66 | 12 | 0 | 54 | see "Remaining" |
| tests/sandbox/test_docker_removal.py | 74 | 10 | 0 | 64 | host-side removal service protocol |
| tests/sandbox/test_docker_removal_client.py | 14 | 0 | 0 | 14 | service bootstrap and binding |
| tests/sandbox/test_docker_removal_worker.py | 60 | 2 | 0 | 58 | in-container worker script, ctypes and descriptor doubles |

Totals: 24 ported, 0 covered, 190 remain python-only in these files.

## 1. Per-agent sandbox resume (`RunState.sandbox`)

`RunState` already round-trips a `sandbox` section opaquely (`src/run_state/sandbox_payload.rs` sanitizes mount authority; `wire.rs` reads and writes it). What was missing was anything that wrote or consumed it. Now:

- New `src/sandbox/resume_payload.rs` (crate-private `ResumeSource`): port of the payload logic in `SandboxRuntimeSessionManager` (`_resume_state_payload_for_agent`, `_assign_resumed_agent_key`, `_resumed_keys_by_name`, `_legacy_session_entries`, `_serialize_sessions_by_agent`, `_ensure_resume_key`). Section shape is Python's: `backend_id`, `current_agent_key`, `current_agent_name`, `session_state` (the agent the run stopped at) and `sessions_by_agent: {key: {agent_name, session_state}}`, keyed by the identity keys (`name`, `name#2`, ...).
- Resume: `SandboxRunner::run*` with `RunInput::State` reads `state.sandbox`; each agent of the graph resumes the session filed under its key (the key first, then the saved current-agent key for the agent the run stopped at, then an unclaimed entry of the same name, never the current agent's reserved entry for another agent). The section wins over `SandboxRunConfig::session_state`, as in Python. A section for another backend is refused with Python's message. Legacy sections (no `sessions_by_agent`, `current_agent_name`/`id` only) resume the current agent only. An entry that carries no `$schemaVersion` (another SDK's state) is skipped, so that agent starts a fresh session. New public `SandboxRunner::prepare_from_state`.
- Write: a run that pauses writes the section into `RunResult::to_state()` (non-streamed), and `SandboxStreamedRun::to_state()` / `sandbox_resume_state()` (streamed). Sessions of the resumed section that the run did not touch are carried unchanged; a session resumed from one key and re-filed under another drops its old entry. A caller-supplied live session clears the section; a run that finishes leaves the section it was resumed from.
- Old snapshots without the section read and resume exactly as before (`session_state` still works); `sandbox_tests::resumes_a_suspended_sandbox_from_persisted_state` still passes.

Tests: `tests/py_core10_r10c_resume_runs.rs` (10 integration cases on the real unix-local client with per-agent sessions: restore of every agent across handoffs, single agent plus tool-choice reset, same-named agents `sandbox`/`sandbox#2`, legacy section, backend mismatch, finished run keeps section, injected session omits it, untouched session carried, streamed run, and the negative control without the section) and 9 unit tests in `resume_payload.rs` (key matching, reservation of the current key, legacy, foreign state, untouched entries, key collisions).

## 2. Docker removal group

Decision followed: no root-privileged host-side removal helper. I read every case of the three files. Python's `DockerRemovalService` pauses the container, spawns a host worker that enters the container's mount namespace, binds descriptors and removes there; the Rust client always removes in the container with `docker exec rm`. Cases that assert request/argument shapes the Rust client produces were ported through a recording `DockerRunner` (`tests/py_core10_r10c_docker_removal_shapes.rs`, 8 tests) and the local client (`tests/py_core10_r10c_removal_local.rs`, 2 tests):

- recursive removal argv and user (`rm -r -f -- <abs>`, `-u developer`; workspace-relative and writable-grant targets; no user means no `-u`): `test_recursive_removal_preserves_unrelated_writable_trees` x2.
- `link/../build` is folded before the resolver and `rm` see it: `test_relative_parent_segments_use_the_normalized_request`.
- a workspace root that is an alias keeps its spelling (`<alias>/build`), accessible variant: 1 of the 2 `test_relative_removal_preserves_workspace_alias_traversal`.
- a failed removal ran as the requested user: `test_missing_target_still_checks_the_requested_user`.
- delete outcomes, container half (removed, not found at lookup, not found at remove, lookup error, remove error): the 5 `test_delete_releases_authority_only_after_confirmed_container_removal`. The authority-release half is service state.
- literal-backslash path in a read-only grant is not a traversal: `test_removal_preserves_protected_literal_backslash_tree`; a read-only grant inside the workspace does not block later removals: non-resume variant of `test_workspace_overridden_grant_does_not_invalidate_later_removal`.

Also: an ancestor of a read-only grant vetoes a recursive removal (argv-level, no `rm` issued); it is Python's own `validate_recursive_remove`, already covered by `py_sandbox_b_removal.rs`, so it adds no ledger credit.

### Source fix

`src/sandbox/sandboxes/docker.rs` `shutdown`: `docker rm -f` of a container that is already gone ("No such container") was reported as `WorkspaceStopError`; Python's `delete` treats `docker.errors.NotFound` as removed. Now the container is considered released (volumes are still removed), and a second shutdown does not call `docker rm` again.

### Remaining (valid python-only)

- test_docker_removal.py (64): pause/unpause ordering, bind/inspect/remove worker protocol, bound-root refusal (`docker_removal_bound_root`, a service feature: without the service Python removes the workspace root too), service resource limits and concurrency bounds, host-service uid/namespace/security-mode checks, cancellation of host threads, live-manifest authority transfer. Reason: monkeypatched Python internals of the host service, plus a privileged component the owner decided not to build.
- test_docker_removal_client.py (14): workspace bootstrap and strict binding happen inside `DockerRemovalService._bind_new`; every case asserts service bindings or worker calls.
- test_docker_removal_worker.py (58): the in-container worker script with ctypes, O_PATH descriptors, `/proc/<pid>` namespace entry and syscall doubles. The one resume variant of the overridden-grant case (`_clear_workspace_dir_on_resume_pruned`) needs the worker too.

## 3. Remaining in tests/sandbox/test_runtime.py (54)

Unchanged groups from r8g2/r9gaps, plus:

- `test_runner_merges_sandbox_instructions_and_tools`, `test_runner_streamed_cleans_runner_owned_session` (only their `_sandbox` assertions): Python writes the section after a run that *completed*; Rust writes it only for a suspended run, since a completed run releases its session and nothing is resumable from it. Kept as a documented deviation, no new source.
- `test_runner_omits_sandbox_resume_state_when_cleanup_fails`: follows from the same rule (a completed run carries no section).
- `test_runner_streamed_run_loop_task_waits_for_sandbox_cleanup_and_persisted_state`: asyncio `run_loop_task` and `asyncio.Event` internals.

## Breaking and additive changes

- Additive: `SandboxRunner::prepare_from_state`, `SandboxStreamedRun::{to_state, sandbox_resume_state}`; `RunResult::to_state()` now carries `sandbox` for a paused client-owned sandbox run (it was `None` before).
- `SandboxRunner::run*` given a `RunState` with a `sandbox` section now resumes from it (before it ignored the section and used only `SandboxRunConfig::session_state`).
- A paused streamed run needs `SandboxStreamedRun::to_state()` (not `streaming.to_state()`) to carry the section.
- Behaviour: `docker rm` of a missing container no longer fails `shutdown`.

## Unresolved

- Streamed runs find the current agent by name only (`StreamCompletion` has no agent handle), so for two same-named agents the section's current key is the first of that name.
- Python also writes the section after a completed run; not mirrored (see above).

## Gates

See the final message of the run for the exact totals.

Status: DONE_WITH_CONCERNS
