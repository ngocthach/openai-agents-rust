# Python→Rust test parity — sandbox-a / memory-snapshot

Python: `/Users/ngocthach/myGit/openai-agents-python/tests/sandbox/`.
Rust tests (all green, 28 + 19 + 31 + 15 = 93 Rust tests, 0 `#[ignore]`):
- `tests/py_sandbox_a_snapshot_1.rs` (28): test_snapshot_defaults.py + test_session_state_roundtrip.py
- `tests/py_sandbox_a_snapshot_2.rs` (19): test_snapshot.py
- `tests/py_sandbox_a_memory_1.rs` (31): test_memory.py, session-free half
- `tests/py_sandbox_a_memory_2.rs` (15): test_memory.py, SandboxRunner-driven half

Counting: `grep -c '^\s*\(async \)\?def test_'` = 10 / 22 / 32 / 64. Parametrize expansions are not
counted separately; they are folded into one Rust test looping over the same cases.

## 1. Per-file table

| file | #tests | ported | covered | python-only | failures found |
| --- | --- | --- | --- | --- | --- |
| test_snapshot_defaults.py | 10 | 10 | 0 | 0 | 0 (API gap: platform/env/home not injectable, added) |
| test_session_state_roundtrip.py | 22 | 8 | 0 | 14 | 3 Rust bugs |
| test_snapshot.py | 32 | 22 | 0 | 10 | 0 (1 backend divergence deferred) |
| test_memory.py | 64 | 47 | 0 | 17 | 7 Rust divergences fixed, 2 deferred |
| total | 128 | 87 | 0 | 41 | |

No test is classified COVERED: existing Rust coverage (src unit tests, sandbox_tests.rs,
sandbox_core_parity_tests.rs) does not assert the Python-exact behavior, so everything was ported.

### test_snapshot_defaults.py (10, all ported, snapshot_1)
8 base-dir tests -> `default_local_snapshot_base_dir_*` (platform/env/home injected through the new
`default_local_snapshot_base_dir_for`); cleanup -> `cleanup_stale_default_local_snapshots_removes_only_old_tar_files`
(mtime set with `File::set_modified` instead of Python's injectable `now`);
resolve keeps stale -> `resolve_default_local_snapshot_spec_keeps_existing_stale_files`.
Extras: 0o700 directory mode, zero-arg resolver agrees with host. Difference: the managed subdir is
`openai-agents-rust/sandbox/snapshots` (Python: `openai-agents-python/...`); asserted as the Rust name.

### test_session_state_roundtrip.py (22)
Ported (8), snapshot_1: #6 `session_state_json_uses_the_documented_keys` (exact serialized key set,
snapshot sub-document `{"type","base_path"}`); #9 `parse_rejects_invalid_payloads`; #16
`exposed_ports_are_normalized`; #17 `exposed_ports_reject_invalid_values`; #18
`serialization_redacts_host_paths_and_rebinds_from_the_trusted_manifest` (+ missing-grant refusal);
#19 `path_only_grants_preserve_a_direct_resume_round_trip`; #21
`removed_redaction_marker_does_not_restore_a_host_backed_grant`; #22
`deserialization_discards_an_unmarked_serialized_host_path`.
Extra Rust tests: full-field round trip, mount-authority redaction + provider-id sanitising,
credential/host-path leak check, manifest digest topology, `verify_manifest`, SDK-owned root.
Python-only (14): subclass registry / `parse()` / `model_validate_json` / `exclude_unset` / type-literal
registration (#1,2,3,7,8,14,15: pydantic subclass machinery; Rust has one concrete state type);
manifest-embedded `EnvValue` subclass and legacy StrEnvValue payloads (#4,5: Rust state never embeds the
manifest); traceback-frame-locals redaction (#10,11,12,13: Python frames); no-deepcopy of extension
state (#20).

### test_snapshot.py (32)
Ported (22): #1 and #4 (state round trip incl. fingerprint + local snapshot, snapshot_1
`session_state_round_trips_through_json_with_every_field`); #2 (snapshot_1 keys test); #5
`snapshot_spec_always_serializes_its_type_discriminator`; #6 `local_snapshot_restorable_requires_file`;
#11 `noop_snapshot_stop_skips_workspace_persist`; #12 `non_noop_snapshot_stop_persists_workspace`;
#14 `non_noop_snapshot_stop_records_snapshot_fingerprint`; #15
`start_skips_snapshot_restore_when_live_workspace_fingerprint_matches`; #16 (2 params)
`start_compares_legacy_fingerprint_before_restoring_workspace` (independent `tar` + sha256 v1 hash);
#18 `start_restores_snapshot_when_live_workspace_fingerprint_mismatches`; #19 (2 params)
`start_restores_snapshot_when_resume_manifest_changes`; #20 `start_applies_full_manifest_for_fresh_non_restorable_backend`;
#21 `start_reapplies_only_ephemeral_manifest_for_preserved_non_restorable_backend`; #22 DEFERRED (below,
`start_with_unready_workspace_state_applies_the_full_manifest`); #23
`start_applies_full_manifest_when_preserved_non_restorable_workspace_unproven`; #25 (9 params)
`local_snapshot_rejects_non_basename_ids`; #26 `local_snapshot_persist_is_atomic_on_failure` (adapted:
Rust persists from a buffer, failure induced with a read-only directory); #27
`remote_snapshot_persist_restore_and_restorable_use_the_registered_client`; #28
`remote_snapshot_spec_builds_remote_snapshot`; #29 (snapshot_1 `remote_snapshot_serializes_through_session_state_without_its_client`;
Rust key is `client_key`, Python `client_dependency_key`, and no `id` in the spec); #31
`session_persist_uploads_through_the_registered_remote_client` (client resolved by registry key, not
`set_dependencies`).
Drives `persist_session_snapshot` / `restore_session_snapshot` through a fake `SandboxSession` that
execs on the host, so the fingerprint helper script runs for real. #20-#23 use the Unix-local backend.
Python-only (10): #3 exclude_unset; #7 #9 #10 #8 custom snapshot type registry / duplicate / missing
`type` / frozen model; #13 #17 stream `close()` call counts; #24 `system_state_preserved` (no Rust
concept; account provisioning is not observable); #30 duck-typed client without `exists` (Rust trait
requires it); #32 `SandboxSession` wrapper delegation (no wrapper type).

### test_memory.py (64)
Ported (47): rollout file name (3); build_rollout_payload (5); phase-one prompt truncation (1); prompts (3);
`_updated_at_sort_key` (1, adapted to the parsed value) + phase-two selection (1); layout dirs (2 defs,
8 params); requires read/generate (1); generate-config limits/defaults/typed+disabled settings (5:
non-positive, too many, defaults, renamed field, typed settings; defaults positional test folded in);
memory capability instructions (6: none without summary, truncated summary, live update, absolute with
cwd, layout spelling x2); runner e2e (17): writes rollouts, no-memory-capability, custom layout,
multiple layouts, validated rollout id, groups segments (adapted to explicit `rollout_id`),
conversation_id + group_id fallback (DEFERRED, one test), per-run rollouts, sanitise/truncate prompt,
caps phase-two selection, phase-two turn limit (via flush), consolidation turn budget (3 params),
turn exhaustion releases session, close runs both phases, enqueue failure (adapted), interrupted runs.
Python-only (17): `_sandbox_memory_input` + streaming input override x4 (Python run internals);
`instructions` without bound session (Rust types enforce it); dict / subclass / type-hint / forward-compat
reasoning / invalid-type model-settings coercion (5, pydantic/dataclass typing); per-session manager
registry conflicts x3 and registry unregister x1 and SDK-session-derived fallback x1 (Rust builds one
manager per prepared session, so the conflicts cannot occur); BaseException / cancellation worker tests x2
(asyncio worker task); log-redaction part of enqueue-failure (counted above as ported for the run/teardown part).
Rust-only additions: `flush_runs_once`, `capped_selection_prefers_dated_rollouts_over_unknown_ones`,
`terminal_metadata_serializes_every_field`, `memory_build_manager_validates_the_layout`.

## 2. Rust bugs fixed (files I own)

- `src/sandbox/state.rs`: `path_grants_require_rebind` was set for any extra path grant; now only for grants
  with a host path or host-path mounts, so path-only grants resume directly (Python #19).
- `src/sandbox/state.rs`: `with_exposed_ports` now dedupes order-preserving; `from_json_str` rejects port 0.
- `src/sandbox/snapshot.rs`: added `default_local_snapshot_base_dir_for(home, env, platform, os_name)` with
  Python's rules (Windows-absolute check hand rolled; relative/posix-absolute env values ignored) and
  `resolve_default_local_snapshot_spec_at` (creates dir, chmod 0700). Zero-arg fns delegate.
- `src/sandbox/memory/config.rs`: layout errors now use Python's three messages (`must be relative` /
  `must not escape root` / `must be non-empty`); limit errors use Python's exact text.
- `src/sandbox/memory/manager.rs`: rollout-id error text is Python's; added `rollout_file_name_for_rollout_id`;
  phase one now writes `rollout_summaries/{rollout_id}_{slug}.md` (was `{slug}.md`), the summary gets Python's
  `session_id/updated_at/rollout_path/terminal_state` header, `updated_at` / `terminal_state` come from the last
  recorded segment (was the wall clock); `enqueue_rollout_payload` writes the canonical `rollout_id` right after
  `updated_at`.
- `src/sandbox/memory/rollouts.rs`: payload key order and shape match Python (`interruptions` only when paused,
  serialized as the raw tool-call dicts via new `build_rollout_payload_with_interruptions`; `final_output` only when
  present); `terminal_metadata` serializes every field including nulls; role filter falls through to the type for
  unknown roles; phase-one terminal metadata JSON is `sort_keys + indent=2 + (",", ":")` like Python (was
  serde pretty `": "`).
- `src/sandbox/memory/storage.rs`: capped selection ranks unknown/missing `updated_at` below every real timestamp
  (new `updated_at_sort_key`; was plain string compare).

## 3. Cross-area bugs deferred / existing tests that need a controller update

- (RESOLVED, edited with controller permission) `tests/sandbox_tests.rs` `memory_generation_writes_artifacts_at_session_close` asserted the old summary path `memories/rollout_summaries/fix-the-build.md` with exact body
  `"Fixed the failing build."`. Python parity (test_memory `rollout_summaries/{id}_{slug}.md`, header) is the
  contract. Minimal fix: read `memories/rollout_summaries/rollout-1_fix-the-build.md` and assert it
  `contains("Fixed the failing build.")`. I may not edit that file.
- `src/sandbox/runtime.rs` (`prepare`, ~line 367, `rollout_id`): Python names the rollout after the SDK session id,
  else `conversation_id`, else `group_id`, else `run-<uuid>`; Rust only uses `SandboxRunConfig::rollout_id`, the
  resumed state, or a bare uuid. Minimal fix: thread the run's conversation id / group id into the fallback and
  prefix generated ids with `run-`. Tests assert the current behavior with a `// DEFERRED` comment
  (`sandbox_memory_without_a_rollout_id_generates_one_per_run`).
- `src/sandbox/runtime.rs`: each run builds its own memory manager, so a caller-owned session reused for N
  runs under one `rollout_id` has N managers; flushing all of them would run phase one N times on the same
  rollout file. Tests flush only the last manager. Python shares one manager per session.
- `src/sandbox/sandboxes/unix_local.rs` (`start`, resume branch): Python re-applies only the ephemeral entries
  for a preserved workspace even when `workspace_root_ready` is false (probe succeeds); Rust re-applies the whole
  manifest. Asserted as current behavior in `start_with_unready_workspace_state_applies_the_full_manifest`
  (DEFERRED comment). Not a regression risk (full apply is the safer reading).
- Not mine, seen while running: `tests/sandbox_core_parity_tests.rs::a_raw_write_file_is_refused_under_a_read_only_grant`
  fails because `unix_local.rs` now rejects `host_path` grants (`SandboxConfigInvalid`); `cargo clippy -D warnings`
  currently fails in `src/extensions/models/mod.rs:1016`, `src/models/openai_responses_ws.rs:553,556`,
  `src/sandbox/dependencies.rs:188`, `src/sandbox/sandboxes/unix_local.rs:1149`, `src/sandbox/types.rs:461`
  (dead `name_after_fields`), and in other areas' test files (`py_mcp_manager.rs`, `py_sandbox_a_unix_local_2.rs`,
  `py_sandbox_a_mount_security_3.rs`).

## 4. `#[ignore]`d tests

None. `local_snapshot_persist_is_atomic_on_failure` returns early when run as root (read-only directory does not
stop root); it is `cfg(unix)`.

## 5. Shared-file edits needed

None for `src/lib.rs` / `Cargo.toml`. New public items are reachable through existing `pub mod` paths
(`sandbox::snapshot`, `sandbox::memory::*`, re-exported from `sandbox::memory`).

## Verification

`c.sh test --all-features --no-fail-fast --test py_sandbox_a_snapshot_1 --test py_sandbox_a_snapshot_2
--test py_sandbox_a_memory_1 --test py_sandbox_a_memory_2`: 28 + 19 + 31 + 15 passed. `cargo test --lib sandbox::`:
299 passed. Clippy `-D warnings` on `--lib` + my four test targets is clean for my files once other areas' lints
(listed above) are allowed.


---

# Extra: test_runtime.py lines 3999-end (runtime tail)

Files: `tests/py_sandbox_a_runtime_5.rs` (17), `_6.rs` (12), `_7.rs` (10) = 39 Rust tests, all green, 0 `#[ignore]`,
clippy clean for them. Count: 82 `def test_` from line 3999 (awk `^(async )?def test_`), 20 parametrize
decorators (expansions folded into one Rust test or noted). Line 3999 itself (`rebinds_redacted_external_mount_authority`)
sits on the boundary with the runtime helper's range; taken here.

| group | py tests | ported | covered | python-only |
| --- | --- | --- | --- | --- |
| session manager (3999-4783) | 22 | 15 | 2 | 5 |
| unix_local exec/file io (5099-5630) | 13 | 4 | 0 | 9 |
| runner/clone/misc (5630-5770) | 7 | 2 | 0 | 5 |
| apply/start/stop/resume lifecycle (5773-5955) | 8 | 6 | 2 | 0 |
| clear on resume + prepare_agent (5955-6525) | 13 | 1 | 12 | 0 |
| spans, handoff, memory, resume (6525-6866) | 9 | 6 | 1 | 2 |
| isolation, hooks, mcp, provider (6866-end) | 10 | 4 | 0 | 6 |
| total | 82 | 38 | 17 | 27 |

(Ported = 38 distinct Python tests; 39 Rust tests because the DEFERRED rows and one adapted test split in two.)

## Ported (Rust test -> Python test)
runtime_5: `resume_receives_the_trusted_manifest_with_its_mount_entries` (rebinds_redacted_external_mount_authority; state half
COVERED by `py_sandbox_a_snapshot_1::mount_authority_is_redacted_and_blocks_a_bare_resume`);
`a_capability_host_path_grant_reaches_resume_exactly_once` (rebinds_capability_host_path_grant_once);
`resume_without_the_trusted_host_path_is_refused_by_name` (rejects_unmarked_serialized_host_path, real Unix-local resume);
`resume_adds_the_run_as_user_to_the_manifest`; `deferred_run_as_user_is_duplicated_when_a_group_already_has_it`
(does_not_duplicate_run_as_user_from_group); `capability_edits_reach_*` x3 (applies_capability_manifest_mutations_with_session_parity,
3 params); `a_stopped_injected_session_is_started_and_given_the_capability_file`;
`deferred_unsafe_stopped_injected_session_manifest_is_not_refused_at_preparation` (rejects_unsafe_stopped..., 2 params, one test);
`deferred_a_changed_host_source_on_an_injected_session_is_not_refused` + `deferred_mixed_duplicate_grant_target_is_not_refused`
(rejects_stopped_injected_session_host_mount_changes, 2 params); `a_running_injected_session_materializes_the_capability_file`;
`a_failed_delta_apply_is_retried_on_the_next_preparation`; `an_unchanged_running_session_is_not_rematerialized`;
`adding_an_account_to_a_running_injected_session_is_refused`; `a_run_over_an_injected_session_returns_no_resume_state`
(omits_existing_payload_for_injected_live_session).
runtime_6: `applying_a_manifest_reports_account_provisioning_failures`; `applying_only_ephemeral_entries_skips_account_provisioning`;
`applying_a_manifest_refuses_unacknowledged_mount_authority`; `starting_a_session_refuses_unacknowledged_mount_authority_before_materializing`;
`a_failing_snapshot_persist_is_reported_once` + `an_unsafe_mount_is_refused_before_snapshot_work` (session_stop_redacts_... persist half,
session_stop_rejects_mutated_unsafe_mount...); `deferred_restore_does_not_reprovision_accounts` +
`restore_without_account_reprovisioning_clears_and_hydrates_once` (resume_reprovisions..., resume_can_skip...);
`unix_local_file_io_allows_an_extra_path_grant`; `unix_local_file_io_rejects_a_write_under_a_read_only_grant`;
`unix_local_exec_runs_in_the_workspace_root` (runs_without_wrapper_on_linux);
`deferred_unix_local_exec_is_not_confined_by_sandbox_exec` (exec_confines_commands_to_workspace_root).
runtime_7: `a_sandbox_run_persists_only_the_new_session_input_items`; `streamed_events_name_the_public_agent`;
`non_function_tool_outputs_name_the_public_agent`; `hooks_and_results_keep_the_public_agent_identity`;
`a_handoff_target_is_prepared_with_its_own_capabilities`; `the_memory_manager_follows_the_prepared_agent_generate_setting` +
`an_agent_that_adds_the_memory_capability_gets_a_manager` (2 memory-handoff tests); `one_sandbox_agent_can_serve_concurrent_runs`
(adapted opposite of rejects_concurrent_reuse); `a_shared_capability_is_bound_per_run`; `provider_defaults_and_capability_settings_are_merged`
(2 params streamed/non-streamed folded; env var half not ported).

## Covered (citations)
- test_session_manager_rejects_unmarked_serialized_host_path (state half): `tests/sandbox_core_parity_tests.rs::persisted_state_drops_the_host_paths_and_demands_them_back`.
- test_session_start_redacts_external_mount_operation_failure: `tests/py_sandbox_a_docker_3.rs::a_failed_create_redacts_the_provider_error_and_removes_generated_volumes`.
- test_session_stop_redacts_external_mount_snapshot_failure: `tests/py_sandbox_a_docker_3.rs::persist_redacts_a_provider_error_for_a_protected_mount`.
- clear_workspace_root_on_resume x3: `py_sandbox_a_runtime_2.rs::clearing_the_workspace_on_resume_preserves_a_nested_mount`,
  `clearing_the_workspace_deletes_a_file_standing_where_a_mount_parent_belongs`, `a_mount_at_the_workspace_root_stops_the_resume_clear_entirely`.
- prepare_agent: rechecks_session_liveness `py_sandbox_a_runtime_3.rs::a_second_preparation_restarts_a_stopped_session`; binds_run_as_to_cloned_capabilities
  `preparation_binds_run_as_without_probing_a_working_directory`; binds_and_validates_run_workspace_scope `preparation_probes_the_working_directory_and_binds_the_scope`;
  serializes_cwd_probe `the_working_directory_probe_uses_posix_sandbox_paths`; rejects_inaccessible `an_inaccessible_working_directory_stops_preparation_after_one_probe`;
  processes_context `capability_context_processing_runs_once_per_turn`; starts_new_live_session `a_runner_created_session_is_started_even_when_the_backend_reports_running`.
- revalidates_cwd_after_restarting_cached_session / when_cached_agent_run_as_changes: Rust has no prepared-agent cache (every `prepare` re-probes), so
  the probe-per-preparation behaviour is `runtime_3::a_second_preparation_restarts_a_stopped_session` + `the_working_directory_probe_uses_posix_sandbox_paths` (counted COVERED, 2).
- runner_restores_sandbox_from_run_state, duplicate-name/legacy payload resume roundtrips (3, macOS-native): resume through a persisted state is
  `tests/sandbox_tests.rs::resumes_a_suspended_sandbox_from_persisted_state` (1 COVERED; the 3 duplicate-name/legacy are PYTHON-ONLY below).

## Python-only (one line per group)
- Session-manager internals / RunState payload plumbing: `redacts_capability_failure` x2, `redacts_authority_added_before_capability_failure`,
  `process_manifest_preserves_value_free_process_control` x2 (Rust `process_manifest` is infallible and exceptions do not exist),
  `validates_running_manifest_update_before_materialization` (no backend validation hook), `preserves_existing_payload...` (RunState._sandbox dict),
  `uses_run_state_starting_agent_for_duplicate_resume_keys`, `restores_duplicate_name_sessions...` and the two macOS-native roundtrips (Rust rejects duplicate
  agent names: `tests/sandbox_tests.rs::rejects_duplicate_agent_names_in_the_handoff_graph`; legacy `current_agent_id` payload is a Python format).
- macOS `sandbox-exec` profile builders (`_confined_exec_command` x4 incl. params, `_darwin_exec_profile` x3, launch-services ctypes check,
  confinement-unavailable monkeypatch): Python-private; the feature itself is absent in Rust (cross-area gap below).
- Capability clone deepcopy x4 and `deep_clones_capability_runtime_state` (pydantic deepcopy of mutable capability fields).
- `rematerializes_cached_tools_when_run_as_changes` (prepared-agent cache identity), `agent_as_tool_uses_runner_sandbox_prep` (Rust `Agent::as_tool` of a SandboxAgent
  is not sandbox-prepared; `PreparedSandboxGraph::process_context` passes nested tool agents through: gap, see below), `runner_rejects_concurrent_reuse` ported inverted.
- Hook-driven agent mutation: `start_hook_state_controls_prepared_sandbox_tools` x4, `start_hooks_refresh_sandbox_application_tools` x16,
  `start_hooks_control_sandbox_mcp_filters` x8, `mcp_names_remain_distinct_from_conditional_handoffs` x2 (hooks mutate the public `Agent` name/tools; Rust hooks take `&Agent`).
- `runner_protected_start_preserves_caller_error_without_exposing_it` x3 (Python exception `__cause__`/`__context__` chaining).
- `sandbox_runtime_emits_high_level_sdk_spans`: Rust sandbox runtime emits no `sandbox.*` custom spans (gap, not python-only; counted here as not ported).

## Rust bugs fixed in this round
None in my files: all runtime-tail divergences are in files I do not own (below).

## Cross-area bugs deferred (file:line, Python expectation, minimal fix) -- tests assert current behavior with `// DEFERRED`
- `src/sandbox/runtime.rs` `apply_live_session_manifest` / `manifest_topology_matches`: early return ignores `extra_path_grants`, so a changed host source on an
  injected session is accepted silently. Fix: compare extra path grants there.
- `src/sandbox/runtime.rs` `host_path_grant_topology`: keeps only host-backed grants; Python keeps every grant whose path is a host-backed target
  (mixed-duplicate-target must be refused). Fix: filter on `mounted_targets.contains(&grant.path)`.
- `src/sandbox/runtime.rs` `prepare`, injected branch: starts the session before validating the processed manifest; Python refuses an unsafe (credentialed)
  stopped session before start (`start_calls == 0`). Fix: authorize mount targets before `session.start()`.
- `src/sandbox/agent.rs` `resolved_manifest` run_as block: does not look at group members; `run_as` user in a group is added to `users` too. Fix: also skip when a group lists it.
- `src/sandbox/session.rs` `restore_session_snapshot`: never reprovisions manifest accounts after hydrate (Python: `resume_reprovisions_manifest_accounts...` raises on failing useradd).
- `src/sandbox/errors.rs` `exec_nonzero` message: Python is "stdout: ...\nstderr: ..."; Rust "Command `...` exited with code N: <stderr>" (context fields match).
- `src/sandbox/runtime.rs` `prepare`: memory manager is built from the starting agent only; Python switches the manager as handoffs change the active agent.
- `src/sandbox/sandboxes/unix_local.rs`: no macOS `sandbox-exec` confinement of `exec` (Python confines reads/writes to the workspace + grants).
- Rust has no `sandbox.prepare_agent` / `sandbox.start` / `sandbox.cleanup` ... custom spans (`runtime.rs`), and `SandboxAgent` used via `Agent::as_tool` is not sandbox-prepared.
- Earlier items still open: rollout id derivation (session/conversation/group id), manager sharing across runs on one session, resume re-applying only ephemeral
  entries on an unready workspace. Flip the memory/unix-local `DEFERRED` assertions once the runtime helper lands them.

## `#[ignore]`
None.

## Final status
Gates: all 7 files (snapshot_1/2, memory_1/2, runtime_5/6/7) pass; `cargo test --test sandbox_tests` 65 pass (after the controller-approved edit of
`memory_generation_writes_artifacts_at_session_close`); clippy on `--lib` + these targets clean (other areas' lints allowed).
