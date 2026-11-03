# py-test-parity core17 r17c

Work list: `tests/sandbox/test_mount_security.py` (120), `tests/sandbox/test_runtime.py` (38),
`tests/extensions/sandbox/test_blaxel.py` (28), `tests/extensions/sandbox/test_vercel.py` (28), plus the changelog gap
"capability-changed path grants are not applied to an injected sandbox session".

## How the cases were found

The ledger has no per-case list. I enumerated every Python `test_*` function of the four files (134, 165, 227 and 104
functions), searched the whole Rust tree (`tests/`, `src/`) for each name, and read the body of every function with no
hit (70 in `test_mount_security.py`, 21 in `test_runtime.py`, 55 in `test_blaxel.py`, 31 in `test_vercel.py`).
Functions whose name does appear were ported by an earlier round and were not re-read. Many names that look missing are
cited by group (for example `test_extract_preview_url*`, `TestCoerceExecTimeout`) and are already ported. The ledger
counts are expanded `parametrize` rows, so the number of functions read does not equal the ledger counts; the CSV
subtracts only what this round moved from the ledger's `python_only` figure.

## Gap fix: capability-changed path grants on an injected session (FIXED)

Python replaces `session.state.manifest` with the processed manifest, so a capability that changes the non-host
`extra_path_grants` of an injected session takes effect (and an unchanged alias stays pinned to the directory it named
first). Rust kept the session's original manifest and the unix-local session pinned its grants at construction, so such a
grant was silently ignored.

- `src/sandbox/session.rs`: new `LiveGrants` cell and three defaulted trait methods, `live_grants`,
  `extra_path_grants` and `update_extra_path_grants`. The default path policy (`workspace_path_policy`) and the host
  copy source check (`entries.rs`) and lazy skill loading (`skills.rs`) now read `extra_path_grants()`.
- All nine session types hold a `LiveGrants` (unix-local, docker, E2B, Daytona, Vercel, Modal, Blaxel, Cloudflare,
  scripted). `InstrumentedSession` forwards the three methods.
- `src/sandbox/sandboxes/unix_local.rs`: pinned grants are re-pinned on update; a grant whose spelling is unchanged keeps
  its first pin (the alias re-point case), a new one is pinned to what it names now, and the per-exec confinement uses
  the live grants.
- `src/sandbox/runtime.rs`: `apply_live_session_manifest` compares and processes against the live grants and applies a
  changed grant list through `update_extra_path_grants`, also when nothing else changed. New
  `Manifest::with_extra_path_grants` replaces the list.
- Test: `tests/py_core17_r17c_injected_grants.rs` (port of
  `test_injected_session_uses_current_grants_without_rebinding_aliases`: upgrade a read-only alias grant, add and later
  make read-only a second grant, re-point the alias and see it refused, clear all grants).

## Source bugs found by porting

1. `InstrumentedSession` did not forward `shutdown_after_failed_close`, so an instrumented Vercel session whose `stop`
   failed (for example a raising event sink) was never shut down by `aclose`, leaving credentialed S3 mounts running
   (`src/sandbox/events.rs`). Mutation-checked: the new test fails without the forward.
2. Vercel did not serialize workspace commands while an S3 mount was being attached or detached (a documented 0.18.0
   gap). Added `s3_mount_operation` (tokio mutex with task-owner re-entry, activation allowed only for the task running
   `start`) around exec, read, bounded read, write, persist, hydrate, shutdown and the mount activation itself
   (`src/extensions/sandbox/vercel/mod.rs`, `mounts.rs`).
3. A manifest whose entry path leaves the workspace root hid every mount from the credential validation (the mount
   walk swallowed the path error), so a credentialed mount behind a bad path was never validated and the later
   failure echoed the path. `try_mount_targets` now reports the invalid path; `validate_mount_credential_boundaries`
   returns it with only the fixed protected-mount sentence when any credentialed mount is present, and
   `manifest_has_protected_mounts` finds mounts regardless of path validity (`manifest.rs`, `docker.rs`).
4. A runner-owned run did not validate the manifest's mount credential boundaries before asking the client for a
   session (the injected-session path did). It does now (`runtime.rs`), and both paths use the full
   `Manifest::validate_mount_credential_boundaries`, which also checks credential files.
5. A Blaxel read that fails with a non-404 status whose body says "not found" or "no such file" is now a
   `WorkspaceReadNotFound`, as Python classifies the SDK error text (`blaxel/sandbox.rs`).

## Tests added (all green)

`py_core17_r17c_injected_grants.rs` (1), `py_core17_r17c_runtime.rs` (3), `py_core17_r17c_mount_security.rs` (3),
`py_core17_r17c_blaxel.rs` (2), `py_core17_r17c_vercel.rs` (4).

## Per file

| file | python_only before | ported | covered | remaining |
|---|---|---|---|---|
| tests/sandbox/test_mount_security.py | 120 | 3 | 1 | 116 |
| tests/sandbox/test_runtime.py | 38 | 2 | 0 | 36 |
| tests/extensions/sandbox/test_blaxel.py | 28 | 2 | 2 | 24 |
| tests/extensions/sandbox/test_vercel.py | 28 | 4 | 1 | 23 |
| tests/sandbox/test_unix_local_file_io.py (gap case only; file is in `core16-r16a`) | 6 | 1 | 0 | 5 |
| **total** | 220 | 12 | 4 | 204 |

### test_mount_security.py (70 functions read)

Ported: `test_preserves_credentialless_hosted_mount_strategies` (Daytona, E2B and Blaxel hosted strategies need no
acknowledgement when credentialless and the exact-path one when credentialed),
`test_operation_error_with_implicit_broad_authority_is_replaced` (Azure blobfuse and S3 Files mounts),
`test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary` (source fix 3).
Covered: `test_manifest_input_cannot_inject_mount_credential_acknowledgement` by
`py_core8_r8g1_manifest.rs` (three of the five parametrized keys are exercised there; all five go through the one
refusal in `Manifest::from_value`).
Remaining, by group:

- Hostile or custom pydantic subclasses of mounts, strategies and patterns (rejected before deepcopy,
  `__getattribute__`, serializer, extra fields, forged class provenance, behaviour-only subclass, self-declared trusted
  boundary, registry replacement, forged `type`): about 23 functions
  (`test_manifest_acknowledgement_rejects_custom_mount_before_deepcopy`, `..._redacts_custom_provenance_traceback_locals`,
  `test_builtin_mount_subclass_*`, `test_direct_custom_mount_*`, `test_behavior_only_*`, `test_custom_*`,
  `test_unknown_sdk_strategy_subclasses_*`, `test_acknowledgement_requires_a_matching_provider_owned_strategy`,
  `test_rejects_configured_custom_mount_strategies_before_side_effects`, `test_session_state_rejects_inherited_...`).
  Reason: pydantic subclass, `__deepcopy__`, `__getattribute__`, descriptor and registry tricks; Rust mounts are trait
  objects and cannot be subclassed to override a built-in. DIVERGENCE RECORDED, NOT FIXED: a third-party
  `impl MountStrategy` is trusted (its `credential_authority` is believed), where Python refuses any custom strategy.
  The trait documents that custom implementations keep working unchanged, so rejecting them is a product decision
  (see Unresolved questions).
- Public error redactor decorators and hostile exception objects (`redact_mount_error_data` descriptors, `__cause__`,
  `__context__`, traceback frame locals, exception groups, `SystemExit`, `GeneratorExit`, `KeyboardInterrupt`,
  cancellation, read-only provider attributes, process-control replacement, ambient caller context, retained start
  failure on repeated calls): about 25 functions (`test_public_mount_error_redactor_*`,
  `test_rejection_redacts_sdk_traceback_frames_*`, `test_protected_*`, `test_operation_error_*`,
  `test_sync_operation_error_*`, `test_cancellation_with_mount_authority_*`,
  `test_start_preserves_retained_failure_and_context_on_repeated_calls`,
  `test_serialization_failure_redacts_mount_authority_from_sdk_traceback_frames`). Reason: Python exception object
  internals; Rust errors are structured values and the redaction contract (code, op and retry flag survive, message and
  context do not) is already asserted by `py_core12_r12a_mount_security.rs`.
- Manifest-in-state serialization and raw-state sanitizers (`serialize_session_state`, `deserialize_session_state`,
  `sanitize_raw_session_state_mount_authority`, `model_dump_json` round trip, slot and property backed state,
  aliased authority carriers): about 20 functions (`test_session_state_serialization_*`, `test_raw_state_*`,
  `test_legacy_non_inline_credential_file_source_*`, `test_deserialization_*`, `test_direct_state_*`,
  `test_slot_backed_*`, `test_property_backed_*`, `test_required_authority_carrier_dominates_aliased_optional_link`,
  `test_untrusted_nested_authority_owner_fails_closed_*`, `test_serialization_redacts_rclone_on_the_fly_remote_name`,
  `test_serialization_redacts_pattern_authority`). Reason: Rust `SandboxSessionState` carries a manifest digest and
  no manifest, so there is no persisted payload to sanitize; the structural guarantee is asserted by
  `py_sandbox_a_snapshot_1::state_json_never_contains_credentials_or_host_paths` and
  `py_sandbox_a_mount_security_3::a_credentialed_docker_volume_mount_is_redacted_from_durable_state`; the on-the-fly
  remote name is refused at activation (`py_sandbox_a_mount_security_2::an_on_the_fly_remote_name_never_reaches_a_command`).
  `Manifest::to_value` is the explicit `model_dump` analogue and is documented as carrying credentials.
- `test_docker_mount_examples_use_supported_external_strategies`: imports Python `examples/` modules.
- `test_custom_entry_at_credential_file_path_is_rejected`: a custom entry subclass; built-in entry kinds at the
  credential path are asserted by `py_sandbox_a_mount_security_3::a_credential_file_materialized_by_the_manifest_is_rejected`.

### test_runtime.py (21 functions read)

Ported: `test_runner_streamed_run_loop_task_waits_for_sandbox_cleanup_and_persisted_state` (the stream stays open while
`stop` blocks, then the finished run carries the released session's state and the agent runs again; the Python
`snapshot["marker"]` assertion has no Rust equivalent because the serialized state carries no custom snapshot field),
`test_session_manager_rebinds_persisted_path_grants_from_current_manifest` (the trusted manifest's grant reaches
`resume`).
Remaining (36 by ledger count): `deepcopy` of custom patterns and capability `clone()` identity (5, Rust capabilities
are stateless and bound by argument, see the trait docs), asyncio cancellation of pre-stop hooks
(`test_pre_stop_cancellation_skips_persistence_and_completes_cleanup`), private-hook failure of
`_aclose_dependencies` (Rust's dependency close is infallible), `SandboxSession` wrapper delegation of private path
helpers (`InstrumentedSession` delegates the whole operation to the inner session), missing packaged prompt resource
(`include_str!`), `SystemExit`/`KeyboardInterrupt` through `_process_manifest`, `SandboxRuntimeSessionManager.cleanup()`
with no session (no manager object), per-turn `prepare_agent` revalidation and re-materialization when `run_as` or the
cached session changes (a documented divergence: Rust prepares once per run and agents are immutable), start hooks that
rename the agent or replace its tools in place (3 functions x parametrization: agents are immutable in Rust; the
state-controlled variant is ported in `py_core16_r16g_sandbox_runtime.rs`), and the already counted covered cases
(`runner_streamed_guardrail_trip...`, `..._public_agent_for_call_model_input_filter`) which `core16-r16g` filed.

### test_blaxel.py (55 functions read)

Ported: `test_read_not_found_via_error_string` (source fix 5), `test_drive_strategy_build_docker_volume_returns_none`.
Covered: `TestValidateTarBytesExtra` (`test_unsupported_member_type`, `test_hardlink_rejected`) by
`py_core6_blaxel_archive::tar_validation_accepts_safe_archives_and_names_each_refusal`.
Already ported under group citations (not re-counted): `test_extract_preview_url*` (`preview_url_is_read_from_every_supported_shape`),
`TestCoerceExecTimeout` (`exec_timeout_is_coerced_into_the_request_body`; the negative case is unrepresentable because
`Duration` is unsigned), `test_exec_maps_sdk_api_error_504_to_timeout` and `test_exec_reraises_exec_timeout_error`
(`exec_maps_408_and_504_to_a_timeout`, `exec_timeout_is_an_exec_timeout`), `test_read_status_attr_on_error`
(`blaxel_read_404_is_a_not_found`).
Remaining, by group: import guards and package re-exports (`_import_blaxel_sdk`, `_import_aiohttp`,
`_has_aiohttp`, `_import_sandbox_api_error`, `test_blaxel_package_re_exports_backend_symbols`, 11 functions);
SDK duck typing and wrong object types (`test_read_returns_str`, `test_write_invalid_payload_type`,
`test_hydrate_invalid_payload_type`, `test_delete_wrong_type`, `test_assert_blaxel_session_wrong_type`,
`test_attach_drive_no_drives_api`, `test_detach_drive_no_drives_api`); pydantic `frozen`/`ge=1`/`exclude_args`
(4); async context manager and `close()` of the SDK client (2); asyncio cancellation during PTY start cleanup (2) and
PTY entry internals driven through private dataclasses (`_terminate_pty_entry`, `_collect_pty_output`,
`_prune_pty_sessions_if_needed`, ws close error: 11, the observable PTY behaviour is covered by `py_core6_blaxel_pty.rs`);
`supports_pty` without a sandbox URL (Rust derives the URL from the workspace and region); `_FakeFs.ls` (a test double).

### test_vercel.py (31 functions read)

Ported: `test_vercel_s3_nested_activation_serializes_workspace_commands` and
`test_vercel_without_s3_mounts_does_not_serialize_workspace_commands` (source fix 2),
`test_vercel_mount_command_timeout_includes_output_collection` (through `exec` with a timeout; the mount helper is
private), `test_vercel_s3_aclose_bypasses_failing_instrumentation` (source fix 1).
Covered: `test_vercel_s3_mount_upgrades_mountpoint_below_minimum` by
`sandbox_e2b_daytona_vercel_tests::vercel_installs_mountpoint_when_the_image_ships_an_old_one`.
Remaining, by group: `from_state`/constructor with a state that embeds the manifest and trusted mount maps (8,
Rust state carries no manifest; resume takes the trusted manifest as an argument and the topology check is
`py_core8_r8g3_vercel_live_manifest`); in-place mutation of `session.state.manifest` or the caller's manifest object
(`..._dynamic_mount_is_rejected_before_materialization`, `..._snapshots_trusted_create_time_configuration`,
`..._rejects_state_topology_changes_...`, `..._rejects_logical_path_and_root_changes_...`, 4; Rust takes the manifest
by value and the session's is immutable); asyncio cancellation and shielding (`..._entry_cancellation_...`,
`..._mount_cancellation_...`, `..._shutdown_cancellation_...`, `..._archive_cancellation_...`, task-local lock leak,
and the two traceback variants: 7); traceback and hostile-descriptor redaction (`..._redacts_full_activation_traceback`,
`..._ignores_hostile_exception_descriptors`, `..._shutdown_redacts_provider_failure`, `from_state` traceback redaction,
6; the Rust redaction contract is `redacted()` and `redact_for_protected_mounts`); SDK client identity
(`test_vercel_resume_does_not_read_token_or_scope_from_env`: the Rust client speaks REST and documents its environment
fallback; `..._deserialize_discards_surviving_resource_identity_after_mount_erasure` and
`..._snapshot_mode_resume_uses_native_snapshot_reference` need the Python SDK's snapshot objects: 3);
`test_vercel_package_re_exports_backend_symbols`.

### test_unix_local_file_io.py (gap case)

`test_injected_session_uses_current_grants_without_rebinding_aliases` is ported by `py_core17_r17c_injected_grants.rs`.
The other five cases of that file keep the reasons in `core16-r16a`.

## Residual differences

- `session_state()` of every backend still serializes `manifest.extra_path_grants()`, not the live grants, so an
  injected session whose grants a capability changed reports the original grants if a caller serializes it directly
  (Python replaces `state.manifest`). Runner-owned cleanup never serializes an injected session, so the impact is low.
- The CSV's fifth row (`test_unix_local_file_io.py`) is the gap case only; the file's remaining five cases are already in
  the `core16-r16a` row, so do not add the two rows' remainders.

## Unresolved questions

1. Third-party `impl MountStrategy` is trusted. Python refuses every custom strategy, mount and pattern with "custom
   mount implementations". Making Rust refuse implementations that are not built in (for example with a sealed marker)
   would break the documented extension point. Needs a product decision.
2. Python re-prepares the sandbox each turn (revalidating `cwd`, re-materializing tools when `run_as` changes); Rust
   prepares once per run. Unchanged, listed in the changelog.
3. Vercel: the `try_id()` of a caller outside a task (a `block_on` body) is treated as one owner, so two concurrent
   operations both started from such a body are not serialized. Spawned tasks are.

## Read-depth caveats

- Bodies of every function whose name had no Rust hit were read; the 70 hostile-exception and state-payload functions of
  `test_mount_security.py` were read in the first 25-40 lines (enough to see the hostile construct) rather than to the
  last assertion, and a few cancellation-heavy `test_vercel.py` bodies likewise. Functions whose names do appear in a
  Rust test were trusted to be ported.
- The `covered` counts rest on the ledger's expanded-row basis, which cannot be reconciled per function; some of the
  functions I list as already ported under group citations may be among the ledger's residue, so the remaining counts
  are upper bounds.
- The full suite result is in the final message of this round.
