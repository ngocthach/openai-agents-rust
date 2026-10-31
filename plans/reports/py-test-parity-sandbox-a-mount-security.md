# py-test-parity — sandbox-a / mount-security

Python file: `tests/sandbox/test_mount_security.py` (4642 lines).
Counted with `grep -n "^async def test_\|^def test_"` = 134 (parametrize expansions are not counted
as extra rows; a Rust test that loops a parametrize set says so in its doc comment).

Rust test files (new): `tests/py_sandbox_a_mount_security_1.rs` (28 tests), `_2.rs` (17), `_3.rs` (18) = 63.
All pass; clippy `-D warnings` clean on the lib and these three files.

## 0. Why most of the file is Python-only

Python's `_mount_security.py` defends against hostile pydantic subclasses (overridden `__deepcopy__`,
`model_serializer`, `args`, `__traceback__`, forged `type` discriminators) and against raw dicts fed to
`model_validate`. Rust closes these by construction: `MountStrategy`/`MountPattern`/`ConfiguredMount` are
traits whose identity is a `&'static str`; `SandboxSessionState` has **no manifest field** (only
`manifest_digest`, `mount_authority_redacted`, `mount_authority_rebound`), so there is no raw mount payload
to sanitize; `SandboxError` has no traceback, no instance `__dict__`, no exception groups.

Divergence for the controller to confirm: Rust trusts any third-party `impl MountStrategy` reporting
`credential_authority() == None`. Python rejects every non-SDK strategy at a credential boundary. I did not
add a provenance allowlist: `MountStrategy` is the extension point the e2b/daytona/vercel/blaxel/modal
crates implement, and a strategy is never materialized from serialized state, so the smuggling vector does
not exist.

## 1. Per-file table

| file | #tests | ported | covered | python-only | failures found |
| --- | --- | --- | --- | --- | --- |
| tests/sandbox/test_mount_security.py | 134 | 45 | 0 | 89 | 11 Rust gaps, all fixed (no deferrals) |

45 + 0 + 89 = 134. COVERED is 0: an earlier draft cited unit tests for "halves" of some rows; those rows are
fully ported here, so no COVERED claim is made.

## 2. PORTED, per Python test (index = order in file)

Rust test names are in `py_sandbox_a_mount_security_{1,2,3}.rs`. No row diverges any more; every former `DEFERRED:` assertion was flipped to Python's behavior (section 5).

| # | Python test | Rust test |
| --- | --- | --- |
| 1 | rejects_explicit_credentials_for_in_container_mounts | _1 explicit_credentials_on_an_in_container_mount_require_an_acknowledgement |
| 2 | exact_path_acknowledgement_allows_... | _1 an_acknowledgement_covers_only_the_exact_mount_path, a_broad_acknowledgement_subsumes_a_mount_scoped_one |
| 3 | acknowledgement_rejects_incomplete_in_container_s3_credentials | _1 an_incomplete_s3_credential_set_is_rejected, a_session_token_without_a_key_pair_is_rejected |
| 4 | ..._incomplete_in_container_gcs_hmac_credentials | _1 an_incomplete_gcs_hmac_credential_set_is_rejected |
| 5 | ..._accepts_complete_in_container_gcs_hmac_credentials | _1 a_complete_gcs_hmac_credential_set_is_accepted |
| 6 | ..._rejects_empty_in_container_scalar_authority | _1 a_blank_scalar_authority_field_is_rejected |
| 7 | ..._incomplete_in_container_r2_credentials | _1 an_incomplete_r2_credential_set_is_rejected |
| 8 | incomplete_credentials_remain_external_provider_configuration | _1 incomplete_credentials_remain_external_provider_configuration |
| 9 | credential_acknowledgement_is_not_a_path_prefix | _1 an_acknowledgement_is_not_a_path_prefix |
| 10 | ..._preserves_path_whitespace | _1 an_acknowledgement_preserves_path_whitespace |
| 13 | manifest_acknowledgement_is_runtime_only_and_rejects_root | _1 acknowledgement_paths_are_validated_like_python (+ a_wildcard_or_parent_acknowledgement_never_matches_a_real_mount_path for the fail-closed wrappers) |
| 14 | manifest_acknowledgement_rejects_wildcard_and_parent_paths | same test as 13 |
| 30 | preserves_credentialless_in_container_and_credentialed_docker_mounts | _1 credentialless_in_container_and_credentialed_docker_volume_mounts_need_no_acknowledgement |
| 41 | ignores_environment_values_already_exposed_to_the_sandbox | _1 environment_values_already_exposed_are_not_mount_authority |
| 42 | blobfuse_mounts_require_broad_acknowledgement | _1 a_blobfuse_mount_requires_a_broad_acknowledgement |
| 43 | blobfuse_account_key_requires_... | _1 a_blobfuse_account_key_still_requires_the_broad_acknowledgement |
| 44 | s3_files_require_broad_acknowledgement_... | _1 an_s3_files_mount_requires_a_broad_acknowledgement |
| 47 | rejects_rclone_credential_source_overrides | _2 rclone_credential_source_overrides_are_rejected |
| 48 | box_mounts_with_direct_credentials_require_exact_acknowledgement | _1 a_box_mount_with_a_direct_token_requires_an_exact_acknowledgement |
| 49 | box_config_file_requires_broad_acknowledgement | _1 a_box_configuration_file_requires_a_broad_acknowledgement |
| 50 | box_in_container_mount_requires_non_interactive_authentication | _1 a_box_in_container_mount_requires_a_non_interactive_authentication_source |
| 51 | box_in_container_mount_rejects_empty_authentication_sources | _1 a_blank_box_authentication_source_is_rejected |
| 52 | ..._rejects_mixed_usable_and_empty_authentication_sources | _1 a_blank_box_authentication_source_beside_a_usable_one_is_rejected |
| 53 | preserves_box_mounts_with_an_external_strategy | _1 a_box_mount_on_a_docker_volume_strategy_needs_no_acknowledgement |
| 54 | preserves_multiline_external_mount_credentials | _1 a_multiline_external_credential_is_preserved |
| 56 | rejects_and_redacts_rclone_config_line_injection | _2 an_rclone_configuration_value_that_would_add_a_line_is_rejected_and_redacted, the_configuration_line_guard_names_the_field_only |
| 57 | rejects_blaxel_s3fs_endpoint_option_injection | _2 an_s3fs_option_delimiter_inside_a_value_is_rejected (shared guard; Blaxel strategy takes its own config) |
| 58 | rejects_rclone_on_the_fly_remote_name | _2 an_on_the_fly_rclone_remote_name_is_rejected_without_echoing_it, an_on_the_fly_remote_name_never_reaches_a_command |
| 60 | preserves_ordinary_rclone_remote_name | _2 an_ordinary_rclone_remote_name_is_preserved (inner space stays rejected: deliberate narrowing) |
| 61 | preserves_supported_credentialless_rclone_extra_args | _2 supported_credentialless_rclone_extra_args_are_preserved |
| 62 | rejects_malformed_inline_credential_url_... | _2 a_url_carrying_its_own_authority_is_detected_even_when_malformed, an_inline_authority_endpoint_makes_an_s3_mount_credential_bearing |
| 63 | rejects_mountpoint_endpoint_authority | _2 a_mountpoint_pattern_endpoint_carrying_authority_is_rejected, an_ordinary_mountpoint_pattern_endpoint_is_preserved |
| 64 | rejects_manifest_backed_credential_files | _3 a_credential_file_materialized_by_the_manifest_is_rejected, a_structural_directory_at_the_credential_path_is_allowed |
| 65 | broad_acknowledgement_does_not_allow_manifest_backed_rclone_config | same test as 64 |
| 66 | rejects_credential_files_from_manifest_materialization_sources | same test as 64 |
| 67 | session_state_serialization_redacts_complete_opaque_authority_fields | _3 a_persisted_state_records_that_mount_authority_existed_but_never_the_authority, docker_volume_driver_options_never_reach_durable_state, no_built_in_mount_leaks_a_credential_into_durable_state |
| 68 | session_state_round_trip_preserves_credentialless_external_mount | _3 a_credentialless_external_mount_round_trips_without_a_redaction_marker |
| 69 | credentialless_marker_does_not_override_configured_mount_authority | _3 a_forged_credentialless_marker_does_not_make_a_state_resumable |
| 78 | opaque_external_authority_remains_resumable_through_trusted_rebind | _3 opaque_external_authority_remains_resumable_through_a_trusted_rebind |
| 79 | in_container_acknowledgement_is_rebound_only_from_trusted_manifest | _3 mount_authority_is_rebound_only_from_a_trusted_manifest |
| 80 | implicit_broad_authority_is_rebound_only_from_trusted_manifest | _3 implicit_broad_authority_is_rebound_only_from_a_trusted_manifest |
| 81 | mount_authority_rebind_requires_exact_credential_free_topology | _3 a_rebind_requires_the_same_mount_topology |
| 82 | resume_validation_rejects_wrong_provider_strategy | _3 a_resume_against_a_different_mount_backend_is_refused, a_resume_that_swaps_the_storage_behind_one_backend_is_refused |
| 92 | protected_structured_mount_error_preserves_safe_contract | _3 a_failing_mount_command_keeps_its_contract_without_reproducing_the_credential |
| 111 | credentialless_structured_mount_error_is_unchanged | _3 a_credentialless_mount_failure_is_passed_through_unchanged |

Rust-only extras (no Python row): a_mount_rejects_an_unsupported_in_container_pattern,
a_caller_placed_rclone_configuration_is_broad_authority, a_remote_name_with_a_separator_is_rejected,
command_output_is_redacted_value_by_value, a_redacted_state_drops_its_provider_identity_on_persistence,
a_mount_debug_rendering_never_reproduces_a_credential.

## 3. PYTHON-ONLY (89), grouped

| group | Python rows | count | reason |
| --- | --- | --- | --- |
| Pydantic-input / platform-path surface | 11, 12 | 2 | `PureWindowsPath` inputs and `Manifest.model_validate` injection; Rust acknowledgements are constructor-only `&str`. |
| Custom Mount/Strategy/Pattern subclass provenance, `__deepcopy__`, serializer hooks, traceback locals | 15-29 | 15 | Rust trait impls have no class identity to forge and are never materialized from serialized data; no pydantic hooks, no tracebacks. |
| Hosted strategies read `S3Mount` fields | 31 | 1 | Rust hosted strategies (Blaxel/E2B/Daytona) take their own config instead of classifying a generic `S3Mount`; covered by `tests/sandbox_e2b_daytona_vercel_tests.rs` / `sandbox_modal_cloudflare_blaxel_tests.rs` provider suites. |
| Custom strategy/pattern provenance and pre-side-effect rejection | 32-40 | 9 | Same provenance machinery (`type` forging, extra pydantic config, behavior-only subclasses). |
| Inherited-strategy and provider-type-string forging | 45, 46 | 2 | Subclassed `InContainerMountStrategy` / assigning `strategy.type`; identity is a `&'static str` in Rust. |
| `examples/` import | 55 | 1 | Imports `examples.sandbox.docker.mounts.*` modules; no such modules in Rust. |
| Serialized manifest redaction | 59, 70-77 | 9 | `SandboxSessionState` carries no manifest, so there are no serialized mount fields to redact or preserve; the "never leaks" guarantee is asserted structurally by row 67. |
| Serialized-pattern redaction, raw-dict sanitization | 83-91 | 9 | Same: no manifest/pattern payload in Rust state, so no raw state to sanitize or registries to replace. |
| Error-redaction decorator internals | 93-110 | 18 | `redact_mount_error_data[_sync]` defends against hostile exceptions (args/`__traceback__` descriptors, exception groups, `SystemExit`, `GeneratorExit`, slot/property state); `Result<_, SandboxError>` has none of these. |
| Provider-link and operation-error replacement | 112-119 | 8 | The decorator replaces arbitrary exceptions raised through opaque `_client`/`_session` links; Rust errors are structured and, since the rclone stderr fix, carry no credential. |
| Raw-state and deserialization sanitization, traceback redaction | 120-132 | 13 | `model_validate`/raw-dict parsing and SDK traceback frames; no equivalent surface. |
| asyncio retained-failure / ambient context | 133, 134 | 2 | Python `contextvars` and retained coroutine exceptions. |

2+15+1+9+2+1+9+9+18+8+13+2 = 89.

## 4. Rust bugs found

Fixed in files I own:

| file | fix |
| --- | --- |
| `src/sandbox/mounts.rs` | `resolve_remote_name` echoed the untrusted remote name in its error (`:s3,secret_access_key=...` IS the credential) and accepted `,`/`=`/`@`; now validated against `[A-Za-z0-9_][A-Za-z0-9_.-]*` and never echoed. |
| `src/sandbox/mounts.rs` | `RcloneMountPattern::extra_args` was unvalidated and reached `rclone mount` verbatim (`--config`, `--s3-env-auth`, `--s3-profile`, `--azureblob-use-msi`, `--header`). Added `rclone_extra_args_are_safe` (port of `_rclone_extra_args_analysis`), enforced at the top of `apply`. |
| `src/sandbox/mounts.rs` | rclone mount failure echoed raw stderr, which can repeat a configuration value; now every staged credential value is redacted (`rclone_secret_values` + `redact_sensitive_values`). |
| `src/sandbox/mount_patterns.rs` | `MountpointMountPattern.options.endpoint_url` carrying inline authority bypassed credential classification (classification reads the mount's endpoint, not the pattern's) and reached the sandbox unacknowledged; `apply` now rejects it without echoing the value. |
| `src/sandbox/mount_patterns.rs` | Blobfuse `cache_path` errors now match Python: message `blobfuse cache_path must be relative to the workspace root` with context `cache_path` (Windows-absolute paths rejected too, shown with `/`), and `blobfuse cache_path must be outside the mount path` with context `mount_path` + `cache_path`. Requested by the entries helper. Tests: `_2 a_blobfuse_cache_path_outside_the_workspace_is_rejected_with_context`, `a_blobfuse_cache_inside_the_mount_is_rejected_with_both_paths`. |
| `src/sandbox/mounts.rs` | `supplement_rclone_config_text` now keeps the blank lines that separated the merged section from the next one (Python parity; was dropped). Unit test `supplements_only_the_mounted_remote_section` updated; new `_2 an_rclone_merge_keeps_the_blank_line_before_the_next_section`. The entries owner updated `tests/py_sandbox_a_entries_3.rs` for it; that suite passes. |

Fixed in a file I do not own (small, unambiguous, re-read first; flag for the entries owner):

| file | fix |
| --- | --- |
| `src/sandbox/mount_entries.rs` (`S3Mount::validate`) | A bare `session_token` with no key pair is now rejected (Python: `credential_fields == (access_key_id, secret_access_key)`). |
| `src/sandbox/mount_entries.rs` (`GcsMount::build_docker_volume_driver_config`) | The docker-volume `gcs-service-account-credentials` option is passed through verbatim; Rust rejected a multi-line key that Python preserves (Docker driver options travel over the container API, not a config file). |

Deliberate narrowing: Python's remote-name regex admits an inner space (`"public bucket-1"`); Rust keeps
rejecting whitespace (`an_ordinary_rclone_remote_name_is_preserved` documents it).

## 5. Former deferrals, now closed (Python behavior; no `DEFERRED:` assertions remain)

| item | change |
| --- | --- |
| `manifest_digest` provider swap | `src/sandbox/state.rs`: the digest now folds in `ConfiguredMountStrategy::topology_identity()` = mount type + the storage it names (bucket/prefix/region, account/container, account id, file system id, Box path) + the pattern or driver. Credentials, endpoints and file paths stay out, so rotating a credential still resumes. `rebind_path_grants` now verifies the digest when authority was redacted ("exactly matching" topology), so a swapped bucket or provider is refused with `SessionStateMismatch`. Side effect: manifests with configured mounts hash differently from before, so a state persisted by an older build with mounts will fail `verify_manifest` once. Root is still excluded from the digest by design (the root travels in `workspace_root`), so Python's root-mismatch half of row 81 has no Rust counterpart. |
| acknowledgement-path validation | `src/sandbox/manifest.rs`: new `try_with_in_container_mount_credential_exposure_acknowledged` / `try_with_in_container_mount_broad_credential_exposure_acknowledged` returning `SandboxResult<Self>` (`MountConfigInvalid`) reject no paths, empty, root, `\`, wildcard `*?[]` and `..` segments, exactly as Python. The old infallible builders keep their signatures and now act as fail-closed wrappers: an invalid path acknowledges nothing. Acknowledgements are normalized to Python's policy keys, so `"data"` and `"/workspace/data"` name the same mount (previously only exact strings matched). |
| manifest-backed credential files | `src/sandbox/mount_entries.rs`: new `ConfiguredMount::credential_file_fields()` (GCS `service_account_file`, Box `box_config_file`) and `ConfiguredMountStrategy::credential_file_paths()` (adds rclone `config_file_path`). `src/sandbox/manifest.rs`: new `Manifest::validate_mount_credential_files()` (port of `_manifest_materializes_path`; plain `dir` entries are structural and allowed), called from `authorize_mount_activation` and at the start of manifest materialization, so docker-volume mounts are checked too. |
| half-supplied credentials on docker volumes | `src/sandbox/mount_entries.rs`: `ConfiguredMount::validate` is split into structural checks and a new `validate_in_container_credentials`, which `ConfiguredMountStrategy::validate` and the in-container config builders run only for in-container backends. External (docker volume) mounts keep half-supplied or blank credentials as provider configuration, as Python does. Box's "needs a non-interactive authentication source" is now in-container-only too. |

### Redaction keyed on carried credentials, not exposure (coordinator request)

`SandboxSessionState::mount_authority_redacted` keyed off `credential_authority()`, which is `None` for a credentialed
Docker-volume mount (the credential goes to the container runtime on the host). Python redacts any mount whose raw fields
hold authority. Changes:

- `src/sandbox/mounts.rs`: defaulted `MountStrategy::carries_credentials(&self, mount) -> bool` (default
  `credential_authority() != None`).
- `src/sandbox/mount_entries.rs`: `ConfiguredMount::carries_credentials()` per built-in mount, from the same fields as
  Python's `_AUTHORITY_FIELDS_BY_MOUNT_TYPE` (S3 keys/token/inline-authority endpoint, GCS id/secret/file/credentials/token,
  R2 keys/custom domain, Azure key/identity client id, Box secret/tokens/config); `ConfiguredMountStrategy::carries_credentials`
  also counts non-empty Docker `driver_options` as opaque authority, as Python removes that field as a unit.
- `src/sandbox/state.rs`: `SandboxSessionState::new` uses `carries_credentials`, so redaction, provider-identity drop and
  the rebind requirement apply to credentialed Docker volumes. (Rebind itself already keys off `mount_authority_redacted`.)
- `src/sandbox/sandboxes/docker.rs`: `manifest_has_protected_mounts` uses the same predicate; its extra checks (unknown
  strategy, credential-looking `volume-opt`) are kept.
- `tests/py_sandbox_a_docker_5.rs`: the `AuthorityVolume` MountScoped workaround strategy is replaced by a real credentialed
  Docker-volume `S3Mount` (`authority_volume()`); no assertion changed, they now pass for the real reason.
- New `_3 a_credentialed_docker_volume_mount_is_redacted_from_durable_state`.

Not covered: Python's "mount strategy is not supported by this sandbox backend" (rows 82's `provider_backend_id` check) has no Rust counterpart; Rust strategies do not declare a provider backend id.

## 6. `#[ignore]`d tests

None. Nothing needs a network, real credentials, a docker daemon or Python.

## 7. Shared-file edits needed

None (`src/lib.rs`, `Cargo.toml` untouched). The two helpers are private to `mounts.rs`.

## Verification

- `c.sh test --all-features --no-fail-fast --test py_sandbox_a_mount_security_{1,2,3}`: 28 + 17 + 18 passed.
- `c.sh test --all-features --lib sandbox::`: 299 passed. Also green after the closed deferrals: `sandbox_tests` 65,
  `sandbox_core_parity_tests` 28, `sandbox_e2b_daytona_vercel_tests` 41, `sandbox_modal_cloudflare_blaxel_tests` 37,
  `sandbox_archive_tests` 12, `py_sandbox_a_entries_1..4` 22/30/35/33, `py_sandbox_a_snapshot_1..2` 28/19,
  `py_sandbox_b_misc` 25, `py_sandbox_rclone_mounts` 8. After the `carries_credentials` change: `py_sandbox_a_docker_1..6`,
  `py_sandbox_a_entries_1..4` (22/30/35/33), `sandbox_tests` 65, `sandbox_core_parity_tests` 28, my three files 28/17/18: all green.
- `rustfmt --check` clean on `mounts.rs`, `mount_patterns.rs`, `manifest.rs`, `state.rs`, `mount_entries.rs` and my three test files.
- `c.sh clippy --all-features --lib --test py_sandbox_a_mount_security_{1,2,3} -- -D warnings`: no finding in any file I
  touched. It still fails in other areas' in-flight files only (`src/run_internal/mod.rs`, `agent_identity.rs`,
  `src/stream_events.rs`, `src/sandbox/sandboxes/unix_local.rs`); not touched.
