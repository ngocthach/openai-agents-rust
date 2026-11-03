# Python test parity, round 20, r20c (sandbox / realtime / voice / codex / hosted multi-agent)

Scope: 46 files, 106 python-only cases (pytest-collect basis, from the ledger). Result: **8 ported,
2 covered, 96 remain python-only.**

## Read depth

To locate the python-only cases, every Python test name in the 46 files was matched against the
Rust test corpus (test names and doc comments). The Python cases that no Rust test names
(about 230) were then read in full, along with the parametrized variants that earlier reports
name, except for the cases below. For those, only the assertions and parametrization were read,
and each was matched to a differently named Rust port: in the Rust function list, or by grepping
the cited Rust file for the behaviour.

- `realtime/test_session_approvals.py`: 14 of 19 unmatched names were read at assertion level; all map to tests in `py_realtime_session_approvals.rs`.
- `hosted_multi_agent/test_model.py`: 13 of 16 at assertion level; all map to `py_ext_hosted_multi_agent.rs`.
- `realtime/test_session_guardrails.py` (2) and `realtime/test_session_history.py` (2): assertion level.
- `extensions/sandbox/test_cloudflare.py`: 3 at assertion level (hydrate remount order, two terminal-delete cases).

Rows that the ledger counts as ported or covered, located by keyword in the cited Rust files rather than by reading the Python bodies:

- `sandbox/test_extract.py`: 13 names, found in `py_sandbox_a_archive_extract.rs`.
- `sandbox/test_snapshot.py`: 12 names, found in `py_sandbox_a_snapshot_2.rs`.
- `sandbox/test_session_state_roundtrip.py`: 10 names, found in `py_sandbox_a_snapshot_1.rs`, `py_core11_r11b_session_state.rs` and `py_core15_r15d_manifest_env_registry.rs`.

For these rows the python-only cases themselves were read in full; only the ledger's existing
ported/covered classification of the other names was not re-audited. One more: the 9 other
unmatched `test_e2b.py` names were read in full. The ledger counts them as ported or covered
(12 covered). Several of them test private waiter/prune helpers, and they were not re-classified
here.

## Per-file table

| file | before | ported | covered | after | reason / note |
|---|---|---|---|---|---|
| extensions/experiemental/codex/test_payloads.py | 2 | 2 | 0 | 0 | ported: the mapping view of a thread item is `ThreadItem::to_value` (keys `id`,`text`,`type`; an unknown key is absent) |
| extensions/experimental/hosted_multi_agent/test_model.py | 1 | 0 | 0 | 1 | `test_config_rejects_non_positive_concurrency[-1/True]`: `max_concurrent_subagents` is a `u32`, so `-1` and `True` cannot be passed (the `0` case and the mapping form, via a struct literal, are ported in `py_ext_hosted_multi_agent.rs`) |
| extensions/sandbox/test_cloudflare.py | 2 | 0 | 0 | 2 | asyncio cancellation of `start()` while a resume mount/restore is blocked, where Python shields the remount (three `resume_start_*_cancellation` cases read). Rust cancellation drops the future |
| extensions/sandbox/test_daytona.py | 3 | 1 | 0 | 2 | ported: the terminal tail is delivered before the exit is reported. Remaining: `agents.extensions.sandbox.daytona` package re-export identity (2) |
| extensions/sandbox/test_e2b.py | 5 | 0 | 0 | 5 | package re-export identity (2), the vendor SDK `connect` vs `_cls_connect` wrapper choice and `__import__` class resolution (live vendor SDK classes), positional dataclass args |
| extensions/sandbox/test_vercel_provider_contract.py | 3 | 1 | 0 | 2 | ported: the `[timeout]` variant of the unadopted reconnect test (a pending session that outlives `wait_for_running` is replaced; the original is not stopped or deleted). Remaining: cancelled creation and the `[cancelled]` reconnect variant (asyncio task cancel) |
| realtime/test_agent.py | 2 | 0 | 0 | 2 | `__post_init__` type checks of the name/tools/handoffs/instructions arguments, and an instructions object with a custom `__str__` assigned after construction; typed in Rust |
| realtime/test_app_server_debug.py | 4 | 0 | 0 | 4 | `examples/realtime/app/server.py` logging (Python example) |
| realtime/test_audio_formats_unit.py | 1 | 0 | 0 | 1 | dict subclass whose `__str__`/`__repr__` raise |
| realtime/test_openai_realtime_conversions.py | 1 | 1 | 0 | 0 | ported: a hosted tool on a realtime agent fails with `Tool {name} is unsupported. Must be a function tool.` (no Rust test asserted it). The other unmatched case, `input_text` parts with missing or non-string text, cannot be expressed: `RealtimeUserInputContent::InputText.text` is a `String`. The ledger appears to count it elsewhere |
| realtime/test_realtime_model_settings.py | 2 | 0 | 0 | 2 | **pending fallible-callback API**: an `is_enabled` that raises, with sibling checks cancelled (`filter_enabled_tools_cancels_sibling_checks_on_error`, `build_model_settings_cancels_cross_group_enablement_on_error`) |
| realtime/test_session_approvals.py | 1 | 0 | 0 | 1 | `test_cancelled_rejection_formatter_leaves_invocation_executed`: a task cancelled inside the formatter, checked through private `_tool_invocation_status` |
| realtime/test_session_close_iteration.py | 1 | 0 | 0 | 1 | `asyncio.CancelledError` variant of the close failure |
| realtime/test_session_guardrails.py | 1 | 0 | 0 | 1 | guardrail callable without `__name__` (diagnostic `guardrail_type` from the class qualname) |
| realtime/test_session_history.py | 1 | 0 | 0 | 1 | duck-typed bogus content entry plus a patched module logger |
| realtime/test_session_payload_and_formats.py | 2 | 1 | 0 | 1 | ported plus a **source fix**: only a realtime session payload retunes the playback format. Remaining: `__bool__`-falsy pydantic models |
| realtime/test_session_tool_error_events.py | 1 | 0 | 0 | 1 | exception whose `__str__`/`__repr__` raise |
| sandbox/capabilities/test_apply_patch_tool.py | 1 | 0 | 0 | 1 | module `Path` monkeypatched to `PureWindowsPath` |
| sandbox/capabilities/test_compaction_capability.py | 1 | 0 | 0 | 1 | `Compaction.model_validate` with an unknown policy type (pydantic) |
| sandbox/capabilities/test_shell_capability.py | 3 | 0 | 2 | 1 | covered (end state, Rust has no `configure_tools` hook): a non-PTY session gets `exec_command` and no `write_stdin` (`py_sandbox_b_capabilities.rs::shell_tools_expose_exec_command_without_a_pty`); the exec tool runs in the bound workspace scope (`::exec_command_runs_as_the_bound_user_in_the_scope_cwd`). Remaining: `tools()` before `bind` (Rust `tools` takes the binding) |
| sandbox/capabilities/test_skills_capability.py | 3 | 0 | 0 | 3 | `Dir` as skill content and literal+`from_`/`lazy_from` combinations; `Skill.content` is bytes and `Skills` holds one source enum |
| sandbox/capabilities/test_tool_input_errors.py | 2 | 0 | 0 | 2 | `validation` (pydantic `ValidationError` raised by the backend) and `cancel` (`asyncio.CancelledError`) failure kinds |
| sandbox/test_client_options.py | 4 | 0 | 0 | 4 | pydantic `parse` passthrough, `model_dump(exclude_unset)`, subclass registry rules |
| sandbox/test_dependencies.py | 1 | 0 | 0 | 1 | asyncio scheduling: a rebind queued with `call_soon` before the factory task starts, the 3.12 eager task factory, `aclose` cancellation (all three read; the Rust rebind contract is ported in `py_sandbox_a_entries_1.rs`) |
| sandbox/test_docker_removal.py | 1 | 0 | 0 | 1 | `[base_exception]`: `BaseException` raised from a worker thread |
| sandbox/test_docker_removal_worker.py | 2 | 0 | 0 | 2 | patched `ExitStack.callback` registration; `KeyboardInterrupt` in a forked child via `os._exit` |
| sandbox/test_entries.py | 3 | 0 | 0 | 3 | monkeypatched `os.open` and `_OPEN_SUPPORTS_DIR_FD` to swap the source root to a symlink between validation and open |
| sandbox/test_external_provider_example.py | 8 | 0 | 0 | 8 | `examples/sandbox/external_provider` package, using `create_autospec` delegates |
| sandbox/test_extract.py | 3 | 0 | 0 | 3 | `zipfile_compatible_stream` over Python streams without `seekable()` (2) and a chunked non-seekable write stream; Rust writes take `&[u8]` |
| sandbox/test_mount_lifecycle.py | 4 | 0 | 0 | 4 | cancellation of `with_ephemeral_mounts_removed` during teardown/operation/restore/shutdown. Rust documents that cancellation is a future drop and the caller's guard restores |
| sandbox/test_pty_output.py | 1 | 0 | 0 | 1 | monkeypatched `time.monotonic` plus injected poll/settle/wait callbacks of a private collector |
| sandbox/test_pty_types.py | 2 | 0 | 0 | 2 | `_settle_pty_cleanup` keeps the `CancelledError` reason |
| sandbox/test_retry.py | 2 | 0 | 0 | 2 | `__cause__`/`__context__` chain walkers; `SandboxError` has no exception chain |
| sandbox/test_sandboxes_import.py | 3 | 0 | 0 | 3 | `sys.modules` / `sys.platform` import behaviour |
| sandbox/test_session_sinks.py | 2 | 1 | 0 | 1 | ported: a sink that defers delivery until flushed has the `stop`/`shutdown` finish events after shutdown. Remaining: cancellation during finish-sink delivery (asyncio). A third case was also read: `test_filesystem_test_session_rejects_process_backed_operations` exercises the Python test helper `FilesystemTestSandboxSession`, so it is python-only too; the ledger appears to count only two of the three |
| sandbox/test_session_state_roundtrip.py | 2 | 0 | 0 | 2 | pydantic subclass registry (`_subclass_registry`, `model_validate_json`, `parse` identity) and no-`deepcopy` of extension state |
| sandbox/test_session_utils.py | 2 | 0 | 0 | 2 | `_best_effort_stream_len` over Python file objects |
| sandbox/test_snapshot.py | 5 | 0 | 0 | 5 | frozen pydantic model, duplicate/missing `type` registration, `exclude_unset`, archive stream `close()` accounting |
| sandbox/test_tar_utils.py | 2 | 0 | 0 | 2 | monkeypatched `TemporaryFile` / `extractfile` failure injection |
| sandbox/test_unix_local_special_files.py | 3 | 0 | 0 | 3 | real `F_SETLEASE` lease with a `SIGIO` handler in a subprocess (Linux only), and a monkeypatched `os.open` EWOULDBLOCK |
| voice/test_input.py | 1 | 0 | 0 | 1 | numpy float64 dtype rejection (`AudioData` is typed) |
| voice/test_openai_model_provider.py | 5 | 0 | 0 | 5 | `openai_client=` plus conflicting kwargs in one constructor call. The Rust builder instead applies a later `with_api_key`/`with_base_url`/... to the explicit client (see concerns) |
| voice/test_openai_stt_client_config.py | 1 | 1 | 0 | 0 | ported: omitted organization/project headers are absent from the STT websocket handshake headers |
| voice/test_openai_stt_session_config.py | 2 | 0 | 0 | 2 | `dataclasses.fields` order; positional subclass construction |
| voice/test_pipeline_config.py | 3 | 0 | 0 | 3 | positional / duplicate-argument dataclass construction |
| voice/test_pipeline_tracing.py | 1 | 0 | 0 | 1 | legacy positional `VoicePipelineConfig` variant. Non-legacy and disabled-provider behaviour is ported in `py_voice_pipeline_tracing.rs` |
| **total (46 files)** | **106** | **8** | **2** | **96** | |

## New tests

- `tests/py_core20_r20c_codex_payloads.rs` (2)
- `tests/py_core20_r20c_realtime.rs` (2, feature `realtime`)
- `tests/py_core20_r20c_sandbox.rs` (3, feature `sandbox`: Daytona terminal tail, Vercel pending-timeout replacement, deferred sink flush)
- `tests/py_core20_r20c_voice.rs` (1, feature `voice`)

## Source fix

- `src/realtime/openai_model.rs`: on `session.created`/`session.updated`, a session whose `type`
  is present and is not `realtime` (for example a transcription session) no longer sets the
  playback/audio tracker format. This matches Python's `_normalize_session_payload`. Not a public
  API change.

## Pending fallible-callback API

- `realtime/test_realtime_model_settings.py`: `test_filter_enabled_tools_cancels_sibling_checks_on_error`,
  `test_build_model_settings_cancels_cross_group_enablement_on_error`. These need an `is_enabled` that can fail. They can be ported once that API lands; the sibling-cancellation part becomes a future drop.

## Gates

- New binaries: 8 tests pass (`--all-features`).
- All 56 realtime/voice/STT test binaries pass (`--all-features`): 601 passed, 0 failed. This is the
  regression surface of the `openai_model.rs` change.
- Adjacent sandbox binaries pass: `py_core6_cfdaytona_daytona`, `py_core12_r12a_vercel`,
  `py_core16_r16d_sandbox_sinks`, `py_sandbox_b_session_sinks`, `py_sandbox_b_sinks`. The full set
  of about 131 sandbox binaries was not re-run, because no `src/sandbox/**` or
  `src/extensions/sandbox/**` source changed.
- `cargo clippy --lib` plus the four new test files, `-D warnings`: clean with `--all-features`
  and with default features.
- rustfmt: the new files and `src/realtime/openai_model.rs` are clean. The cargo output dir was
  deleted.

## Suggested changelog paragraph

> Realtime: a `session.created`/`session.updated` event for a non-realtime (for example
> transcription) session no longer changes the playback audio format, as in the Python SDK.
> Added parity tests for codex thread-item field access, realtime hosted-tool rejection, the
> Daytona terminal tail before exit, the Vercel pending-session timeout fallback, deferred sink
> flushing on shutdown and omitted STT websocket headers.

## Concerns / unresolved

- Voice provider (5 cases): Python rejects `openai_client` combined with `api_key`/`base_url`/
  `organization`/`project`. The Rust builder deliberately applies those to the explicit client.
  Matching Python would need a fallible builder, which is a public API change. Left as a
  documented divergence for a maintainer decision.
- Cloudflare/Vercel/mount-lifecycle cancellation: when a Rust future is dropped mid-resume or
  mid-create, the remount or allocation cleanup that Python shields does not run. This is
  documented, but a dropped Vercel `create` can leave a running sandbox.
- Linux file-lease cases cannot be verified on the macOS host. Two ledger rows (conversions,
  session_sinks) count fewer cases than the unmatched bodies read; the note column above says how
  each was resolved.
