# py-test-parity core15 / r15d: not-reached and name-level files of r14c

Work list: the 97 files of `r14_r14c.txt` whose r14c CSV row had `moved_to_ported = 0`, minus the CI-script,
example and release files (`scratchpad/r15d_list.txt`, about 1,200 Python-only cases). Rust tests:
`tests/py_core15_r15d_*.rs` (7 files, 21 tests). CSV: `py-test-parity-core15-r15d.csv` (81 rows, one per file
whose Python-only candidates I read; `before - ported - covered = remaining` on every row).

## Honest coverage statement

Second pass (after the coordinator's rejection) is complete for 81 of the 97 files; 16 remain at name level and are
listed below. Method per file: every Python test name was checked against Rust citations and reuse
(`scratchpad/cand.py`); the uncited names were listed, the bodies of every uncited case that looked portable
(user-visible behaviour, as opposed to pydantic/inspect/asyncio-task/gc/import machinery) were read, and the Rust
behaviour was checked in source and tests. Where the uncited names were a long list of private-helper, hostile-object
or task-cancellation cases (`test_mount_security`, `test_error_logging_redaction`, the provider files), I read the
portable-looking subset in full and classified the rest by name and group; that is not a body-level read of every case,
and the CSV reflects it (those rows move only what I ported or proved).

Still not read (16): `test_agent_runner` and `test_agent_runner_streamed` beyond the 15 bodies named below,
`mcp/test_mcp_util` (86 uncited names, all in the groups `py_mcp_util.rs`/`py_core6_mcpmisc_mcp_util.rs` cite), `mcp/test_mcp_imports`,
`extensions/memory/test_memory_imports`, `sandbox/test_sandboxes_import` (Python packaging imports),
`memory/test_session_helpers`, `sandbox/capabilities/test_tool_input_errors`, `extensions/experimental/hosted_multi_agent/test_model`
(16 uncited), `extensions/test_tool_output_trimmer` (43 uncited, `py_ext_tool_output_trimmer.rs` exists),
`deferred_session_writes/test_pending_write`, `test_run_internal_items` (16 uncited), `test_agent_instructions_signature`,
`test_default_tool_error_redaction` (one body read), `sandbox/test_mount_lifecycle` (one body read),
`realtime/test_session_tool_error_events`, `tracing` leftovers.

Partly read, not in the CSV: `test_agent_runner` (7 bodies: blocked-output group, session rejection, conversation-id and
trace-env cases) and `test_agent_runner_streamed` (8 bodies: empty input, tool-not-found, cancelled sibling, dynamic tools,
max-turns events, handoff guardrail results, persisted count). All are covered in Rust except the reasoning-item case,
ported below.

## Totals (CSV rows only)

| | cases |
|---|--:|
| Python-only before (81 files) | 779 |
| Moved to ported | 19 |
| Moved to covered | 10 |
| Python-only after | 750 |

Plus one port from a partly read file (`test_agent_runner::test_blocked_reasoning_item_discards_the_complete_response`)
that is not in the CSV totals.

## Ported

| Rust file | Python source | cases |
|---|---|--:|
| `py_core15_r15d_manifest_env_registry.rs` (4 tests) | `sandbox/test_manifest.py`: tagged env values round trip without resolved secrets, custom-serializer type kept and internal value hidden, `type` survives narrowed dumps (2 ids: the Rust payload always carries it), duplicate type registration refused | 5 |
| `py_core15_r15d_models.rs` (2) | `models/test_openai_chatcompletions_stream.py`: dropping the event stream closes the provider connection (TCP mock keeps the stream open and detects the peer close); streamed tool call without provider metadata has no `provider_data` | 2 |
| `py_core15_r15d_realtime_formats.rs` (2) | `realtime/test_session_payload_and_formats.py`: `session.created` GA formats reach the playback tracker as `pcm16`/`g711_ulaw`/`g711_alaw`, string and mapping spellings pass through, a missing format leaves the tracker alone | 2 |
| `py_core15_r15d_memory_registry.rs` (5) | `sandbox/test_memory.py`: a different generate config for the same layout, the same `memories_dir` with another `sessions_dir`, a shared `sessions_dir` for different `memories_dir`s (all refused with Python's messages), unregister on session close; plus a session-scoping check | 4 |
| `py_core15_r15d_mount_payload.rs` (3) | `sandbox/test_mount_security.py`: unknown strategy/pattern `type` refused without the value (2 locations), malformed entry container, credential content under an unknown entry type | 5 |
| `py_core15_r15d_tool_constraints.rs` (1) | `test_function_schema.py::test_annotated_constraints_reject_arguments_before_tool_execution`: range and length constraints are enforced before the tool body runs (already true in Rust; pinned) | 1 |
| `py_core15_r15d_runner.rs` (1) | `test_agent_runner.py`: a reasoning item next to the terminal call makes a tripped output guardrail discard the whole response from the session | (1) |

## Covered (verified by reading)

- `sandbox/test_session_state_roundtrip` (2): manifest-embedded env values and the legacy discriminator-free form, now
  `py_core15_r15d_manifest_env_registry.rs` and `py_core8_r8g1_manifest.rs` (Rust session state carries a manifest digest,
  not the manifest).
- `test_programmatic_tool_calling` (2): inferred object output schema (`with_output_type`, `py_core4_programmatic_schema_runtime.rs`)
  and the declared-output mismatch error (`core_tool_output_schema.rs`).
- `mcp/test_streamable_http_session_id` (2): id captured on connect and cleared on cleanup (`mcp_tests.rs::streamable_http_records_the_session_id_and_sends_the_auth_header`).
- `mcp/test_list_page_limits` (3): zero limit rejected before connect and the page-boundary group (`py_mcp_pagination.rs`, `py_core8_r8g4_mcp_errors.rs`).

## Source changes

| file | change |
|---|---|
| `src/sandbox/capabilities/memory.rs` | `Memory::manager_for_session` (per-session, per-layout registry of generation managers with Python's three `UserError`s), `Memory::unregister_manager` |
| `src/sandbox/memory/config.rs` | `PartialEq for MemoryGenerateConfig` (model instances by identity) |
| `src/sandbox/memory/manager.rs` | `flush` unregisters the manager (session close) |
| `src/sandbox/runtime.rs` | the starting agent's manager comes from `manager_for_session` instead of `build_manager` (handoff targets of the active agent still use `build_manager`) |
| `tests/py_core8_r8g1_memory.rs` | two agents sharing a session and layout now share one `Memory` configuration, as the Python test does (distinct model instances would be a config conflict) |
| `src/sandbox/environment.rs` | `EnvValue::to_payload` (default `None`) |
| `src/sandbox/persisted_registry.rs` | `register_env_value_type`, `EnvValueParser`; `parse_env_value_mapping` consults the registry (the unknown-type error lists registered names); `env_value_to_value` writes a registered type's payload when its `type` matches `env_value_type()` |
| `src/realtime/openai_model.rs` | `read_format_type` maps GA format objects `audio/pcm`/`audio/pcmu`/`audio/pcma` to `pcm16`/`g711_ulaw`/`g711_alaw` before the trackers see them (the playback tracker reported the raw GA name before) |
| `docs/project-changelog.md` | two Unreleased bullets |

Additive public API only (a defaulted trait method, one function, one type alias). Behaviour change: the playback
tracker's `audio_format()` now returns the legacy names for GA objects.

## Valid Python-only reasons confirmed (remaining 122)

- `test_entries` (3): monkeypatched `os.open`/`O_DIRECTORY` flags of a Python fallback path.
- `test_session_state_roundtrip` (2): pydantic subclass registry, `exclude_unset`, no-deepcopy of extension state.
- `test_programmatic_tool_calling` (5): `Awaitable`/`Coroutine` return-annotation introspection.
- `test_cancel_streaming` (4): consumer-task cancellation while awaiting guardrail/run-loop/cleanup tasks.
- `test_function_tool_approval_arguments` (4): `**kwargs` reconstruction collisions; Rust arguments are typed.
- `test_openai_chatcompletions` (5): falsy `Reasoning` object, `httpx` timeout in `extra_args`, `with_options(max_retries=0)`
  on a vendor client (the Rust HTTP client has no provider-managed retries to switch off), dictionary agent settings.
- `test_openai_chatcompletions_stream` (4): dictionary agent settings, in-flight `aclose` after cancellation, private
  `Part`/`_StreamOutputLayout`/`_finish_reasoning_summary_part` state.
- `test_streamable_http_session_id` (2): private `_get_session_id` callback injection.
- `test_list_page_limits` (4): `bool`/`float` limit type errors (Rust takes `usize`), `CancelledError` mid-list.
- `test_mcp_approval` (3): invalid `require_approval` strings/mappings; the Rust policy is the typed `RequireApproval` enum.
- `models/test_map` (3): falsy provider, invalid prefix-mode strings (typed enum), hybrid `CancelledError`.
- one-case realtime/tracing/voice files: hostile-`__str__` logging objects, typed realtime user input and tools, callable
  without `__name__`, `atexit`/logging-handler monkeypatches, `omit` sentinels of the OpenAI client, positional-argument
  legacy constructors, falsy pydantic models, `contextvars` task ownership.
- `test_shell_capability`, `test_docker_removal`, `test_pty_types`, `test_session_utils`, `test_docker_network_mode`,
  `test_runtime_agent_preparation`: `isinstance` checks on Python tool classes, thread/asyncio cleanup under cancellation,
  Python `io` stream helpers, an AST import check, dictionary agent configuration.
- `test_handoff_tool` (6), `test_tool_exception_formatter` (4), `test_run_hooks` (3): dynamic validation of `on_handoff`
  and hook objects, `CancelledError` policies, sibling-task cancellation, positional constructors.
- `memory/test_session` (11): per-thread sqlite connection reaping, rollback/quarantine eviction, post-commit cancellation.
- `test_run_state_pending_input` (11): the server-managed groups are ported in `py_core3_runstatectx_pending_a/b.rs`; the
  rest are `CancelledError`/`BaseException` ownership and private step fields.

## Unresolved

- Memory registry follow-ups: the active handoff target in a shared session still builds its own manager
  (`active_memory_manager`), and a caller-owned session that is never flushed keeps its registry entry until the session
  is dropped (the entry holds a `Weak`, so it is pruned on the next lookup). Python's weak registry behaves the same.
- `sandbox/test_memory` (11 left): dictionary/typed `MemoryGenerateConfig` normalization (6), worker `BaseException` and
  parent-cancellation (2), private rollout-payload helpers (3).
- `realtime/test_openai_realtime`: Python aggregates several interrupt failures into one error object; Rust returns the first
  failure after still attempting every step. Valid private exception type, noted as a visible difference.
- `unix_local` user-scoped `mkdir`/`rm`: Python checks permissions through `sudo` and then touches the local file system;
  Rust runs the command as the user. Both need a real `sudo`, so neither is testable here.
- The Rust session-state value does not carry the manifest, so Python's manifest-in-state env value tests are covered
  at the manifest level only.
- Gate: my seven binaries plus the realtime, manifest, snapshot and core-parity binaries that touch the edited source
  pass; clippy `-D warnings` is clean for `--lib` and my tests on both feature sets. The full `--all-targets` clippy
  currently fails on another agent's in-flight file (`tests/py_core15_r15c_realtime_settings.rs`, `err_expect`) and, with default features, on `tests/py_core15_r15b_call_id_reuse.rs`; neither is mine.

Status: DONE_WITH_CONCERNS
Summary: 81 of 97 files processed (779 Python-only before, 19 ported, 10 covered, 750 remain valid); per-session memory
manager registry implemented in `src/sandbox` with Python's errors and unregister-on-close; custom env value types persist;
GA realtime audio format names normalized. 16 files are still at name level (listed above), so the list is not finished.
Concerns: the unread files; the shared-session handoff target still builds its own memory manager; only my binaries and the
sandbox/realtime/memory binaries were re-run, not the full workspace suite.
