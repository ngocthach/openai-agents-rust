# Core14 r14b: re-audit of the python-only bucket, work list r14b (119 files, 1,293 cases)

Crate 0.16.0. Totals: python-only before 1,293; moved to ported 16; moved to covered 0; remaining 1,277. The CSV (`py-test-parity-core14-r14b.csv`) lists all 119 files; rows changed: `sandbox/test_runtime.py` (6), `sandbox/test_workspace_paths.py` (4), `test_tool_context.py` (4), `test_run_impl_resume_paths.py` (2).

## Source change (additive public API)

`SandboxAgent::as_tool(name, description, SandboxRunConfig)` and `SandboxAgent::as_tool_with(AgentAsToolOptions, SandboxRunConfig)` (`src/sandbox/agent.rs`). A sandbox agent used as a tool of a plain parent now runs its nested call through `SandboxRunner::run_with_options` with the given sandbox config: capability instructions and tools, manifest, session create or reuse, and teardown, and resume from a nested `RunState` goes through the same path. Python reads that config from the parent's `RunConfig.sandbox`; Rust's `RunConfig` has no sandbox field, so the config is passed to `as_tool`. Implementation: `Agent::try_as_tool_with_runner` (crate-private, `src/agent.rs`) takes an optional runner replacing the plain `Runner::run_with_options` call; `AgentAsToolOptions` is unchanged (a new public field would break struct-update users). `on_stream` nested runs still use the plain streaming runner (not routed through the sandbox).

Rust's `UnixLocalSandboxClient::delete` keeps returning the mount-detach error and leaves the workspace on disk (Python logs it and returns). Documented, not changed.

## Ported (16 cases, 12 tests in 7 new binaries, all green with `--all-features`; clippy `-D warnings` clean on lib and the new tests under all-features and default; rustfmt clean)

| Python case | Rust test |
|---|---|
| `test_runtime.py::test_unix_local_client_delete_unmounts_workspace_mounts_before_rmtree` | `py_core14_r14b_unix_delete.rs` (detach runs while the root exists, root removed after) |
| `..::test_unix_local_client_delete_skips_rmtree_when_unmount_fails[redacted,diagnostic]` (2) | same file, on-disk half only (the log record has no Rust counterpart) |
| `..::test_remote_realpath_empty_success_output_is_transport_error` | `py_core14_r14b_remote_realpath.rs` |
| `..::test_runner_streamed_guardrail_trip_blocks_running_injected_session_mutation` | `py_core14_r14b_injected_guardrail.rs` (streamed and non-streamed). Also disproved a stale "DEFERRED" note in `py_sandbox_a_runtime_4.rs` (comment fixed) |
| `..::test_sandbox_agent_as_tool_uses_runner_sandbox_prep` | `py_core14_r14b_sandbox_as_tool.rs` (new `as_tool`) |
| `test_workspace_paths.py`: windows-drive relative path, two posix-normalisation cases, recursive remove | `py_core14_r14b_workspace_paths.rs` (3 tests, Windows inputs as Windows-looking strings) |
| `test_tool_context.py`: from-agent-context fields, agent none by default, namespace from tool call, explicit run config | `py_core14_r14b_tool_context.rs` (3 tests) |
| `test_run_impl_resume_paths.py::test_fresh_streamed_handoff_publishes_agent_update_before_session_append_failure`, `..._drains_agent_update_event_for_slow_consumer` | `py_core14_r14b_streamed_handoff_session.rs` (2 tests) |

## Files read (Python bodies skimmed for every candidate not obviously ported, each Rust counterpart located)

For each file below, every remaining case is either a valid reason from the list or was found already ported/covered under a different name (names differ: the Rust ports mostly drop the `test_` prefix and rename, so the reports' counts stay authoritative; I did not move numbers for cases that were already covered).

1. `test_run_state.py` (134): context (de)serializer for Python objects, `get_interruptions` deepcopy/pydantic/descriptor hardening, private `_deserialize_*`, falsy objects, object identity. Items that looked portable (`deserialize_*item*`, tool approval item, `from_json_agent_not_found`, JSON decode error redaction) are in `py_core2_runstate_items.rs`, `py_core7_a_runstate_corpus.rs`, `py_core8_r8g8_runstate_identity.rs`.
2. `mcp/test_server_errors.py` (72): exception-group / anyio / traceback-locals / log-record shape, `patch.object` of private retry methods, constructor validation of a dynamically typed timeout. Prompt/resource/list/call credential redaction is ported in `py_core13_r13b_mcp_errors.rs`.
3. `sandbox/test_runtime.py` (51, now 45): SandboxRuntime/session-manager internals (cached `prepare_agent`, clone identity, start hooks mutating the agent in place), package-resource lookup, asyncio cancellation. Ported items above.
4. `test_function_tool_decorator.py` (50): callable objects, `__wrapped__`, `inspect`, generic/annotation resolution. `defer_loading`/`timeout*` decorator kwargs exist as `FunctionTool::with_*`.
5. `test_run_step_execution.py` (45): tool-task cancellation and BaseException ordering (asyncio), fallible `is_enabled` (Rust's returns `bool`), exception objects as tool output, hand-built `ProcessedResponse` for hosted-MCP approval collection (private `collect_manual_mcp_approvals`).
6. `test_agent_scoped_approvals.py` (40): same-named agents and distinct same-named tools told apart by object identity; `weakref`/gc; context converters (python/json/temporal). Rust identifies agents by name.
7. `extensions/sandbox/test_vercel.py` (35): state-class construction, asyncio lock/cancellation/traceback redaction, SDK `get` call arguments. Mounts, dynamic-mount refusal, resume project/team and snapshot resume are in `py_core6_modalvercel_vercel_mounts.rs`, `py_core8_r8g3_vercel_live_manifest.rs`, `py_core13_r13a_vercel_state.rs`. Rust resumes over HTTP and must read the token from the environment itself.
8. `models/test_openai_responses.py` (34): stream close / cancel / websocket close handshakes; prompt-id, request id, usage counting and unpaired outputs are ported (`py_models_responses_*.rs`).
9. `realtime/test_session.py` (34): close/cleanup/tracked-task races (asyncio); duplicate-tool-name validation, call-id checks, tool timeout are ported (`py_realtime_session_*.rs`, `py_realtime_openai_sequencing.rs`).
10. `test_tool_batch_failure_history.py` (32): hooks that raise (Rust hooks return `()`), opaque-output deepcopy identity, asyncio races.
11. `test_scripted_model.py` (29), `test_scripted_sandbox.py` (25), `realtime/test_testing.py` (21): helper tests about aliasing snapshots, span contextvars, `__str__` raising, dict-keyed step specs.
12. `sandbox/test_compatibility_guards.py` (27): `__all__`, dataclass field order, pydantic field order.
13. `codex/test_codex_tool.py` (26) and `codex/test_codex_exec_thread.py` (18): pydantic/dataclass contexts, option coercion, private helpers, `monkeypatch` of subprocess; span budget, `find_codex_path`, stream limit, web_search flags are covered by `src/extensions/codex/*` unit tests and `py_codex_*.rs`. "no stdin/no stdout" cannot occur with piped tokio stdio.
14. `sandbox/test_docker.py` (23): docker-py sockets, stream framing, spool files, deferred cleanup tasks, image-ref parsing by the Python client. Persist pruning, volume names, labels are in `py_sandbox_a_docker_*.rs`.
15. `test_agent_as_tool.py` (21): `Runner.run` monkeypatched with dummy results, run-registry scope ids.
16. `extensions/memory/test_async_sqlite_session.py` (20), `test_advanced_sqlite_session.py` (14), `test_sqlalchemy_session.py` (9), `test_dapr_session.py` (14): private connection/quarantine, rollback injection, cancellation, and `close()` lifecycle (Rust sessions have no `close`; drop releases them and ownership forbids use after close). Behaviour cases (corrupt rows, branches, usage, same-timestamp order, first access) are in `py_memory_sqlx.rs`, `py_memory_advanced_sqlite*.rs`, `py_core13_r13b_*`.
17. `models/test_any_llm_model.py` (14): any-llm vendor provider internals (Google `http_options`, content roles), module import bindings, stream close failures; the rest is in `py_ext_models_any_llm.rs`, `py_ext_models_ordering.rs`.
18. `test_trace_processor.py` (14): non-JSON Python keys/values, unpaired surrogates, cyclic values, monotonic-clock scheduling, atexit subprocess script; Retry-After and truncation are unit-tested in `src/tracing/processors.rs`.
19. `sandbox/test_manifest_config.py` (13), `sandbox/test_workspace_payloads.py` (9), `sandbox/test_unix_local_create.py` (8), `sandbox/test_unix_local_user_file_io.py` (19): dictionary-typed manifests, IO stream adapters, `os` monkeypatching of the worker.
20. `mcp/test_mcp_v2_http.py` (12): MCP Python SDK v2 / httpx2 objects.
21. `memory/test_openai_responses_compaction_session.py` (12): `MagicMock` sessions, falsy default client, model-name validation; rollback budget, restore and threshold cases are ported (`py_core6_final7_compaction.rs`, `py_memory_responses_compaction_*.rs`).
22. `memory/test_openai_conversations_session.py` (9): mocked Python client internals; clear/add/pop/lazy-create semantics are in `py_memory_conversations*.rs`.
23. `test_hitl_error_scenarios.py` (12): hand-built `ProcessedResponse`, falsy items, dynamically typed `needs_approval`.
24. `test_run_impl_resume_paths.py` (11, now 9): `monkeypatch` of run-loop internals, `SingleStepResult` construction, redaction through nested callbacks and CancelledError.
25. `test_tool_context.py` (11, now 7): hashing, required-field `ValueError`, dict-typed run config, inheriting from a parent `ToolContext` (no Rust API), string-annotation introspection.
26. `test_handoff_history_duplication.py` (10): private ownership markers, dataclass-field introspection, object identity.
27. `test_tool_approval_call_id_reuse.py` (10): hooks that raise, `model_dump` signature probing, hosted-MCP private collection helpers.
28. `voice/test_openai_stt.py` (10): asyncio cancel/cleanup; setup happens in `create_session` in Rust, so a transcript buffered before setup cannot be delivered ahead of a setup error.
29. `test_source_compat_constructors.py` (17): positional constructor binding.
30. `test_check_optional_truthiness.py` (71), `test_code_change_verification_runner.py` (19): Python CI scripts under `.agents/` and `scripts/`.
31. Name-level only (bodies of the portable-looking cases read): `test_tool_custom_data.py`, `test_usage.py`, `test_agents_logging.py`, `test_tool_origin.py`, `test_run_context_approvals.py`, `test_shell_call_serialization.py`, `test_visualization.py`, `sandbox/test_tar_utils.py`, `sandbox/test_extract.py`.

## Gap found, not closed

Rust `ToolContext` has no "derive from a parent tool context and inherit agent/run config" constructor (Python `from_agent_context(parent_tool_context, ...)`); two cases remain python-only for that reason.

## Files not reached (no per-case read, counts unchanged)

All 119 files not listed in "Files read" or the item-31 name-level group: `test_agent_runner_sync.py`, `sandbox/test_workspace_paths.py` (done), the voice/realtime/tracing singletons (`voice/test_testing.py`, `voice/test_input.py`, `voice/test_openai_model_provider.py`, `voice/test_openai_stt_session_config.py`, `voice/test_pipeline_tracing.py`, `realtime/test_session_tool_outputs.py`, `realtime/test_realtime_handoffs.py`, `realtime/test_app_server_debug.py`, `realtime/test_conversion_helpers.py`, `realtime/test_playback_tracker.py`, `realtime/test_session_close_iteration.py`, `realtime/test_session_history.py`), `models/test_openai_retry_helpers.py`, `models/test_agent_registration.py`, `models/test_litellm_*.py`, `models/test_responses_websocket_session.py`, `models/test_anthropic_thinking_blocks.py`, `models/test_model_payload_iterators.py`, `tracing/*`, `mcp/test_streamable_http_client_factory.py`, `extensions/sandbox/test_cloudflare.py`, `extensions/sandbox/test_vercel_provider_contract.py`, `sandbox/test_docker_removal_worker.py`, `sandbox/test_client_options.py`, `sandbox/test_session_sinks.py`, `sandbox/test_pty_output.py`, `sandbox/test_workspace_jsonl_sink.py`, `sandbox/test_retry.py`, `sandbox/test_dependencies.py`, `sandbox/capabilities/*`, `sandbox/test_unix_local_file_io.py`, `sandbox/test_unix_local_special_files.py`, `test_run_state_pydantic_context.py`, `test_asyncio_progress.py`, `test_asyncio_stability_command.py`, `test_code_change_verification_policy.py`, `test_run_serial_tests.py`, `test_tool_name_collision_policy.py`, `test_server_conversation_tracker.py`, `test_extension_filters.py`, `test_items_helpers.py`, `test_guardrails.py`, `test_run_context_wrapper.py`, `test_run_internal_approvals.py`, `test_computer_tool_lifecycle.py`, `test_doc_parsing.py`, `test_pretty_print.py`, `test_tool_guardrails.py`, `test_agent_hooks.py`, `test_agent_tool_input.py`, `test_call_model_input_filter_unit.py`, `test_handoff_failure_error_function.py`, `test_image_generation_tool_reference.py`, `test_runner_model_provider_lifecycle.py`, `test_runner_tracing_disabled.py`, `test_tool_output_conversion.py`, `deferred_session_writes/test_compaction.py`, `memory/test_compaction_suffix.py`, `memory/test_session_context_wrapper.py`, `codex/test_payloads.py`.

## Unresolved questions

- Should `RunConfig` gain a sandbox field so `Runner::run` on a plain parent applies it to nested sandbox agents (Python behaviour), instead of passing the config to `SandboxAgent::as_tool`?
- Should nested `on_stream` runs of a sandbox agent tool also prepare the sandbox?

Status: DONE_WITH_CONCERNS
Summary: 16 cases ported (12 tests, 7 binaries); `SandboxAgent::as_tool` added; about 40 files read by body skim with Rust counterparts located, about 60 not reached; 1,277 remain python-only.
Concerns: the "read every case of 40 files" bar was met at skim depth (candidate bodies read, not every parametrized case line by line) for the large files; the unreached list above is exact.
