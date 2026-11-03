# py-test-parity core16 r16a

Work list: 31 files, every python-only case body read (read_depth=full unless stated).

| file | python_only before | ported | covered | remaining | notes |
|---|---|---|---|---|---|
| tests/sandbox/test_mount_security.py | 120 | 0 | 0 | 120 | all remaining rows are hostile pydantic subclass/__deepcopy__/descriptor/exception-group/traceback/raw-dict cases or manifest-in-state sanitizers (Rust state carries no manifest; structural guarantee already ported) |
| tests/test_scripted_model.py | 29 | 3 | 0 | 26 | ported: recorded settings/input snapshot and detached call history (py_core16_r16a_scripted.rs). Rest: mutable aliasing at queue time, dict steps, aclose/__str__/BaseException failure modes, contextvars span scope |
| tests/test_scripted_sandbox.py | 25 | 0 | 1 | 24 | covered: only-configured-methods by py_core5_leftover1_scripted_sandbox::scripted_sandbox_reports_structured_payload_free_failures. Rest: io.BytesIO/StringIO snapshots, dict step validation, deepcopy failure, matcher exceptions |
| tests/test_shell_examples.py | 21 | 0 | 0 | 21 | Python examples/ scripts (monkeypatched stdin/subprocess/input); the Rust example is different code |
| tests/test_source_compat_constructors.py | 17 | 0 | 0 | 17 | positional constructor binding of Python dataclasses; non-bool dynamic validation |
| tests/test_example_workflows.py | 14 | 2 | 0 | 12 | ported: nested-run failure keeps sibling output (default message) and None policy propagates (py_core16_r16a_agent_tools.rs). Rest: examples/ modules (web search urls, financial research, docker runner, sandbox examples) |
| tests/test_hitl_error_scenarios.py | 12 | 4 | 1 | 7 | ported: deferred-lookup resume, reject+pending+finished sibling, plain sibling without output re-runs; covered: local MCP resume without binding (py_core6_mcpmisc_resume_binding). Rest: hashable item, dynamic needs_approval type, __bool__ falsy items, same-named sibling agent identity, public-vs-execution agent object identity, hand-built ProcessedResponse |
| tests/voice/test_openai_stt.py | 10 | 1 | 0 | 9 | ported: buffered transcript delivered before listener failure (py_core16_r16a_voice_stt.rs). Rest: asyncio cancellation/aclose with mocks, numpy dtype, Python 3.10 timeout class |
| tests/deferred_session_writes/test_pending_write.py | 8 | 5 | 0 | 3 | ported: pairing guard (2, unit tests in src/run_internal/items.rs, with a source fix for hosted MCP pairing) and remote-history reasoning-id policy (3, py_core16_r16a_held_policy.rs). Rest: settled-count of private save helpers |
| tests/realtime/test_session_tool_outputs.py | 7 | 0 | 0 | 7 | tool-end asyncio cancel, pydantic/dataclass/model_dump serialization fallbacks, serialization-failure patching of a private helper (Rust tools return typed text) |
| tests/sandbox/test_unix_local_file_io.py | 6 | 0 | 0 | 6 | os.open/os.scandir monkeypatch TOCTOU, fd-leak and thread cancel cases (Rust read returns bytes, RAII fds). GAP (not a valid reason): Python applies capability-changed non-host path grants to an injected session; Rust apply_live_session_manifest keeps the original manifest and pinned_grants, so the grants are silently ignored (src/sandbox/runtime.rs, unix_local.rs pinned_grants). Needs interior-mutable session manifest; not fixed |
| tests/test_tool_name_collision_policy.py | 6 | 0 | 0 | 6 | asyncio cancellation during formatter (2), approval mutation during an awaiting get_all_tools (2) |
| tests/tracing/test_import_side_effects.py | 5 | 0 | 0 | 5 | fresh-interpreter import checks (httpx2, sqlite3, lazy export) and atexit/module-global bootstrap state; Rust has OnceLock providers (partly covered by py_tracing_core default-provider tests) |
| tests/realtime/test_app_server_debug.py | 4 | 0 | 0 | 4 | examples/realtime/app server module logging (Python example) |
| tests/test_extension_filters.py | 4 | 2 | 0 | 2 | ported: unknown-type item and structured content summaries (py_core16_r16a_handoff_nest.rs). Rest: string input_history (typed Vec in Rust), objects with raising __str__ |
| tests/tracing/test_traces_impl.py | 4 | 0 | 1 | 3 | covered: reattach restores scope (src tracing/trace_state.rs and tests/py_tracing_core.rs). Lifecycle and double-start are already in py_tracing_core.rs (ledger stale). Rest: contextvar foreign-token ValueError, NoOpTrace double enter logging, processor raising in on_trace_end |
| tests/models/test_responses_websocket_session.py | 3 | 0 | 0 | 3 | dict run_config coercion (2); monkeypatched Runner.run/run_streamed injection. NOTE: Rust ResponsesWebSocketSession has no run/run_streamed wrappers (callers pass run_config()), an API difference not a behaviour gap |
| tests/test_run_context_wrapper.py | 3 | 0 | 0 | 3 | _to_str_or_none with raising __str__, raw-dict name/call-id fallbacks, falsy ToolApprovalItem (Rust items are typed) |
| tests/test_usage.py | 3 | 0 | 0 | 3 | raw-usage snapshot of pydantic/non-JSON objects and detached aliasing (Rust values are JSON and owned) |
| tests/sandbox/test_unix_local_special_files.py | 3 | 0 | 0 | 3 | Linux-only fcntl F_SETLEASE kernel lease and an os.open monkeypatch injecting EWOULDBLOCK (not runnable on the macOS dev host; no injection seam) |
| tests/extensions/sandbox/test_cloudflare.py | 2 | 0 | 0 | 2 | asyncio cancellation of start()/resume with blocking snapshot/mount; hydrate remount order and terminal-delete failure are already covered in py_core6_cfdaytona_cloudflare.rs |
| tests/models/test_model_payload_iterators.py | 2 | 0 | 0 | 2 | single-use Python iterables inside request payloads; monkeypatched converters |
| tests/sandbox/test_retry.py | 2 | 0 | 0 | 2 | Python __cause__/__context__ exception-chain helpers (no Rust equivalent API) |
| tests/test_items_helpers.py | 2 | 0 | 0 | 2 | weakref/gc agent release; other listed names are ported in py_core_b_items_helpers.rs |
| tests/voice/test_input.py | 2 | 0 | 1 | 1 | covered: float32 base64 does not mutate (py_voice_misc.rs::audio_input_to_base64_does_not_mutate_the_buffer). Rest: numpy float64 dtype rejection (typed AudioData) |
| tests/realtime/test_openai_realtime_conversions.py | 1 | 0 | 0 | 1 | missing/non-string text parts and non-function tool objects (typed RealtimeUserInputContent / tool structs) |
| tests/sandbox/capabilities/test_compaction_capability.py | 1 | 0 | 0 | 1 | unsupported policy type passed to a pydantic field (typed policy in Rust) |
| tests/sandbox/test_workspace_paths.py | 1 | 1 | 0 | 0 | ported: inaccessible root is a remote path (py_core16_r16a_workspace_root.rs) |
| tests/test_exception_exports.py | 1 | 0 | 0 | 1 | agents.__all__ / module export identity; MCPToolCancellationError is an AgentsError variant in Rust |
| tests/test_stream_events.py | 1 | 0 | 0 | 1 | private stream_step_items_to_queue / stream_step_result_to_queue helpers with asyncio.Queue and a duck-typed StepResult; local MCP title case is covered in py_core8_r8g8_stream_events.rs |
| tests/tracing/test_tracing_env_disable.py | 1 | 0 | 0 | 1 | falsy (__bool__) span parent and caplog debug message; env cache cases are already ported in py_tracing_core.rs |
| **total (31 files)** | 320 | 18 | 4 | 298 | |

## Tests added (all green, clippy -D warnings clean on both feature sets for these files)

`tests/py_core16_r16a_scripted.rs` (2), `py_core16_r16a_agent_tools.rs` (5), `py_core16_r16a_voice_stt.rs` (1), `py_core16_r16a_held_policy.rs` (2), `py_core16_r16a_handoff_nest.rs` (2), `py_core16_r16a_workspace_root.rs` (1), plus two unit tests in `src/run_internal/items.rs` (held-batch pairing guard).

## Source fix

`src/run_internal/mod.rs::settle_safe_items`: a held batch settled an unpaired hosted MCP approval request. It now settles a request only when its response is present or its approval is still open (port of `_held_items_safe_to_settle`). Covered by `run_internal::items::tests::a_held_batch_settles_only_requests_that_can_pair`.

## Divergences found and NOT fixed (decision needed)

1. `sandbox/test_unix_local_file_io.py::test_injected_session_uses_current_grants_without_rebinding_aliases`: Python applies capability-changed non-host path grants to an injected session. Rust `apply_live_session_manifest` (src/sandbox/runtime.rs) keeps the session's original manifest and `UnixLocalSandboxSession::pinned_grants` is fixed at construction, so a grant added or changed by a capability is silently ignored. Fixing needs an interior-mutable manifest/grant set on sessions.
2. `ResponsesWebSocketSession` has no `run`/`run_streamed` wrappers (callers pass `run_config()`); API difference only.
3. Rust trusts any third-party `impl MountStrategy` (carried over from the earlier mount-security rounds; unchanged).

## Notes on counts

For several files the ledger's python_only count is lower than the number of Python test names with no Rust name match (the ledger had already moved cases to ported/covered under other names, e.g. `tests/py_tracing_core.rs`, `tests/py_core_b_items_helpers.rs`). The CSV subtracts only what this round moved; remaining = ledger python_only minus this round's moves. For `test_hitl_error_scenarios.py`, `test_extension_filters.py` and `test_stream_events.py` the number of bodies read exceeds the ledger remainder for that reason.

Mount-security: bodies of every hostile-subclass, decorator-internals, raw-dict and manifest-in-state group were read (about 70 test functions covering the 120 expanded rows); none has a Rust surface (Rust `SandboxSessionState` carries only a manifest digest, strategies are trait impls, errors are structured).

Status: DONE_WITH_CONCERNS. Concerns: the injected-session grant divergence above; the cross-agent `py_core16_r16e_encrypt_interleave.rs` has an unused import that fails `clippy --all-targets` (not mine).
