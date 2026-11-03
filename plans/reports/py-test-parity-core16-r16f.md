# Core16 r16f: re-audit of the python-only bucket, work list r16_5 (33 of 33 files done)

Crate 0.17.0 plus other agents' uncommitted work. Python-only before 313; moved to ported 7; moved to covered 8; remaining 298. read_depth is `full` only where every python-only body was read.

## Source changes

1. `src/voice/result.rs`: when a voice session fails, `StreamedAudioResult::stream` now aborts and awaits the producer, the dispatcher and every segment task before it hands the error to the consumer (Python: the public stream awaits sibling task cleanup after a TTS failure). Before, a sibling TTS segment kept running until the result was dropped. The segment tasks a cancelled producer was awaiting are handed back to the writer (`Joining` guard) so they are not detached. Test: `tests/py_core16_r16f_voice.rs`.
2. `src/sandbox/session.rs` (`write_file_as_with`, new default trait method; `write_file_as` calls it with `exclusive = false`) and `src/sandbox/sandboxes/unix_local.rs` (`write_new_file`): Add File for a run-as user stages the file and hard-links it into place, so a name that appears between validation and write, or a dangling symlink leaf, is refused with `already exists` and never replaced (before, `mv` replaced it). The after-validate hook now runs on that path too. Tests: `tests/py_core16_r16f_unix_local_create_user.rs`.

No public API is removed or changed; (2) adds one public default trait method.

## New tests
`py_core16_r16f_native_and_decorator.rs` (2), `py_core16_r16f_voice.rs` (1), `py_core16_r16f_memory_flush_cancel.rs` (2), `py_core16_r16f_unix_local_create_user.rs` (2).

Not mine: `py_core16_r16e_unix_local_user` (mkdir_as/rm_as) failed once while r16e was editing.


## Per-file table

| file | before | ported | covered | after | what remains and why |
|---|--:|--:|--:|--:|---|
| tests/test_function_tool_decorator.py | 50 | 0 | 0 | 50 | callable-object / __wrapped__ / inspect shapes (dynamic Python callables; Rust tools are explicit FunctionTool values); timeout config extended in py_core16_r16f_native_and_decorator.rs (not a moved case) |
| tests/extensions/memory/test_redis_session.py | 44 | 0 | 0 | 44 | redis-py pool/WATCH/release internals (Rust writes are single Lua scripts) / close() terminal state (Rust has no close; drop releases) / asyncio cancel / decode_responses / live redis |
| tests/test_native_tool_failure_history.py | 32 | 2 | 0 | 30 | ported approved_native_output_survives_custom_data_failure x2 (non-object extractor as the failing extractor); rest: CancelledError tool / cancelled child / cleanup that converts cancel to error / late finalization budgets under asyncio |
| tests/voice/test_pipeline.py | 27 | 1 | 0 | 26 | ported voice_pipeline_awaits_task_cleanup_after_tts_failure (source fix: StreamedAudioResult now stops and awaits the producer and segment tasks before delivering a failure); rest asyncio cancel / private queue+dispatcher / numpy dtype / dict config |
| tests/extensions/memory/test_mongodb_session.py | 21 | 0 | 0 | 21 | close() terminal state / non-string ids (typed) / asyncio cancel waits / injected-client driver info kwargs |
| tests/test_agent_runner.py | 20 | 0 | 4 | 16 | covered: reasoning-blocked and accepted-prefix (py_core2_runner_guardrail_sessions.rs) |
| tests/test_decisions_examples.py | 15 | 0 | 0 | 15 | imports the Python examples/decisions recipes (examples) |
| tests/mcp/test_mcp_util.py | 13 | 0 | 3 | 10 | covered by py_core16_r16gaps_mcp_cancellation.rs (inner cancellation x3); rest outer asyncio cancel / legacy no-agent signature / live FastMCP |
| tests/sandbox/test_memory.py | 11 | 2 | 0 | 9 | ported flush cancel + fatal worker (py_core16_r16f_memory_flush_cancel.rs); rest dict/typed config normalization / private payload helpers / CancelledError raised by worker |
| tests/extensions/memory/test_sqlalchemy_session.py | 9 | 0 | 0 | 9 | patched fake SQLAlchemy sessions / dialect fallbacks / cross-loop engines / engine registry gc / asyncio cancel waits for transaction exit (a dropped Rust future cannot wait) |
| tests/sandbox/test_unix_local_create.py | 8 | 2 | 0 | 6 | ported bound-user creator-between-validation-and-write and nested+dangling-leaf (source fix: exclusive hard-link create for run-as users); rest: bound-user partial-write and close-failure injected into Python's worker/os.fdopen/copyfileobj (Rust stages then links atomically so no partial destination exists) |
| tests/test_tool_context.py | 7 | 0 | 0 | 7 | identity hash / required-field ValueErrors and dict run_config normalization (typed constructor in Rust) / from_agent_context taking a parent ToolContext (polymorphic Python argument; Rust builds from_call) / annotation-based context-kind detection of on_invoke_tool |
| tests/test_run_serial_tests.py | 6 | 0 | 0 | 6 | tests of .github/scripts/run_serial_tests.py (Python CI script) |
| tests/memory/test_session_context_wrapper.py | 5 | 0 | 0 | 5 | signature-inspection opt-in rules (legacy kwargs / failed inspection / partial opt-in / named wrapper parameter / all-operations rule); Rust opts in by overriding *_with_context methods |
| tests/test_programmatic_tool_calling.py | 5 | 0 | 0 | 5 | return-annotation inference of the output schema (awaitable/PEP 695 alias/Annotated/TypedDict/dataclass); Rust takes output_type or output_json_schema explicitly (ported) |
| tests/models/test_openai_chatcompletions_stream.py | 4 | 0 | 0 | 4 | provider stream aclose / cancel during in-flight close (asyncio; drop closes the connection in Rust |
| tests/test_asyncio_stability_command.py | 4 | 0 | 0 | 4 | tests of .github/scripts/run-asyncio-teardown-stability.sh (CI shell script) |
| tests/test_tool_exception_formatter.py | 4 | 0 | 1 | 3 | covered: MCP inner cancellation keeps the default policy and never calls the formatter (py_core16_r16gaps_mcp_cancellation.rs); rest tool raising CancelledError / old positional ToolErrorFormatterArgs constructor |
| tests/models/test_litellm_extra_body.py | 3 | 0 | 0 | 3 | Python truthiness of a Reasoning object; litellm-library kwarg _skip_mcp_handler x2 (the Rust LitellmModel speaks HTTP to a proxy) |
| tests/sandbox/test_extract.py | 3 | 0 | 0 | 3 | Python io stream objects (zipfile_compatible_stream over seekable-less streams; chunked non-seekable binary write); Rust writes byte slices |
| tests/test_guardrails.py | 3 | 0 | 0 | 3 | non-callable guardrail UserError x2 (typed closure in Rust) / Temporal compat patch of should_cancel_parallel_model_task_on_input_guardrail_trip |
| tests/test_shell_call_serialization.py | 3 | 0 | 0 | 3 | bytes/bytearray commands (no JSON form) and the string-outcome branch (typed executor output) |
| tests/extensions/experiemental/codex/test_payloads.py | 2 | 0 | 0 | 2 | mapping protocol (__getitem__/get/keys/in/KeyError) of Python dataclasses |
| tests/memory/test_compaction_suffix.py | 2 | 0 | 0 | 2 | asyncio cancellation of a suffix transaction settling before a newer append (Rust cancels by dropping futures) |
| tests/sandbox/test_docker_removal_worker.py | 2 | 0 | 0 | 2 | patched ExitStack.callback registration step (no Rust analogue: OwnedFd drop) / KeyboardInterrupt in a forked child via os._exit |
| tests/test_agents_logging.py | 2 | 0 | 0 | 2 | sys.stdout replaced at runtime and stderr fallback of a logging.StreamHandler (Rust prints at call time; idempotent install |
| tests/test_tool_guardrails.py | 2 | 0 | 0 | 2 | non-callable tool guardrail UserError x2 (typed closure in Rust) |
| tests/realtime/test_audio_formats_unit.py | 1 | 0 | 0 | 1 | hostile dict subclass whose __str__/__repr__ raise (serde_json values have none) |
| tests/realtime/test_session_history.py | 1 | 0 | 0 | 1 | patched module logger with a bogus content entry that trips an assertion (typed items in Rust) |
| tests/sandbox/test_pty_output.py | 1 | 0 | 0 | 1 | monkeypatched time.monotonic plus injected poll/settle callbacks |
| tests/test_call_model_input_filter.py | 1 | 0 | 0 | 1 | filter returning a str (the Rust filter returns a typed value) |
| tests/test_run_context_approvals.py | 1 | 0 | 0 | 1 | raw deferred_top_level: ledger key of a permanent deferred approval (Rust keeps the same decisions under its own key; the observable status for the item and the no-alias-to-bare-name behaviour are covered by py_core2_approvals_context.rs) |
| tests/tracing/test_reasoning_export.py | 1 | 0 | 0 | 1 | tuple/UserList/UserDict message sequences handed to generation_span (dynamic Python containers) |
