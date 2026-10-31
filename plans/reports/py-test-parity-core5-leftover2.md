# Py test parity core5: leftover2

Python files (under `openai-agents-python/tests/`) that no earlier round ported, plus two verify-then-finish files. Counted per test case: `def test_` plus parametrize expansion.

## Per-file totals

| Python file | cases | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| test_invalid_final_output_handler.py | 15 | 15 | 0 | 0 | 0 |
| test_agent_tool_enablement_snapshot.py | 4 | 3 | 0 | 1 | 0 |
| test_agent_config.py | 19 | 12 | 0 | 7 | 0 |
| test_logprobs.py | 5 | 4 | 1 | 0 | 0 |
| test_disabled_trace_provider_flush.py | 1 | 1 | 0 | 0 | 0 |
| test_tool_converter.py | 6 | 6 | 0 | 0 | 0 |
| test_computer_action.py | 40 | 37 | 0 | 3 | 0 |
| test_call_model_input_filter.py | 12 | 9 | 2 | 1 | 0 |
| test_agent_as_tool.py | 88 | 58 | 6 | 24 | 0 |
| **Total** | **190** | **145** | **9** | **36** | **0** |

Rust test files (all new, `tests/py_core5_leftover2_*.rs`): `invalid_final_output` (12 fns), `tool_enablement` (2), `agent_config` (9), `logprobs_converter` (12), `computer_actions` (10), `spans` (5), `input_filter` (4), `agent_tool_config` (24), `agent_tool_stream` (17), `agent_tool_approvals` (5). 100 tests, all green, stable over 3 runs.

## Classification notes

- **enablement_snapshot**: tests 1 and 3 ported as non-mutating contracts (original tool returned, `agent.tools` untouched, enablement re-evaluated per run/turn, blocking + streamed). Test 2 python-only: replaces `agent.tools` while `get_all_tools` is suspended; a Rust `Agent` is shared immutably behind an `Arc` and `is_enabled` gets only the agent name.
- **agent_config python-only (7)**: `Agent.__post_init__` type checks (name, tool_use_behavior, hooks, tools/handoffs list types = 4) and "no constructor kwarg for model settings" (3); all compile-time in Rust. The dict form of `model_settings` maps to the new `ModelSettings::from_value`.
- **logprobs covered (1)**: Responses `top_logprobs` -> `py_models_responses_core::top_logprobs_adds_the_logprobs_include`. Chat cases assert on `build_request_body`.
- **flush**: also covered at provider level by `py_core3_tracing_processor::default_trace_provider_force_flush_still_flushes_when_disabled`; ported at the global `flush_traces` level.
- **computer_action python-only (3)**: legacy driver, non-introspectable driver, `**kwargs` driver (duck-typed `keys` kwarg filtering via `inspect.signature`); the Rust `Computer` trait always takes `keys: Option<&[String]>`. Sync vs async driver parameter sets (9+9) both run through the one `async_trait`.
- **call_model_input_filter**: covered (2) = `py_core3_runstatectx_context::call_model_input_filter_{sync_non_streamed,async_streamed}_replaces_input_and_instructions`. Python-only (1) = invalid return type (typed in Rust). The old `py_core_a_runner.rs` header cited this file but held no filter test.
- **agent_as_tool covered (6)**: `py_core_a_agent_as_tool::{agent_as_tool_is_enabled_bool, agent_as_tool_is_enabled_callable, agent_as_tool_returns_final_output, agent_as_tool_failure_error_function_none_reraises, agent_as_tool_failure_error_function_custom_handler}` and `py_core2_runstate_resume::resume_recursively_nested_agent_as_tool_decision` (approve=false = rejected nested approval resumes). The earlier "11 ported" in py_core_a was weaker than the Python assertions (only `custom_output_extractor`, `structured_input`, `custom_input_builder` overlap); those were re-ported with the full assertions.
- **agent_as_tool python-only (24)**:
  - truthiness/typing: falsey extractor, falsey input builder, invalid builder output (3)
  - Python's global agent-tool run registry and scope ids (`record_agent_tool_run_result`, `get_agent_tool_state_scope`): scope for nested ToolContext / RunContextWrapper, namespace on nested ToolContext (the nested run gets a `RunContext`, no `ToolContext`) (3); cross-scope ambiguous approval identity guard x4 (the nested state lives in the run context, nothing aliases two runs); legacy approval rebuild (`_rebuild_approvals`, deferred legacy binding) (1); hosted_tool_call `provider_data` wrapper exact decision (1)
  - cancellation/BaseException: resume survives cancellation x2 (relies on in-place mutation of the same `RunState` object across a cancelled run; Rust `RunState` is passed by value), handler `BaseException`/`CancelledError` x2, handler `CancelledError` with backlog (1), MCP `CancelledError` (1), stream-close failure injection into `RunResultStreaming.stream_events` x4 (Python internals)
  - sync-vs-async handler (1), negative `on_stream_max_pending_events` (unrepresentable in `usize`) (1)
  - Adapted ports: the 4 `hosted_mcp_nested_sticky` cases run with a function tool instead of a hosted MCP request (legacy flag not applicable).

## Rust bugs fixed (source)

| Divergence | Fix |
|---|---|
| Empty structured turn + `invalid_final_output` handler that declines ended the run with an error; Python runs another turn | `src/run_internal/mod.rs`: `handled_final` split into `try_handled_final` (None when no handler or it declines) |
| `call_model_input_filter` failure left no span error | `src/run_internal/mod.rs`: records `Error in call_model_input_filter` (redacted unless `trace_include_sensitive_data`) on the current turn span |
| Computer span error lacked `tool_name`, no log; agent span listed `computer_use_preview`; warning text missing the `Configure ComputerTool.on_safety_check ...` sentence | `src/run_internal/local_tools.rs` (span error via `record_local_tool_error`, `Failed to execute computer action` log, warning text), `src/tool.rs` (`Tool::trace_name`), `src/run_internal/mod.rs` (agent span uses it) |
| Chat converter: `defer_loading` label, tool name without namespace, `output_json_schema` not rejected | `src/models/openai_chatcompletions.rs` |
| `as_tool`: no argument validation (Python validates `{"input": str}` / `parameters`, drops undeclared fields); plain tool inherited stale `tool_input`; nested run ignored the parent `run_config`; `include_input_schema` honoured without `parameters`; builder could not return input items | `src/agent.rs` (`normalize_agent_tool_input`, inheritance, `input_items_builder`), `src/run_context.rs` (`without_tool_input`, crate-private) |
| `as_tool` `on_stream`: handler awaited inline (stalled the nested run), event agent never followed handoffs, no backlog limit, handler failure could not be isolated, parent drop did not stop the nested run | `src/agent.rs` `run_nested_streamed`: producer/consumer in one task, bounded queue (`on_stream_max_pending_events`, default 1024), panic-isolated handler logged through `log_model_and_tool_action_error`, drain-before-error, overflow cancels the run, drop-guard cancels on caller cancellation |
| Dict-form model settings: no strict validation, partial `retry`/`backoff` objects failed, reasoning extras dropped | `src/model_settings.rs`: `ModelSettings::from_value`, serde defaults on `ModelRetryBackoffSettings`/`max_retries`, `Reasoning.extra` |

## Breaking / API changes

- `Reasoning` gained a public `extra: Map<String, Value>` (struct literals need `..Default::default()`). One in-repo literal fixed: `tests/py_model_settings_serialization.rs` (+1 line).
- `Agent::as_tool` arguments are now validated: a missing/mistyped required field is `ModelBehaviorError: Invalid JSON input for tool X`, and undeclared fields no longer reach the nested agent. Two tests in `tests/py_core3_guardrails_agent_setup.rs` encoded the old pass-through (they exercised Python's `resolve_agent_tool_input` helper end to end, which the real tool never does); edited: `agent_as_tool_input_without_an_input_field_...` now asserts rejection, `..._extra_fields_...` asserts the field is dropped, and three builder tests now send valid arguments.
- New: `Agent::try_as_tool_with`, `AgentAsToolOptions::{input_items_builder, on_stream_max_pending_events}`, `DEFAULT_ON_STREAM_MAX_PENDING_EVENTS`, `ModelSettings::from_value`, `Tool::trace_name`.

## Shared-file edits

None to `src/lib.rs` or `Cargo.toml`. Source files touched: `src/agent.rs`, `src/model_settings.rs`, `src/models/openai_chatcompletions.rs`, `src/run_context.rs`, `src/run_internal/{mod,local_tools}.rs`, `src/tool.rs`; docs: `docs/project-changelog.md` (Unreleased).

## Gates

- Own tests: 100/100 green (3 runs). `cargo test --all-features --lib`: 651 passed.
- Regression batches (no fail-fast): `py_core_*`, `py_core4_collision*`, `py_core4_tracerunner*`, `core_*`, `runner_tests`, `local_tools_tests`, `py_mcp_*`, `py_memory_*`, `py_realtime_*`, `tracing_*`, `py_core5_*`, `py_models_*`, `py_ext_models_*`, `models_*`, `sessions_tests`, `mcp_tests`, codex/realtime/extension suites: green.
- Failures seen in other agents' in-flight areas (not touched by this work; `src/testing.rs` automatic-streaming limits and streamed-usage `cache_write_tokens`): `py_core4_programmatic_guardrails_approvals::typed_programmatic_tool_preserves_approval_rejection`, `py_core2_runner_streamed::raw_responses_carry_the_request_id_and_optional_raw_usage`, `py_core2_runstep_processing::process_model_response_classifies_tool_search_items`, `py_core2_toolerrors_batch` (3), `py_core3_shelltools_shell_tool::shell_tool_empty_structured_output_reaches_next_model_unchanged_streamed`.
- Clippy `-D warnings` on lib + own test targets: clean except `src/testing.rs:933` (`field_reassign_with_default`, another agent's in-flight edit; checked with `-A clippy::field_reassign_with_default`).
- rustfmt on own test files; hand-formatted own edits in shared src files (`rustfmt --check` clean for them).

## Unresolved

- `RunState` is passed by value, so a resume cancelled after a nested tool committed cannot be re-resumed from the same in-memory object without re-running the tool (Python mutates the shared object). A persisted JSON state has the same limit in both SDKs.
- Async input builders and a ready-made hosted-MCP nested approval flow were not ported (sync builder and function-tool equivalents used).
