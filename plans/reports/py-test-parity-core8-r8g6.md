# Py test parity, round 8, agent r8g6 (top-level tests/*.py, first third)

Scope: 41 Python files, 701 python-only cases before. Method: read the Python tests and the earlier reports' per-test classifications, grep Rust for existing assertions, port what Rust can express.

## Totals

- python_only before: 701; moved to ported: 41; moved to covered: 2; remaining: 658.
- New Rust tests: 7 files `tests/py_core8_r8g6_*.rs`, 18 test fns (all pass with `--all-features`; the websocket file is gated on the `websocket` feature).

## Per-file table (files with movement)

| file | before | ported | covered | after | note |
|---|--:|--:|--:|--:|---|
| test_agent_instructions_signature.py | 11 | 6 | 0 | 5 | async/sync/stateful-callable/static/None instructions ported (`py_core8_r8g6_agent_basics.rs`); arity and non-callable checks are compile-time |
| test_agent_memory_leak.py | 1 | 1 | 0 | 0 | `Arc`/`Weak` strong-count after a run |
| test_agent_runner.py | 48 | 3 | 2 | 43 | 3 session-input-callback cases (`py_core8_r8g6_session_input.rs`); covered: env default -> `py_core2_runconfig_settings::trace_include_sensitive_data_follows_env_value`, reasoning blocked response -> `py_core2_runner_guardrail_sessions::a_reasoning_bearing_terminal_response_is_dropped_whole_on_a_trip` |
| test_agent_runner_streamed.py | 27 | 3 | 0 | 24 | WebSocket-model terminal failure x2 and request-lock handoff (`py_core8_r8g6_streamed_ws.rs`, local WS server) |
| test_debug.py | 9 | 9 | 0 | 0 | env-var flags (`py_core8_r8g6_debug_env.rs`; the earlier "set_var is unsafe" reason is not valid in a single-test binary) |
| test_doc_parsing.py | 4 | 1 | 0 | 3 | Google/NumPy/Sphinx detection (`py_core8_r8g6_docstrings.rs`); `test_functions_are_ok` is fixture-only, instance/classmethod need a bound receiver (the macro does not accept associated functions) |
| test_error_logging_redaction.py | 104 | 18 | 0 | 86 | trace-processor failure identity under both data policies, 6 operations x 3 policies (`py_core8_r8g6_trace_processor_logging.rs`) |

## Remaining python-only groups (one reason each)

- test_agent_as_tool 24: monkeypatched `Runner.run`/dummy results, global agent-tool run registry and scope ids, asyncio cancellation/BaseException, Python truthiness, typed builder output, `usize` limit.
- callback_validation 1: legacy positional-argument call signature.
- clone_shallow_copy 2: list aliasing/identity.
- agent_config 7: `__post_init__` type checks, constructor kwargs.
- agent_hooks 1: handoff cycle built by mutating `handoffs`.
- instructions_signature 5: `inspect` arity checks and non-callable instructions.
- agent_runner 43: private blocked-output/snapshot helpers with hash/eq impostors and BaseException, truthiness, pydantic `model_dump`, typed input filters, session rewind helpers (Rust never rewinds, see `a_model_retry_does_not_rewind_committed_session_input`), callback matching by object identity (replace-in-place, rebuilt copy), hand-built approval items.
- agent_runner_streamed 24: typed input filters, monkeypatched finalizer/save, private queue, asyncio cancel, `Fn -> Option<String>` blocked-message formatter.
- agent_runner_sync 5, asyncio_progress 7, asyncio_tasks 12, cancel_streaming 4: asyncio event-loop/task internals.
- scoped_approvals 48: agent object identity (Rust keys approvals by name), Temporal converter, gc/weakref.
- tool_state 6, enablement_snapshot 1: gc/weakref and in-place `agent.tools` mutation.
- tool_input 1: private schema helper.
- agents_logging 2: swaps `sys.stdout` on a live handler.
- asyncio_stability_command 4, change_detection 62, check_optional_truthiness 71, code_change_verification_* 25, dapr/daytona/decisions/example_workflows/financial_research examples 83: Python-only CI scripts, packaging and examples.
- call_model_input_filter 1 + unit 1: wrong return type is a compile error.
- computer_action 3, computer_tool_lifecycle 3: duck-typed drivers, replacing `agent.tools` in a hook.
- config 13: httpx2 client type, falsy client, `openai.OpenAIError` at construction, per-event-loop websocket cache and foreign-loop aclose.
- decorators 3: Python decorator module identity.
- default_tool_error_redaction 5: asyncio `CancelledError` variants and a hostile `__str__`.
- error_logging_redaction 86: traceback locals, `__cause__`/`__context__`/notes, exception groups, BaseException/hybrid cancellation, hostile objects, LogRecord extras, raising hooks/formatters/MCP callbacks, `_deserialize_items` warnings.
- exception_exports 2: `agents.__all__` re-exports; extension_filters 4: unserializable objects; function_schema 46: pydantic/inspect/typing introspection.

## Source bugs fixed

- `src/tracing/provider.rs` `isolate`: a failing trace processor's log always included the processor `Debug` text. It now follows the mixed model/tool data policy (fixed message only when either flag is set; identity and panic text otherwise) and uses Python's messages for flush and shutdown.
- `macros/src/lib.rs` `parse_docs`: the description now stops at the first section, so a Google `Returns:` body no longer leaks into the tool description; NumPy (`Parameters` over dashes) and Sphinx (`:param x:`) doc comments are parsed.

## Breaking changes

None to public API. Behavior change: tool descriptions generated from doc comments no longer include text after the first section header (`Returns:`, `Note:`, `# Examples`).

## Unresolved questions

- Approvals are keyed by agent name, so two same-named agents share decisions (Python distinguishes by identity). Design decision needed.
- `RunState` restore keeps items whose agent name is unknown instead of skipping with a redacted warning (12 cases).
- `#[function_tool]` cannot be applied to associated functions (struct generated inside `impl`).
- Ledger accounting: `test_agents_logging`, `test_config`, `test_function_schema` were not moved because the earlier reports do not say which cases were python-only. The added tests `py_core8_r8g6_logging_concurrent.rs` and `py_core8_r8g6_agent_basics.rs::a_mapping_parameter_cannot_be_made_strict` harden already-counted cases.
- Gates: the final `cargo test` over my binaries and both clippy runs were still in flight when this was written, see the status below.

Status: DONE_WITH_CONCERNS
Summary: 41 cases ported, 2 covered, 658 remain python-only; two source fixes (trace-processor log redaction, doc-comment parsing).
Concerns: final test/clippy gate results not yet confirmed; open design questions above.
