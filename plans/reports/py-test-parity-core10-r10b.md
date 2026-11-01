# Python test parity, round 10, r10b (fallible async failure formatters, tracked model-settings edits)

Scope: make `failure_error_function` match Python (awaitable, may re-raise) for function tools, handoffs, agent tools, MCP tools and realtime tool calls, port the cases that were blocked by it, and decide the "in-place writes to `agent.model_settings`" cases.

## API change

`tool::ToolErrorFn<C>` is now `Arc<dyn for<'a> Fn(&'a RunContext<C>, &'a AgentsError) -> BoxFuture<'a, Result<String>> + Send + Sync>`. An `Err` ends the run (or the realtime call) with that error instead of the original one, which is what a Python formatter that re-raises does. Function, agent and MCP tools surface it as `Error running tool <name>: <formatter error>` (the same wrapping the runner applies to any tool failure); a handoff formatter's error is raised as is.

- Constructors: `tool::tool_error_fn` (plain closure, the old shape), `tool::try_tool_error_fn` (plain closure returning `Result`), `tool::async_tool_error_fn` (async).
- Builders that kept their closure shape and so need no change in existing code: `FunctionTool::with_failure_error_function`, `with_timeout_error_function`, `Handoff::with_failure_error_function`, `MCPConfig::with_failure_error_function`, `AgentAsToolOptions::failure_error_function`.
- New builders taking a `ToolErrorFn`: `FunctionTool::with_failure_error_handler` / `with_timeout_error_handler`, `Handoff::with_failure_error_handler`, `MCPConfig::with_failure_error_handler`, `AgentAsToolOptions::failure_error_handler`. `MCPToolPolicy::with_failure_error_function` already took a `ToolErrorFn`.
- Wired through: `src/tool.rs` (`FunctionTool::invoke`), `src/run_internal/tools.rs` (runner, streamed and not: failure and timeout formatters), `src/run_internal/mod.rs` (handoff recovery, shared by streamed runs), `src/agent.rs` (agent-as-tool, `MCPConfig`), `src/mcp/util.rs`, `src/realtime/session.rs` (failure and timeout formatters).
- Also added: `Agent::model_settings_mut` (see below).
- Docs: `docs/project-changelog.md` (Unreleased).

## Breaking changes

- Code that built a `ToolErrorFn` with `Arc::new(closure)`, called one directly, or passed a plain closure to `MCPToolPolicy::with_failure_error_function` must wrap the closure with `tool::tool_error_fn` and `.await` the call. Three existing tests were adjusted (`tests/py_mcp_util.rs`, `tests/mcp_tests.rs`, `tests/py_core2_toolerrors_formatter.rs`); no other source in the repo needed a change.

## Per-file table

| file | python_only before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| test_handoff_failure_error_function.py | 4 | 2 | 0 | 2 | `failure="cancel"` (asyncio `CancelledError` raised inside `on_handoff`; Rust cancels by dropping the future), `failure="filter"` (the Rust handoff input filter returns a plain value, so it cannot fail; the filter type is a separate sync API, not the formatter) |
| test_function_tool.py | 24 | 1 | 0 | 23 | `test_async_failure_error_function_is_awaited` ported; the rest are the asyncio, monkeypatch, `copy`/`deepcopy`/`dataclasses.replace`, `**kwargs` groups recorded in round 9 |
| test_run_config.py | 16 | 6 | 0 | 10 | the 6 `field="verbosity"` cases of `test_equal_in_place_assignment_preserves_explicit_settings` ported; the rest are dictionary coercion and typed-literal groups from round 9 |
| test_native_tool_failure_history.py | 34 | 0 | 0 | 34 | none of the 34 depend on the formatter: every case is a tool raising `CancelledError`, a cancelled child task, a native invocation turning cancellation into an error or suppressing it, or a raising `on_tool_end` hook (Rust hooks return `()`). All of them set `failure_error_function=None`, so the formatter change does not unblock them |

The task text listed `tests/test_function_tool*.py` and `tests/test_native_tool_failure_history.py` as blocked by this; reading them, the only blocked case was `test_async_failure_error_function_is_awaited`. The other formatter tests there (`normalizes_cancelled_error_to_exception`, `survives_dataclasses_replace`, `survives_deepcopy`, `resolved_at_invoke_time`) test asyncio cancellation normalization or Python object copying and stay valid python-only. `test_function_tool_decorator.py` formatter uses only sync handlers.

## Ported tests

`tests/py_core10_r10b_failure_formatters.rs` (13 tests, direct and streamed): async handoff formatter; a failing handoff formatter (sync and async, called exactly once, the model is not called again) and a successful handoff never reaching the formatter; async `invoke` of a function tool; a failing formatter error from `invoke`; async and failing function-tool formatter in runs; async and failing timeout formatter; async and failing agent-tool formatter; async and failing MCP formatter.

`tests/py_core10_r10b_realtime.rs` (4 tests): `test_function_tool_timeout_uses_async_error_function_result` (async timeout formatter) plus async failure formatter, failing failure formatter and failing timeout formatter in both the awaited and the background dispatch paths (no output is sent, the session error is the formatter's).

`tests/py_core10_r10b_model_settings.rs` (2 tests): the tracked in-place edit, all 12 parameter combinations (reasoning and verbosity, clone timing none/before/after, direct and streamed), and the untouched / directly assigned contrast.

## In-place `agent.model_settings` writes

A Rust-idiomatic equivalent exists: an explicit mutation accessor. `Agent::model_settings_mut()` is the tracked way to edit in place. On the first call it folds the defaults captured when the agent was created (`implicit_model_settings`) under whatever is set and marks the settings explicit, so the edit, equal to the default or not, no longer follows the model the run selects. The Python test passes with the new test as its analogue (the captured `gpt-5.6-luna` default `reasoning=none` survives a run whose selected model is `gpt-5`).

Not equivalent, and documented on the method: assigning a field of the public `model_settings` field directly is untracked (Rust cannot observe a field write), so unset fields still follow the selected model, and `with_model_settings` keeps its explicit meaning. Python's shared-object semantics with `clone()` (an edit through the original reaching the clone) do not exist because a Rust clone is an independent copy; the test pins the clone-before case to editing the clone that runs.

## Gates

- `cargo test --all-features` on the three new binaries (13 + 4 + 2) and 98 test binaries covering function tools, handoffs, runner approvals/run steps/run config, tool errors, agent-as-tool, MCP, codex and realtime: 1365 passed, 0 failed. The full workspace run was not completed: it needs more than the 4 GB target-dir budget on the shared disk, so it was stopped and the touched-area subset run instead.
- clippy `-D warnings` on the lib and all my test binaries, all-features and default: clean. The whole-workspace `--all-targets` run currently fails on two other agents' in-flight files (`src/sandbox/runtime.rs:1713` `too_many_arguments`, `tests/py_core10_r10a_item_skips.rs:105` `MutexGuard` held across await, default features); with `too_many_arguments` allowed the lib is clean.
- rustfmt `--check` clean for `src/tool.rs` and every test file I added or edited. Target dir deleted.

## Unresolved questions

1. `HandoffInputFilter` (and `handoff_input_filter`) are still plain `Fn(HandoffInputData) -> HandoffInputData`, so `failure="filter"` of the handoff failure test cannot be expressed. Making it fallible is the same kind of change as this one; not done because it is outside the formatter scope given.
2. `tool_error_formatter` on `RunConfig` (the run-level formatter) is still a sync `Fn -> Option<String>`; Python allows a raising or async one (`test_tool_exception_formatter.py` `response=raise`). Not changed here.

Status: DONE_WITH_CONCERNS
Summary: failure formatters are async and fallible everywhere (function, handoff, agent-tool, MCP, timeout, realtime, streamed and not); 2 handoff cases + 1 function-tool case ported, 6 in-place model-settings cases ported via `Agent::model_settings_mut`; totals 9 moved out of 78 python-only in the four files (69 remain, CSV balances). 19 new tests.
Concerns: breaking `ToolErrorFn` type (noted in the changelog); full workspace gate not run (disk budget), subset of 98 binaries passed; direct field assignment on `agent.model_settings` stays untracked by design; native-tool-failure-history cases were not actually formatter-blocked and stay python-only.
