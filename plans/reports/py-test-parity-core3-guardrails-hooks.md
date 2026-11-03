# Python test parity: core3 guardrails and hooks

Scope: the 10 Python files in `openai-agents-python/tests/` listed in the task. Rust tests are in
`tests/py_core3_guardrails_*.rs` (9 files, 124 test functions). `tests/py_core_b_guardrails.rs`
(24 functions) is reused as the coverage source for the Python cases it already ports.

## Counting

- Python cases = `grep -c 'def test_'` per file, plus parametrize expansion (`run_in_parallel`,
  `streamed`, `guardrail_delay`, `hook_kind`, `change`, `call_removed`, ...). Total 143 cases, which
  matches core-b's count (119 + 24).
- A Python case is COVERED when a named test in `tests/py_core_b_guardrails.rs` asserts the same
  behaviour. Each Rust test counts once even when it covers several cases (the two tracing cases
  share one function in `py_core3_guardrails_tracing.rs`).

## Per-file table

| file | cases | ported | covered | python-only | failures found |
| --- | ---: | ---: | ---: | ---: | --- |
| `test_guardrails.py` | 60 | 40 | 17 | 3 | 0 observed; 4 source divergences (see fixes 2-4) |
| `test_agent_as_tool_output_guardrails.py` | 20 | 20 | 0 | 0 | 0 observed; 1 source divergence (fix 5) |
| `test_start_hook_tools.py` | 20 | 20 | 0 | 0 | 0 observed |
| `test_agent_tool_input.py` | 10 | 9 | 0 | 1 | 0 observed; 2 source divergences (fixes 6, 7) |
| `test_agent_hooks.py` | 9 | 8 | 0 | 1 | 4 observed (`on_handoff` fired on the target agent; fix 1) |
| `test_stream_input_guardrail_timing.py` | 9 | 9 | 0 | 0 | 0 observed |
| `test_agent_prompt.py` | 6 | 6 | 0 | 0 | 0 observed; 1 source divergence (fix 4) |
| `test_agent_clone_shallow_copy.py` | 5 | 3 | 0 | 2 | 0 |
| `test_agent_llm_hooks.py` | 3 | 3 | 0 | 0 | 0 |
| `test_output_guardrail_cancellation.py` | 1 | 1 | 0 | 0 | 0 observed; covered by fix 2 |
| **total** | **143** | **119** | **17** | **7** | |

Checks: 119 + 17 + 7 = 143.

Covered cases (17), by core-b test name: `input_guardrail_run_in_parallel_default_and_override`
(2 cases: default and `run_in_parallel=False`), `parallel_guardrail_runs_concurrently_with_agent`
(+ `_streaming`), `blocking_guardrail_prevents_agent_execution` (+ `_streaming`),
`blocking_guardrail_prevents_tool_execution` (+ `_streaming`), `parallel_guardrail_passes_agent_continues`,
`blocking_guardrail_passes_agent_continues`, `multiple_blocking_guardrails_one_triggers`,
`guardrail_via_agent_and_run_config_are_equivalent`, `input_guardrail_tripwire_reports_results_sync`,
`input_guardrail_results_reported_on_success`, `output_guardrail_tripwire_reports_results`
(+ `_streamed`), `output_guardrail_results_reported_on_success`.

Python-only cases (7):

- `test_guardrails.py` (3): passing a non-callable guardrail raises `UserError` (2 cases, the Rust
  constructor takes a typed closure); `test_parallel_guardrail_trip_compat_mode_does_not_cancel_model_task`
  checks a Temporal replay-compatibility switch that Rust does not have.
- `test_agent_tool_input.py` (1): `AgentAsToolInput.model_validate` is pydantic validation.
- `test_agent_hooks.py` (1): `test_falsy_agent_hooks_are_invoked` checks `__bool__` on the hooks
  object; Rust has no object truthiness.
- `test_agent_clone_shallow_copy.py` (2): the two cases that assert two agents hold one Python list
  object and that mutating one changes the other. Rust agents own their `Vec`s.

## Rust bugs fixed

All fixes are in `src/run_internal/mod.rs` and `src/agent.rs`. No other src file, `src/lib.rs`
or `Cargo.toml` was edited by this work. Both files are shared with other agents' in-flight edits;
the changes are scoped to the functions named below.

1. **`on_handoff` hook target** (`run_internal/mod.rs`, handoff execution). The agent-level
   `on_handoff` ran on the target agent's hooks. Python calls the source agent's hooks with
   `agent=new_agent, source=agent` (`run_internal/turn_resolution.py`). Covered by
   `test_agent_hooks.py` (4 cases).
2. **Guardrail batches** (`race_input_guardrails`, `race_output_guardrails`, both new free
   functions). `join_all` ran every guardrail to completion in declaration order, dropped the
   completed results on error, and never cancelled siblings. The batch now collects results in
   completion order, stops at the first tripwire or error, drops the guardrails still running, and
   returns the completed results with the error. Every guardrail is polled once before any result
   is consumed, and simultaneous completions are ordered by declaration, as Python's task
   creation order does. Covers the blocking/parallel cancel tests, the raise-cancels-siblings
   tests (input and output), completion-order reporting, and the output-cancellation test.
3. **Parallel input guardrail vs model call** (`run_turn`, parallel branch). The guardrail is
   raced with the model call. A trip or error cancels the in-flight model call. A non-streamed
   model error cancels the guardrail. A streamed model error reports the guardrail outcome instead.
   A guardrail still running when the model answers keeps running through the turn, and is settled
   at the end of the turn (non-streamed) or at the next terminal point (streamed, across
   `RunAgain` turns). This lets a guardrail that waits on a tool run, and that trips after the tool
   ran, complete as in Python (`may_not_prevent_tool_execution`). New field
   `pending_input_guard` (a `Mutex` slot, so the run future stays `Sync`), and
   `settle_pending_input_guard`. The guardrail is polled once before the turn continues, as
   Python's task creation starts it right away; without this a guardrail that was never polled
   did not start (`py_core2_runconfig_cancel::cancel_with_pending_parallel_input_guardrail_finishes_cleanup`).
4. **Settling before max-turns** (`apply_error_handler`). The pending guardrail is settled after
   the max-turns handler runs and before `MaxTurnsExceeded` is raised, so a tripwire wins
   (`max_turns_does_not_clobber_input_guardrail_tripwire`).
5. **Agent-tool default output** (`agent.rs`, `default_agent_tool_output`). Python's `as_tool`
   returns the latest non-empty message or string tool output when the final output is empty and
   no output guardrail ran (`agents/agent.py`). Rust returned the empty final text. Covered by the
   unguarded fallback cases in `test_agent_as_tool_output_guardrails.py`.
6. **Agent-tool input unwrapping** (`agent.rs`, as_tool). Only a single-key `{"input": "<text>"}`
   object is unwrapped. `{"input": ..., "target": ...}` was reduced to `input` and lost `target`.
   Python's `_has_only_input_field` gates this.
7. **Structured input summary** (`agent.rs`, `build_schema_summary`, `describe_schema_field`).
   (a) The summary is always derived, so the builder gets it next to the schema when
   `include_input_schema` is set (Python passes both). (b) Enum and const labels follow Python:
   `enum("a" | "b" | ...)` with a `" | ..."` suffix past five values, and `literal(...)`. (c)
   `null` is not a simple type, and a non-simple string type yields no summary.
8. **Prompt and instructions** (`run_turn` / `run_loop` prefetch). Instructions and the prompt
   resolve concurrently with `try_join!`, so a failing prompt cancels the instructions
   (`test_agent_prompt.py`, 2 cases). New field `prefetched_system_prompt`.

## Deviations kept (with reason)

- **In-hook `agent.tools` mutation** (`test_start_hook_tools.py`, the 8 "replace" cases, and the
  approval-tool case). Rust hooks receive `&Agent`, so a hook cannot change the tools of a running
  agent. The "replace" cases install the post-hook tool set through `RunOptions::prepare_agent`,
  which runs at the same point (after run start, before the first model call), and the replacement
  agent keeps the hooks. The approval case declares the tool on the original agent and keeps the
  reconstruction assertions. The "disable" cases use the run context, as Python does.
- **Cancellation cleanup is not awaited.** Python's cancelled coroutines run async cleanup before
  the error surfaces. A dropped Rust future runs no code, so the tests assert the observable part:
  the sibling is dropped and its results are not recorded (`test_guardrails.py` streaming
  exception and slow-sibling cases).
- **Streamed trip cancels the model call.** Python's streamed path lets the stream finish and only
  blocks tool execution. Rust cancels the in-flight call in both modes. The observable tool
  outcome is the same. The test asserts that the model started, not that its stream finished.
- **Guardrail vs tool execution inside one turn.** Rust polls a parallel guardrail at the
  model-call race and at turn end, not concurrently with tool execution. A trip raised during tool
  execution takes effect when the tools finish. Python would cancel the tool task.
- **Null agent-tool output** is exported as the text `null`, not `None` (`test_agent_as_tool_output_guardrails.py`,
  the `none` cases). Tool outputs are text in Rust.
- **Streamed result accessors.** `RunResultStreaming` has no `input_guardrail_results` or
  `output_guardrail_results` accessor. Tests read them from `into_result()` after success and from
  `run_data()` on error. Adding the accessors is an API change, so it is left open.
- **Builder return type.** `as_tool` input builders return `String`. Python allows a list of
  input items, so the "builder returns items" case asserts the string form.
- **Clone and aliasing** (2 Python-only cases above). The clone tests assert value equality.

## Tests and gates

- New tests: 124 functions. Ported cases are listed in the file headers.
- No `#[ignore]` in the new files. No test needs a network or credentials. The prompt test uses a
  local TCP mock server (`responses_mock`, same pattern as `tests/openai_http_tests.rs`).
- Timing: the new tests use 5-second timeouts. They passed six consecutive runs.
- `cargo clippy --all-features --lib --test <each new file> -- -D warnings`: clean. A whole-workspace
  clippy run fails on two other agents' files (`tests/py_core3_functiontools_mcp_custom_data.rs`,
  `tests/py_core3_tracing_errors_streamed.rs`), which this work does not touch.
- `rustfmt --edition 2024` on the new test files. The shared src files were not reformatted.
- Regressions: `py_core_b_guardrails`, `py_core_a_streamed`, `py_core_a_run_loop`,
  `py_core_a_hooks`, `py_core_a_agent_as_tool`, `py_core_a_tool_guardrails`,
  `py_core2_runner_guardrail_sessions`, `py_core2_runconfig_cancel`,
  `py_core3_tracing_errors_streamed`: all green on the final source.
- Full `cargo test --workspace --all-features --no-fail-fast` (one run, mid-round, before the last
  two edits): 5098 passed, 53 failed. Two of the failures were in this work and are fixed above
  (`py_core2_runconfig_cancel`, `py_core3_tracing_errors_streamed`, both green on rerun). The rest
  are doc-tests in `src/extensions/**` and one macros unit test (`parses_doc_sections`, a doc-section
  whitespace difference), which this work does not touch. I did not re-run the whole workspace
  after the final edits.
- Build directory `$HOME/.cache/oars-out/c3-guardrails` was cut to the SDK-free state (`cargo clean -p`)
  on the coordinator's request, and is deleted at the end of the round.

## Open items for the reviewer

- The tracing agent edited `race_input_guardrails` in parallel to add a `parent` span argument.
  The current signature includes it. The tests passed against that version.
- `RunResultStreaming` guardrail accessors (see deviations).
