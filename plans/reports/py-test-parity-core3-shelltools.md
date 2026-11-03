# Python test parity: core3 / shelltools

Round: TEST-PARITY, core round 2 (shelltools area). Output dir `c3-shelltools`. No git commands run, nothing committed.

Counting: `pytest --collect-only -q` on the Python venv for each of the nine files (real parametrized IDs): 142 / 36 / 36 / 24 / 23 / 13 / 11 / 8 / 7 = 300.

Buckets: PORTED = a Rust test runs the same scenario and asserts the same expectation (parametrized cases counted individually). COVERED = an existing Rust test asserts it (none claimed). PYTHON-ONLY = cannot exist in Rust. NOT PORTED = in scope but still open, with the reason.

## Per-file table

| file | #tests | ported | covered | python-only | not ported | failures found |
| --- | --- | --- | --- | --- | --- | --- |
| `test_programmatic_tool_calling.py` | 142 | 10 | 0 | 10 | 122 | 1 (src/run_internal/tools.rs) |
| `test_tool_name_collision_policy.py` | 36 | 4 | 0 | 0 | 32 (1 ignored: divergence) | 2 divergences (run_internal) |
| `test_computer_tool_lifecycle.py` | 13 | 10 | 0 | 3 | 0 | 2 divergences (fixed, see below) |
| `test_scripted_sandbox.py` | 36 | 2 | 0 | 0 | 34 | out of scope (sandbox-facing) |
| `test_strict_schema_oneof.py` | 7 | 7 | 0 | 0 | 0 | 0 |
| `test_shell_call_serialization.py` | 23 | 20 | 0 | 3 | 0 | 1 (src/tool/local.rs) |
| `test_shell_tool.py` | 24 | 24 | 0 | 0 | 0 | 3 (validation, span tool_name, hosted local) |
| `test_apply_patch_tool.py` | 11 | 11 | 0 | 0 | 0 | 0 |
| `test_local_shell_tool.py` | 8 | 6 | 0 | 0 | 2 | 1 cross-area (2 ignored cases) |
| **total** | **300** | **94** | **0** | **16** | **190** | |

Per-file detail:

- `test_programmatic_tool_calling.py` (142). Ported 10: converter wire shape; typed output type; raw output schema rejection (2 cases); single-element `object` type array; declared output validation; schema-backed programmatic failure bypasses the default formatter; direct argument error uses the default text (through the runner); runner preserves a direct error for a schema-backed tool; and one extra (`schema_backed_programmatic_output_is_returned_as_json`, not counted). Python-only 10: annotation-based output inference (`Awaitable`/`Coroutine`/awaitable aliases 6 cases, PEP 695 alias 1, `Annotated` plain-return and `Field` metadata 2, loose `dict[str, Any]` annotation 1), and `does_not_infer_non_programmatic_output` (inference, 1). Not ported 122: the test file lines 600-2836 (timeouts, guardrails, approvals, `process_model_response` program ownership and caller rules, runner and streaming replay, SQLite/session round trips, handoff summaries) were not read or ported in this session. Also not ported: `output_type` plus `output_json_schema` exclusivity (1), and the two late-timeout cases (2) plus the direct default-timeout case (1) that need a timeout race that the Python test reproduces by monkeypatching `asyncio.wait_for`.
- `test_tool_name_collision_policy.py` (36). Ported 4 in `py_core3_shelltools_collision_policy.rs`: resume with a colliding name in error mode; a queued handoff with no recipient (refused before side effects, but with Rust's message, see divergence); a queued handoff rebound to the current warn winner; a cross-kind duplicate call id. Not ported 32: MCP-server cases (2), nested agent-as-tool approval cases (6), reclassified handoff/function cases (3), apply_patch-prefixed queued functions (4), malformed approval payloads (4), and the cases that edit Python private RunState fields (`_authoritative_interruption`, `tool_lookup_key`, `_last_processed_response`, `_model_responses`) which were not yet translated to serialized-JSON edits. One ignored divergence: model-order resume with a missing tool under `return_error_to_model`. Most cases read Python RunState internals (`_model_responses`, `tool_lookup_key`, `_authoritative_interruption`, `_last_processed_response`). The plan is the same as for the run-state tests: build the state through the public API, serialize it, edit the JSON, and call `RunState::from_json`.
- `test_computer_tool_lifecycle.py` (13). Ported 10: fake computer interface; runner disposes after run; runner creates and disposes for a computer call (extra); streamed run disposes; concurrent hook identity; resolve per run context; concurrent runs keep provider instances isolated (gated model with a barrier); dispose failure is logged with redaction (2 cases); start-hook failure never creates the computer (2 cases, streamed and not). Python-only 3: the duck-typed `create` attribute case (1), and the two start-hook cases that replace `agent.tools` inside a hook, which Rust agents cannot do during a run (2).
- `test_scripted_sandbox.py` (36). Ported 2: the apply_patch create path through `SandboxEditor`, and the black-box `SandboxAgent` workflow (scripted via `PtyExecStart`, since Rust's `exec_command` runs through PTY start). The other 34 test the scripted session itself and are left to the sandbox area as the task specified.
- `test_strict_schema_oneof.py` (7). Ported 7. The pydantic discriminator metadata assertion inside `test_discriminated_union_with_pydantic` is not checked: schemars emits no OpenAPI `discriminator` block (the strict `anyOf` shape is checked).
- `test_shell_call_serialization.py` (23). Ported 20 (parametrized cases expanded). Python-only 3: `bytes` and `bytearray` commands (no JSON form), and the `"success"` string-outcome branch (executor output is typed in Rust).
- `test_shell_tool.py` (24). Ported 24 (22 functions in `py_core3_shelltools_shell_tool.rs`, 2 span functions in `py_core3_shelltools_tool_spans.rs`). The `ShellTool()` no-argument case is a compile-time error in Rust, so only the local-without-executor case is asserted.
- `test_apply_patch_tool.py` (11). Ported 11 (9 in `py_core3_shelltools_apply_patch.rs`, 2 span functions).
- `test_local_shell_tool.py` (8). Ported 6. Not ported 2: `run_state_rejects_id_only_local_shell_output` (both cases) is `#[ignore]`d, see the cross-area divergence below.

## Rust bugs fixed

1. `src/tool/local.rs` `normalize_shell_output_entries`: an explicit `"exit_code": null` in a dict outcome was overwritten with `0`. Python's `setdefault` keeps it. Covered by `py_core3_shelltools_shell_serialization.rs::shell_rejection_payload_preserves_missing_exit_code`.
2. `src/tool/local.rs` `ShellTool::validate` (new), called from `src/agent.rs` `get_all_tools` (`Tool::Shell` arm). Python's construction-time errors ("ShellTool with local environment requires an executor.", "... hosted environment does not accept an executor.", "... does not support needs_approval or on_approval.") were missing. Rust builders are infallible, so the check runs when the agent resolves its tools, before any model call. `ShellTool::hosted` for a local environment now also says "requires an executor". Covered by `shell_tool_rejects_local_executor_and_approval_for_hosted_environment` and `shell_tool_rejects_local_mode_without_executor`.
3. `src/run_internal/local_tools.rs`: failed shell, apply_patch and custom tool spans now carry `data.tool_name` next to the redacted error, as Python's `SpanError` does (new helper `record_local_tool_error`). Covered by both redaction span tests.
4. `src/tool.rs` `with_output_json_schema`: a JSON Schema type array holding only `"object"` is accepted and normalized to `"object"`, as Python does. Covered by `function_tool_accepts_single_element_object_type_array_output_schema`.
5. `src/run_internal/tools.rs`: a failure or timeout that the tool's error function turned into model text is no longer validated against the declared output type (a schema-backed direct tool was failing the run with a "does not match its declared output type" error on the default text). Covered by `runner_preserves_direct_error_for_schema_backed_tool`.
6. Computer lifecycle (decided: per run context, not per tool). `src/run_context.rs` gets a per-run computer store (shared by clones like `uncommitted`, fresh for nested agent-tool contexts). `src/tool/local.rs`: `ComputerTool::resolve(&ctx)` and `to_responses_param(&ctx)` now take the run context; the lazy slot on the tool is gone. Each run gets its own computer. `ComputerTool::dispose()` is removed; the run loop (`src/run_internal/mod.rs` `dispose_computers`) takes and disposes the run's computers. Breaking change: the `resolve`/`to_responses_param` signatures and the removal of `dispose()`. Covered by `resolve_computer_per_run_context` and `concurrent_runs_keep_computer_provider_instances_isolated`.
7. Computer dispose failure (decided: make it reportable, not silently dropped). `ComputerTool::from_provider`'s dispose hook may now return `()` (existing callers unchanged) or a `Result<(), E: Display>`. A failure is logged as "Failed to dispose computer for run context", with the error text only when tool data logging is allowed (`debug::dont_log_tool_data`), as Python does. Covered by `dispose_computer_failure_respects_tool_data_policy`.

## Cross-area divergences (not fixed here)

- `src/run_state/wire.rs` (around line 1085): an id-only `local_shell_call_output` is refused in every schema with "Run state has an invalid local_shell_call_output item". Python drops the item: the current schema then fails the completed-invocation check ("...does not match a restored tool call and output"), and the 1.13 schema resumes without replaying the pair. Test: `py_core3_shelltools_local_shell.rs::run_state_rejects_id_only_local_shell_output` (ignored). Fix sketch: skip such items in the two `run_item_from_wire` callers (about lines 1417 and 1975) instead of erroring.
- `src/tool/local.rs` `ComputerTool::from_provider` documents `dispose` as returning `()` or `Result`; the scripted sandbox session (`src/sandbox/testing.rs`) advertises PTY support, so `exec_command` takes the PTY path in the scripted black-box test. Recorded here for the sandbox area; the test scripts the PTY step.
- `src/tool.rs` `FunctionTool::invoke` returns argument errors raw; Python's `on_invoke_tool` applies the failure formatter. The runner path applies it, so the test checks the runner path.

## Gates

- My test binaries (11 files) all pass in the last run; two flakes are documented below. Counts: schema_oneof 7, shell_serialization 10, shell_tool 22, apply_patch 9, tool_spans 4, local_shell 4 (+1 ignored), computer 10, scripted_sandbox 2, programmatic_schema 6, programmatic_errors 4, collision_policy 4 (+1 ignored).
- `cargo test --all-features --lib`: 651 passed, 0 failed.
- `cargo clippy --all-features --lib -- -D warnings` and clippy `-D warnings` on my test targets: clean.
- `rustfmt --edition 2024` applied to my test files only (not src/lib.rs).
- Output dir `$HOME/.cache/oars-out/c3-shelltools` deleted at the end (it had reached 2.0 GB).

## Not finished

- `test_programmatic_tool_calling.py`: 122 cases (source lines 600-2836) were not read or ported in this session. Remaining groups: timeout race cases (3, need a deterministic way to reproduce Python's `asyncio.wait_for` race), `process_model_response` program ownership and caller rules, runner and streaming replay of program items, approvals for programmatic calls, previous_response_id continuation, RunState and SQLite round trips, nested handoff summaries, non-function program outputs. `output_type` plus `output_json_schema` exclusivity (1) is also open.
- `test_tool_name_collision_policy.py`: 32 cases not ported (listed in the file row above).
- `test_scripted_sandbox.py`: 34 sandbox-facing cases left to the sandbox area, as the task specified.

## Ignored tests

- `run_state_rejects_id_only_local_shell_output` (cross-area: `src/run_state/wire.rs` refuses id-only `local_shell_call_output`; Python drops it).
- `resume_preserves_model_order_for_function_outcomes` (cross-area: `src/run_internal/mod.rs`, the `ReturnErrorToModel` branch near line 3994 only emits the not-found output when the call is not in `skip`; a resumed approved call is in `skip`, so its output is lost. Python emits it.)

Neither needs network or credentials. Un-ignore each once its cross-area fix lands.

## Flaky tests (documented, not retried)

- `py_core2_runconfig_cancel.rs::cancel_with_pending_parallel_input_guardrail_finishes_cleanup` (another agent's file, not changed). The test waits 2 seconds of wall-clock time for a guardrail to start (`tokio::time::timeout(Duration::from_secs(2), guardrail_started.notified())`). Under heavy parallel build load the spawned run can be scheduled later than that. The failure site is that wait (`guardrail started: Elapsed`), not the assertions. Reason: wall-clock bound, not a logic race. Suggested fix for the owner: raise the bound (for example to 30 seconds), which does not change what is asserted.
- `py_core3_shelltools_local_shell.rs::local_shell_output_survives_run_state_resume` failed once in a combined run of several test binaries, panicking because the replayed request had no `local_shell_call`. 120 runs of this binary with six parallel loops (20 each) and 55 earlier isolated runs all passed, so the cause is not reproduced. The assertions now print the replayed input on failure, so the next occurrence will show what was sent. Root cause unknown; not fixed.

## Changelog

`docs/project-changelog.md` (Unreleased 0.6.0) records the shelltools round, including the breaking per-run computer change (`ComputerTool::resolve`/`to_responses_param` take the run context, `ComputerTool::dispose()` removed), the dispose-hook `Result` option, the `ShellTool` validation timing, and the output-schema type-array change. The entry says the round is partial: programmatic and collision parity are not complete.

## Shared-file edits

None to `src/lib.rs` or `Cargo.toml`. No new dependencies.
