# Py test parity core4: programmatic tool calling

Source: `/Users/ngocthach/myGit/openai-agents-python/tests/test_programmatic_tool_calling.py` (142 cases, counted with `pytest --collect-only -q`, 142 collected). Scope of this round: the 122 cases not ported by the shelltools round (lines 348 and 540-2836). The 20 earlier cases (lines before 540, other than 348) are the shelltools round's 10 ported and 10 python-only.

## Totals for the whole file

| Bucket | Count |
|---|---|
| Ported (own Rust test, incl. subcases) | 110 (10 shelltools + 100 core4) |
| Covered (as_mapping twin, same Rust test) | 22 (core4) |
| Python-only | 10 (shelltools round: annotation-based output inference) |
| Not ported | 0 |
| Total | 142 |

## Per-file table

| File | #tests (cases) | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| `test_programmatic_tool_calling.py` (lines 348, 540-2836) | 122 | 100 | 22 | 0 | 13 Rust bugs (below); harness fixes in the tests (schema-backed fixtures, output filtering) |
| `test_programmatic_tool_calling.py` (lines 1-539, earlier round) | 20 | 10 | 0 | 10 | not this round |

Counting: `pytest --collect-only` gives 142 IDs. Parametrize expansion is included (each ID is one case). Twin rule: Python's `as_mapping` parameter passes model objects or dicts; Rust carries every item as a JSON `Value`, so the dict twin is the same Rust test with the same input and is counted COVERED. The 22 covered cases are: shell output (1), program-owned call with incomplete retained program (1), parent with program call (2), server-owned parent after incomplete output (1), submitted delta (2), disallowed shell output (2), duplicate completed output (2), orphan output (1), child before parent (2), program items without programmatic tool (2), call for completed retained program (1), call completes program (2), direct-only hosted calls (3).

## Python-only sub-assertions (inside ported tests)

- `test_function_output_preserves_caller_and_uses_declared_json_schema`: the legacy `"{'sku': 'A-1'}"` assertion. Python's `str(dict)` fallback has no Rust counterpart; Rust serializes structured outputs as JSON. The JSON and caller assertions are ported.
- `test_schema_backed_function_output_rejects_plain_error_text`: the second assertion calls `ItemHelpers.tool_call_output_item` with a non-strict `{"type": "object"}` schema. Rust's item helper takes no schema, and `with_output_json_schema` requires a strict schema, so no Rust surface exists. The first assertion (plain error text for a schema-backed tool) is ported through `FunctionTool::invoke` and the runner.
- `test_runner_validates_only_user_output_after_late_tool_timeout`: the monkeypatched `asyncio.wait_for` race is asyncio internals. The observable path (the tool's own error, then the default text or a schema check on the custom text) is ported directly.

## Rust bugs fixed (src)

| # | Divergence | Source file |
|---|---|---|
| 1 | Unsupported caller type message printed `Some("unknown")` / Rust Debug quotes; Python reprs `'unknown'` or `None` | `src/run_internal/tool_caller.rs` |
| 2 | Hosted `file_search_call`, `web_search_call`, `image_generation_call` (direct only) and `code_interpreter_call` (its config) were not checked against `allowed_callers` | `src/run_internal/mod.rs` |
| 3 | `shell_call_output` from a program was not checked against the shell tool's `allowed_callers` | `src/run_internal/mod.rs` |
| 4 | Invalid `allowed_callers` (empty, unsupported value, duplicate) was accepted; Python rejects at construction | `src/run_internal/tool_caller.rs` (`validate_allowed_callers`), `src/run_internal/mod.rs` (checked per request before the model call) |
| 5 | Server-managed submitted input was not parent evidence for program children (`process_model_response`) | `src/run_internal/mod.rs` |
| 6 | Resumed approval of a server-owned program call was refused (`Persisted caller ... does not match`) | `src/run_internal/mod.rs` |
| 7 | Rejections (input guardrail, approval, output guardrail) and default failure text for schema-backed tools called by a program were not wrapped as `{"error": ...}` | `src/run_internal/tools.rs` (`error_outcome`) |
| 8 | Timeouts of schema-backed tools called by a program were propagated instead of answered with the default timeout text | `src/run_internal/tools.rs` |
| 9 | A custom failure handler's output was not checked against the declared output type on the runner path; `FunctionTool::invoke` ignored the custom handler | `src/run_internal/tools.rs`, `src/tool.rs` (`validate_failure_text`) |
| 10 | `output_type` and `output_json_schema` could both be set | `src/tool.rs` (`with_output_type`, `with_output_json_schema`) |
| 11 | `output_type` schema used deserialize names; a `#[serde(rename(serialize = ...))]` field must use the serialize name | `src/tool.rs` (`with_output_type`, serialize-mode schema) |
| 12 | Chat Completions accepted `tool_choice="programmatic_tool_calling"` | `src/models/openai_chatcompletions.rs` |
| 13 | Nested handoff summary rendered transcript items as compact JSON; Python uses `json.dumps` spacing | `src/run_internal/history.rs` (`python_style_json`) |

## Behaviour notes and remaining differences

- `FunctionTool::with_allowed_callers` is still infallible, so invalid caller lists fail when the tool is put in a request, before the model call, not at construction as in Python. Changing the builder to return `Result` would be a public API break; the tests assert the `UserError` through `Runner::run` with a model that must not be called.
- `ItemHelpers::tool_call_output_item` has no schema argument. Python's programmatic-error wrapping lives in the runner in Rust (item 7).
- Failure message wording: Python's failure-path message says "declared output schema"; Rust says "declared output type". The tests assert the shared substring.
- Breaking changes (0.6.0 tree): `with_output_json_schema` after `with_output_type` (and the reverse) returns an error; a `with_output_type` schema for a `rename(serialize = ...)` field now uses the serialized name; Chat Completions rejects `tool_choice="programmatic_tool_calling"`; invalid `allowed_callers` lists now fail at request time.

## Tests added

- `tests/py_core4_programmatic_process_items.rs` (29 tests): program ordering, parent/child rules, caller permissions, hosted and shell caller checks, server-owned parents.
- `tests/py_core4_programmatic_schema_runtime.rs` (13 tests): output types, failure and timeout output, aliases, converter and chat configuration, allowed-callers validation, late timeout, caller on output.
- `tests/py_core4_programmatic_guardrails_approvals.rs` (8 tests): input, output and pre-approval guardrails, default timeout, approval rejection (8 combinations), rebuilt approvals and parent checks.
- `tests/py_core4_programmatic_runner_state.rs` (10 tests): replay (streamed and not), multiple pauses, concurrent batch, `previous_response_id` continuation, state round trip, SQLite session, nested handoff, non-function outputs.

Shared files: none edited (`src/lib.rs`, `Cargo.toml` untouched). No commit made.

## Verification

- My four binaries: `py_core4_programmatic_process_items` 29, `py_core4_programmatic_schema_runtime` 13, `py_core4_programmatic_guardrails_approvals` 8, `py_core4_programmatic_runner_state` 10. All pass. The shelltools programmatic binaries (6 + 4) pass.
- `cargo test --all-features --lib`: 651 passed, 0 failed.
- `cargo clippy --all-features --lib --test <my four> -- -D warnings`: clean. rustfmt applied to my four files only.
- Full `cargo test --all-features --tests --no-fail-fast`: all binaries run except the in-flight `py_core4_tracerunner_agent` (another agent's file, which hung; its process was stopped in my output dir and it was not re-run). Failures outside my area, observed in that run:
  - `core_run_state::reads_python_tool_search_items_and_hosted_mcp_interruptions` (`RunState::from_json` gives an empty call_id).
  - `py_core2_approvals_reuse_b` (6 sticky/reuse MCP approval tests).
  - `py_core2_runner_routing::persistent_hosted_mcp_approval_does_not_cross_servers`.
  - `py_core2_runstate_resume::nested_agent_tool_hitl_resume_with_duplicate_agent_names`.
  - `py_deferred_writes_e::a_checkpointed_mcp_approval_survives_callback_failure_and_json_retry`.
  I isolated three of my edits by temporarily disabling them: the nested-summary JSON format (history.rs), the hosted-caller block (mod.rs), and the allowed-callers request validation (mod.rs). The failures were unchanged, so those edits are not the cause. The remaining suspects are concurrent in-flight edits by other agents in `src/run_internal/` and `src/run_state/`; the owners should triage. Not re-verified after the owners' edits.
- `clippy -D warnings` on the whole `--all-targets` set fails only in another agent's `tests/py_core3_functiontools_schema.rs` (dead fields); not touched.
- Shared files: none edited (`src/lib.rs`, `Cargo.toml` untouched). No commit made. No `git` commands run.
