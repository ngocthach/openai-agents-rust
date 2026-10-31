# Python -> Rust test parity: area core-b

Scope: the 83 Python test files in `scratchpad/top_b.txt`, under
`/Users/ngocthach/myGit/openai-agents-python/tests/`.

## How tests were counted

`pytest` is not installed in the scratchpad venv (`pyenv/bin/python -m pytest` -> *No module named
pytest*), and there is no network to install it. Counts come from an AST pass over each file
(`scratchpad/coreb_count.py`, output in `scratchpad/core-b-counts.json`): every top-level or class
`def test*` / `async def test*`, multiplied by the length of each `pytest.mark.parametrize`
argvalues list. That matches pytest collection for every file in scope (none uses dynamic
parametrize generation). **Total in scope: 2370 tests across 83 files.**

## Status

This round was run under a hard constraint: the orchestrator's subagent pool was globally
exhausted (`Concurrent subagent limit reached` on every spawn), so the planned 11-way fan-out was
not possible and the whole area was ported by a single agent. The work below is what was actually
completed and verified green; the remainder is listed explicitly as NOT YET PORTED so the next
round can pick it up without re-triage.

### Rust test files added

| Rust file | Python source | Rust tests | Result |
| --- | --- | --- | --- |
| `tests/py_core_b_max_turns.rs` | `test_max_turns.py` | 9 | green |
| `tests/py_core_b_items_helpers.rs` | `test_items_helpers.py` | 24 | green |
| `tests/py_core_b_pretty_print.rs` | `test_pretty_print.py` | 7 | green |
| `tests/py_core_b_tool_output_result.rs` | `test_tool_output_conversion.py`, `test_result_cast.py` | 12 | green |
| `tests/py_core_b_guardrails.rs` | `test_guardrails.py` (behavioral subset) | 24 | green |
| `tests/py_core_b_handoff_tool.rs` | `test_handoff_tool.py` | 17 | green |
| `src/apply_diff.rs` (in-file `mod tests`) | `test_apply_diff_helpers.py` | 10 | green |

93 integration tests verified green with `--all-features` (9 + 24 + 24 + 7 + 12 + 17 across six `py_core_b_*` binaries), plus 10 in-file apply_diff tests in `src/apply_diff.rs`.

## Rust bugs found and fixed

1. **`src/items.rs` — `created_by` was only stripped at the top level.**
   `shell_call_output.output` is a list of chunks that each carry their own server-assigned
   `created_by`; Python strips both levels (`src/agents/items.py:242-251`) because the Responses
   *input* schema has no such field. Rust's `strip_created_by` only removed the top-level key, so a
   replayed shell output carried `created_by` inside every chunk.
   Fix: `strip_created_by` now also walks `output` chunks when the item type is
   `shell_call_output`. Covered by
   `tests/py_core_b_items_helpers.rs::to_input_items_strips_nested_created_by_from_shell_call_output`.

2. **`src/items.rs` — hosted shell outputs replayed SDK book-keeping fields.**
   Python's `ToolCallOutputItem.to_input_item` (`src/agents/items.py:462-487`) drops `status`,
   `shell_output` and `provider_data` from a `shell_call_output` before replay, because the
   Responses API rejects them. Rust replayed the raw item unchanged.
   Fix: new `sanitize_shell_call_output` applied in `RunItem::to_input_item`. Covered by
   `tests/py_core_b_items_helpers.rs::shell_call_output_item_to_input_item_sanitizes_without_mutation`.

Both fixes are additive, keep the public API unchanged, and leave the caller's original value
untouched (verified by the same tests).

## Divergences accepted (Rust-idiomatic, documented in the test files)

- **`pretty_print` type label.** Python prints the pydantic class name
  (`- Final output (Foo):`); Rust has no runtime type name for a deserialized `serde_json::Value`
  and prints the JSON kind (`- Final output (dict):` / `(list)` / `(str)` / `(NoneType)`).
  Everything else in the rendering — layout, counts, 4-space indentation, CRLF normalization and
  C0/C1 terminal-control escaping — matches byte for byte.
- **`final_output_as`.** Python's cast is unchecked unless `raise_if_incorrect_type=True`; Rust's
  is always checked and returns `Err`. The "bad cast raises" tests port; the "bad cast is silently
  allowed" tests do not.
- **`ToolOutput` is a typed enum in Rust.** Python's `tool_call_output_item` has to guess whether a
  returned `dict` is a structured content part and otherwise stringifies it with Python `repr`
  (`"{'type': 'text'}"`). That whole branch is unrepresentable in Rust.

## Cross-area bug deferred (not fixed)

- **Duplicate agent names in the tool-use tracker snapshot.**
  Python disambiguates same-named agents reachable from the starting agent with a `#2` / `#3`
  suffix when serializing the tracker (`serialize_tool_use_tracker` /
  `hydrate_tool_use_tracker`, exercised by `test_tool_use_tracker.py`). Rust keys the snapshot by
  bare agent name (`src/run_internal/mod.rs:1471-1480` serialize, `:978-983` hydrate), so two
  agents called `duplicate` share one entry and a resumed run can see tools it never used.
  Minimal fix: walk the handoff graph from the starting agent while serializing, and append
  `#<n>` for the n-th distinct agent object with a given name; mirror the same walk on hydrate.
  Not applied here because it touches the run loop that core-a is editing concurrently.

## Per-file classification

### PYTHON-ONLY in full (254 tests)

Repo CI tooling — these load scripts from `.github/scripts/` or the pytest conftest and assert on
the Python repo's own release/lint automation. No Rust counterpart exists or should:

| file | tests |
| --- | --- |
| `test_check_optional_truthiness.py` | 71 |
| `test_release_automation.py` | 56 |
| `test_code_change_verification_runner.py` | 16 |
| `test_release_provenance.py` | 14 |
| `test_update_rclone_pin.py` | 13 |
| `test_run_serial_tests.py` | 6 |
| `test_test_environment.py` | 6 |
| `test_asyncio_stability_command.py` | 4 |

Example scripts — they import modules under `examples/`, which the Rust crate does not ship as a
testable library:

| file | tests |
| --- | --- |
| `test_run_examples_script.py` | 33 |
| `test_decisions_examples.py` | 15 |
| `test_trace_redaction_example.py` | 6 |
| `test_hosted_mcp_connectors_example.py` | 4 |
| `test_usaspending_setup_db.py` | 4 |
| `test_daytona_usaspending_example.py` | 2 |

Python import/reflection semantics:

| file | tests | reason |
| --- | --- | --- |
| `test_exception_exports.py` | 3 | asserts `agents.__all__` re-exports every exception class; Rust re-exports are checked by the compiler |
| `test_function_schema_logger_restore.py` | 1 | restores the `griffe` logger level; Rust has no griffe and no global logging-level stack |

### Ported / classified this round

| Python file | tests | ported | covered | python-only | failures found |
| --- | --- | --- | --- | --- | --- |
| `test_max_turns.py` | 52 | 9 | 0 | 0 | 0 |
| `test_guardrails.py` | 60 | 24 | 0 | 0 | 0 (remaining 36 are mostly asyncio task-cancellation tests; triage pending) |
| `test_handoff_tool.py` | 31 | 17 | 0 | 0 | 0 (3 handoff tests first failed on a wrong assertion, the handoff list is `request.handoffs`, not `request.tools`; fixed in the test) |
| `test_items_helpers.py` | 40 | 24 | 0 | 8 | 2 (both fixed, see above) |
| `test_pretty_print.py` | 16 | 7 | 0 | 0 | 0 |
| `test_tool_output_conversion.py` | 27 | 9 | 0 | 14 | 0 |
| `test_result_cast.py` | 16 | 3 | 0 | 11 | 0 |
| `test_apply_diff_helpers.py` | 10 | 10 | 0 | 0 | 0 |

PYTHON-ONLY reasons for the partially-ported files, one line per group:

- `test_items_helpers.py` (8): `weakref`/`gc` agent-release checks (Rust holds agents in `Arc`,
  there is no `release_agent`), pydantic `model_construct`/`ValidationError` checks, and
  `TypeAdapter` round-trips of validated input items.
- `test_tool_output_conversion.py` (14): untyped-`dict` tool outputs and Python `repr`
  stringification — unrepresentable against Rust's typed `ToolOutput` enum.
- `test_result_cast.py` (11): `weakref`/`gc` `release_agents` lifecycle, `dataclasses.asdict`,
  `FrozenInstanceError`, and pydantic `model_rebuild`.
- `test_max_turns.py` (43 not yet reached): see below.

### NOT YET PORTED (1980 tests)

These were triaged as behavioral (i.e. they should be ported, none is PYTHON-ONLY as a group) but
the single-agent round did not reach them. Listed in the priority order the next round should take,
with the Rust test file name each should land in:

| Rust file to create | Python files | tests |
| --- | --- | --- |
| `py_core_b_run_step.rs` | `test_run_impl_resume_paths.py` (198), `test_run_step_execution.py` (111) | 309 |
| `py_core_b_tool_errors.rs` | `test_error_logging_redaction.py` (170), `test_tool_batch_failure_history.py` (64), `test_tool_failure_boundaries.py` (30), `test_default_tool_error_redaction.py` (17), `test_model_error_logging_redaction.py` (1) | 282 |
| `py_core_b_approvals.rs` | `test_function_tool_approval_arguments.py` (108), `test_agent_scoped_approvals.py` (104) | 212 |
| `py_core_b_run_config.rs` | `test_run_config.py` (123), `test_integration_runner.py` (58), `test_cancel_streaming.py` (18), `test_runner_model_provider_lifecycle.py` (11), `test_agent_runner_sync.py` (6), `test_repl.py` (4), `test_run.py` (2) | 222 |
| `py_core_b_programmatic_tools.rs` | `test_programmatic_tool_calling.py` (142) | 142 |
| `py_core_b_shell_tools.rs` | `test_scripted_sandbox.py` (36), `test_tool_name_collision_policy.py` (36), `test_shell_tool.py` (24), `test_shell_call_serialization.py` (23), `test_computer_tool_lifecycle.py` (13), `test_apply_patch_tool.py` (11), `test_local_shell_tool.py` (8), `test_strict_schema_oneof.py` (7) | 158 |
| `py_core_b_function_tool.rs` | `test_function_tool.py` (91), `test_tool_identity.py` (29), `test_tool_context.py` (18), `test_tool_custom_data.py` (9), `test_tool_use_tracker.py` (9), `test_custom_tool.py` (2) | 158 |
| `py_core_b_handoffs.rs` | `test_handoff_history_duplication.py` (94), `test_handoff_tool.py` (31) | 125 |
| `py_core_b_tracing.rs` | `test_agent_tracing.py` (49), `test_tracing.py` (19), `test_tracing_errors.py` (18), `test_provider_span_errors.py` (17), `test_responses_tracing.py` (12), `test_extra_headers.py` (2), `test_disabled_trace_provider_shutdown.py` (1), `test_image_generation_tool_reference.py` (1), `test_oaiconv_resume_response_id.py` (1) | 120 |
| `py_core_b_guardrails_hooks.rs` | rest of `test_guardrails.py` (36), `test_agent_as_tool_output_guardrails.py` (20), `test_start_hook_tools.py` (20), `test_agent_tool_input.py` (10), `test_agent_hooks.py` (9), `test_stream_input_guardrail_timing.py` (9), `test_agent_prompt.py` (6), `test_agent_clone_shallow_copy.py` (5), `test_agent_llm_hooks.py` (3), `test_output_guardrail_cancellation.py` (1) | 143 |
| `py_core_b_run_state_ctx.rs` | `test_source_compat_constructors.py` (20), `test_run_state_hook_identity.py` (15), `test_run_context_wrapper.py` (14), `test_run_state_aliases.py` (10), `test_run_state_pydantic_context.py` (8), `test_runtime_symmetry_contract.py` (6), `test_run_internal_error_handlers.py` (5), `test_call_model_input_filter_unit.py` (3), `test_hitl_session_scenario.py` (2), `test_streaming_logging.py` (1) | 84 |
| (finish) `py_core_b_max_turns.rs` | rest of `test_max_turns.py` | 43 |
| (finish) the four files above | rest of `test_items_helpers.py` etc. | 17 |

Note on `test_scripted_sandbox.py`: it belongs to the sandbox area rather than core, but it is in
this agent's file list, so it is kept here per the task instruction.

## Totals

| bucket | tests |
| --- | --- |
| PORTED | 103 |
| COVERED by an existing Rust test | 0 |
| PYTHON-ONLY | 287 |
| NOT YET PORTED | 1980 |
| **total in scope** | **2370** |

## Shared-file edits

None. `src/lib.rs` and `Cargo.toml` were not touched. No new dependencies.

## Ignored tests

None. No test needed network, credentials, docker or a Python runtime.

## Gates

- `cargo test --all-features` for the six `py_core_b_*` binaries: 9 + 24 + 24 + 7 + 12 + 17 = 93 passed, 0 failed.
- `cargo test --all-features --lib apply_diff`: 15 passed (10 new, 5 existing), 0 failed.
- `cargo clippy --all-features --test <my six binaries>` (without `-D warnings`): no warnings in my files. Three warnings were fixed in place.
- `cargo clippy --all-features --all-targets -- -D warnings`: fails in other areas, not in my files: `src/extensions/models/mod.rs`, `src/sandbox/**`, `src/models/openai_responses_ws.rs`, `src/lib.rs`. These are the files another agent was editing during this session. I did not fix them.
- `rustfmt --edition 2024` applied to my six test files. `src/items.rs` and `src/apply_diff.rs` were not reformatted (shared files).
- Build dir `/Users/ngocthach/.cache/oars-build/tp-coreB` deleted at the end of the session.

Status: DONE_WITH_CONCERNS
Summary: 103 Rust tests ported from 8 Python files, 2 real `src/items.rs` replay bugs found and
fixed; the remaining 1980 tests in scope are triaged, grouped and listed but not ported, because
the subagent pool this round was planned around was globally unavailable.
Concerns/Blockers: single-agent throughput against 2370 Python tests; one cross-area bug
(duplicate agent names in the tool-use tracker snapshot) left for the run-loop owner.
