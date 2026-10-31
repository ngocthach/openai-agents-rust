# Python test parity, core round 6, area `streamedleft`

Slice: every top-level `tests/*.py` row of `py-test-parity-ledger.csv` with `not_ported > 0` (except
`test_run_impl_resume_paths.py` and `test_native_tool_failure_history.py`, owned by another agent)
plus the rows whose total does not equal the sum of their buckets (only `test_tool_guardrails.py`).

## How counted

Case counts are the ledger's `total` (pytest-collected cases, parametrize ids folded), re-checked
per file by listing every Python `def` and mapping it to a Rust test by reading both. Where the
ledger was stale (round-2 deferrals that later rounds ported, cases the earlier port had put in
one Rust test) the bucket was corrected; where the ledger total was wrong the real total is used:

- `test_tool_guardrails.py` is 22 cases (19 defs, three parametrized x2), not the 25 the buckets
  summed to; that is the "unaccounted" row.
- `test_stream_events.py`: the ledger's `covered 4` is replaced by a name-by-name mapping.
- `test_items_helpers.py`: 5 python-only (not 8), the rest ported.

## Per-file table

| file | cases | ported | covered | python-only | not ported |
| --- | --: | --: | --: | --: | --: |
| test_agent_runner_streamed.py | 208 | 166 | 15 | 27 | 0 |
| test_agent_scoped_approvals.py | 104 | 56 | 0 | 48 | 0 |
| test_error_logging_redaction.py | 170 | 66 | 0 | 104 | 0 |
| test_hitl_error_scenarios.py | 88 | 69 | 3 | 16 | 0 |
| test_items_helpers.py | 40 | 35 | 0 | 5 | 0 |
| test_local_shell_tool.py | 8 | 8 | 0 | 0 | 0 |
| test_pretty_print.py | 16 | 14 | 0 | 2 | 0 |
| test_process_model_response.py | 29 | 26 | 1 | 2 | 0 |
| test_result_cast.py | 16 | 3 | 2 | 11 | 0 |
| test_run_internal_items.py | 54 | 37 | 4 | 13 | 0 |
| test_run_state.py | 469 | 294 | 10 | 165 | 0 |
| test_run_step_execution.py | 111 | 51 | 7 | 53 | 0 |
| test_stream_events.py | 14 | 10 | 1 | 3 | 0 |
| test_strict_schema.py | 69 | 58 | 3 | 8 | 0 |
| test_tool_approval_call_id_reuse.py | 130 | 118 | 0 | 12 | 0 |
| test_tool_batch_failure_history.py | 64 | 18 | 0 | 39 | 7 |
| test_tool_guardrails.py | 22 | 20 | 0 | 2 | 0 |
| test_tool_output_conversion.py | 27 | 26 | 0 | 1 | 0 |
| test_tool_use_behavior.py | 17 | 11 | 0 | 6 | 0 |
| test_usage.py | 23 | 16 | 2 | 5 | 0 |
| **total** | **1679** | **1102** | **48** | **522** | **7** |

Before this round the same rows held 164 not-ported cases. 43 new Rust tests in
`tests/py_core6_streamedleft_*.rs` (10 files, all green).

## What each former not-ported group became

- **runner_streamed (34)**: 26 serialized-ownership refusals were already ported by
  `py_core2_runstate_boundary.rs` (2 modes x 5 corruptions, 2 x 8 ownership shapes; the ledger
  was stale). The 8 chat-completions adapter-policy cases are new
  (`py_core6_streamedleft_chat_policy.rs`: mode x strict x agent/run-config model, local mock
  server, asserts the warning, the strict refusal, no model call and the dropped
  `previous_response_id`). Ported.
- **scoped_approvals (4), hitl_error (10), call_id_reuse (2)**: the first 14 live in
  `py_core2_approvals_binding.rs` (stale ledger). The two legacy changed-pending cases are new
  (`py_core6_streamedleft_approvals.rs`, schema 1.13 and 1.14) and exposed a real bug (below).
- **error_logging_redaction (17)**: agent-tool validation redaction (2) and Responses model
  request failure logging (3, `get_response` redacted/enabled and `stream_response` redacted) are
  new (`py_core6_streamedleft_logging.rs`, 4 tests). The 12 `_deserialize_items` warning cases are
  python-only: Rust `RunItem.agent` is a name, never resolved to an object, so no item is skipped
  and nothing is logged.
- **items_helpers (8), pretty_print (9), tool_output_conversion (4), result_cast (2),
  usage (2), tool_use_behavior (7), tool_guardrails (5), stream_events (5), strict_schema (18),
  local_shell (2), run_state (1), batch_failure_history (2 of 9)**: new tests, see the Rust files
  below; leftovers are classified under python-only or covered.
- **run_step_execution (14)**, hosted MCP approval matrix: covered 7 by
  `py_core2_approvals_reuse_b.rs` (`sticky_mcp_approval_rejects_same_id_on_another_server`,
  `..._for_changed_tool_name`, `unbindable_*` x3, `runner_omits_completed_mcp_approval_request_replay`
  x2 cases) and python-only 7 (hand-built `processed_response` / pending items that disagree,
  private `collect_manual_mcp_approvals`, `approve_tool` raising on an incomplete identity, the
  public-vs-execution agent identity).
- **run_internal_items (10)**: programmatic structured tool errors (2) covered by
  `py_core4_programmatic_guardrails_approvals.rs` (`{"error": ...}` JSON for program callers);
  digest fixed size and assistant-status equivalence (2) covered by
  `py_core2_runstate_sidecars.rs`; python-only 6 (private fingerprint helper, legacy tool-search
  type, object-identity occurrence resolution x4, clean-history normalization helper).
- **process_model_response (1)**: local MCP tool title on the run item, covered by
  `py_core_a_tool_origin.rs` (streamed and non-streamed).

## New Rust tests

| file | tests | covers |
| --- | --: | --- |
| `py_core6_streamedleft_chat_policy.rs` | 1 | runner_streamed chat-completions policy (8 cases) |
| `py_core6_streamedleft_approvals.rs` | 1 | call_id_reuse legacy changed pending (2) |
| `py_core6_streamedleft_items.rs` | 12 | items_helpers (9), pretty_print (5), tool_output_conversion (19) |
| `py_core6_streamedleft_stream_events.rs` | 5 | stream_events (5) |
| `py_core6_streamedleft_strict_schema.rs` | 13 | strict_schema (15 cases) |
| `py_core6_streamedleft_tool_behavior.rs` | 2 | tool_use_behavior (5 cases) |
| `py_core6_streamedleft_guardrails_usage.rs` | 3 | tool_guardrails (3), usage (1) |
| `py_core6_streamedleft_logging.rs` | 4 | error_logging_redaction (5) |
| `py_core6_streamedleft_batch.rs` | 1 | batch_failure_history (2) |
| `py_core6_streamedleft_legacy_mcp.rs` | 1 | run_state (1) |

## Python-only groups (this slice, new classifications)

- gc / weakref / `release_agents` / repr after release (result_cast 8, items_helpers 3).
- pydantic-only validation (`None` refusal, `None` text, `model_validate` copies) and `__bool__` /
  `__str__` overrides on user types (items_helpers 3, tool_output_conversion 1, pretty_print 2).
- Private helpers with no Rust seam: `check_for_final_output_from_tools` with an empty list,
  `stream_step_items_to_queue` x3, `snapshot_usage` / `usage_delta`, fingerprint / occurrence
  helpers (run_internal_items), `_MAX_SCHEMA_NODES` monkeypatching (a faithful port needs a
  100k-node chain, about 30 s), mapping subclasses.
- Typed-API cases: wrong `tool_use_behavior` types (5), invalid guardrail callables (2).
- `_deserialize_items` unknown-agent warnings (12), described above.

## Rust bugs fixed (source)

1. `src/run_state/wire.rs` `reconstruct_legacy_pending_bindings`: a legacy snapshot whose pending
   call reuses an id that already completed with other arguments kept the pending binding, so the
   changed call ran or re-interrupted. The completed record now wins and the resume refuses with
   "unique call ID".
2. `src/run_state/wire.rs` `from_wire`: an id-only `local_shell_call_output` is dropped like
   Python's strict reader instead of failing the whole snapshot (current schema then fails the
   completed-invocation check, 1.13 resumes without replay). Un-ignored
   `py_core3_shelltools_local_shell::run_state_rejects_id_only_local_shell_output`.
3. `src/run_state.rs` `from_json`: legacy (<1.15) per-call decisions with no ledger binding are
   asked again (`_mark_restored_unbound_approval_call_ids`); before, a schema 1.13 snapshot
   auto-approved a hosted MCP request from an orphaned decision.
4. `src/items.rs`: untyped JSON tool outputs of the `{"type": "text"|"image"|"file"}` shape (or a
   non-empty list of them) are converted to `input_*` parts like
   `ItemHelpers._convert_tool_output_as_structured`; an empty list and invalid shapes stay
   stringified. New `RunItem::tool_name()` and `RunItem::call_id()` (`ToolCallItem.tool_name`,
   `ToolCallItem.call_id`, `ToolCallOutputItem.call_id`).
5. `src/result.rs`: `RunResultStreaming::pretty_print` prints `Is complete: True/False` as Python
   does (an earlier test had pinned `true`; updated in `py_core_b_pretty_print.rs`).
6. `src/agent.rs`: the `Agent::as_tool` argument validation error echoes the offending input while
   tool data may be logged (Python's validation error does); still redacted otherwise.

## Cross-area deferred / not ported (7 cases)

`test_tool_batch_failure_history.py`: `streamed_partial_history_waits_for_input_verdict` x3 and
`nonstreamed_partial_history_waits_for_input_verdict` x3 (pass, reject, error verdicts) and
`nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail`. Python runs a turn's
tools while a parallel input guardrail is still pending and releases partial history only after
the verdict; the tests make guardrail and tools wait on each other. Rust (`src/run_internal/mod.rs`
`run_turn`) awaits the guardrail before running tools, so those tests would deadlock by design
(never a speculative tool run). The observable rejection contract is ported
(`py_core6_streamedleft_batch.rs`). Minimal fix: keep the guardrail future owned, race it with
`execute_side_effects`, on tool failure await the verdict only when outputs were committed.
Not attempted: it rewrites the turn's concurrency in a file other agents are editing.

## Breaking changes

- `ToolOutput::Json` values shaped like a structured part are now sent as `input_*` parts instead
  of a JSON string (Python parity). A tool that returned such a value as data will see the model
  receive a part list.
- `RunResultStreaming::pretty_print` spelling `True` / `False`.
- The `Agent::as_tool` validation error text carries `(input: ...)` when tool data logging is on.

## Gates

- New binaries: 10 files, 43 tests, all green; `py_core_b_pretty_print` (7) green.
- Regression over `py_core_a_*`, `py_core_b_*`, `py_core2_*`, `py_deferred_writes_*`,
  `py_core3_functiontools_*`, `py_core3_shelltools_local_shell`, `runner_tests`,
  `local_tools_tests`, `core_python_run_state` and the lib tests (651): green after fixing one
  regression my wire change caused in `py_core2_runstate_sidecars` (an id-only shell output is
  dropped only for a snapshot that has a ledger or predates it, otherwise still refused).
- `cargo clippy --all-features --lib --test <mine> -- -D warnings`: clean for my files. Three lints
  in other agents' in-flight code fail the lib without them (`src/realtime/debug.rs:29`
  dead code, `src/run_internal/mod.rs:2291` collapsible `if`, `src/strict_schema.rs:340`
  `nonminimal_bool`); I linted my targets with those three allowed.
- rustfmt applied to my new test files only.
