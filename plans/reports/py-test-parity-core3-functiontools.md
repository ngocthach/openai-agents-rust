# Python test parity: area core3-functiontools

Scope: `openai-agents-python/tests/` files `test_function_tool.py`, `test_tool_identity.py`,
`test_tool_context.py`, `test_tool_custom_data.py`, `test_tool_use_tracker.py`,
`test_custom_tool.py`, plus the open rows of `test_function_schema.py` and
`test_function_tool_decorator.py` from `py-test-parity-core-a.md`.

## Counting

AST scan with parametrize expansion (`scratchpad/count_tests_corea.py`, same script as core-a).
Totals: 8 files, **227 defs -> 292 cases**. Per-file counts match core-a and core-b.
Each case is in exactly one bucket. "Ported" cases are asserted by the new Rust files;
"covered" cases are asserted by an existing Rust test; "not ported" cases are open.

## Rust test files added

| file | tests | ports |
| --- | --: | --- |
| `tests/py_core3_functiontools_schema.rs` | 31 | `test_function_tool.py` construction, schema, namespace, enablement, guardrail tests |
| `tests/py_core3_functiontools_invoke.rs` | 18 | `test_function_tool.py` failure policies, timeouts, invalid input, logging, clone |
| `tests/py_core3_functiontools_identity_context.rs` | 15 | `test_tool_identity.py`, `test_tool_context.py`, `test_custom_tool.py` |
| `tests/py_core3_functiontools_custom_data_tracker.rs` | 7 | `test_tool_custom_data.py`, `test_tool_use_tracker.py` |
| `tests/py_core3_functiontools_decorator_schema.rs` | 21 | open rows of `test_function_schema.py`, `test_function_tool_decorator.py` |

All five binaries are green: 92 tests.

## Per-Python-file table

| file | cases (defs) | ported | covered | python-only | not ported | failures found (observed / by reading) |
| --- | --: | --: | --: | --: | --: | --- |
| `test_function_tool.py` | 91 (71) | 67 | 0 | 24 | 0 | 3 / 3 |
| `test_tool_identity.py` | 29 (27) | 14 | 6 | 8 | 1 | 0 / 0 |
| `test_tool_context.py` | 18 (18) | 7 | 0 | 11 | 0 | 0 / 2 |
| `test_tool_custom_data.py` | 9 (7) | 3 | 2 | 4 | 0 | 0 / 0 |
| `test_tool_use_tracker.py` | 9 (9) | 6 | 1 | 0 | 2 | 1 / 0 |
| `test_custom_tool.py` | 2 (2) | 2 | 0 | 0 | 0 | 0 / 0 |
| `test_function_schema.py` (open rows) | 70 (52) | 15 | 1 | 46 | 8 | 0 / 1 |
| `test_function_tool_decorator.py` (open rows) | 64 (41) | 12 | 0 | 50 | 2 | 1 / 0 |
| **total** | **292 (227)** | **126** | **10** | **143** | **13** | **5 / 7** |

Notes on the counts:
- `test_function_schema.py` and `test_function_tool_decorator.py` replace core-a's rows
  (`covered 8 / python-only 40 / not ported 22` and `covered 10 / python-only 30 / not ported 24`).
  Core-a had no per-test triage for those rows; the table above is per test.
- `test_function_tool.py` "ported" includes the sync/async pairs that collapse in Rust
  (`test_sync_custom_error_function_works` + `test_async_custom_error_function_works`, one case each),
  and `test_function_tool_timeout_seconds_must_be_positive_number`, of which only the zero check
  is ported (str, inf, nan are type-level in Rust).
- `covered` is a file-level claim, verified by reading the test names and asserts:
  - duplicate-name tracker: `tests/py_core_a_run_loop.rs::a_snapshot_names_the_current_duplicate_agent_by_identity` (asserts `duplicate#2`) and `a_duplicate_agent_snapshot_resumes_on_the_agent_it_named`;
  - computer custom data: `tests/core_audit_fixes_tools.rs::computer_tool_custom_data_reaches_the_output_item`;
  - apply_patch custom data: `tests/core_audit_fixes_tools.rs::apply_patch_custom_data_carries_the_operations_and_status` (values checked by name only; the replay exclusion is asserted by this port's custom-data tests on the shared replay path);
  - tool-approval lookup keys (identity cases 6 of 29): `tests/py_core2_runstate_items.rs` (`tool_approval_item_tracks_namespace`, `tool_approval_item_collapses_synthetic_deferred_namespace_in_qualified_name`) and `tests/py_core2_runstate_sidecars.rs` (`round_trip_serialization_preserves_allow_bare_name_alias_and_tool_lookup_key`). These were not re-run in this round, so treat them as the nearest existing coverage;
  - `test_function_with_context` schema: `tests/function_tool_macro_tests.rs` (ctx excluded from the schema).

## Not ported (13 cases) and why

- `test_tool_identity.py::test_deserialize_invalid_returns_none`: malformed `tool_lookup_key`
  handling in the wire decoder (`src/run_state/wire.rs`) is not asserted yet.
- `test_tool_use_tracker.py::test_record_processed_response_counts_output_only_tools_without_shifting_names`:
  needs a scripted shell tool call; not built this round.
- `test_tool_use_tracker.py::test_serialize_and_hydrate_tool_use_tracker_preserves_duplicate_agent_identity` is covered;
  `test_tool_use_tracker_handles_literal_suffix_names_without_collision` (`sandbox#2`) is not ported.
- `test_function_schema.py`: `test_no_args_function`, `test_no_args_function_with_context`, `test_simple_function`,
  `test_nested_data_function`, `test_complex_args_and_docs_function` (schema-shape asserts for nested
  and no-argument tools), `test_function_with_field_multiple_constraints`,
  `test_google_docstring_blank_line_form_is_unchanged` (exact docstring output), and
  `test_schema_with_mapping_raises_strict_mode_error` (typed `FunctionTool::new` falls back to non-strict instead of raising; see deferred items).
- `test_function_tool_decorator.py`: `test_sync_no_context_override_invocation` (override name plus invocation)
  and `test_extract_descriptions_from_docstring` (exact description output).

These are open rows. They are behavioural and portable in principle, and they remain in this round's
open list, not classified as Python-only.

## Python-only groups (one line each)

- `asyncio` and monkeypatch: `to_thread` swap, event-loop blocking, `CancelledError` normalization,
  `monkeypatch` of module defaults, `get_all_tools` sibling cancellation on a raising `is_enabled`
  (Rust `is_enabled` returns `bool`, cannot raise).
- Callable objects and wrapped-callable identity (`__wrapped__`, `__call__`, slotted classes, class docstrings,
  class-scoped and inherited annotations, awaitable return from sync functions, `copy`/`deepcopy` of subclass instances, `__post_init__`).
- `*args` / `**kwargs` signatures and their pydantic constraints; `Annotated[...]` / `Field(...)` typing and precedence;
  `Mapping` / variadic schema rules; `inspect.signature`, sentinel equality, `to_call_args` internals.
- Python typing: dict-vs-object extractors, dict `run_config`, `ToolContext` hashability, required-field and
  missing-default errors (compile-time in Rust), `from_agent_context` parent overloads, annotation-based
  dispatch between `RunContextWrapper` and `ToolContext`.
- Optional arguments in Python decorators (`function_tool(strict_mode=...)`, `timeout_seconds` keyword-only,
  decorator-only guardrail kwargs); Rust uses builders for these.
- Async failure handlers: `failure_error_function` may return an awaitable in Python; the Rust closure is sync by type.

## Rust bugs fixed

Observed failures (a test failed on the first run before the fix):

1. `src/strict_schema.rs` `resolve_ref`: `$ref` fragments were not percent-decoded, so `#/$defs/A%20B` resolved to a literal `A%20B`. Now decoded before the JSON pointer split, as Python's `unquote`. Invalid UTF-8 after decoding is an error. (`test_function_tool.py`: 3 failing tests.)
2. `src/tool.rs` `FunctionTool::with_strict_json_schema(false)` only flipped the flag, leaving a strictified schema; the macro's `#[function_tool(strict = false)]` therefore kept `additionalProperties`. The raw schema is now stored (`raw_params_json_schema`, `pub(crate)`) and restored when strictness is turned off. `with_params_json_schema` updates the raw schema. (`test_function_tool_decorator.py::test_non_strict_mode_function`.)
3. `src/run_internal/mod.rs` (one line): a function call's tool-use name was the bare name; Python records the trace name (`get_tool_call_trace_name`, `turn_resolution.py`). Now `crate::tool_context::call_trace_name(fc)`. (`test_tool_use_tracker.py::test_record_used_tools_uses_trace_names_for_namespaced_and_deferred_functions`.)

Found by reading the Python code and fixed before the first run (each is asserted by a test that passes now):

4. `src/tool.rs` `FunctionTool::new` typed invoke: a non-object JSON argument (`[]`, `"x"`, `123`, `null`, `true`) went through serde's positional/scalar decoding. Now `Invalid JSON input for tool X: expected a JSON object`.
5. `src/tool.rs` `FunctionTool::try_with_timeout` (new, additive): a zero `Duration` is rejected as Python does (`timeout_seconds must be greater than 0`).
6. `src/tool.rs` `FunctionTool::invoke_body`: the tool-data debug log `Tool call args: ...` is now emitted when tool data is not redacted, as Python logs it.
7. `src/tool_context.rs` `ToolContext::from_call`: the namespace is inferred from the call's `namespace` (Python infers it from `tool_call`); `qualified_tool_name` no longer prefixes a namespace equal to the tool name (the synthetic deferred namespace). Adds `pub(crate) fn call_trace_name`.
8. `macros/src/lib.rs` `parse_docs`: wrapped argument descriptions are joined with `\n` instead of a space (Python keeps the line break).

New additive API (no existing signature changed):
- `FunctionTool::try_new_raw(name, description, schema, strict, f) -> Result<Self>`: a schema that cannot be made strict is a `UserError` when `strict` is set; the non-strict path keeps the schema untouched. The infallible `new_raw` keeps its silent non-strict fallback (MCP-style callers).
- `FunctionTool::try_with_timeout(Duration) -> Result<Self>`.

Behaviour changes to note:
- `with_strict_json_schema(true)` now re-strictifies from the raw schema; before it only set the flag.
- `$ref` fragments are percent-decoded; a `%` escape that was previously taken literally now decodes.
- `ToolContext::qualified_tool_name` no longer shows a namespace equal to the tool name.

## Deferred / cross-area (not changed here)

- Resolved by coordinator decision: MCP `result_meta` is added (`FunctionToolCustomDataContext::result_meta`, `ToolContext::result_meta()`). Breaking for struct-literal users; recorded in the changelog. Ported in `tests/py_core3_functiontools_mcp_custom_data.rs`.
- Resolved by coordinator decision (documented divergence): an explicit `#[arg(desc = "...")]` wins over the docstring; Python gives the docstring precedence over `Annotated`. Recorded in the changelog.
- Resolved by coordinator decision: a typed `FunctionTool::new` with a non-strict-convertible argument raises (panics, with Python's message) instead of falling back silently. `FunctionTool::try_new` returns the error. `try_new_with_strictness(..., false, ...)` is the analog of Python's `strict_mode=False`; three tests in `py_core2_scopedapprovals_fn_args.rs` with map arguments use it. `test_schema_with_mapping_raises_strict_mode_error` is covered by `typed_tool_with_open_map_argument_is_rejected_by_try_new` and `typed_tool_with_open_map_argument_panics_in_new`.
- `ToolUseTracker` snapshots: duplicate-name identity (`duplicate#2`) is already fixed and covered (see above); not re-touched here.
- `get_all_tools` evaluates enablement checks sequentially; Python gathers them concurrently. Only a latency difference, since Rust `is_enabled` cannot fail.
- Async failure handlers (awaitable return) are not supported by the sync Rust closure type.
- `tests/py_core3_shelltools_computer.rs` (another agent, in progress) does not compile (`with_dynamic_instructions` closure signature, lines 314 and 333). It blocked the full `cargo test --all-features` run in this round.

## Shared-file edits

- No edits to `src/lib.rs` or `Cargo.toml`.
- Edited shared source (all within tool and run-loop surface, each a targeted edit): `src/tool.rs`, `src/tool_context.rs`, `src/strict_schema.rs`, `src/run_internal/mod.rs` (one line), `src/mcp/util.rs`, `src/run_internal/tools.rs` (one field), `macros/src/lib.rs`, `docs/project-changelog.md`.
- `src/strict_schema.rs`: a boolean `true` sub-schema (schemars for `serde_json::Value`) is accepted unchanged, matching Python's `Any`. Found by the full-suite run: typed tools with a `Value` field panicked after the typed constructor became strict.

## Verification

- `cargo test --all-features` for the five new binaries: 92 passed, 0 failed.
- `cargo clippy --all-features --test <five new binaries> -- -D warnings`: clean (covers the lib).
- `cargo test --all-features --lib` after the changes: 649 passed, 1 failed. The failure is
  `tracing::error_tracing::tests::records_a_redacted_error_on_a_span` (`span.error()` is `None`);
  it fails in isolation too. It sits in the tracing area, which has uncommitted edits from another agent, and this round did not touch it.
- Integration sweep of about 30 binaries that exercise the changed code (tool, MCP, run-state,
  approvals, schema, macro, runner): started, not finished when this report was written.
  Shared-machine builds were slow and lock-contended. Result lines go to
  `$HOME/.cache/oars-out/c3-focus.log`.
- `cargo test --all-features` (whole workspace) cannot build right now: `tests/py_core3_shelltools_computer.rs`
  (another agent, in progress) has two closure-signature errors at lines 314 and 333.
- rustfmt applied to the five new test files only.
- Build dir: `$HOME/.cache/oars-out/c3-functiontools`. Still in use by the running sweep; delete after it finishes.
