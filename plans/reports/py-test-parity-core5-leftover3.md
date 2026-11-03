# Python test parity: core5 leftover3

Scope (Python files relative to `openai-agents-python/tests/`): `test_extension_filters.py`,
`test_mcp_tool_metadata.py`, `test_visualization.py`, the 13 open rows of `test_function_schema.py`,
`test_function_tool_decorator.py`, `test_tool_identity.py` and `test_tool_use_tracker.py` listed in
`py-test-parity-core3-functiontools.md`, and a completeness sweep of every top-level `tests/*.py`.

Counting: `def test_` plus parametrize expansion (same method as core-a/core3). Each case sits in one bucket.

## Rust test files added (105 tests, all green)

| file | tests | ports |
| --- | --: | --- |
| `tests/py_core5_leftover3_visualization.rs` | 20 | `test_visualization.py` |
| `tests/py_core5_leftover3_extension_filters.rs` | 41 | `test_extension_filters.py` |
| `tests/py_core5_leftover3_mcp_tool_metadata.rs` | 27 | `test_mcp_tool_metadata.py` |
| `tests/py_core5_leftover3_function_schema.rs` | 10 | 9 schema/decorator rows plus one constraint-enforcement test |
| `tests/py_core5_leftover3_function_tools.rs` | 4 | `test_tool_identity.py` / `test_tool_use_tracker.py` rows |
| `tests/py_core5_leftover3_sweep.rs` | 3 | sweep finds: `test_exception_exports.py`, `test_agent_as_tool_callback_validation.py` |

## Per-file table

| file | cases | ported | covered | python-only | not ported | failures found |
| --- | --: | --: | --: | --: | --: | --- |
| `test_extension_filters.py` | 55 (44 defs) | 51 | 0 | 4 | 0 | 1 (compaction items removed) |
| `test_mcp_tool_metadata.py` | 27 | 27 | 0 | 0 | 0 | 2 (no JSON resolvers; falsy `raw_item`) |
| `test_visualization.py` | 26 (21 defs) | 25 | 0 | 1 | 0 | 1 (no previsited-name support) |
| 13 open function-tool rows | 13 | 12 | 1 | 0 | 0 | 2 (lookup key decoder; constraint enforcement) |
| `test_exception_exports.py` (sweep) | 3 | 1 | 0 | 2 | 0 | 1 (`RunErrorDetails` had no `Display`) |
| `test_agent_as_tool_callback_validation.py` (sweep) | 5 (2 defs) | 4 | 0 | 1 | 0 | 0 |
| **total** | **129** | **120** | **1** | **8** | **0** | **7** |

Python-only reasons:
- `test_extension_filters`: a bare `str` `input_history` is a list in Rust (the three string-input
  cases are ported with a one-message list, not counted here); four cases use `Unserializable`
  objects or `patch.object(json, "dumps")` (`Value` always serializes).
- `test_visualization::test_draw_graph_renders_filename`: monkeypatches `graphviz.Source.render`;
  Rust `draw_graph` writes the DOT source to a path (asserted in `draw_graph_writes_the_dot_source`).
  `test_cycle_detection` is ported as a diamond: `Arc` agents are immutable, so an A<->B cycle cannot
  be built; the "draw each agent once" rule is asserted instead.
- `test_exception_exports`: two cases assert `agents.__all__` re-exports (the compiler checks Rust paths).
- `test_agent_as_tool_callback_validation`: the legacy positional-`run_config` rejection is a Python
  call-signature check. The sync, async and callable-instance handler cases collapse to one Rust closure
  type (ported as "no handler" and "handler").
- Covered: `test_schema_with_mapping_raises_strict_mode_error` is
  `typed_tool_with_open_map_argument_is_rejected_by_try_new` / `..._panics_in_new` (core3).

## Rust bugs fixed

1. `src/extensions/handoff_filters.rs`: `remove_all_tools` removed compaction items; Python keeps
   `CompactionItem`. Also re-exports `nest_handoff_history` and `default_handoff_history_mapper`
   (Python's `__all__`).
2. `src/extensions/visualization.rs`: added `get_all_nodes_with_visited` / `get_all_edges_with_visited`
   (Python's `visited` set). A name already in the set is skipped but its DOT id stays reserved, so an
   included tool named like a skipped agent gets a generated id. The previsited root renders `""`.
3. `src/mcp/logging.rs`, `src/mcp/mod.rs`: `resolve_mcp_tool_title_value`,
   `resolve_mcp_tool_description_value`, `resolve_mcp_tool_description_for_model_value` (Python
   resolvers take dicts; only `MCPTool` forms existed). `collect_mcp_list_tools_metadata` treats a
   falsy `raw_item` as absent (`raw_item or item`).
4. `src/run_state/wire.rs`: a persisted `tool_lookup_key` is now validated like
   `deserialize_function_tool_lookup_key` (string `kind`, non-empty `name`, `namespaced` needs a
   non-empty namespace, known kinds only) and rebuilt canonically; before, any object was kept verbatim.
5. `src/error.rs`: `RunErrorDetails::pretty_print` and `Display` (Python `str(details)`).
6. `src/strict_schema.rs` (`validate_value_constraints`) and `src/tool.rs` (typed `FunctionTool` invoke):
   typed arguments are checked against the schema's value bounds before deserializing, as pydantic does.
   Keywords: `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `minLength`,
   `maxLength`, `pattern`, `minItems`, `maxItems`, `uniqueItems`, `enum`, `const`, through `$ref`,
   `allOf`, `anyOf`, `oneOf`, `properties`, `additionalProperties`, `items`/`prefixItems`. Found by
   `test_function_with_field_multiple_constraints` (out-of-range values reached the function).
   Violation message: `Invalid JSON input for tool X: <path>: <reason>`. Raw-schema tools
   (`new_raw`, MCP) are not validated, as in Python.

Deliberate divergence (not changed): Python schemas carry pydantic `title` annotations (`<func>_args`,
`City`); Rust schemas omit them (the typed constructor strips the root title on purpose). The `title`
assertions of the ported schema tests were dropped; changing it would alter the wire schema of every tool.

## Completeness sweep

174 top-level files in `tests/*.py`. All are cited by at least one `plans/reports/py-test-parity-*.md`
report. Before this round 71 of them (3 of those hold no tests) were not named in any `tests/*.rs`
file; 53 still are after it (this round's ports name theirs). Those were triaged:

- Ported this round (partial files above): `test_extension_filters`, `test_mcp_tool_metadata`,
  `test_visualization`, `test_exception_exports` (1 case), `test_agent_as_tool_callback_validation` (4 cases).
- Owned by other agents (not touched): `test_agent_config`, `test_agent_tool_enablement_snapshot`,
  `test_computer_action`, `test_disabled_trace_provider_flush`, `test_invalid_final_output_handler`,
  `test_logprobs`, `test_start_hook_tool_concurrency`, `test_streamed_terminal_output_backfill`,
  `test_tool_converter`.
- Ported earlier under another name (Rust file does not repeat the Python name):
  `test_apply_diff_helpers` (`src/apply_diff.rs` unit tests), `test_prompt_cache_key`
  (`core_prompt_cache_key.rs`), `test_run_state_agent_identity` (`py_core_a_run_loop.rs`),
  `test_oaiconv_resume_response_id` (unit test in `src/run_internal/oai_conversation.rs`),
  `test_runner_guardrail_resume` (`runner_tests.rs`), `test_extra_headers` (`py_models_*`),
  `test_tool_metadata` (`core_audit_fixes_tools.rs`, `local_tools_tests.rs`), `test_agents_logging`
  (`logger_verbose_stdout.rs`; two handler-stream cases Python-only), `test_output_guardrail_cancellation`
  (Python-only: `CancelledError` handler).
- Python-only, re-read and confirmed (no Rust analogue):
  - gc / weakref / asyncio internals: `test_agent_memory_leak`, `test_agent_tool_state` (module-level
    weakref registries), `test_asyncio_progress`, `test_asyncio_stability_command`, `test_asyncio_tasks`.
  - reflection and typing: `test_agent_instructions_signature` (`inspect.signature` arity),
    `test_decorators`, `test_doc_parsing` (Python docstring styles; Rust parses doc comments at compile
    time), `test_image_generation_tool_reference`, `test_image_generation_tool_typing`,
    `test_released_api_contract`, `test_function_schema_logger_restore` (griffe logger),
    `test_check_optional_truthiness`.
  - pydantic context models: `test_run_state_pydantic_context`.
  - logging-handler streams: `test_streaming_logging`, `test_tracing_provider_safe_debug`.
  - test helpers and environment: `test_hitl_utils`, `test_test_environment`; `conftest`,
    `_fake_workspace_paths`, `model_test_helpers` hold no tests.
  - examples and repo CI/release scripts: `test_example_workflows`, `test_live_example`,
    `test_dapr_session_example`, `test_daytona_usaspending_example`, `test_decisions_examples`,
    `test_financial_research_budget`, `test_hosted_mcp_connectors_example`,
    `test_human_in_the_loop_server_example`, `test_redis_session_example`, `test_shell_examples`,
    `test_trace_redaction_example`, `test_usaspending_setup_db`, `test_integration_runner`,
    `test_change_detection`, `test_code_change_verification_policy`, `test_code_change_verification_runner`,
    `test_release_automation`, `test_release_please`, `test_release_provenance`,
    `test_repository_workflow_interfaces`, `test_run_examples_script`, `test_run_serial_tests`,
    `test_update_rclone_pin`.

No behavioural top-level file is left without a Rust port, apart from the Python-only reasons above.

## Shared-file edits

- `Cargo.toml`: one additive dependency, `regex = { version = "1", default-features = false, features = ["std", "unicode", "perf"] }` (already in `Cargo.lock`, used for `pattern`).
- `src/lib.rs`: not touched.
- Source files edited: `src/extensions/handoff_filters.rs`, `src/extensions/visualization.rs`,
  `src/mcp/logging.rs`, `src/mcp/mod.rs`, `src/run_state/wire.rs`, `src/error.rs`, `src/strict_schema.rs`,
  `src/tool.rs` (typed invoke, one block), `docs/project-changelog.md` (new `Unreleased` section).

## Breaking changes

- Typed function tools now reject arguments that violate schema bounds (before: accepted).
- `remove_all_tools` keeps compaction items (before: dropped).
- A malformed persisted `tool_lookup_key` reads as none (before: kept).

## Verification

- All six new binaries: 105 passed.
- `cargo test --all-features --lib`: 651 passed.
- Every other `tests/*.rs` binary except `py_model_settings_serialization` (does not compile: another
  agent's in-flight `Reasoning.extra` change): passed, except 8 failures in other agents' areas:
  `py_core2_runner_streamed::raw_responses_carry_the_request_id_and_optional_raw_usage` (`cache_write_tokens`
  usage field), and streaming tests failing with "Automatic streaming does not support ..."
  (`py_core2_runstep_processing`, `py_core2_toolerrors_batch`, `py_core3_shelltools_shell_tool`,
  `py_core4_programmatic_*`): from the in-flight `ScriptedModel` change in `src/testing.rs`.
- `cargo clippy --all-features --all-targets -- -D warnings` fails only in other agents' files
  (`src/testing.rs:933` field_reassign_with_default, `src/model_settings.rs:443` items_after_test_module).
  With `-D warnings` off there are no warnings in any file this round touched.
- rustfmt applied to the new test files and the edited `src/` files (not `src/lib.rs`, not `mod.rs` files).
- Output dir `c5-leftover3` deleted at the end.

## Unresolved

- Schema `title` annotations (see above) stay a documented divergence.
- `py_model_settings_serialization` and the streaming failures above need their owners.
