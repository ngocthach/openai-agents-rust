# Python test parity, round 8, r8g7 (top-level tests/*.py, middle third)

41 files, 839 python-only cases before. Method: read each file's group list in the earlier reports, matched Python test names to Rust tests by token similarity (not exact name, since ports are renamed), read the Python test and the candidate Rust test for every suspect, ported what had no Rust counterpart.

## Per-file table

| file | python_only before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| test_function_schema_logger_restore.py | 1 | 0 | 0 | 1 | griffe/logging module internals |
| test_function_tool.py | 24 | 0 | 0 | 24 | asyncio (to_thread, cancel), monkeypatch of defaults, shallow-copy/deepcopy/dataclass replace of Python objects, `**kwargs` signatures, awaitable failure handlers (see questions) |
| test_function_tool_approval_arguments.py | 58 | 0 | 0 | 58 | pydantic validators/plugins/patterns/`model_post_init`; see questions on `approval_args_check` |
| test_function_tool_decorator.py | 50 | 0 | 0 | 50 | decorator, `__wrapped__`, callable objects, inspect/typing introspection |
| test_guardrails.py | 3 | 0 | 0 | 3 | non-callable guardrail (typed closure), Temporal compat switch |
| test_handoff_failure_error_function.py | 4 | 0 | 0 | 4 | asyncio cancel, async formatter |
| test_handoff_history_duplication.py | 13 | 0 | 0 | 13 | object-identity occurrence tracking, private helpers, legacy 1.12 layout |
| test_handoff_tool.py | 6 | 0 | 0 | 6 | callable-object dispatch, dynamically typed argument validation, raising `is_enabled` |
| test_hitl_error_scenarios.py | 16 | 0 | 1 | 15 | falsy `__bool__`, `set_public_agent` clone identity, same-named sibling by object identity, two-phase rule evaluation, private planner re-entry, dynamically typed `needs_approval` |
| test_hitl_utils.py | 1 | 0 | 0 | 1 | Python test helper |
| test_hosted_mcp_connectors_example.py | 4 | 0 | 0 | 4 | Python example |
| test_human_in_the_loop_server_example.py | 13 | 0 | 0 | 13 | Python example |
| test_image_generation_tool_reference.py | 1 | 0 | 0 | 1 | griffe docstring introspection |
| test_image_generation_tool_typing.py | 3 | 0 | 0 | 3 | pyright typing canaries |
| test_integration_runner.py | 58 | 0 | 0 | 58 | `.github/scripts` CI script |
| test_items_helpers.py | 4 | 0 | 0 | 4 | gc/weakref, pydantic |
| test_live_example.py | 16 | 0 | 0 | 16 | Python example |
| test_native_tool_failure_history.py | 34 | 0 | 0 | 34 | asyncio task cancel, hooks that raise |
| test_output_tool.py | 3 | 0 | 0 | 3 | pydantic/typing introspection |
| test_pretty_print.py | 2 | 0 | 0 | 2 | `__bool__`/`__str__` overrides |
| test_process_model_response.py | 2 | 0 | 0 | 2 | private helper, mocks |
| test_programmatic_tool_calling.py | 10 | 0 | 0 | 10 | typing/awaitable-alias inference |
| test_provider_span_errors.py | 1 | 0 | 0 | 1 | broken `__str__` span monkeypatch |
| test_redis_session_example.py | 8 | 0 | 0 | 8 | Python example |
| test_release_automation.py | 56 | 0 | 0 | 56 | CI script |
| test_release_please.py | 2 | 0 | 0 | 2 | packaging metadata |
| test_release_provenance.py | 14 | 0 | 0 | 14 | CI script |
| test_released_api_contract.py | 126 | 0 | 0 | 126 | inspect/signature introspection of the Python package |
| test_repository_workflow_interfaces.py | 18 | 0 | 0 | 18 | Makefile/workflow CI |
| test_result_cast.py | 11 | 0 | 0 | 11 | gc/weakref/release_agents |
| test_run_config.py | 22 | 0 | 0 | 22 | dict coercion, enum literal validation, `ModelSettings.__setattr__` tracking (verbosity half probed, see below) |
| test_run_context_approvals.py | 3 | 0 | 0 | 3 | non-string ids (typed), Python deferred key spelling (see questions) |
| test_run_context_wrapper.py | 3 | 0 | 0 | 3 | raising `__str__`, raw-dict fallbacks, falsy item |
| test_run_examples_script.py | 33 | 0 | 0 | 33 | examples runner script |
| test_run_hooks.py | 3 | 0 | 0 | 3 | dynamic type validation of hooks |
| test_run_impl_resume_paths.py | 24 | 0 | 0 | 24 | asyncio cancel, raising hooks, monkeypatched private functions |
| test_run_internal_approvals.py | 3 | 0 | 0 | 3 | private helper on arbitrary dict/objects |
| test_run_internal_error_handlers.py | 2 | 0 | 0 | 2 | monkeypatched schema, unserializable Python objects |
| test_run_internal_items.py | 13 | 0 | 0 | 13 | private helpers, object-identity occurrences |
| test_run_serial_tests.py | 6 | 0 | 0 | 6 | CI script |
| test_run_state.py | 165 | 10 | 5 | 150 | pydantic/deepcopy snapshot internals of `get_interruptions` (about 35), live-object identity (approval ownership by `id()`), private `_deserialize_processed_response` with live agents/tools (about 20), merge of `last_processed_response.new_items` into `generated_items` (Rust keeps them in `generated_items` by construction), datetime/bytes/pydantic output coercion, typed-None state fields, falsy objects |
| **total** | **839** | **10** | **6** | **823** | |

## Ported (10 cases, 2 files)

`tests/py_core8_r8g7_run_state_binding.rs` (5 tests) and `tests/py_core8_r8g7_run_state_misc.rs` (2 tests):

- `legacy_schema_orphaned_per_call_approval_requires_reapproval` (1.13/1.14 x safe/changed arguments: 4 cases).
- `schema_1_14_sticky_approval_binds_pending_hosted_mcp_invocation` (1).
- `serialized_apply_patch_approval_binds_plural_operations` (sticky/not: 2).
- `current_schema_unbindable_hosted_pending_requires_reapproval` (server_label half: 1).
- `a_statusless_model_response_message_round_trips_unchanged` (1).
- `from_json_warns_when_the_context_needs_restoration` (1).
- Extra task: `schema_1_13_hosted_mcp_orphaned_decisions_need_a_fresh_decision` adds the context-level half of `test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval` (approve after restore then Some(true); orphaned exact rejection stays undecided but keeps its message). The run-level half and the fix (legacy per-call decisions with no ledger binding are asked again) were already in place from round 6 (`py_core6_streamedleft_legacy_mcp.rs`, `src/run_state.rs` `from_json`); both pass. It was not part of the 165, so it moves no count.

## Covered (6 cases)

- run_state `to_json(strict_context=True)` requires a serializer: `core_python_run_state.rs::a_non_object_context_is_refused_by_a_strict_serialization`.
- run_state `from_json(strict_context=True)` requires a deserializer, and `context_deserializer` restoring: `core_python_run_state.rs::a_context_that_needs_restoring_is_an_error_only_under_strict_context` (override read).
- run_state `from_json` agent not found: `py_core2_runstate_items.rs::missing_agent_in_map_error`.
- run_state output item id cannot complete an unrelated invocation: `py_core2_runstate_approvals.rs::current_schema_rejects_completed_invocation_with_only_output_item_id`.
- hitl_error `resume_rejects_local_mcp_approval_without_recipient_binding`: `py_core6_mcpmisc_resume_binding.rs::legacy_pending_mcp_call_requires_new_run` (same message, no server call).

(That is 5 covered in run_state and 1 in hitl_error.)

## Probed, Rust differs, test dropped (not a Rust bug or not a small fix)

- `test_run_config ... field="verbosity"`: Rust agent settings start empty and the selected model's defaults fill unset fields, so an explicit verbosity equal to the default does not pin the captured reasoning default. Architectural difference, not changed.
- run_state `last_processed_response.new_items` merged into `generated_items`: Rust always holds them in `generated_items` (indexed by `turn_start_index`), so the case cannot arise.
- Duplicates found already ported earlier (approval precedence, tool output name, function-call-output input, bare-string input items) were removed from my files.

## Source bugs fixed

None. No `src/` edits.

## Breaking changes

None.

## Unresolved questions

1. A hosted MCP interruption restored from a current-schema snapshot with `arguments` removed is still honoured when a matching decision and ledger entry exist (Python: unbindable, asked again; `test_current_schema_unbindable_pending_*` arguments half and `..._with_ledger_...`). Rust's wire reader treats hosted requests as legitimately argument-less, and an empty string cannot be told from a missing key. Only a hand-corrupted snapshot triggers it. Options: carry a "missing arguments" marker on the item, or leave as is.
2. `FunctionTool::new` with a dynamic `needs_approval` runs the argument type's `Deserialize` (`approval_args_check`) before approval, so a side-effecting custom `Deserialize` runs before the user approves. Python skips inspection for models with user code (`can_prepare_without_user_code`). Rust cannot detect custom `Deserialize`; derived impls are pure. Decision needed: accept, or replace the check with a schema-only inspection (would change the already-ported invalid-typed-input behaviour).
3. Deferred top-level approvals are keyed by the bare tool name in Rust, so a permanent approval of a deferred `get_weather` is visible to a name-only query (`is_tool_approved("get_weather", ...)` returns Some(true); Python returns None). Matters only if a deferred and a bare tool of the same name coexist on one agent.
4. Async `failure_error_function` (awaitable return) has no Rust equivalent (sync closure); a feature gap rather than a Python-only behaviour.

## Gates

- `cargo test --all-features --test py_core8_r8g7_run_state_binding --test py_core8_r8g7_run_state_misc`: 7 passed.
- clippy `-D warnings` on both of my test targets: default features clean; `--all-features` clean once the lib's existing lints in other agents' in-flight code (`src/mcp/mod.rs:1399,1429`, `src/memory/advanced_sqlite_session.rs:800-852`, `needless_borrow`/`explicit_auto_deref`) are allowed. Those are not mine.
- rustfmt applied to my two files.

Status: DONE_WITH_CONCERNS
Summary: 839 python-only cases audited across 41 files; 10 ported, 6 covered, 823 remain valid python-only.
Concerns: questions 1 to 3 above (Rust divergences found but not changed); lib clippy failures in other agents' files.
