# Python test parity, core round 20, r20a: python-only re-audit

Scope: the 69 files and 279 python-only cases in `r20a_files.txt`. Each case was classified from its Python body. Rules: preamble_r8/r9.

Totals: **3 ported, 6 covered, 270 remain python-only.** One group of 3 is not really python-only. It is a recorded Rust divergence that needs a user decision (see Concerns). Three more cases are pending the fallible-callback API.

## New test files (all green, `--all-features` and default)

- `tests/py_core20_r20a_doc_parsing.rs` (gated on `macros`): `test_functions_are_ok`. The five `#[function_tool]` functions can still be called directly: Google, NumPy and Sphinx docstrings, a `&self` method and an `associated` function.
- `tests/py_core20_r20a_agent_tool_reviews.rs`: `test_sandbox_agents_as_tools_example_serializes_structured_reviews`. Two agents with structured output types are exposed as tools with a custom output extractor. Their extractor strings reach the orchestrator verbatim and in call order, next to a function tool called in the same turn.
- `tests/py_core20_r20a_deferred_approval.rs`: `test_deferred_top_level_permanent_approval_does_not_alias_to_bare_name`. A permanent approval of a deferred top-level tool is stored under `deferred_top_level:<name>`. It does not grant the bare name, and it does cover later calls of the deferred tool.

No source changes were needed. No public API changed.

## Per-file table

| file | before | ported | covered | remaining | reason for what remains (all bodies read) |
| --- | --: | --: | --: | --: | --- |
| docs/test_translate_docs.py | 9 | 0 | 0 | 9 | tests `docs/scripts/translate_docs.py` (Python docs script) |
| extensions/test_tool_output_trimmer.py | 4 | 0 | 0 | 4 | negative `recent_turns`/`preview_chars` (Rust uses `usize`), `bytes` allowlist (dynamic typing). The 4th case could not be identified by name; the str/frozenset coercion candidates are matched by `py_ext_tool_output_trimmer::{trimmable_tools_become_a_set,a_single_allowlist_entry_matches_a_whole_tool_name}` but are left uncounted |
| test_agent_as_tool_callback_validation.py | 1 | 0 | 0 | 1 | legacy positional `run_config` argument (dynamic typing) |
| test_agent_clone_shallow_copy.py | 2 | 0 | 0 | 2 | Python list aliasing: clone shares the original list, and mutating it affects both. A Rust `Vec` is owned |
| test_agent_hooks.py | 1 | 0 | 0 | 1 | falsy hooks object (`__bool__`) |
| test_agent_instructions_signature.py | 5 | 0 | 0 | 5 | arity 0/1/3, `*args/**kwargs`, non-callable instructions (ruled out by the closure type) |
| test_agent_runner_sync.py | 4 | 0 | 0 | 4 | asyncio event-loop policy: reuse, create, replace, KeyboardInterrupt task cancel, async-generator finalisation |
| test_agent_tool_enablement_snapshot.py | 1 | 0 | 0 | 1 | replaces `agent.tools` while `get_all_tools` is suspended. A Rust agent is shared immutably |
| test_agents_logging.py | 2 | 0 | 0 | 2 | Python `logging` handlers with `sys.stdout` replaced or stderr fallback, threaded handler init |
| test_asyncio_stability_command.py | 4 | 0 | 0 | 4 | `.github/scripts/run-asyncio-teardown-stability.sh` |
| test_call_model_input_filter.py | 1 | 0 | 0 | 1 | filter returns a `str` (the return type is fixed) |
| test_cancel_streaming.py | 2 | 0 | 0 | 2 | asyncio consumer cancellation at the terminal wait (monkeypatched `_await_task_safely`), registered-cleanup wait, cancelled guardrail dependency |
| test_code_change_verification_policy.py | 6 | 0 | 0 | 6 | `.agents/skills` files, `detect-changes.sh`, conftest helper |
| test_code_change_verification_runner.py | 19 | 0 | 0 | 19 | `.agents/skills/code-change-verification/scripts/run.sh` and its harness |
| test_computer_action.py | 3 | 0 | 0 | 3 | `inspect.signature` probing of legacy, non-introspectable and `**kwargs` drivers |
| test_computer_tool_lifecycle.py | 1 | 0 | 0 | 1 | start hook assigns `agent.tools` (hooks get `&Agent`) |
| test_daytona_usaspending_example.py | 2 | 0 | 0 | 2 | `examples/sandbox/.../usaspending_text2sql` |
| test_decorators.py | 3 | 0 | 1 | 2 | covered: bare and configured tool name, `function_tool_macro_tests::generates_tools_from_functions`. Remaining: module identity, and guardrail decorator forms with `assert_type` (Rust guardrails take an explicit name) |
| test_default_tool_error_redaction.py | 5 | 0 | 0 | 5 | `asyncio.CancelledError` raised in a tool (4), `__str__` that raises (1) |
| test_doc_parsing.py | 1 | 1 | 0 | 0 | ported |
| test_example_workflows.py | 12 | 1 | 0 | 11 | ported: structured agent-tool reviews. Remaining are example helpers: web-search URL helpers (3), financial manager (5), docker SDK import (1), docker runner formatters (2) |
| test_exception_exports.py | 1 | 0 | 1 | 0 | `py_core14_r14c_public_api_contract::every_exception_class_is_a_variant_reachable_from_the_crate_root` (includes `MCPToolCancellationError`) |
| test_extension_filters.py | 2 | 0 | 1 | 1 | covered: unserializable item / `json.dumps` TypeError, `py_core16_r16a_handoff_nest` (its doc names both Python tests). Remaining: content object with custom `__str__` (`serde_json::Value` always serializes) |
| test_financial_research_budget.py | 3 | 0 | 0 | 3 | `examples/financial_research_agent` |
| test_function_schema_logger_restore.py | 1 | 0 | 0 | 1 | griffe logger level |
| test_function_tool_approval_arguments.py | 4 | 0 | 0 | 4 | compiled `re.Pattern` (2), `tuple[int, ...]` annotation (1), `**kwargs` reconstruction (1): pydantic/typing |
| test_guardrails.py | 3 | 0 | 0 | 3 | non-callable guardrail (2), Temporal replay patch switch (vendor SDK) |
| test_handoff_tool.py | 4 | 0 | 0 | 4 | invalid `on_handoff` signature, `input_type` without `on_handoff`, non-callable `on_handoff` (typed builder). Sibling `is_enabled` cancellation on error is **pending fallible-callback API** (6 bodies read, including the pydantic lenient and callable-object cases) |
| test_hitl_utils.py | 1 | 0 | 0 | 1 | Python test helper |
| test_image_generation_tool_reference.py | 1 | 0 | 0 | 1 | griffe docstring introspection |
| test_image_generation_tool_typing.py | 3 | 0 | 0 | 3 | `__all__` identity, pydantic `TypeAdapter`, dataclass fields |
| test_items_helpers.py | 2 | 0 | 0 | 2 | pydantic rejects a `None` refusal; `ValidatorIterator` copy |
| test_pretty_print.py | 2 | 0 | 0 | 2 | dataclass output rendered by `__str__` (streamed or not) |
| test_process_model_response.py | 2 | 0 | 0 | 2 | falsy `ShellTool` / `ApplyPatchTool` (`__bool__`) |
| test_programmatic_tool_calling.py | 5 | 0 | 0 | 5 | return-annotation inference through `Awaitable`/`Coroutine`/PEP 695 aliases |
| test_provider_span_errors.py | 1 | 0 | 0 | 1 | monkeypatched private `attach_error_to_span` |
| test_redis_session_example.py | 8 | 0 | 0 | 8 | `examples/memory/redis_session_example.py` |
| test_release_please.py | 2 | 0 | 0 | 2 | release-please metadata / workflow YAML |
| test_release_provenance.py | 14 | 0 | 0 | 14 | `.github/scripts/verify_release.py` |
| test_repository_workflow_interfaces.py | 18 | 0 | 0 | 18 | Makefile, workflows, `.agents` skills |
| test_run_context_approvals.py | 1 | 1 | 0 | 0 | ported |
| test_run_context_wrapper.py | 3 | 0 | 0 | 3 | `_to_str_or_none` with a raising `__str__`, raw-dict name/call-id fallbacks (typed fields), falsy approval item |
| test_run_hooks.py | 3 | 0 | 0 | 3 | hooks of the wrong class, `validate_run_hooks(object())` |
| test_run_internal_approvals.py | 3 | 0 | 0 | 3 | private helpers called on heterogeneous lists, duck-typed steps, dict/object namespaces |
| test_run_internal_items.py | 5 | 0 | 0 | 5 | `json.dumps` monkeypatch, legacy `tool_search_result` coercion, `Sequence` read counting, non-mapping entries, `extract_mcp_request_id` on objects (8 candidate bodies read) |
| test_run_serial_tests.py | 6 | 0 | 0 | 6 | `.github/scripts/run_serial_tests.py` |
| test_run_state_agent_identity.py | 3 | 0 | 0 | 3 | legacy `.agent` handoff wrappers (2), capability identity of Python object config |
| test_run_state_hook_identity.py | 2 | 0 | 0 | 2 | `hook_type` / `hook_category` discriminators (Python class name, dataclass-ness) |
| test_run_state_pending_input.py | 3 | 0 | 0 | 3 | **not valid python-only**: recorded Rust divergence, user decision pending (see Concerns) |
| test_runner_model_provider_lifecycle.py | 1 | 0 | 0 | 1 | provider `aclose` raising; Rust `ModelProvider::close` cannot fail |
| test_server_conversation_tracker.py | 4 | 0 | 1 | 3 | covered: the reasoning-id omit policy on items the tracker prepares (`py_core2_runner_routing::the_omit_policy_strips_follow_up_reasoning_ids` + `py_core15_r15a_server_tracker`). Remaining: monkeypatched `id()` and stale-address identity (2 params) |
| test_shell_call_serialization.py | 3 | 0 | 0 | 3 | `bytes`/`bytearray` commands, string executor outcome (typed enum) |
| test_stream_events.py | 1 | 0 | 0 | 1 | private `stream_step_items_to_queue` with a duck-typed `object()` |
| test_strict_schema.py | 4 | 0 | 2 | 2 | covered: the recursion-depth probes named by r16d, `py_core_a_pure::{deeply_nested_schema_is_rejected_before_recursive_conversion,deeply_chained_refs_are_rejected_before_recursive_conversion}`. Remaining: `OrderedDict` equality, `_MAX_SCHEMA_NODES` monkeypatch (the budget is covered by an in-crate unit test) |
| test_test_environment.py | 6 | 0 | 0 | 6 | conftest proxy-environment helper |
| test_tool_approval_call_id_reuse.py | 3 | 0 | 0 | 3 | pydantic web-search schema drift, `model_dump` signature probing (2). The cancelled async formatter was also read: valid |
| test_tool_custom_data.py | 4 | 0 | 0 | 4 | `object()` and NaN/±inf custom data (`serde_json::Value` cannot hold them) |
| test_tool_exception_formatter.py | 3 | 0 | 0 | 3 | formatter returns a non-string, tool raises `CancelledError`, positional `ToolErrorFormatterArgs` constructor |
| test_tool_failure_boundaries.py | 6 | 0 | 0 | 6 | tool-local and native `CancelledError` (4). `mcp_callback` boundary (2) is **pending fallible-callback API** |
| test_tool_guardrails.py | 2 | 0 | 0 | 2 | non-callable tool guardrail |
| test_tool_name_collision_policy.py | 6 | 0 | 0 | 6 | async formatter cancellation (3), agent subclass mutating approvals inside an awaiting `get_all_tools` (3) |
| test_tool_output_conversion.py | 1 | 0 | 0 | 1 | falsy structured output (`__bool__`) |
| test_tool_use_behavior.py | 6 | 0 | 0 | 6 | private helper with an empty result list, invalid behavior string, non-`ToolsToFinalOutputResult` return (typed) |
| test_trace_redaction_example.py | 6 | 0 | 0 | 6 | `examples/basic/trace_redaction.py` exporter |
| test_update_rclone_pin.py | 19 | 0 | 0 | 19 | `.github/scripts/update_rclone_pin.py` |
| test_usage.py | 3 | 0 | 0 | 3 | private `_raw_usage_snapshot` detachment, unset pydantic fields, non-JSON value |
| test_usaspending_setup_db.py | 4 | 0 | 0 | 4 | example `setup_db` with monkeypatched ssl/urllib |
| test_visualization.py | 1 | 0 | 0 | 1 | monkeypatched `graphviz.Source.render` |
| utils/test_json.py | 2 | 0 | 0 | 2 | `_to_dump_compatible` on iterators / `MappingProxyType` |

The repository-tooling and example files were confirmed by their imports and bodies. They test `.github`/`.agents` scripts, `docs/scripts`, conftest helpers or `examples/` modules, not SDK behaviour.

## Pending fallible-callback API

These were left alone because `src/tool.rs` and `src/handoffs.rs` belong to the agent adding the fallible API:

- `test_handoff_tool.py::test_get_handoffs_cancels_sibling_enablement_checks_on_error` (1)
- `test_tool_failure_boundaries.py::test_completed_history_at_failure_boundaries[mcp_callback]` (streamed and blocking, 2)

`HostedMCPTool::with_on_approval_request_fallible` already exists in `src/tool.rs` but has no tests yet. These cases can be ported once that API settles.

## Gates

- New binaries pass with `--all-features` and with default features (`doc_parsing` compiles to nothing without `macros`).
- Related binaries pass: `py_core2_approvals_context`, `py_core14_r14c_example_workflows`, `py_core16_r16gaps_function_tool_methods`, `function_tool_macro_tests`.
- `clippy -D warnings` is clean for the lib and the new test files, on both feature sets.
- `rustfmt --edition 2024` was run on the new files only.

## Concerns / unresolved questions

1. **`test_run_state_pending_input.py` (3 cases):** `test_server_tracker_distinguishes_identical_input_occurrences_after_restore` and `test_server_managed_resume_sends_identical_late_input_in_later_occurrence` (2 params).
   - In Python, under server-managed conversations, a later input identical to one already delivered is sent again, because the tracker selects items by object identity.
   - Rust selects by content fingerprint, so it does not resend that input. `py_core3_runstatectx_pending_b::identical_late_input_is_not_resent_by_fingerprint_*` asserts this. The fingerprint was chosen so that a retry or a resumed state can neither resend nor skip an item.
   - This is user-visible, so it is not a valid python-only reason. It is left as a recorded decision, not reversed.
   - Options: (a) keep the fingerprint behaviour and document the divergence; (b) add per-occurrence identity, e.g. an occurrence counter or `input_id`, to the tracker, then port the 3 cases.
2. The ledger does not map python-only counts to test names. For trimmer, extension_filters, strict_schema, cancel_streaming, handoff_tool, run_internal_items and server_conversation_tracker, more candidate bodies were read than the count, and the remaining names were inferred from earlier reports. Moves to covered were made only where an earlier report named the group and a Rust test could be cited.

## Suggested changelog paragraph (not applied to docs/project-changelog.md)

> Python parity, round 20 (r20a): re-audited 279 python-only cases across 69 Python test files by reading every case body. Ported 3: `#[function_tool]` functions, methods and associated functions stay directly callable; structured agent-as-tool outputs pass through a custom extractor in call order; a deferred top-level permanent approval does not alias to the bare tool name. Six more are now cited as covered by existing tests. No source changes. The pending-input identical-occurrence resend behaviour remains a recorded divergence awaiting a decision.

Status: DONE_WITH_CONCERNS
Summary: All 279 case bodies across 69 files read: 3 ported, 6 covered, 270 remain. That includes 3 that are a recorded divergence needing a user decision and 3 pending the fallible-callback API.
