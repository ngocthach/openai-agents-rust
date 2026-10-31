# Python test parity — area core-a

Scope: the 85 Python test files in `top_a.txt` (`openai-agents-python/tests/`), plus the shared
helpers they import (`conftest.py`, `test_responses.py`, `agents.testing.ScriptedModel`;
this fork has no `fake_model.py`).

## How the counts were produced

AST scan of each file: every `def test_*` / `async def test_*` is one "def"; a
`@pytest.mark.parametrize` whose argvalues are a list/tuple literal multiplies by its length
("cases"). Script: `scratchpad/count_tests_corea.py`. Totals: **2065 defs → 3052 cases**.
(`pytest --collect-only` was not available: the scratchpad venv has no pytest and no network.)

## Rust test files added

| file | tests | ports |
| --- | --: | --- |
| `tests/py_core_a_pure.rs` | 83 | `test_apply_diff.py`, `test_strict_schema.py`, `test_usage.py`, `test_transforms.py`, `test_handoff_prompt.py`, `test_debug.py` |
| `tests/py_core_a_tool_behavior.rs` | 22 | `test_tool_use_behavior.py`, `test_tool_choice_reset.py`, `test_output_tool.py` |
| `tests/py_core_a_hooks.rs` | 14 | `test_run_hooks.py`, `test_global_hooks.py`, `test_run_error_details.py` |
| `tests/py_core_a_runner.rs` | 14 | `test_agent_runner.py` (end-to-end half) |
| `tests/py_core_a_stream_events.rs` | 6 | `test_stream_events.py`, `test_streaming_tool_call_arguments.py` |
| `tests/py_core_a_tool_guardrails.rs` | 11 | `test_tool_guardrails.py` |
| `tests/py_core_a_agent_as_tool.rs` | 11 | `test_agent_as_tool.py` |
| `tests/py_core_a_tool_origin.rs` | 12 | `test_tool_origin.py` |
| `tests/py_core_a_run_loop.rs` | 8 | empty/refused turns (`test_agent_runner.py`), duplicate agent identity (`test_run_state_agent_identity.py`) |
| `tests/py_core_a_streamed.rs` | 15 | `test_agent_runner_streamed.py` |
| `tests/py_core_a_approvals.rs` | 10 | `test_run_context_approvals.py` |

## Per-file table

Buckets: **ported** = a Rust test with the same inputs/expectations now exists;
**covered** = an existing Rust test already asserts it; **python-only** = cannot exist in Rust;
**not ported** = behavioural but out of this session's budget, or blocked on a private Rust
module (honest remainder, not padding).

**Accuracy caveat:** *ported* and *python-only* counts are exact (one row per Rust test / per
reason group). *covered* counts are **approximate file-level mappings** — they say "an existing
Rust test file asserts this area", not "test N maps to Rust test M". The cited Rust files are in
the notes column; treat the covered column as a lower-bound estimate, not a precise figure.

| Python file | defs | cases | ported | covered | python-only | not ported | notes |
| --- | --: | --: | --: | --: | --: | --: | --- |
| `test_agent_as_tool_callback_validation.py` | 2 | 5 | 0 | 0 | 5 | 0 | callback signature validation; Rust types enforce it at compile time |
| `test_agent_as_tool.py` | 67 | 88 | 11 | 3 | 0 | 74 | 11 ported in py_core_a_agent_as_tool.rs; nested approvals covered by runner_tests::agents_as_tools_and_nested_approvals |
| `test_agent_config.py` | 14 | 19 | 0 | 6 | 13 | 0 | instructions/handoff/clone behaviour covered by runner_tests.rs + core_tool_handoff_rows.rs; __post_init__ type validation is Python-only |
| `test_agent_instructions_signature.py` | 11 | 11 | 0 | 0 | 11 | 0 | inspect.signature arity probing |
| `test_agent_memory_leak.py` | 1 | 1 | 0 | 0 | 1 | 0 | gc/weakref |
| `test_agent_runner_streamed.py` | 95 | 214 | 15 | 12 | 0 | 187 | 15 ported in py_core_a_streamed.rs (basic runs, handoffs and final agent, guardrail tripwire/exception, save-error precedence, max-turns pending events, tool-guardrail results across a handoff, event order); the rest is retry/ws/terminal-event/session-persistence variants |
| `test_agent_runner.py` | 203 | 240 | 14 | 20 | 0 | 206 | 14 ported in py_core_a_runner.rs; ~60 of the remainder call private run_internal helpers (prepare_input_with_session, save_result_to_session, rewind, blocked-batch internals) and need a pub(crate) test seam before they can be ported |
| `test_agent_tool_enablement_snapshot.py` | 3 | 4 | 0 | 2 | 0 | 2 | is_enabled snapshotting covered by py_core_a_agent_as_tool.rs |
| `test_agent_tool_state.py` | 6 | 6 | 0 | 0 | 6 | 0 | gc/weakref module-level tool state |
| `test_agents_logging.py` | 5 | 5 | 0 | 3 | 2 | 0 | logger_verbose_stdout.rs; the rest asserts Python logging handlers |
| `test_apply_diff.py` | 21 | 21 | 21 | 0 | 0 | 0 | all ported |
| `test_asyncio_progress.py` | 7 | 7 | 0 | 0 | 7 | 0 | asyncio task progress deadlines |
| `test_asyncio_tasks.py` | 9 | 12 | 0 | 0 | 12 | 0 | asyncio gather/producer-consumer helpers |
| `test_call_model_input_filter.py` | 11 | 12 | 0 | 4 | 0 | 8 | runner_tests::call_model_input_filter_and_session_callback |
| `test_change_detection.py` | 16 | 62 | 0 | 0 | 62 | 0 | shells .github/scripts/detect-changes.sh |
| `test_code_change_verification_policy.py` | 3 | 6 | 0 | 0 | 6 | 0 | asserts .agents/skills files |
| `test_computer_action.py` | 17 | 40 | 0 | 6 | 0 | 34 | local_tools_tests.rs + core_audit_fixes_tools.rs |
| `test_config.py` | 27 | 30 | 0 | 0 | 0 | 30 | OpenAI client/provider configuration - models area (other agent) |
| `test_dapr_session_example.py` | 19 | 33 | 0 | 0 | 33 | 0 | runs examples/ scripts |
| `test_debug.py` | 10 | 10 | 1 | 0 | 9 | 0 | override API ported as one test in py_core_a_pure.rs; the 9 env-var monkeypatch variants are Python-only (std::env::set_var is unsafe in edition 2024 and races across the test binary) |
| `test_decorators.py` | 3 | 3 | 0 | 0 | 3 | 0 | Python decorator module identity |
| `test_disabled_trace_provider_flush.py` | 1 | 1 | 0 | 0 | 0 | 1 | tracing area (other agent) |
| `test_doc_parsing.py` | 4 | 4 | 0 | 0 | 4 | 0 | Python docstring (google/numpy/sphinx) parsing |
| `test_example_workflows.py` | 30 | 30 | 0 | 0 | 30 | 0 | runs examples/ workflows |
| `test_extension_filters.py` | 44 | 55 | 0 | 4 | 0 | 51 | handoff filters - extensions area (other agent); nesting covered by runner_tests::handoff_input_filter_and_nested_history |
| `test_financial_research_budget.py` | 1 | 3 | 0 | 0 | 3 | 0 | examples/financial_research_agent |
| `test_function_schema.py` | 52 | 70 | 0 | 8 | 40 | 22 | schema shape covered by function_tool_macro_tests.rs + core_tool_output_schema.rs; the rest is inspect/pydantic introspection |
| `test_function_tool_decorator.py` | 41 | 64 | 0 | 10 | 30 | 24 | function_tool_macro_tests.rs; the rest is Python decorator/inspect behaviour |
| `test_global_hooks.py` | 4 | 4 | 0 | 4 | 0 | 0 | py_core_a_hooks.rs covers the same counters end to end |
| `test_handoff_failure_error_function.py` | 6 | 11 | 0 | 7 | 0 | 4 | core_audit_fixes_handoff.rs |
| `test_handoff_prompt.py` | 1 | 1 | 1 | 0 | 0 | 0 | ported |
| `test_hitl_error_scenarios.py` | 67 | 90 | 0 | 14 | 0 | 76 | approval/resume paths partly covered by runner_tests.rs + core_python_run_state.rs |
| `test_hitl_utils.py` | 1 | 1 | 0 | 0 | 1 | 0 | test-helper class |
| `test_human_in_the_loop_server_example.py` | 6 | 13 | 0 | 0 | 13 | 0 | examples/agent_patterns |
| `test_image_generation_tool_typing.py` | 3 | 3 | 0 | 0 | 3 | 0 | dataclass/typing introspection |
| `test_invalid_final_output_handler.py` | 12 | 15 | 0 | 4 | 0 | 11 | runner_tests::invalid_output_and_refusal_handlers |
| `test_live_example.py` | 12 | 16 | 0 | 0 | 16 | 0 | examples/live |
| `test_logprobs.py` | 5 | 5 | 0 | 0 | 0 | 5 | chat-completions logprobs - models area (other agent) |
| `test_mcp_tool_metadata.py` | 27 | 27 | 0 | 0 | 0 | 27 | mcp area (other agent) |
| `test_native_tool_failure_history.py` | 8 | 72 | 0 | 8 | 0 | 64 | hosted-tool failure history partly covered by local_tools_tests.rs |
| `test_openai_provider_client_options.py` | 3 | 10 | 0 | 0 | 0 | 10 | models area (other agent) |
| `test_output_tool.py` | 15 | 15 | 10 | 2 | 3 | 0 | ported in py_core_a_tool_behavior.rs; Literal/TypedDict naming is Python-only |
| `test_process_model_response.py` | 29 | 29 | 0 | 0 | 0 | 29 | run_loop.process_model_response is private in Rust; needs a pub(crate) test seam. Observable results are partly asserted through Runner in py_core_a_runner.rs |
| `test_prompt_cache_key.py` | 14 | 14 | 0 | 14 | 0 | 0 | core_prompt_cache_key.rs |
| `test_redis_session_example.py` | 3 | 8 | 0 | 0 | 8 | 0 | examples/memory |
| `test_release_please.py` | 2 | 2 | 0 | 0 | 2 | 0 | repo release metadata |
| `test_released_api_contract.py` | 95 | 126 | 0 | 0 | 126 | 0 | importlib/inspect sweep of the Python public API |
| `test_repository_workflow_interfaces.py` | 15 | 18 | 0 | 0 | 18 | 0 | Makefile / GitHub workflow files |
| `test_responses.py` | 0 | 0 | 0 | 0 | 0 | 0 | helper module, no tests |
| `test_resume_tool_use_behavior.py` | 1 | 48 | 0 | 4 | 0 | 44 | runner_tests::approval_pause_and_resume_with_serialization + core_python_run_state.rs |
| `test_run_context_approvals.py` | 25 | 31 | 10 | 0 | 0 | 21 | 10 ported in py_core_a_approvals.rs; the hosted-MCP identity records, deferred top-level lookup keys and legacy bare-name aliasing (about 21 cases) have no Rust model: approvals are keyed by (agent, qualified tool name) |
| `test_run_error_details.py` | 2 | 2 | 2 | 0 | 0 | 0 | ported in py_core_a_hooks.rs |
| `test_run_hooks.py` | 15 | 15 | 12 | 0 | 3 | 0 | ported in py_core_a_hooks.rs; validate_run_hooks type checks are Python-only |
| `test_run_internal_approvals.py` | 7 | 7 | 0 | 0 | 0 | 7 | agents.run_internal.approvals is private in Rust; needs a pub(crate) test seam |
| `test_run_internal_items.py` | 43 | 54 | 0 | 0 | 0 | 54 | agents.run_internal.items is private in Rust (`mod run_internal`, src/lib.rs:54); needs a pub(crate) test seam |
| `test_run_state_agent_identity.py` | 18 | 18 | 3 | 12 | 0 | 3 | 3 ported in py_core_a_run_loop.rs (duplicate-name identity round trip, tracker keys, refusal without identity); core_python_run_state.rs; the sandbox-manifest/capability signature cases belong to the sandbox area |
| `test_run_state_compatibility_corpus.py` | 44 | 84 | 0 | 20 | 0 | 64 | core_run_state.rs + core_python_run_state.rs read the Python fixtures |
| `test_run_state_pending_input.py` | 25 | 54 | 0 | 12 | 0 | 42 | core_python_run_state.rs |
| `test_run_state.py` | 313 | 470 | 0 | 60 | 0 | 410 | core_run_state.rs (10) + core_python_run_state.rs (24) + core_trace_resume.rs cover the envelope, approval ledger, resume and interop corpus |
| `test_run_step_processing.py` | 17 | 17 | 0 | 0 | 0 | 17 | run_loop single-step processing is private in Rust; needs a pub(crate) test seam |
| `test_runner_guardrail_resume.py` | 3 | 6 | 0 | 6 | 0 | 0 | runner_tests guardrail + resume tests |
| `test_runner_tracing_disabled.py` | 8 | 23 | 0 | 0 | 0 | 23 | tracing area (other agent) |
| `test_scripted_model.py` | 62 | 97 | 0 | 7 | 0 | 90 | testing_model_tests.rs; src/testing.rs is not in this area's ownership |
| `test_server_conversation_tracker.py` | 23 | 24 | 0 | 6 | 0 | 18 | core_server_conversation.rs + py_core_a_runner::conversation_id_only_sends_new_items_multi_turn |
| `test_shell_examples.py` | 5 | 21 | 0 | 0 | 21 | 0 | examples/tools shell scripts |
| `test_soft_cancel.py` | 23 | 25 | 0 | 3 | 0 | 22 | RunResultStreaming::cancel covered by runner_tests.rs |
| `test_start_hook_tool_concurrency.py` | 12 | 39 | 0 | 4 | 0 | 35 | hook concurrency partly covered by py_core_a_hooks.rs |
| `test_stream_events.py` | 14 | 14 | 5 | 4 | 0 | 5 | ported in py_core_a_stream_events.rs; ordering helpers in private streaming module |
| `test_streamed_terminal_output_backfill.py` | 5 | 5 | 0 | 0 | 0 | 5 | hosted terminal backfill - not ported (budget) |
| `test_streaming_tool_call_arguments.py` | 4 | 4 | 1 | 0 | 0 | 3 | ported in py_core_a_stream_events.rs |
| `test_strict_schema.py` | 60 | 69 | 43 | 0 | 8 | 18 | ported in py_core_a_pure.rs; OrderedDict / int-vs-bool identity cases are Python-only |
| `test_tool_approval_call_id_reuse.py` | 85 | 130 | 0 | 20 | 0 | 110 | call-id rebinding covered by core_python_run_state.rs approval-ledger tests |
| `test_tool_choice_reset.py` | 6 | 6 | 7 | 0 | 0 | 0 | ported in py_core_a_tool_behavior.rs |
| `test_tool_converter.py` | 6 | 6 | 0 | 0 | 0 | 6 | chatcmpl converter - models area (other agent) |
| `test_tool_exception_formatter.py` | 13 | 33 | 0 | 8 | 0 | 25 | runner_tests::tool_errors_timeouts_and_not_found + core_audit_fixes_tools.rs |
| `test_tool_guardrails.py` | 22 | 25 | 16 | 0 | 4 | 5 | 11 Rust tests in py_core_a_tool_guardrails.rs cover 16 Python cases (2 of them parametrized over streaming); the invalid-guardrail-function and exception-str cases are Python-only |
| `test_tool_metadata.py` | 3 | 3 | 0 | 3 | 0 | 0 | hosted tool names covered by core_audit_fixes_tools.rs + local_tools_tests.rs |
| `test_tool_origin.py` | 14 | 14 | 11 | 0 | 3 | 0 | ported in py_core_a_tool_origin.rs; weakref/gc, positional dataclass arg and the SDK-internal json_tool_call tool are Python-only |
| `test_tool_use_behavior.py` | 10 | 17 | 6 | 0 | 4 | 7 | ported in py_core_a_tool_behavior.rs; invalid-value and return-contract cases cannot exist in Rust (enum + typed closure) |
| `test_trace_processor.py` | 69 | 89 | 0 | 0 | 0 | 89 | tracing area (other agent) |
| `test_tracing_errors_streamed.py` | 10 | 11 | 0 | 0 | 0 | 11 | tracing area (other agent) |
| `test_tracing_provider_safe_debug.py` | 1 | 1 | 0 | 0 | 0 | 1 | tracing area (other agent) |
| `test_transforms.py` | 2 | 5 | 2 | 0 | 3 | 0 | ported in py_core_a_pure.rs; caplog warning assertions are Python-only |
| `test_usage.py` | 23 | 23 | 15 | 2 | 4 | 2 | ported in py_core_a_pure.rs; pydantic model_construct/None-normalisation cases cannot exist in Rust |
| `test_visualization.py` | 21 | 26 | 0 | 0 | 0 | 26 | graphviz export - extensions area (other agent) |
| **TOTAL** | **2065** | **3052** | **206** | **302** | **517** | **2028** | |

## python-only reason groups (one line per group)

- **Repo/metadata tests** (`test_release_please`, `test_repository_workflow_interfaces`,
  `test_change_detection`, `test_code_change_verification_policy`): assert the Python repo's
  `Makefile`, `.github/` workflows, shell scripts and `.agents/skills` files.
- **Example-script tests** (`test_example_workflows`, `test_dapr_session_example`,
  `test_redis_session_example`, `test_live_example`, `test_shell_examples`,
  `test_financial_research_budget`, `test_human_in_the_loop_server_example`): `runpy`/import of
  `examples/` modules that have no Rust counterpart.
- **Python reflection** (`test_released_api_contract`, `test_agent_instructions_signature`,
  `test_decorators`, `test_doc_parsing`, `test_image_generation_tool_typing`, most of
  `test_function_schema` / `test_function_tool_decorator`): `importlib`, `inspect.signature`,
  docstring parsing, `assert_type`, pydantic internals. Rust derives schemas from `schemars` at
  compile time, so there is no runtime signature to probe.
- **gc / weakref** (`test_agent_memory_leak`, `test_agent_tool_state`, parts of `test_config`,
  `test_tool_origin`): refcount/finaliser assertions; Rust `Arc` drop is deterministic and these
  checks have no analogue.
- **asyncio internals** (`test_asyncio_tasks`, `test_asyncio_progress`): `gather_with_cancel`,
  `run_producer_consumer`, task progress deadlines — tokio provides the equivalents and they are
  not part of the public Rust surface.
- **Dynamic-type validation** (`Agent.__post_init__` checks in `test_agent_config`,
  `test_tool_use_behavior::test_invalid_tool_use_behavior_raises`,
  `test_run_hooks::test_validate_run_hooks_*`, `ToolGuardrail(guardrail_function="foo")`):
  Rust's enums and typed closures make these states unrepresentable.
- **caplog / logging-handler assertions** (`test_transforms`, parts of `test_agents_logging`):
  assert `pytest.LogCaptureFixture` records.
- **Python value identity** (`test_strict_schema` `OrderedDict` / `True == 1` cases,
  `test_usage` `model_construct` None-normalisation): `serde_json::Value` already distinguishes
  bools from numbers and has no mapping subclasses.

## Rust bugs found and fixed

1. **`src/strict_schema.rs` — `$ref` sibling allow-list was incomplete.**
   Python's `_REF_NON_CONSTRAINING_SIBLINGS` is
   `{$anchor, $comment, $defs, $schema, contentEncoding, contentMediaType, contentSchema,
   default, definitions, deprecated, description, examples, readOnly, title, writeOnly}`
   (root adds `$id`). Rust listed only 8 of them, so
   `{"$ref": "...", "$anchor": "..."}` and `{"$ref": "...", "contentMediaType": "..."}` were
   rejected as "incompatible sibling". Both lists now match Python.
   Tests: `py_core_a_pure::ref_with_anchor_is_still_expanded`,
   `ref_with_annotation_sibling_is_still_expanded`.

2. **`src/strict_schema.rs` — a singleton `allOf` holding a bare `$ref` could not be merged.**
   `{"type": "object", "allOf": [{"$ref": "#/$defs/Outer"}]}` failed with
   "`$ref` with incompatible sibling keyword(s) (type)". Ported Python's
   `_resolve_non_constraining_ref_chain` (follows `$ref` aliases, carries annotation siblings,
   raises on a circular chain, spends node budget) and used it for that case.
   Tests: `all_of_single_ref_entry_merging`, `all_of_single_ref_entry_preserves_annotated_aliases`,
   `all_of_single_circular_ref_is_rejected`, `all_of_single_ref_entry_decodes_json_pointer_tokens`.

3. **`src/strict_schema.rs` — `merge_single_all_of` silently overwrote conflicting parent
   keywords.** It unioned `properties`/`required` and let every other parent key win. Python
   raises `UserError("...singleton \`allOf\` entry with incompatible parent keyword(s)...")`
   when the parent and the entry disagree, and refuses `additionalProperties: false` on a parent
   whose own `properties` is empty while the entry declares some. Ported, including Python's
   `_json_values_equal` (booleans never equal numbers; `1 == 1.0`).
   Test: `all_of_single_entry_cannot_overwrite_strict_object`.

4. **`src/strict_schema.rs` — nested `$id` resources were resolved against the document root.**
   Python raises `_NESTED_RESOURCE_REF_ERROR` when a `$ref` is owned by, or crosses, a schema
   node that declares its own `$id`. Rust happily resolved through it (wrong target, and a way to
   smuggle a `$defs` shadow past strict conversion). `resolve_ref` now rejects a traversed node
   that declares a string `$id`, and `ensure_strict` threads an `inside_nested_resource` flag that
   rejects any `$ref` below such a node. A `$defs` member *named* `$id` is still allowed
   (Python checks `isinstance(..., str)`).
   Tests: `root_ref_with_id_is_still_expanded`, `ref_allows_unrelated_id_definition_name`.

5. **`src/run_internal/mod.rs` — `RunContext::turn_input()` was empty inside `on_agent_start`.**
   Python sets `context_wrapper.turn_input` *before* the agent-start hooks
   (`run_loop.py:2468` and `:2820`); Rust only set it just before the model call, so every
   `on_agent_start` hook saw an empty list. Now set from the run input before the hooks, and
   still refreshed with the real model input afterwards.
   Tests: `py_core_a_hooks::run_hooks_receives_turn_input_{string,list,streamed}`.

6. **`src/agent.rs` — `Agent::as_tool_with(...input_builder(..))` ignored the builder unless
   `parameters` was also set.** Python's `should_capture_tool_input` is
   `has_custom_parameters or include_schema or input_builder is not None`, so a builder alone opts
   into structured input. Rust used `parameters.is_some()` only. Now
   `parameters.is_some() || input_builder.is_some()`.
   Test: `py_core_a_agent_as_tool::agent_as_tool_input_builder_alone_opts_into_structured_input`.

## Cross-area gaps recorded, not fixed

- **`ToolOrigin` does not exist in the Rust crate** (`grep -rn ToolOrigin src` → no hits).
  `test_tool_origin.py` (14 defs) asserts that `ToolCallItem` / `ToolCallOutputItem` carry a
  `tool_origin` (`function` | local MCP server, with credentials stripped from the URL | hosted
  MCP), that it survives `RunState` round-trips, that legacy 1.5/1.6 snapshots without it still
  load, and that it is attached on the streamed path too. Implementing it touches `src/items.rs`,
  `src/run_internal/**`, `src/run_state/wire.rs` *and* `src/mcp/**` (another agent's area), so it
  is recorded rather than started. Minimal shape: add
  `ToolOrigin { r#type: ToolOriginType, server_label: Option<String>, server_url: Option<String> }`
  to `RunItemKind::ToolCall`/`ToolCallOutput` as an `Option`, default `None` on read.
- **`src/testing.rs::FakeModel::stream_response`** synthesizes only
  `response.created` / `output_item.added` / `output_text.delta` / `output_item.done` /
  `response.completed`. It never emits `response.function_call_arguments.delta|done`, which the
  richer `ScriptedModel` (same file, `scripted_stream_events`) does. `src/testing.rs` is not in
  this area's ownership; the ported test uses `ScriptedModel` instead. Suggested fix:
  `response_to_stream_events` should emit the two argument events for
  `OutputItem::FunctionCall`, mirroring `scripted_stream_events` (src/testing.rs:836-847).
- **`Usage::to_span_usage` shape differs from Python's `model_usage_to_span_usage`.**
  Rust (`src/usage.rs:205`) emits
  `{"input_tokens", "output_tokens", "details": {requests, total_tokens, cached_input_tokens,
  cache_write_input_tokens, reasoning_output_tokens}}`; Python emits
  `{"requests", "input_tokens", "output_tokens", "total_tokens",
  "input_tokens_details": {cached_tokens, cache_write_tokens},
  "output_tokens_details": {reasoning_tokens}}` (usage.py:415-427). `src/usage.rs` is not in this
  area's ownership and three model adapters call it, so this is recorded only; it changes exported
  generation-span data.
- **`serialize_usage` wire shape.** Python wraps the detail objects in a one-element *list*
  (`"input_tokens_details": [ {...} ]`, usage.py:429-441) and `deserialize_usage` accepts both a
  list and a bare object. Rust's serde derive writes a bare object. Round-tripping Rust→Rust is
  fine and the ported test asserts the Rust shape; a Python consumer reading a Rust-written usage
  blob is unaffected (Python's `_coerce_input_token_details` accepts a mapping), but a Rust
  consumer reading a Python-written blob would fail on the list. Fix belongs with whoever owns
  `src/usage.rs` / `src/run_state/wire.rs`: accept both shapes on read.

## Deviations in the ports (deliberate)

- Python names several agents `test` in the same test (`test_handoffs`, `test_structured_output`);
  the Rust ports use distinct names so the derived `transfer_to_<name>` tool stays unambiguous.
- `test_strict_schema::test_deeply_nested_schema_is_rejected...` builds a 1000-deep schema.
  Constructing a 1000-deep `serde_json::Value` overflows the 2 MiB test-thread stack in a debug
  build *before* the library is called (verified with a standalone probe that only builds and
  drops the value), so the port uses depth 300 — still far past `MAX_SCHEMA_DEPTH = 100`.
- `test_debug.py` monkeypatches process environment variables. `std::env::set_var` is `unsafe`
  in edition 2024 and races the other tests in the binary, so the port drives the equivalent
  `set_dont_log_*` / `reset_dont_log_*` override API instead.
- Tests that call `run_loop.*` / `run_internal.items.*` directly (`test_process_model_response`,
  `test_run_step_processing`, `test_run_internal_items`, `test_run_internal_approvals`, large
  parts of `test_agent_runner`) have no Rust equivalent entry point: `mod run_internal` is
  private (`src/lib.rs:54`). Their observable behaviour is asserted through `Runner` where a port
  exists; they are counted as *covered* only where an existing Rust test asserts the same result.

## Shared-file edits

None. `src/lib.rs` and `Cargo.toml` were not touched; no new dependency was added.

## Files changed

- `src/strict_schema.rs` (not in this area's listed ownership — see fixes 1-4; small targeted
  edits, no reformatting of unrelated code)
- `src/run_internal/mod.rs` (fix 5, 3 added lines)
- `src/agent.rs` (fix 6, one line; rustfmt left the rest of the file unchanged)
- `tests/py_core_a_pure.rs`, `tests/py_core_a_tool_behavior.rs`, `tests/py_core_a_hooks.rs`,
  `tests/py_core_a_runner.rs`, `tests/py_core_a_stream_events.rs`,
  `tests/py_core_a_tool_guardrails.rs`, `tests/py_core_a_agent_as_tool.rs` (new)

## Breaking changes

None. `strict_schema::ensure_strict_json_schema` now *accepts* schemas it used to reject
(fixes 1-2) and *rejects* two previously-accepted malformed shapes (fix 3: a singleton `allOf`
that contradicts its parent; fix 4: a `$ref` crossing a nested `$id` resource) — both match
Python and both were silent-wrong-answer paths before.

## Verification

- `cargo test --all-features` for `--lib` (639), the 7 `py_core_a_*` files (83+22+14+14+6+11+11 =
  161) and the existing consumers of the code I touched (`core_tool_output_schema`,
  `function_tool_macro_tests`, `runner_tests`, `local_tools_tests`, `core_run_state`,
  `core_python_run_state`, `core_tool_handoff_rows`): all green.
- `cargo clippy --all-features --all-targets -- -D warnings`: the lib target currently fails only
  in other agents' files (`src/extensions/models/mod.rs`, `src/models/openai_responses_ws.rs`,
  `src/sandbox/**`). A clippy run (without `-D`) over my 7 test targets plus `strict_schema.rs`,
  `run_internal/mod.rs`, `agent.rs` reports nothing in my files. One real clippy error in
  `strict_schema.rs` (identical `if` blocks) was fixed.
- No `#[ignore]` tests; nothing needed network or credentials.
- Several runs were blocked for minutes by other agents' in-flight edits (voice, sandbox); they
  were retried, never patched.

## Round 2: deferred bugs and run_internal gaps (all fixed)

Source fixes, all matching Python:

7. **Empty model turn** (`src/run_internal/mod.rs`, turn resolution): a turn with no message and no
   tool activity looped until `MaxTurnsExceeded`. It now ends a plain-text run with `""`
   (Python `final_output=potential_final_output_text or ""`); a refusal raises
   `ModelRefusalError` with or without an output type; an empty *structured* turn runs again, or
   reaches the invalid-final-output handler with "Model returned no final output for the
   structured output type." when one is set. A tool-not-found turn answered with
   `ReturnErrorToModel` counts as tool activity (`has_tool_activity`). Tests: `py_core_a_run_loop`.
8. **Tool-use tracker and agent identity** (new `src/run_internal/agent_identity.rs`,
   `RunState::current_agent_identity`, `wire.rs`): the tracker is keyed by agent object at runtime
   and by identity key (`name`, `name#2`, ...) in snapshots; identity follows Python's
   `_build_agent_identity_map` (root first, then configuration signature, then graph order,
   skipping literal `name#n` collisions). `current_agent` serializes `{name, identity}`; resume
   locates the agent by identity and still refuses a duplicate-name snapshot that has none.
9. **`ToolOrigin` / `ToolOriginType`** (`src/tool.rs`, `src/items.rs`, `wire.rs`, re-exported at the
   crate root): `RunItem` gains `description`, `title`, `tool_origin`; `ToolApprovalItem` gains
   `tool_origin`. Function tools, local MCP tools (log-safe server name, credentials stripped),
   hosted `mcp_call` and `Agent::as_tool` set it; malformed or unknown origins are dropped on read
   and legacy snapshots load with `None`. 11 of the 14 `test_tool_origin.py` tests ported (3 are
   Python-only).
10. **Pre-filter compaction exchange**: `notify_session_completed` recorded the history *before*
    `call_model_input_filter`, so hidden history was compacted. The loop now records the input the
    model actually saw plus its answer (`compaction_exchange`), and the `store` the call ran with
    (the `RunConfig` override), not `agent.model_settings.store`.
11. **Orphan output pruning**: `drop_orphan_function_calls_in` ports `pruning_indexes` /
    `output_pruning_indexes`; stored history loaded for a run is pruned (with a session limit,
    outputs whose call the limit cut off are dropped too).
12. **Deferred interruption write** (`save_session_through`, `register_held_write`,
    `RunState::pending_session_write`): when an output guardrail and a non-`run_llm_again` tool-use
    behavior make the parked response possibly terminal, the interruption writes only the input and
    earlier turns, records a held batch on the checkpoint, and the batch settles with the resolved
    turn in one ordered append (a call whose output a handoff filter dropped is dropped with it;
    calls still awaiting approval settle). A lost-ack append is recognized at the session tail and
    not repeated. Resume refuses another session id and refuses re-gating after current-turn items
    were persisted. The handoff filter sees the parked response as pre-handoff context, and an
    after-turn cancel no longer pre-empts a resumed turn that still owes approved tools.
13. **Guardrail crash**: a crashing output guardrail still persists the turn that ran (call and
    output together); a failing save wins over the guardrail error (Python precedence).
14. **Retry re-running an approved tool**: tool outputs committed before a failure (custom-data
    extractor error, later tool failure, a failing tool-use-behavior decision) are kept in
    history, and a failed run leaves a resumable checkpoint. Python mutates the `RunState` it was
    given, a Rust run consumes it, so the checkpoint is returned through the new
    `run::RunCheckpoint` slot (`RunOptions::checkpoint`); a retry re-processes the response and
    skips completed calls. New public API; no existing signature changed.
15. **Pre-approval tool-input guardrails** ran on every pass (3 times on a resumed run); they now
    run once before the approval is offered and once after (Python 2).
16. **`Agent::as_tool`** with only an `input_builder` now opts into structured input.

Un-ignored tests (now green): `py_deferred_writes_{a,b,d,e,compaction}` (all 35),
`py_memory_deferred_writes`, `py_memory_compaction_visibility` (2), `py_memory_limits_helpers`,
`py_mcp_runner_tracing::pre_approval_guardrails_run_once_before_and_once_after_approval`. Only
`#[ignore]` lines, stale "cross-area gap" doc comments and the guardrail that makes the deferral
apply (`always_fine`, as Python's helpers declare it) were edited in those files; the retry tests
read the checkpoint from `RunCheckpoint` where Python re-serializes the mutated state.

Still ignored (not the five run_internal gaps; belong to other areas): `py_mcp_resume_binding` (2:
approved MCP call not bound to its server; handoff-name reservation), `py_mcp_util_schemas` (5:
strict-schema reject-open-objects mode), `py_mcp_caching_approval` (3: `list_tools` refresh
generation in `src/mcp`).

## Not ported in this round

`test_run_state.py` (470 cases), `test_agent_runner.py` private-helper groups,
`test_tool_approval_call_id_reuse.py` (130), `test_run_internal_*`: these drive Python-private
helpers (`run_internal.*`, `RunState._...`, pydantic/dataclass shapes) or hosted-MCP/deferred-lookup
identity records that Rust does not model. `mod run_internal` stays private; I did not add a
`pub(crate)` test seam because the behavior each group asserts is reachable through `Runner` and
`RunState`, and the ported files cover it that way. Their counts stay in the "not ported" column.

## Verification (round 2)

- `cargo test --all-features --lib --tests`: everything green except other areas' in-flight sandbox
  tests (`py_sandbox_b_*`), which fail independently of this work.
- `cargo clippy --all-features --all-targets -- -D warnings`: the library and all `py_core_a_*`
  targets are clean; remaining errors are in other areas' test files
  (`py_memory_sqlx_compaction.rs:150`, `py_mcp_util_schemas.rs:237`).
- `StreamEvent` carries `#[allow(clippy::large_enum_variant)]` because `RunItem` grew three
  optional fields.

## Breaking changes (round 2)

`RunItem` and `ToolApprovalItem` gained fields (struct literals must add them; `RunItem::new` and
`ToolApprovalItem` builders are unaffected for callers using `new`), `RunState` gained
`current_agent_identity` and `pending_session_write` (both `#[serde(default)]`),
`strict_schema` behaviors as in round 1.

## Round 3: model-call gaps, sandbox hand-offs, stack use, sticky approvals

### Model call (`src/run_internal/model_call.rs`), 20 models tests un-ignored

17. **Retry-disable flags scoped per attempt** (`should_disable_provider_retries`,
    `should_disable_websocket_retry`): ports `_should_disable_provider_managed_retries` /
    `_should_disable_websocket_pre_event_retry`, so a provider sees `false` on the first stateless
    attempt and `true` on explicit no-retry budgets, stateful replays and every replay the runner
    owns. The task-local scope is only entered when the flag differs from what the task already sees
    (each scope costs a large debug poll frame).
18. **`replay_unsafe_request`**: set when a `ProgrammaticToolCalling` tool is present; disables hidden
    provider retries and websocket pre-event retries, skips the `conversation_locked` compatibility
    retry, and `retry::evaluate_retry_for` vetoes a provider-unsafe failure before the policy runs and
    any retry the provider does not approve afterwards. (`retry::evaluate_retry` stays as the
    no-veto wrapper for the unit tests.)
19. **`conversation_locked` with `max_retries=0`** is no longer retried.
20. **`ModelTimeoutError` / `ToolTimeoutError`** messages end with a period, as in Python.
21. **Tool-approval gating before transport**: `FunctionToolSpec` gains `needs_approval`; the hosted
    multi-agent model rejects approval-gated tools ("does not support SDK tool approval
    interruptions ... Remove needs_approval from these tools: a, b.") before any transport.
22. **Streamed usage entries**: the streamed path keeps `usage.requests` and
    `usage.request_usage_entries`, and a hosted logical pause counts no request of its own.

Un-ignored: `py_models_retry_stream` (all 16), `py_models_retry_get_response` (1),
`py_ext_hosted_multi_agent` (2). The live-network test stays ignored (needs an API key).

### Sandbox-a hand-offs

23. `StreamCompletion::last_agent_name` (result of the run, or the agent the stream had reached when
    it failed).
24. `RunOptions::prepare_agent` (new): awaited after the first turn's blocking input guardrails and
    before the first model call; the returned agent replaces the starting agent. A tripped guardrail
    ends the run without preparing. `tests/py_core_a_run_loop.rs::the_prepare_hook_runs_after_blocking_input_guardrails`.
    **Not switched over in `src/sandbox/runtime.rs`** (another agent is editing it): `SandboxRunner`
    still prepares up front. To finish: build the prepared graph inside the hook
    (`options.prepare_agent(move |_| async move { prepare(...).await; Ok(graph.starting_agent().clone()) })`)
    and read `completion.last_agent_name` in the streamed completion hook to pick the memory manager.

### Stack use (debug builds, thread stack at which each scenario still passes)

Scenarios (`tests/py_core_a_stack.rs`): a `Runner::run` with a tool call, a handoff and a final
message; the same traced and streamed; and `SandboxRunner` runs (plain, traced, streamed) with a tool
call. `cargo test --test py_core_a_stack -- --ignored measure_minimum_stack --nocapture` re-executes the
binary at falling stack sizes (an overflow aborts the process, so it cannot be probed in-process).

| scenario | before | after |
| --- | --: | --: |
| run | 1024 KiB | 640 KiB |
| run, tracing on | 1024 KiB | 640 KiB |
| run, streamed | 768 KiB | 448 KiB |
| sandbox run | 2048 KiB | 896 KiB |
| sandbox run, tracing on | 2048 KiB | 896 KiB |
| sandbox run, streamed | 768 KiB | 768 KiB |

Findings: the runner futures themselves are small (`Runner::run` 1.6 KB, `SandboxRunner::run` 37 KB);
the cost is the poll-frame chain in debug builds (individual `async` frames of 5 to 20 KB, and each
tokio task-local scope adds two ~15 KB frames). Fixes: `RunResult::state` is boxed (RunResult 1928 to
about 420 bytes, so every `Result<RunResult>` temporary shrinks), and the retry-flag task-locals are
no longer entered when they would not change anything. Boxing the already-boxed turn futures further
does not help: stack depth is the sum of the live poll frames, not future size.
The `on_a_big_stack` helper (32 MiB thread) is gone from `tests/py_sandbox_a_runtime_tracing.rs`;
both of its tests now run on the default 2 MiB test thread. New regression tests run each scenario on a
1 MiB thread (`a_*_fits_in_one_mebibyte`, 6 tests, 2 gated on `sandbox`). The sandbox scenarios have
12.5 percent margin at 1 MiB in a debug build; release builds use far less.

### Approval keying (`src/run_context.rs`, `src/run_internal/local_tools.rs`, `src/items.rs`)

25. Sticky approvals for `apply_patch` are scoped by invocation type (`apply_patch:apply_patch_call`,
    `apply_patch:custom_tool_call`, `apply_patch:function_call`), as
    `RunContextWrapper._approval_keys_for_scope` does; lookups fall back to the bare name for
    decisions recorded before scoping. New public `scoped_tool_key` / `approval_tool_key`. Python's
    scope is a payload hash; Rust's is the type, which separates the same tools but means a Python
    snapshot's scoped `apply_patch` entry will not match (the bare fallback still applies). The shared
    `RunContext` applies it to the runner, `RunState::approve/reject` and realtime alike; agents with
    the same name still share a key (unchanged).
26. `FunctionCall::to_value` no longer overwrites a stand-in call's own `type`, so an interrupted
    custom/native tool call serializes as `custom_tool_call` / `apply_patch_call` (Python's wire
    shape) and keeps its approval scope across a snapshot round trip.
    Tests: `py_core_a_sticky_approvals.rs` (5 key/round-trip tests plus the 16-case native-sticky
    matrix of `test_multi_operation_checker_ignores_native_sticky_approval`).

## Verification (round 3)

- `cargo test --all-features --lib --tests`: green except the in-flight
  `sandbox::agent::tests::prepares_an_agent_with_capability_tools_and_instructions` (capability tool
  order, sandbox area).
- `cargo clippy --all-features --all-targets -- -D warnings`: lib and every `py_core_a_*` target
  clean; the remaining errors are in other areas (`src/sandbox/http.rs:110`,
  `py_sandbox_a_unix_local_1.rs`, `py_mcp_runner_tracing.rs`, `py_memory_redis.rs`).

## Breaking changes (round 3)

`FunctionToolSpec` gains `needs_approval` (literals must use `..Default::default()`);
`StreamCompletion` gains `last_agent_name`; `ModelTimeoutError` / `ToolTimeoutError` messages gain a
trailing period; `RunOptions` gains `checkpoint`/`prepare_agent` (private fields, builder methods).

Status: DONE_WITH_CONCERNS
Summary: Model-call gaps (flags, replay_unsafe_request, conversation_locked, timeout message,
approval gating, streamed usage) fixed with 20 models tests un-ignored; stack use cut roughly in half
(sandbox run 2048 to 896 KiB) with 1 MiB regression tests and the sandbox big-stack workaround removed;
apply_patch approvals scoped by invocation type with the native-sticky matrix ported; earlier rounds
as above.
Concerns/Blockers: `SandboxRunner` does not yet use `RunOptions::prepare_agent` /
`last_agent_name` (sandbox-owned `runtime.rs`); the sandbox run has only ~12 percent stack margin at
1 MiB in debug; the huge unported groups (`test_run_state`, `test_tool_approval_call_id_reuse`,
`test_run_internal_*`) remain unported; workspace clippy is red only in other areas' files; one
sandbox unit test is failing from another agent's in-flight edit.
