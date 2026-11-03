# Python test parity, round 9, r9h2 (top-level run-state and runner files)

13 files, 274 python-only cases before. Every case was read in the Python test and matched against the Rust tests (by name, by docstring citations and by behaviour). Result: 10 ported, 7 covered, 257 remain valid python-only. Three Rust source bugs were found and fixed on the way.

## Per-file table

| file | python_only before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| test_released_api_contract.py | 126 | 0 | 0 | 126 | tests of the Python repo's `integration_tests` release-contract tooling (inspect/typing/pydantic introspection of the Python package, Python-only CI) |
| test_function_tool_decorator.py | 50 | 0 | 0 | 50 | `@function_tool` on callable objects, `__wrapped__`/`inspect.unwrap` identity, class-scoped annotations, pydantic `ImageGenerationTool` parameter schema, keyword-only signature (`inspect.signature`) |
| test_function_tool.py | 24 | 0 | 0 | 24 | asyncio (`to_thread`, event-loop blocking, sibling cancel on a raising `is_enabled`, `CancelledError` normalization), monkeypatch of module defaults, `dataclasses.replace` / `copy.copy` / `deepcopy` of Python tool objects and subclasses, `**kwargs` / `*args` signatures, awaitable `failure_error_function` (see questions) |
| test_run_config.py | 22 | 2 | 4 | 16 | `RunConfig(**dict)` / `run_config={...}` dictionary coercion and unknown-key errors for `tool_execution`, string-literal validation of the collision policy (Rust enum), non-string blocked-message values, object identity of typed settings, `ModelSettings.__setattr__` tracking of in-place verbosity writes (see questions) |
| test_hitl_error_scenarios.py | 15 | 3 | 0 | 12 | falsy `__bool__` pending items (2), dynamically typed `needs_approval="always"` (2), `set_public_agent` clone identity (3), same-named sibling agents by object identity (1), second policy evaluation in a hand-built `_resolve_interrupted_turn` (2), deferred lookup key with a missing raw namespace in a hand-built state (1), `ToolApprovalItem` hashability (1) |
| test_run_internal_items.py | 13 | 2 | 3 | 8 | `json.dumps` monkeypatch (1), legacy `tool_search_result` type string (1), `Sequence` read counting (1), objects/mappings that are not valid typed items (2), `normalize_resumed_input` no-op on clean mappings (1), `extract_mcp_request_id` on arbitrary objects/dicts with `provider_data` (2) |
| test_handoff_tool.py | 6 | 0 | 0 | 6 | callable-object `__call__` dispatch (1), signature-checked / non-callable `on_handoff` and `input_type` without `on_handoff` (3, typed builder), asyncio cancel of sibling `is_enabled` (1), pydantic lax-mode `validate_json` (1) |
| test_run_serial_tests.py | 6 | 0 | 0 | 6 | the repo's `.github/scripts/run_serial_tests.py` CI script |
| test_guardrails.py | 3 | 0 | 0 | 3 | non-callable guardrail (2, typed closure), `patch` of a private compat switch (1) |
| test_output_tool.py | 3 | 3 | 0 | 0 | |
| test_run_hooks.py | 3 | 0 | 0 | 3 | hooks of the wrong class (typed), `validate_run_hooks(object())` |
| test_release_please.py | 2 | 0 | 0 | 2 | Python release metadata / workflow YAML |
| test_hitl_utils.py | 1 | 0 | 0 | 1 | test helper class of the Python suite |

## Ported (10), `tests/py_core9_r9h2_core.rs`

- test_output_tool (3): `a_generic_mapping_output_is_not_wrapped`, `a_generic_mapping_output_rejects_the_wrapper_shape`, `output_schema_names_read_as_the_type` (the Literal-name cases: the name is the readable type, `Vec<Verdict>`, `HashMap<String, i64>`), plus `the_agent_debug_form_shows_the_readable_output_type`.
- test_hitl_error_scenarios (3): `an_exact_mcp_retry_reuses_the_stored_decision` (shared run context, two runs), `an_abandoned_mcp_callback_is_not_retried` (the Rust analogue of a failing callback: the run is dropped mid-callback), `a_resume_keeps_the_saved_item_count` (sqlite feature; streamed and not).
- test_run_config (2): `a_serialized_manifest_without_path_grants_is_accepted_as_configuration` (python dump and json dump, sandbox feature).
- test_run_internal_items (2): unit tests in `src/run_internal/oai_conversation.rs`: `a_fingerprint_can_ignore_the_item_id`, `a_fingerprint_ignores_internal_tool_call_metadata`.

Extra tests that move no count (the cases were already in the ported bucket under other names): `a_default_handoff_name_with_whitespace_does_not_warn` (found the handoff warning bug below), the three `get_all_tools` + `ToolSearchTool` cases of test_function_tool, and the decorator timeout/defer-loading configuration.

## Covered (7)

- test_run_config: blocked message `""` -> `py_core2_runconfig_settings.rs::run_config_rejects_empty_output_guardrail_blocked_message`; untrusted manifest path grants -> `py_core8_r8g1_manifest.rs::dictionary_manifest_rejects_non_empty_path_grants_but_allows_an_empty_list`; unknown model settings -> `py_core5_leftover2_agent_config.rs` (`Unknown model settings: temperatur`); unknown session settings -> `py_core7_c_memory_settings.rs` (`unknown field limitt`).
- test_run_internal_items (3): occurrence helpers, by behaviour through `Runner`, in `py_core3_handoffs_owned_replay.rs`: `nested_history_filter_does_not_own_an_equal_replacement` (exact input occurrence, ambiguous equal occurrences after a rewrite), `identical_messages_across_turns_are_all_kept` (fully owned equal occurrences kept).

## Source bugs fixed

1. `src/agent_output.rs`: `AgentOutputSchema::name()` for a generic type was the last `::` segment of `std::any::type_name`, so `Vec<String>` was named `String>` (visible as `output_type` in agent spans and `Debug`). Names are now the type with every module path dropped (`Vec<String>`, `HashMap<String, i64>`).
2. `src/handoffs.rs`: `default_tool_name` warned about whitespace in the agent name (`Refund agent`); Python passes `warn_on_whitespace=False` and warns only for other invalid characters.
3. `src/run_internal/mod.rs` (hosted MCP approval requests): a callback was invoked every time, ignoring a decision already on the run context and recording nothing. Now a stored decision (from an earlier answer, or set by the caller) wins and the callback is not run; the callback's answer is recorded on the context; a callback that started but never recorded its answer is not re-run for the same request (`already ran ... not committed`, the pre-existing uncommitted-invocation check refuses first).

## Breaking changes

- `AgentOutputSchema::name()` strings change for generic output types (`String>` -> `Vec<String>`); nothing in the repo asserted the old text.
- A hosted MCP approval callback is no longer called for a request id that already has a decision on the context.

## Unresolved questions

1. `FunctionTool` / handoff / MCP `failure_error_function` is a sync closure; Python allows an awaitable handler (`test_async_failure_error_function_is_awaited`, counted python-only in test_function_tool). A feature gap rather than a Python-only behaviour: supporting it needs an async error-handler type across the tool, handoff, MCP and agent-as-tool call sites.
2. In-place writes to `agent.model_settings` (`test_equal_in_place_assignment_preserves_explicit_settings`, `field="verbosity"`, 6 cases): Python tracks the write and then keeps the captured default model settings; Rust cannot tell a field write from a replacement and fills unset fields from the selected model's defaults. Making any non-empty settings "touched" would also change the meaning of `agent.model_settings = ModelSettings {..}`; left as is.
3. A hosted MCP callback panic or abandoned future is the only way the "callback failure" case arises in Rust (the callback is infallible by type); the abandoned-future variant is tested.

## Gates

- `cargo test --all-features` on `py_core9_r9h2_core` (13 passed), the unit tests of the lib (658 passed) and the binaries that touch the changed code (`py_core2_approvals_binding`, `py_core2_scopedapprovals_hitl`, `py_core2_scopedapprovals_hitl_decisions`, `py_core2_approvals_reuse_b`, `py_core2_runstep_execution`, `py_core2_runstep_processing`, `py_core2_runstate_sidecars`, `py_core2_runner_routing`, `py_core_a_tool_origin`, `py_core6_streamedleft_legacy_mcp`, `py_core8_r8g7_run_state_binding`, `py_core3_handoffs_tool`, `py_core_b_handoff_tool`, `py_core_a_pure`, `py_core_a_tool_behavior`, `py_core3_tracing_errors`, `py_core4_tracerunner_agent`): all pass.
- clippy on the lib plus `py_core9_r9h2_core`, both feature sets, `-D warnings`: clean for my files. The lib itself currently fails `-D warnings` on two other agents' in-flight lines (`src/run_state/wire.rs:2656` needless borrow, `src/sandbox/agent.rs:756` type complexity); with those two lints allowed both feature sets finish clean. The default-feature build of the test file has no unused imports (sqlite and sandbox items are gated).
- rustfmt applied to `tests/py_core9_r9h2_core.rs` and `src/agent_output.rs`. `src/run_internal/mod.rs` has a pre-existing rustfmt difference at line 3256 that is not mine and was left alone.

Status: DONE_WITH_CONCERNS
Summary: 274 python-only cases read across 13 files: 10 ported, 7 covered, 257 remain valid python-only (CSV rows balance). 3 Rust source bugs fixed (output-schema type names, handoff whitespace warning, hosted MCP callback decision reuse).
Concerns:
- The `get_all_tools` + `ToolSearchTool` ports are not counted as moves: counting the python-only candidates of `test_function_tool.py` one by one gives exactly 24 without them, so the earlier report must have counted them as ported. If they were in the 24, that row is 21 remaining and ported goes to 3.
- The hosted MCP change makes a stored decision win over the callback (a decision restored without a ledger binding does not count). Another round-9 file that registers hosted MCP callbacks was not found to assert otherwise.
- Questions 1 and 2 above (async `failure_error_function`, in-place `model_settings` writes) are real Rust divergences left unchanged.
