# Python test parity — core round 2, area `runner`

Scope: `tests/test_agent_runner.py`, `tests/test_agent_runner_streamed.py`.

## How counted

`pytest --collect-only` in the scratchpad venv (pytest installed there, no network): **232** cases in
`test_agent_runner.py` (195 defs; the AST count of 197 includes the `@function_tool`-decorated
`test_tool_one/two` helpers, which pytest does not collect) and **208** in
`test_agent_runner_streamed.py` (89 defs). The parametrize ids are folded per def. The ledger maps
every collected def to one bucket (split defs listed below); the generator script
`scratchpad/c2r/ledger.py` checks that no def is unmapped.

## Per-file table

| Python file | cases | ported | covered | python-only | deferred (cross-area) | failures found |
| --- | --: | --: | --: | --: | --: | --: |
| `test_agent_runner.py` | 232 | 167 | 17 | 48 | 0 | 13 |
| `test_agent_runner_streamed.py` | 208 | 132 | 15 | 27 | 34 | 6 |
| **total** | **440** | **299** | **32** | **75** | **34** | **19** |

Parametrized Python cases are iterated inside one Rust test (e.g. `streamed` x `policy`), so 299
ported cases map to 144 Rust tests.

## Rust test files added (144 tests, all green)

| file | tests |
| --- | --: |
| `tests/py_core2_runner_routing.rs` | 27 |
| `tests/py_core2_runner_session_input.rs` | 30 |
| `tests/py_core2_runner_approvals.rs` | 15 |
| `tests/py_core2_runner_handoffs.rs` | 14 |
| `tests/py_core2_runner_guardrail_sessions.rs` | 20 |
| `tests/py_core2_runner_streamed.rs` | 30 |
| `tests/py_core2_runner_extra.rs` | 8 |

Private-helper tests (`prepare_input_with_session`, `save_result_to_session`,
`execute_approved_tools`, `normalize_resumed_input`, …) are ported through the public API (real
runs; "prepared input" = first model request, "session items" = what the run appended). No
`#[doc(hidden)]` re-export was needed. Expected values that were not obvious were checked by running
the same public-API flow on the Python SDK (scratchpad `c2r/py/probe*.py`).

## Python-only groups (75)

- `_retained_items_for_blocked_output` / snapshot-owner unit tests (dict-key hash/eq impostors,
  monkeypatched BaseException/cancellation) — 15. Observable outcome ported.
- Python truthiness / legacy positional args / env monkeypatch (`falsey runner`, `falsy agent`,
  `legacy positional previous_response_id`, `trace_include_sensitive_data` env) — 4.
- pydantic `model_dump` helpers, fingerprint of broken model_dump — 3.
- non-callable / wrong-type input filters, wrong `on_handoff` arity — 5 (types forbid them).
- `session_input_callback` identity semantics (callback mutating/aliasing the objects it got,
  non-callable callback, private include_history flags) — 7.
- session rewind / cleanup-verification helpers (monkeypatched sleep/logger) — 11: Python writes
  input before the model call and must rewind on a `conversation_locked` retry; Rust writes the input
  with the first turn's output, nothing to rewind (verified by `a_model_retry_does_not_rewind_…`).
- cancellation of the run task during an output guardrail (Python saves in `except CancelledError`;
  a dropped Rust future runs no code) — 1.
- hand-built approval items the public API cannot express (non-function ComputerTool approved as
  function call, missing call_id, dict raw_item, approval for a removed tool) — 4.
- streamed internals: monkeypatched finalizer logger records, private event queue/_stored_exception,
  `stream_step_items_to_queue`, monkeypatched `save_resumed_turn_items`, asyncio
  cancel/BaseException semantics of session saves and immediate cancel — 8.
- `OpenAIResponsesWSModel` monkeypatched websocket internals — 3.
- `output_guardrail_blocked_message` formatter returning non-string / str subclass / awaitable /
  raising / cancelling (Rust formatter is typed `Fn -> Option<String>`) — 12 of the 16 cases of
  `test_terminal_tool_trip_falls_back_when_blocked_message_formatter_fails` (none/empty ported).

## Covered (32)

`py_core_a_runner`, `py_core_a_streamed`, `runner_tests::input_guardrail_tripwire_stops_run`,
`py_core2_approvals_reuse_a` (changed pending invocation), `py_memory_conversations` (Rust strips
conversation ids/placeholder reasoning inside `OpenAIConversationsSession::add_items`; Python does it
in `save_result_to_session` — 7 cases).

## Deferred, cross-area (34)

- `test_serialized_mixed_approval_guardrail_preserves_only_accepted_outputs` (16),
  `test_ambiguous_serialized_approval_state_fails_before_tool_execution` (10): Python refuses
  corrupted / legacy-1.16 ownership snapshots with "current response boundary cannot be proven".
  `src/run_state/wire.rs` has no such validation (Rust resumes them). Fix belongs to the run_state
  area: validate `current_response_generated_item_ownership` against `generated_items` /
  `last_processed_response` / `generated_session_item_indexes` in `from_wire` (or on resume) and
  raise `UserError("… current response boundary cannot be proven …")` before any tool runs. The
  valid-resume halves of the sibling tests (`later_turn`, `filtered_handoff`) are ported.
- `test_chat_completions_output_guardrails_use_adapter_conversation_policy` (8): models area —
  `OpenAIChatCompletionsModel` strict feature validation / "Ignoring unsupported server-managed
  conversation state" warning, tested through monkeypatched `get_response`.

## Rust bugs fixed (source)

All in `src/run_internal/mod.rs` unless noted.

1. Resumed input not normalized: restore now sets the model input to the snapshot input minus
   orphan calls (`normalize_resumed_input`).
2. Tool-name collisions: Python's "Ambiguous {agent tool|handoff|agent routing|function tool|tool
   routing} configuration…" messages (derived agent-name pairs via `as_tool`/handoff defaults) and
   the redacted warning under `DONT_LOG_TOOL_DATA`.
3. `src/run_internal/items.rs` `drop_orphan_function_calls_in`: anonymous `tool_search_call`s pair
   positionally with later anonymous outputs (unpaired ones dropped); program-owned pending hosted
   shell calls kept.
4. `src/run_internal/tools.rs` `execute_function_tools`: FIFO refill under
   `max_function_tool_concurrency` (was `buffered`, which head-of-line blocked), lowest-order
   failure wins. (Another agent later reworked the same loop by hand on top of this.)
5. `prepare_input_with_session` port in `load_session_history`: callback output matched back to
   history/new input by content and frequency so only new-turn items (incl. callback extras) are
   persisted; remote-history assistant messages lose `id`/`provider_data`; latest duplicate output
   wins.
6. Agent-start hooks and the dynamic prompt run before handoff `is_enabled` is evaluated (prompt
   prefetched, not re-resolved); `src/agent.rs`: MCP reserved names (which evaluate handoff
   `is_enabled`) computed only when `include_server_in_tool_names` and MCP servers exist.
7. Input-guardrail trip persists the user input to the session.
8. `reasoning_item_id_policy=omit` strips reasoning ids on session save (local stores) and in
   `RunResult::to_input_list` (`src/result.rs`).
9. A resumed turn that runs the model again no longer reports its tool guardrail results (Python
   accounting); terminal / handoff / interruption resumes still do.
10. Nested handoff history: `RunResult.input` / state `original_input` become the nested history;
    items forwarded verbatim are excluded from `to_input_list()` (preserve_all); normalized view
    replays model history; model-history start persisted as `RunState.model_input_start`.
11. Server-managed handoff input filter error uses Python's text.
12. `block_output`: a tripped terminal response containing reasoning or a non-function tool variant
    is dropped whole (history, model transcript, response, turn tool-output guardrail results),
    with `agent_output` = placeholder.
13. `src/run_internal/model_call.rs`: `response.incomplete`/`response.failed`/`error`/
    `response.error` stream events fail the run with Python's terminal-failure message after the
    raw event is delivered (was "stream ended without response.completed").
14. `src/testing.rs`: ScriptedModel stream carries the step's `request_id` on the terminal event.
15. Session + resumed state carrying conversation_id/previous_response_id/auto is rejected.
16. `src/run_internal/oai_conversation.rs`: items already delivered in this run are not resent
    under `conversation_id` (fingerprint check no longer limited to state-primed runs).

## Shared / cross-area edits

- `src/run_state.rs` (+ field `model_input_start: usize`, `serde(default)`), `src/run_state/wire.rs`
  (`rust.model_input_start` write/read) — run_state area, additive.
- `src/agent.rs` `get_all_tools` (MCP reserved-name gating).
- `src/result.rs` (`nested_history_owned_items` field, reasoning-id policy in `to_input_list_with`).
- `src/testing.rs` (`completed_response_value` request id).
- `tests/runner_tests.rs::tool_name_collisions_follow_policy`: assertion updated to the Python
  message ("Ambiguous tool routing configuration").
- No edits to `src/lib.rs` or `Cargo.toml`. No breaking API change beyond the additive pub field on
  `RunState` (struct literals without `..Default::default()` would need it).

## Gates

- Own 7 binaries: 144/144 pass. `cargo test --all-features --lib`: 642 pass.
- Regression set run after changes: `runner_tests` (except below), `py_core_a_runner/streamed/run_loop`,
  `core_run_state`, `core_python_run_state`, `core_audit_fixes_handoff`, `core_server_conversation`,
  `py_memory_session*` — green.
- clippy `--all-features` on lib + own test targets: no warnings in my files. `-D warnings` currently
  fails only on other areas' in-flight code: `src/run_internal/tool_invocation.rs:328`,
  `src/run_state/sandbox_payload.rs:363`.
- Failing at report time, not from my files (source edited by other agents during the round):
  `runner_tests::structured_output_is_validated` (invalid-JSON error now data-redacted →
  no `run_data`; `src/error.rs`/`src/agent_output.rs` changed 10:54), `py_deferred_writes_e` (3, tool
  callback-failure retry), `py_core2_toolerrors_batch` (1), `py_core2_approvals_reuse_a/b` (3, in-flight
  DBG prints). Workspace run: 5215 passed / 11 failed before my follow-up fixes (4 of the 11 were mine
  and are fixed: lib oai_conversation unit test, `py_memory_session*` repeating-callback x2,
  `runner_tests` collision message).
- rustfmt applied to my test files only. Build dir deleted.

## Known gaps

- `nested_history_owned_items` is not serialized; a state resumed after a nested handoff replays
  verbatim-forwarded items in `to_input_list()` (preserve_all) once.
- Callback-output matching is content based; Python's identity-only cases are python-only above.

## Ledger (def → cases, bucket, Rust location / reason)

| file | Python def | cases | bucket | where / why |
| --- | --- | --: | --- | --- |
| runner | `test_async_handoff_on_input` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_async_input_filter_supported` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_auto_previous_response_id_multi_turn` | 1 | ported | py_core2_runner_streamed |
| runner | `test_auto_previous_response_id_multi_turn_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_auto_previous_response_id_retries_when_policy_approves_unsafe_replay` | 1 | ported | py_core2_runner_extra |
| runner | `test_blocked_function_batch_accepts_exact_typed_direct_caller` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_function_batch_ignores_hash_collision_key_hooks` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_function_batch_is_rebuilt_from_allowlisted_fields` | 2 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_function_batch_rejects_equality_impostor_discriminators_without_hooks` | 3 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_function_batch_rejects_non_direct_typed_callers` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_reasoning_item_discards_the_complete_response` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_snapshot_application_baseexception_severs_every_owner` | 2 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_snapshot_application_exception_becomes_fixed_error` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_snapshot_cancellation_severs_replay_graph_and_propagates` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_snapshot_preserves_accepted_prefix_with_reused_provider_id` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_blocked_unknown_tool_variant_discards_the_complete_response` | 1 | python-only | private _retained_items_for_blocked_output / snapshot owner-plan unit tests with dict-key hash/eq impostors, BaseExcepti |
| runner | `test_call_model_input_filter_can_reintroduce_reasoning_ids` | 1 | ported | py_core2_runner_routing |
| runner | `test_collect_retry_owned_tail_serializations_returns_empty_for_empty_session` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_collision_warning_preserves_tool_diagnostics_when_enabled` | 1 | ported | py_core2_runner_routing |
| runner | `test_collision_warning_redacts_tool_data` | 1 | ported | py_core2_runner_routing |
| runner | `test_conversation_id_only_sends_new_items_multi_turn` | 1 | ported | py_core2_runner_streamed |
| runner | `test_conversation_id_only_sends_new_items_multi_turn_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_conversation_lock_rewind_skips_when_no_snapshot` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_default_multi_turn_drops_orphan_hosted_shell_calls` | 1 | ported | py_core2_runner_streamed |
| runner | `test_default_multi_turn_streamed_drops_orphan_hosted_shell_calls` | 1 | ported | py_core2_runner_streamed |
| runner | `test_default_send_all_items` | 1 | ported | py_core2_runner_streamed |
| runner | `test_default_send_all_items_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_default_trace_include_sensitive_data_env` | 1 | python-only | Python truthiness (`__bool__` falsy runner/agent), legacy positional-argument call signature, env-var default via monkey |
| runner | `test_deferred_collision_rejection_prefers_explicit_message` | 1 | ported | py_core2_runner_approvals |
| runner | `test_dynamic_tool_addition_run` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_ensure_api_input_item_avoids_pydantic_serialization_warnings` | 1 | python-only | model_dump/pydantic conversion helpers and fingerprint failure of a broken model_dump object |
| runner | `test_ensure_api_input_item_handles_model_dump_objects` | 1 | python-only | model_dump/pydantic conversion helpers and fingerprint failure of a broken model_dump object |
| runner | `test_ensure_api_input_item_preserves_object_output` | 1 | ported | py_core2_runner_session_input |
| runner | `test_execute_approved_tools_does_not_fallback_from_namespaced_approval_to_bare_tool` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_does_not_resolve_explicit_namespaced_tool_by_bare_name` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_instance_method` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_prefers_visible_top_level_function_over_deferred_same_name_tool` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_rejects_changed_pending_invocation` | 1 | covered | py_core2_approvals_reuse_a (changed pending invocation under a reused call id is rejected with 'unique call ID') |
| runner | `test_execute_approved_tools_rejects_missing_call_id` | 1 | python-only | hand-built approval items the public API cannot produce: a non-function ComputerTool approved as a function call, a raw_ |
| runner | `test_execute_approved_tools_timeout_can_raise_exception` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_timeout_returns_error_as_result` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_uses_internal_lookup_key_for_deferred_top_level_calls` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_uses_last_duplicate_top_level_function` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_invalid_raw_item_type` | 1 | python-only | hand-built approval items the public API cannot produce: a non-function ComputerTool approved as a function call, a raw_ |
| runner | `test_execute_approved_tools_with_missing_tool` | 1 | python-only | hand-built approval items the public API cannot produce: a non-function ComputerTool approved as a function call, a raw_ |
| runner | `test_execute_approved_tools_with_non_function_tool` | 1 | python-only | hand-built approval items the public API cannot produce: a non-function ComputerTool approved as a function call, a raw_ |
| runner | `test_execute_approved_tools_with_rejected_deferred_tool_uses_display_name` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_rejected_tool` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_rejected_tool_formatter_none_uses_default` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_rejected_tool_prefers_explicit_message` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_rejected_tool_uses_run_level_formatter` | 1 | ported | py_core2_runner_approvals |
| runner | `test_execute_approved_tools_with_unclear_status` | 1 | ported | py_core2_runner_approvals |
| runner | `test_falsey_per_handoff_input_filter_takes_precedence` | 2 | ported | py_core2_runner_handoffs |
| runner | `test_fingerprint_input_item_returns_none_when_model_dump_fails` | 1 | python-only | model_dump/pydantic conversion helpers and fingerprint failure of a broken model_dump object |
| runner | `test_function_tool_queue_preserves_fifo_across_multiple_refills` | 1 | ported | py_core2_runner_routing |
| runner | `test_get_new_response_uses_agent_retry_settings` | 1 | ported | py_core2_runner_session_input |
| runner | `test_handoff_enablement_uses_initialized_turn_context` | 2 | ported | py_core2_runner_routing |
| runner | `test_handoff_filters` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_handoff_on_input` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_handoffs` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_input_guardrail_no_tripwire_continues_execution` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_input_guardrail_tripwire_does_not_save_assistant_message_to_session` | 1 | ported | py_core2_runner_session_input |
| runner | `test_input_guardrail_tripwire_triggered_causes_exception` | 1 | covered | runner_tests::input_guardrail_tripwire_stops_run |
| runner | `test_invalid_handoff_input_json_causes_error` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_invalid_input_filter_fails` | 1 | python-only | input filter returning a non-HandoffInputData / non-callable filter / on_handoff with wrong arity: ruled out by Rust's t |
| runner | `test_manual_pending_shell_call_input_is_preserved_non_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_manual_pending_shell_call_input_is_preserved_non_streamed_with_session` | 1 | ported | py_core2_runner_streamed |
| runner | `test_manual_pending_shell_call_input_is_preserved_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_manual_pending_shell_call_input_is_preserved_streamed_with_session` | 1 | ported | py_core2_runner_streamed |
| runner | `test_model_settings_override` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_multi_turn_previous_response_id_passed_between_runs` | 1 | ported | py_core2_runner_streamed |
| runner | `test_multiway_mixed_collision_reports_every_owner_must_be_unique` | 1 | ported | py_core2_runner_routing |
| runner | `test_nested_handoff_filters_model_input_but_preserves_session_items` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_nested_handoff_filters_reasoning_items_from_model_input` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_non_callable_input_filter_causes_error` | 1 | python-only | input filter returning a non-HandoffInputData / non-callable filter / on_handoff with wrong arity: ruled out by Rust's t |
| runner | `test_non_streamed_model_retry_does_not_rewind_committed_session_input` | 2 | ported | py_core2_runner_session_input |
| runner | `test_non_streamed_trip_preserves_prior_run_state_side_effect` | 1 | ported | py_core2_runner_extra |
| runner | `test_non_streamed_trip_uses_safe_items_for_sandbox_memory_after_session_failure` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_normalize_resumed_input_drops_orphan_function_calls` | 1 | ported | py_core2_runner_routing |
| runner | `test_normalize_resumed_input_drops_orphan_tool_search_calls` | 1 | ported | py_core2_runner_routing |
| runner | `test_normalize_resumed_input_matches_latest_anonymous_tool_search_call` | 1 | ported | py_core2_runner_routing |
| runner | `test_normalize_resumed_input_preserves_hosted_tool_search_pair_without_call_ids` | 1 | ported | py_core2_runner_routing |
| runner | `test_omit_policy_strips_reasoning_ids_already_stored_in_the_session` | 1 | ported | py_core2_runner_routing |
| runner | `test_opt_in_handoff_history_accumulates_across_multiple_handoffs` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_opt_in_handoff_history_nested_and_filters_respected` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_output_guardrail_cancellation_preserves_final_output_in_session` | 1 | python-only | cancelling the run task while the output guardrail is pending: Python saves in an `except CancelledError` handler; a dro |
| runner | `test_output_guardrail_error_preserves_final_output_in_session` | 2 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_output_guardrail_no_tripwire_continues_execution` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_output_guardrail_tripwire_does_not_save_assistant_message_to_session_sync` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_output_guardrail_tripwire_triggered_causes_exception` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_parallel_tool_call_with_cancelled_sibling_reaches_final_output` | 1 | ported | py_core2_runner_routing |
| runner | `test_pending_approval_skips_tool_input_guardrails_by_default` | 1 | ported | py_core2_runner_routing |
| runner | `test_persist_session_items_for_guardrail_trip_uses_original_input_when_missing` | 1 | ported | py_core2_runner_session_input |
| runner | `test_persistent_hosted_mcp_approval_does_not_cross_servers` | 2 | ported | py_core2_runner_routing |
| runner | `test_pre_approval_tool_input_guardrails_can_reject_before_pending_approval` | 1 | ported | py_core2_runner_routing |
| runner | `test_pre_approval_tool_input_guardrails_rerun_after_resume` | 1 | ported | py_core2_runner_routing |
| runner | `test_prepare_input_with_openai_conversation_callback_keeps_user_ids_distinct` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_openai_conversation_callback_matches_assistant_no_ids` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_openai_conversation_strips_assistant_history_ids` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_regular_session_preserves_history_ids` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_async_callback_moves_repeated_history_item` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_awaits_async_callback` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_callback_accepts_extra_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_callback_drops_new_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_callback_reorders_new_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_callback_replaces_history_item` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_drops_limited_orphan_history_function_call_outputs` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_drops_orphan_function_calls` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_drops_output_for_program_owned_call_pruned_with_parent` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_extra_reconstructed_history_item_stays_new` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_history_moved_to_new_input_stays_history` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_ignores_callback_without_history` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_keeps_orphan_output_without_limit` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_keeps_paired_history_function_call_outputs` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_leaves_custom_callback_output_unchanged` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_matches_copied_items_by_content` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_prefers_latest_function_call_output` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_preserves_new_function_call_outputs` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_preserves_pending_new_shell_calls` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_rejects_non_callable_callback` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_rejects_non_list_callback_result` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_repeated_history_keeps_equal_new_item` | 1 | python-only | session_input_callback matching by Python object identity (callback mutating/aliasing the list objects it received, a no |
| runner | `test_prepare_input_with_session_sanitizes_new_tool_call_session_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_strips_internal_tool_call_metadata` | 1 | ported | py_core2_runner_session_input |
| runner | `test_prepare_input_with_session_uses_sync_callback` | 1 | ported | py_core2_runner_session_input |
| runner | `test_previous_response_id_only_sends_new_items_multi_turn` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_only_sends_new_items_multi_turn_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_passed_between_runs` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_passed_between_runs_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_passed_between_runs_streamed_multi_turn` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_retry_does_not_resend_initial_input_multi_turn` | 1 | ported | py_core2_runner_streamed |
| runner | `test_previous_response_id_retry_does_not_resend_initial_input_multi_turn_streamed` | 1 | ported | py_core2_runner_streamed |
| runner | `test_reasoning_item_id_policy_omits_follow_up_reasoning_ids` | 1 | ported | py_core2_runner_routing |
| runner | `test_resume_preserves_filtered_model_input_after_handoff` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_resumed_final_output_persists_once_after_passing_output_guardrail` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_resumed_final_tool_persists_call_and_output_after_output_guardrail` | 2 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_resumed_run_uses_serialized_reasoning_item_id_policy` | 1 | ported | py_core2_runner_routing |
| runner | `test_resumed_state_updates_agent_after_handoff` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_resumed_stream_debug_extra_preserves_falsy_current_agent` | 1 | python-only | Python truthiness (`__bool__` falsy runner/agent), legacy positional-argument call signature, env-var default via monkey |
| runner | `test_resumed_trip_preserves_accepted_turns_and_turn_budget` | 4 | ported | py_core2_runner_extra |
| runner | `test_rewind_debug_logging_respects_model_and_tool_policies` | 2 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_rewind_failure_uses_placeholder_free_shared_logger_message` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_rewind_handles_id_stripped_sessions` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_rewind_preserves_unrelated_tail_items_when_server_tracker_cleanup_runs` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_rewind_skips_mismatched_tail_suffix` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_rewind_strips_only_retry_owned_tail_items_before_known_server_item` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_run_config_defaults_nested_handoff_history_opt_in` | 1 | ported | py_core2_runner_routing |
| runner | `test_run_rejects_duplicate_function_tools_in_error_mode` | 2 | ported | py_core2_runner_routing |
| runner | `test_run_rejects_session_with_resumed_conversation_state` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_run_rejects_session_with_server_managed_conversation` | 3 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_run_reports_derived_agent_name_collisions_before_model_call` | 12 | ported | py_core2_runner_routing |
| runner | `test_run_streamed_preserves_legacy_positional_previous_response_id` | 1 | python-only | Python truthiness (`__bool__` falsy runner/agent), legacy positional-argument call signature, env-var default via monkey |
| runner | `test_run_streamed_rejects_session_with_resumed_conversation_state` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_run_streamed_rejects_session_with_server_managed_conversation` | 3 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_run_warns_and_keeps_last_duplicate_function_tool` | 2 | ported | py_core2_runner_routing |
| runner | `test_run_warns_once_for_repeated_source_agent_name` | 1 | ported | py_core2_runner_routing |
| runner | `test_save_result_to_openai_conversation_drops_placeholder_id_reasoning_item` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_openai_conversation_drops_unpersistable_reasoning_item` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_openai_conversation_keeps_reasoning_encrypted_content` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_openai_conversation_preserves_reasoning_id_when_policy_is_omit` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_openai_conversation_strips_placeholder_reasoning_id` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_session_counts_sanitized_openai_items` | 1 | covered | py_memory_conversations::{the_runner_strips_message_ids_before_writing_to_the_conversation,conversation_writes_keep_ids_ |
| runner | `test_save_result_to_session_does_not_increment_counter_when_nothing_saved` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_keeps_latest_output_after_its_call` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_keeps_tool_call_before_its_output` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_keeps_tool_call_payload_api_safe` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_omits_reasoning_ids_when_policy_is_omit` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_prefers_latest_duplicate_function_outputs` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_preserves_function_outputs` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_returns_count_and_updates_state` | 1 | ported | py_core2_runner_session_input |
| runner | `test_save_result_to_session_sanitizes_original_input_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_server_conversation_tracker_tracks_previous_response_id` | 1 | ported | py_core2_runner_routing |
| runner | `test_server_managed_handoff_history_auto_disables_with_warning` | 4 | ported | py_core2_runner_handoffs |
| runner | `test_server_managed_handoff_input_filters_still_raise` | 4 | ported | py_core2_runner_handoffs |
| runner | `test_session_add_items_called_multiple_times_for_multi_turn_completion` | 1 | ported | py_core2_runner_extra |
| runner | `test_session_persists_only_new_step_items` | 1 | ported | py_core2_runner_session_input |
| runner | `test_set_default_agent_runner_preserves_falsey_runner` | 1 | python-only | Python truthiness (`__bool__` falsy runner/agent), legacy positional-argument call signature, env-var default via monkey |
| runner | `test_set_default_agent_runner_roundtrip` | 1 | ported | py_core2_runner_routing |
| runner | `test_simple_first_run` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_single_tool_call_with_cancelled_tool_reaches_final_output` | 1 | ported | py_core2_runner_routing |
| runner | `test_structured_output` | 1 | ported | py_core2_runner_handoffs |
| runner | `test_subsequent_runs` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_tool_call_context_includes_current_agent` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_tool_call_runs` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_tool_not_found_behavior_handles_mixed_function_tool_calls` | 1 | ported | py_core2_runner_extra |
| runner | `test_tool_not_found_behavior_returns_error_to_model` | 1 | covered | py_core_a_runner::{simple_first_run,subsequent_runs,tool_call_runs,tool_call_context_includes_current_agent,handoffs_swi |
| runner | `test_tool_not_found_behavior_uses_tool_error_formatter` | 1 | ported | py_core2_runner_extra |
| runner | `test_tool_use_behavior_custom_function` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_tool_use_behavior_first_output` | 1 | ported | py_core2_runner_guardrail_sessions |
| runner | `test_wait_for_session_cleanup_logs_when_targets_linger` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_wait_for_session_cleanup_retries_after_get_items_error` | 1 | python-only | session rewind / cleanup-verification helpers (monkeypatched asyncio.sleep, mocked logger) exist because Python writes t |
| runner | `test_without_previous_response_id_and_auto_previous_response_id_no_chaining` | 1 | ported | py_core2_runner_streamed |
| runner | `test_wrong_params_on_input_causes_error` | 1 | python-only | input filter returning a non-HandoffInputData / non-callable filter / on_handoff with wrong arity: ruled out by Rust's t |
| runner | `testdrop_orphan_function_calls_removes_orphans` | 1 | ported | py_core2_runner_routing |
| runner | `testnormalize_input_items_for_api_preserves_provider_data` | 1 | ported | py_core2_runner_routing |
| streamed | `test_ambiguous_serialized_approval_state_fails_before_tool_execution` | 10 | deferred | 'current response boundary cannot be proven' validation of corrupted / legacy (1.16) serialized ownership: run_state wir |
| streamed | `test_async_handoff_on_input` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_async_input_filter_supported` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_blocked_final_persists_tool_items_but_not_the_message` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_blocked_message_final_output_is_not_persisted` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_blocked_tool_final_discards_reasoning_response_suffix_on_trip` | 4 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_blocked_tool_final_discards_suffix_with_multiple_reasoning_groups` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_chat_completions_output_guardrails_use_adapter_conversation_policy` | 8 | deferred | OpenAIChatCompletionsModel strict_feature_validation / server-managed-state warning: models area (src/models/openai_chat |
| streamed | `test_dynamic_tool_addition_run_streamed` | 1 | covered | py_core_a_runner::dynamic_tool_addition_run (Rust agents are immutable during a run; the streamed variant shares the too |
| streamed | `test_empty_list_input_reaches_model` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_failing_output_guardrail_keeps_the_whole_final_turn` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_handoff_filters` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_handoff_on_input` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_handoffs` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_input_guardrail_streamed_does_not_save_assistant_message_to_session` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_input_guardrail_streamed_persists_user_input_for_async_sequential_guardrail` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_input_guardrail_streamed_persists_user_input_for_sequential_guardrail` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_input_guardrail_tripwire_triggered_causes_exception_streamed` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_invalid_input_filter_fails` | 1 | python-only | non-HandoffInputData / non-callable input filter: ruled out by Rust's types |
| streamed | `test_legacy_approval_checkpoint_uses_current_session_ownership` | 8 | ported | py_core2_runner_streamed |
| streamed | `test_mixed_final_turn_session_order_and_committed_items` | 4 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_non_callable_input_filter_causes_error` | 1 | python-only | non-HandoffInputData / non-callable input filter: ruled out by Rust's types |
| streamed | `test_output_guardrail_exception_raises_from_run_loop_task_before_stream_consumption` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_output_guardrail_tripwire_raises_from_run_loop_task_before_stream_consumption` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_output_guardrail_tripwire_triggered_causes_exception_streamed` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_output_guardrails_fail_closed_with_server_managed_history` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_persisted_run_llm_again_checkpoint_rejects_terminal_behavior_change` | 6 | ported | py_core2_runner_streamed |
| streamed | `test_public_immediate_cancel_during_guardrail_recovery_save_stays_prompt` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_resumed_approved_tool_final_persists_complete_post_verdict_batch` | 6 | ported | py_core2_runner_streamed |
| streamed | `test_run_input_guardrail_tripwire_triggered_causes_exception_streamed` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_run_llm_again_approval_persists_completed_sibling` | 2 | ported | py_core2_runner_streamed |
| streamed | `test_run_output_guardrail_tripwire_triggered_causes_exception_streamed` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_serialized_filtered_handoff_approval_with_empty_prefix_resumes` | 8 | ported | py_core2_runner_extra |
| streamed | `test_serialized_later_turn_approval_with_output_guardrail_resumes` | 12 | ported | py_core2_runner_extra |
| streamed | `test_serialized_mixed_approval_guardrail_preserves_only_accepted_outputs` | 16 | deferred | 'current response boundary cannot be proven' validation of corrupted / legacy (1.16) serialized ownership: run_state wir |
| streamed | `test_simple_first_run` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_slow_input_guardrail_still_raises_exception_streamed` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_stop_on_first_tool_final_persists_committed_tool_items_on_tripwire` | 8 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_stream_input_persistence_saves_only_new_turn_input` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_stream_input_persistence_strips_ids_for_openai_conversation_session` | 1 | covered | py_memory_conversations::the_runner_strips_message_ids_before_writing_to_the_conversation (Rust strips in the session, P |
| streamed | `test_stream_step_items_to_queue_handles_tool_approval_item` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streamed_cancel_during_output_guardrail_starts_no_final_turn_write` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streamed_finalizer_failure_follows_both_data_policies` | 3 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streamed_interruption_reports_tool_guardrail_results` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_model_retry_does_not_rewind_committed_session_input` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_nested_handoff_filters_reasoning_items_from_model_input` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_streamed_omit_policy_strips_reasoning_ids_already_stored_in_the_session` | 1 | ported | py_core2_runner_routing (both modes) |
| streamed | `test_streamed_parallel_tool_call_with_cancelled_sibling_reaches_final_output` | 1 | ported | py_core2_runner_routing (both modes) |
| streamed | `test_streamed_reasoning_item_id_policy_omits_follow_up_reasoning_ids` | 1 | ported | py_core2_runner_routing (both modes) |
| streamed | `test_streamed_resume_handoff_turn_reports_tool_guardrail_results` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_resume_terminal_turn_reports_tool_guardrail_results` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_resume_tool_guardrail_results_match_non_streamed` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_again_persists_tool_items_to_session` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_counts_retry_attempts_when_terminal_usage_missing` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_exposes_request_id_on_raw_responses` | 3 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_preserves_request_usage_entries_after_conversation_locked_retry` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_preserves_request_usage_entries_after_retry` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_rejects_failed_terminal_response_payload_events` | 2 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_rejects_failed_terminal_response_payload_events_from_ws_model` | 2 | python-only | OpenAIResponsesWSModel internals (monkeypatched _open_websocket_connection and request lock); the Rust WS model's termin |
| streamed | `test_streamed_run_rejects_response_error_terminal_event` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_run_reports_tool_guardrail_results` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_session_save_cancellation_is_not_a_public_immediate_cancel` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streamed_session_save_direct_base_exception_is_terminal` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streamed_session_save_error_takes_precedence_over_output_guardrail_error` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_streamed_single_tool_call_with_cancelled_tool_reaches_final_output` | 1 | ported | py_core2_runner_routing (both modes) |
| streamed | `test_streamed_terminal_failure_closes_the_model_stream` | 4 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_tool_guardrail_results_match_non_streamed` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_tool_guardrail_results_persist_into_run_state` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streamed_tool_guardrail_results_survive_handoff` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_streamed_tool_not_found_behavior_returns_error_to_model` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_streamed_trip_preserves_accepted_tool_prefix` | 2 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_streamed_trip_replaces_current_tool_output_guardrail_results` | 2 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_events` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_hitl_resume_enforces_max_turns` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_hitl_resume_with_approved_tools` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_hitl_server_conversation_tracker_priming` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_max_turns_emits_pending_tool_output_events` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_streaming_non_max_turns_exception_does_not_emit_queued_events` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streaming_resume_carries_persisted_count` | 1 | python-only | internal stream plumbing: monkeypatched finalizer/logger records, private event queue + _stored_exception, stream_step_i |
| streamed | `test_streaming_resume_persists_tool_outputs_on_run_again` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_streaming_resume_preserves_filtered_model_input_after_handoff` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_streaming_resume_with_session_does_not_duplicate_items` | 1 | ported | py_core2_runner_streamed |
| streamed | `test_structured_output` | 1 | ported | py_core2_runner_handoffs (both modes) |
| streamed | `test_subsequent_runs` | 1 | covered | py_core_a_runner::subsequent_runs + py_core2_runner_routing (streamed variants run in both modes there) |
| streamed | `test_terminal_behaviors_defer_completed_approval_siblings` | 6 | ported | py_core2_runner_streamed |
| streamed | `test_terminal_tool_trip_uses_custom_blocked_message_everywhere` | 4 | ported | py_core2_runner_guardrail_sessions (both modes) |
| streamed | `test_tool_call_runs` | 1 | covered | py_core_a_streamed::{simple_first_run,empty_list_input_reaches_the_model,tool_not_found_behavior_returns_error_to_the_mo |
| streamed | `test_ws_terminal_failure_frees_the_request_lock_for_a_waiting_run` | 1 | python-only | OpenAIResponsesWSModel internals (monkeypatched _open_websocket_connection and request lock); the Rust WS model's termin |
| streamed | `test_terminal_tool_trip_falls_back_when_blocked_message_formatter_fails` | 16 | ported 4 / python-only 12 | py_core2_runner_guardrail_sessions::a_formatter_without_a_message_falls_back_to_the_default (none, empty); other outcomes need a non-`Option<String>` formatter |
