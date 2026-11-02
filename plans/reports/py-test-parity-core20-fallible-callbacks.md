# Python test parity: fallible enablement / approval callbacks (core20, r20d)

Date: 2026-11-02. Scope: let `is_enabled`, `needs_approval` and hosted MCP `on_approval_request` callbacks fail, as Python lets them raise. All existing signatures stay the same.

## API added (no existing signature changed)

| Surface | New API |
|---|---|
| `tool::IsEnabled<C>` | variant `Fallible(IsEnabledTryFn<C>)`, `IsEnabled::fallible(f)`, `try_evaluate(..) -> Result<bool>` |
| `tool::NeedsApproval<C>` | variant `Fallible(NeedsApprovalTryFn<C>)`, `NeedsApproval::fallible(f)`, `try_evaluate(..) -> Result<bool>`, `is_callable()` |
| `FunctionTool` | `with_is_enabled_fallible`, `with_needs_approval_fallible`, `try_evaluate_needs_approval` |
| `Handoff` | `with_is_enabled_fallible` |
| `AgentAsToolOptions` | `is_enabled_fallible`, `needs_approval_fallible` |
| `HostedMCPTool` | `with_on_approval_request_fallible`, `has_on_approval_request` |
| `Agent` | `try_get_enabled_handoffs`, `try_mcp_reserved_tool_names` (mcp feature) |

The infallible forms are unchanged. `IsEnabled::evaluate`, `NeedsApproval::evaluate`, `FunctionTool::evaluate_needs_approval`, `Agent::get_enabled_handoffs` and `Agent::mcp_reserved_tool_names` keep their return types. When a `Fallible` callback fails inside one of them, they fail closed: the tool or handoff counts as disabled, and approval is required. The runner always uses the `try_` forms. `HostedMCPTool::on_approval_request()` still returns only a callback set with the infallible builder. The runner reads both kinds through the crate-private `approval_callback()`.

**Semver caveat:** `IsEnabled` and `NeedsApproval` are exhaustive public enums, and both gain a `Fallible` variant. Downstream code that matches on them exhaustively will no longer compile. No crate code or test does this; they only use `matches!` and constructors. This is the only change that is not purely additive.

## Behaviour (matches Python's call sites)

- **Turn tool resolution** (`Agent::get_all_tools`, Python `get_all_tools`): function-tool enablement checks now run at the same time (`try_join_all`). Before this they ran one after another. The first error comes back unchanged and the checks still running are cancelled (dropped), like `gather_with_cancel`. The resolved order does not change.
- **Handoff resolution** (`try_get_enabled_handoffs`, Python `get_handoffs`): the checks also run at the same time, with the same error and cancel behaviour. Both run loops (fresh turn and resumed turn) use it.
- **Enablement re-check before execution** (`execute_function_tools`, Python `_FunctionToolBatchExecutor.execute` → `resolve_enabled_function_tools`): this now checks every function tool configured on the agent, at the same time. Before, it checked only the tools the model called. A failing check aborts the batch with its own error before any call starts. A tool that was called but is now disabled still gets `ModelBehaviorError "Tool X is currently disabled for agent Y."`, using Python's "enabled if any configured instance is enabled" rule. Side effect: a turn now runs `is_enabled` for configured tools the model did not call, as Python does.
- **`needs_approval`** (`execute_function_tool`, Python `evaluate_function_tool_approval`): the error ends the run unchanged. It is not wrapped in "Error running tool", and the tool body never runs. Arguments that are not a JSON object, and argument-validator tools, still skip the rule and require approval, the same as the `Dynamic` form.
- **Hosted MCP callback** (run loop, Python `execute_mcp_approval_requests`): the error ends the run unchanged. The request is marked as executed before the callback runs and is never marked committed, so a retry of the same request on the same context is refused.
- **Failed-turn history** (`items::completed_tool_step_items`): an `mcp_approval_request` is now kept only if its `mcp_approval_response` is among the turn's outputs. If its callback failed, the request is dropped, which matches Python's `_completed_tool_step_items` for the `mcp_callback` boundary. Answered requests are still kept, as before.

## Ported (6 Python ids moved, plus 1 exact port of a case already counted) → `tests/py_core20_fallible_callbacks.rs`

| Python file | Python case | Rust test | Count |
|---|---|---|---|
| test_function_tool.py | `test_get_all_tools_cancels_sibling_enablement_checks_on_error` | `get_all_tools_cancels_sibling_enablement_checks_on_error` | 1 |
| test_handoff_tool.py | `test_get_handoffs_cancels_sibling_enablement_checks_on_error` | `get_handoffs_cancels_sibling_enablement_checks_on_error` | 1 |
| test_run_step_execution.py | `test_function_tool_enablement_error_cancels_sibling_checks_before_execution` | `function_tool_enablement_error_cancels_sibling_checks_before_execution` (through `Runner`, streamed and not) | 1 |
| test_tool_failure_boundaries.py | `test_completed_history_at_failure_boundaries[mcp_callback-False/True]` | `completed_history_at_mcp_callback_failure_boundary` | 2 |
| test_decisions_examples.py | `test_tool_classifier_failure_never_invokes_tool[True]` (SDK behaviour; the example module stays Python-only) | `approval_classifier_failure_never_invokes_tool` (streamed and not) | 1 |
| test_hitl_error_scenarios.py | `test_mcp_callback_failure_is_not_retried_for_same_request` | `mcp_callback_failure_is_not_retried_for_same_request` (streamed and not) | 0 (r9h2 already counted it through the abandoned-callback analogue; this port is exact) |

Rust-only coverage in the same file: `failing_handoff_enablement_aborts_the_run` (no model call, streamed and not) and `agent_as_tool_fallible_callbacks`.

Message divergence: in `mcp_callback_failure_is_not_retried_for_same_request`, the run-level response validation refuses the retry first, with "A tool call already executed, but its output was not committed". Python calls the private executor directly and gets "...callback already ran...". The error type (`ModelBehaviorError`) and the single callback call match.

The `[False]` half of `test_tool_classifier_failure_never_invokes_tool` uses a failing tool input guardrail. It does not depend on this API and stays where r19c classified it.

## Checked, nothing to port here

These files were named in the brief, but an AST scan found no case where an enablement or approval callback raises anything other than an `AssertionError` guard ("must not be called"): test_agent_runner.py, test_tool_use_behavior.py, test_agent_tool_enablement_snapshot.py, test_run_internal_approvals.py. The two `test_agent_tool_enablement_snapshot.py` cases are about snapshot identity and are not affected.

## Left for other owners / out of scope

- Realtime (owned by the realtime agent): `realtime/test_realtime_handoffs.py::test_collect_enabled_handoffs_cancels_sibling_checks_on_error`, `realtime/test_realtime_model_settings.py::test_filter_enabled_tools_cancels_sibling_checks_on_error` (and `test_build_model_settings_cancels_cross_group_enablement_on_error`, `realtime/test_session.py::test_realtime_session_boundaries_cancel_cross_group_enablement_on_error`). `src/realtime/**` still calls `IsEnabled::evaluate`, which treats a failing `Fallible` check as disabled. Porting these needs `try_evaluate`, concurrent evaluation, and error propagation in `src/realtime/{agent,session}.rs`.
- Local tools (`ShellTool` / `ApplyPatchTool` / custom tool `needs_approval` and `on_approval`) and local MCP `RequireApproval::Dynamic`: Python allows them to raise, but no Python test does, and `src/mcp` belongs to another agent. Not changed (YAGNI).

## Suggested changelog paragraph (do not paste into `docs/project-changelog.md` without review)

> **Fallible enablement and approval callbacks.** `FunctionTool::with_is_enabled_fallible`, `FunctionTool::with_needs_approval_fallible`, `Handoff::with_is_enabled_fallible`, `AgentAsToolOptions::{is_enabled_fallible, needs_approval_fallible}` and `HostedMCPTool::with_on_approval_request_fallible` take callbacks that return `Result`. Their errors end the run unchanged, as a raising Python callable does: no tool runs, sibling enablement checks still in flight are cancelled, and a hosted MCP request whose callback failed is not retried. `IsEnabled` and `NeedsApproval` gain a `Fallible` variant, plus `try_evaluate` (an exhaustive `match` on them needs a new arm). The new `Agent::try_get_enabled_handoffs` and `Agent::try_mcp_reserved_tool_names` return the error, and the existing infallible accessors fail closed. Enablement checks now run concurrently. The pre-execution re-check covers every configured function tool, not only the ones the model called. A failed turn drops a hosted MCP approval request that was never answered.
