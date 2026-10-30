# Core-runner parity report (Python openai-agents 0.24 -> Rust openai-agents-sdk)

Date: 2026-10-30. Scope: the core-runner rows of
`plans/reports/parity-audit-261030-0505-python-to-rust-report.md` (run_state, turn_resolution,
tool invocation/identity, tool_caller, session_persistence, oai_conversation, tool_execution,
tool_use_tracker, agent_bindings/agent_runner_helpers, run.py, run_loop, run_internal/items,
items.py, tool_planning, blocked_output, run_context, result, prompt_cache_key/run_grouping,
handoffs/history, util/_transforms, run_config/tool_context, util/_custom_data, computer,
repl, agent/agent_tool_input, agent_tool_state, util/_pretty_print, tool.py typed configs).

Verification (`default = []`, so feature-gated code needs explicit runs):

- `cargo test` and `cargo test --features sqlite` green; `cargo build --examples` clean.
- `cargo clippy --all-targets -- -D warnings` and
  `cargo clippy --all-targets --features mcp,realtime,sandbox -- -D warnings` clean (also
  checked with `macros,mcp,realtime,websocket,voice,sqlite`).
- `cargo check --all-targets` clean for `macros,mcp`, `realtime,websocket,voice` and
  `sandbox,encrypt,sqlx,redis`; `cargo test` green for `--features realtime,websocket`
  (lib + `realtime_tests`), `--features mcp` (`mcp_tests`) and `--features sandbox`
  (`sandbox_tests`), so no cross-agent file needs an update for these changes — the realtime
  and sandbox suites already carry the lowercase tool names.
- `--features mcp,realtime` test suites green: `mcp_tests` (4), `realtime_tests` (33), plus
  every `core_*` suite. `--features sandbox`: `sandbox_tests` green (62);
  `sandbox_modal_cloudflare_blaxel_tests` has 3 failures owned by the sandbox agent (see
  Concerns).
- All touched files `rustfmt --edition 2024`.

## Done

### Wire-visible behaviour

- **`util/_transforms.py`** — `transform_string_function_style` now lowercases and warns like
  Python (`src/util.rs`), so the default handoff tool name and the default `as_tool` name match
  the Python SDK byte for byte. Added `transform_string_function_style_with(name,
  warn_on_whitespace)` and re-exported both from the crate root.
- **`run_internal/turn_resolution.py`** — programmatic tool calling: `program` items validate a
  non-empty `call_id`; `program_output` items validate the parent program, that it is not
  already completed, a `completed`/`incomplete` status and a string `result`. `compaction`
  output items become `RunItemKind::Compaction` and emit **no** stream event. Client-executed
  `tool_search_call` is rejected. `StopAtTools` now matches `tool.name` **or**
  `tool.qualified_name`.
- **`run_internal/tool_caller.py`** — ported as `src/run_internal/tool_caller.rs`:
  `ensure_tool_caller_allowed`, `ensure_programmatic_tool_call_parent`,
  `collect_program_parent_state`. Enforced in `process_response` for function calls and in
  `match_local_call` for computer, shell, local shell, apply_patch and custom calls, so a
  program-owned call to a tool that only allows `direct` is now rejected at runtime instead of
  only being advertised in the tool param.
- **`tool.py`** — `allowed_callers` added to `ShellTool`, `ApplyPatchTool` and `CustomTool`
  (field + `with_allowed_callers` + emission in their Responses params), matching Python.
- **`run_internal/tool_use_tracker.py`** — `tool_search_call` is excluded from tool-use
  tracking (`_TOOL_USE_RESET_TRACKING_ITEM_TYPES`), so it no longer resets `tool_choice`. The
  tracker is serialized into `RunState.tool_use_tracker` and rehydrated on resume
  (`serialize_tool_use_tracker` / `hydrate_tool_use_tracker`).
- **`run_internal/items.py`** — ported as `src/run_internal/items.rs`:
  `drop_orphan_function_calls` (including the reasoning items tied to dropped calls and the
  orphan-program rules), `_dedupe_key`, `deduplicate_input_items`,
  `deduplicate_input_items_preferring_latest`, `order_current_turn_tool_outputs`. Wired into
  `build_model_input` (generated history only — caller input is replayed verbatim like Python's
  `prepare_model_input_items`), the model input after `call_model_input_filter`,
  `save_session`, and tool-output ordering after a turn's side effects, so outputs now follow
  the order of their calls instead of the fixed function/MCP/local phase order.
- **`run_internal/tool_planning.py`** — one model response can no longer schedule the same
  `call_id` twice (`_dedupe_processed_response_invocations` / `_dedupe_tool_call_items`).
- **`handoffs/history.py`** — nested handoff history now forwards non-assistant, non
  tool-traffic items verbatim alongside the summary, in order, like
  `_build_ordered_default_history`; `function_call`, `function_call_output`, `reasoning` and
  every program-owned item stay inside the summary. Added the public `nest_handoff_history`,
  `default_handoff_history_mapper`, `DEFAULT_HISTORY_START`/`DEFAULT_HISTORY_END` and
  `set_/get_/reset_conversation_history_wrappers`.
- **`run_internal/oai_conversation.py`** — `ServerConversationTracker`
  (`src/run_internal/oai_conversation.rs`) replaces the two integer cursors for server-managed
  conversations. It dedupes by server item id, tool call id and canonical content fingerprint,
  records what each model call delivered (`mark_input_as_sent`), records what the provider
  produced (`track_server_items`), and hydrates from `RunState` on resume. Hydration honours
  `get_unsent_tool_call_ids_for_interrupted_state`: tool outputs for the latest response are
  kept unsent, so an approval resume still delivers them.
- **`run_internal/blocked_output.py`** — tool *output* guardrail results of the blocked turn
  now have their `output_info` and any `reject_content` message replaced with the placeholder
  (`_data_free_tool_output_guardrail_results`), and the blocked tool output item is rebuilt
  from allowlisted fields (`type`, `call_id`, `id`, `status`, and a `direct` caller only)
  instead of being overwritten in place (`blocked_function_output_payload`).
- **`run_internal/tool_execution.py`** — the function-tool batch now stops at the first failure
  in call order and drops the stream, which cancels every sibling still in flight, instead of
  letting them run to completion (`_select_function_tool_failure` +
  `_cancel_function_tool_tasks`). Since every Rust tool failure is an `AgentsError`, the
  priority ladder collapses to call order, which `buffered`'s in-order yielding gives directly.
- **`run_internal/agent_runner_helpers.py`** — output guardrails combined with
  `conversation_id` / `previous_response_id` / `auto_previous_response_id` are now rejected up
  front (`validate_output_guardrails_with_server_managed_conversation`), because rejected
  output cannot be removed from server-managed history.
- **`memory/openai_responses_compaction_session.py`** (controller row 127b) —
  `_ignore_ids_for_matching` is propagated: the run loop strips provider-assigned item ids from
  the `model_exchange` digests when the session manages its history remotely, and the
  compaction session inherits the wrapped store's policy for its suffix matching
  (`src/run_internal/mod.rs::strip_item_id`,
  `src/memory/openai_responses_compaction_session.rs::ignore_ids_for_matching`).
  **Caveat:** with the current trait surface neither branch can fire. The compaction session
  rejects a remotely-managed store, so its flag is always `false`, and the run-loop branch
  needs `wants_model_exchange() && manages_history_remotely()`, which no built-in session has.
  The faithful fix is a defaulted `Session::ignore_ids_for_matching()` in `src/memory/mod.rs`
  (the memory agent's file) overridden by `OpenAIConversationsSession`; the plumbing here is
  ready for it.

### `RunState` wire format (`run_state.py` + `_run_state_agent_identity.py`)

`RunState::to_json` now emits Python's schema **`"1.21"`** keyed envelope and `from_json` reads
it (`src/run_state/wire.rs`, tests in `tests/core_run_state.rs`):

- `$schemaVersion`, `current_turn`, `current_agent` (an agent reference object),
  `original_input`, `pending_input`, `model_responses`, `max_turns`, `no_active_agent_run`,
  `conversation_id`, `previous_response_id`, `auto_previous_response_id`,
  `generated_prompt_cache_key`, `reasoning_item_id_policy`,
  `nested_history_owned_session_item_refs`, `generated_session_item_indexes`,
  `generated_items`, `session_items`, `current_step`, `last_model_response`,
  `last_processed_response`, `current_turn_persisted_item_count`, `trace`.
- `context` = `{usage, approvals, function_tool_approvals, tool_invocations, context,
  context_meta}`; per-agent approvals go to `function_tool_approvals` as
  `{agent: {name}, tool_key, decision}` with Python's `{approved, rejected,
  rejection_messages, sticky_rejection_message}` decision shape, and Python's agent-agnostic
  `approvals` bucket is read back as a fallback that applies to every agent.
- `tool_use_tracker` = `{agent name: [tool names]}`.
- Guardrail results use Python's `{guardrail: {type, name}, output: {tripwireTriggered,
  outputInfo}, agentOutput, agent}` shape; tool guardrail results use
  `{guardrail: {type, name}, output: {outputInfo, behavior}}`.
- Usage uses Python's shape, where the top-level token details are one-element lists.
- `trace` is `{object: "trace", id}`, matching `TraceState.to_json`; `from_json` accepts `id`
  or `trace_id`.
- Items carry `{type, raw_item, agent: {name}, output?, source_agent?, target_agent?,
  tool_name?}`. Python's `tool_search_call_item` / `tool_search_output_item` are read back into
  the merged Rust tool-call kinds; hosted-MCP approval interruptions (`raw_item.id` instead of
  `call_id`) are read back; an item type the port cannot read is an **error**, not a silent
  drop.
- Rust-only run state (`server_sent_items`, `server_input_sent`, `session_saved_items`,
  `input_saved_to_session`, `nested_agent_tool_states`, `pending_tool_processing`,
  `turn_start_index`, `final_output`, `server_conversation`) travels under the top-level `rust`
  key, which Python ignores.
- `from_json` accepts every schema version Python supports (`1.0`–`1.21`) and still reads the
  flat snapshots this crate used to write. Those are detected **by shape** (a top-level
  `current_agent_name` string with no `current_agent` object), because Python also has a
  `"1.0"`.
- New `RunState` fields: `tool_use_tracker`, `generated_prompt_cache_key`,
  `server_conversation` (`ServerConversationState`).
- Agent identity: resuming a graph that contains two agents with the same name is now an error
  naming them, instead of silently binding the first match (Python's
  `_has_ambiguous_hook_identities` check).

### Tracing / redaction helper wiring (controller follow-up)

- **Run trace creation** (`src/run_internal/mod.rs`) now goes through
  `crate::tracing::create_trace_for_run(workflow_name, trace_id, group_id, metadata,
  tracing_api_key, disabled)`. A run with `tracing_disabled` always gets its own disabled
  trace, and it additionally holds `crate::tracing::masked_run_span()` for the whole run, so a
  disabled run nested inside an enabled trace neither exports spans nor lets its children
  attach to the caller's span. Covered by `tests/core_tracing_masking.rs`.
- **Tool error spans** now use `crate::tracing::record_tool_error_on_span(span, message, error,
  trace_include_sensitive_data)` instead of writing the raw error into the span:
  `src/run_internal/tools.rs` (the non-fatal failure-function path and the tool span),
  `src/run_internal/local_tools.rs` (computer, shell, apply_patch, custom),
  and the run span, turn span and handoff invocation in `src/run_internal/mod.rs` use
  `tr::model_error_text`, whose redacted wording is the generic "Error details are redacted."
  rather than the tool-specific one. The `tool_name` field the old payloads carried is dropped,
  because the shared helpers write only `{"error": ...}`.
- **Tool argument payloads** no longer reach an error message unredacted:
  `src/tool.rs` and `src/agent.rs` now format the payload through
  `crate::logger::may_log_model_data(true)`. When the policy forbids it, **neither** the
  payload **nor** the serde error (which echoes the offending value) reaches the message, which
  is what `validate_json`'s `should_redact` branch does; the allowed branch follows Python's
  "Invalid JSON when parsing `<payload>` for ..." wording.

### API surface

- **`run_config.py`** — `ToolErrorKind::ToolException`. **Final `ToolErrorFormatterArgs<'a, C>`
  shape** (for the realtime alias): `kind: ToolErrorKind`, `tool_type: String`,
  `tool_name: String`, `call_id: String`, `default_message: String`,
  `run_context: &'a RunContext<C>`, `error: Option<&'a AgentsError>`. The formatter now also
  runs for tool exceptions whenever the tool still uses the SDK default failure function, like
  Python's `_use_default_failure_error_function` path.
  `OutputGuardrailBlockedMessageArgs<'a, C>` gained the full `agent: &'a Arc<Agent<C>>` and
  `run_context: &'a RunContext<C>` (with `agent_name()` kept as a helper).
- **`tool_context.py`** — `ToolContext` gained `tool_namespace`, `agent`, `run_config`,
  `qualified_tool_name()`, and `set_custom_data` / `custom_data`.
- **`run_context.py`** — `turn_input()` / `set_turn_input()` (set from the exact model input of
  each turn, after the input filter and dedupe) and `get_approval_status()` returning the new
  `ApprovalStatus { approved, rejection_message }`.
- **`result.py`** — `ToInputListMode` + `to_input_list_with(mode)`, `AgentToolInvocation` +
  `RunResult::agent_tool_invocation`, `RunResultStreaming::{current_turn, max_turns,
  run_loop_error}`.
- **`util/_pretty_print.py`** — `RunResult::pretty_print`, `RunResultStreaming::pretty_print`
  and `Display` for both, with C0/DEL/C1 terminal-control escaping.
- **`repl.py`** — `run_demo_loop_with(agent, stream, context, max_turns)`; the loop keeps one
  run context across turns so usage and approvals accumulate.
- **`run.py`** — `AgentRunner<C>` trait, `DefaultAgentRunner`, `set_default_agent_runner` /
  `get_default_agent_runner` (per context type, keyed by `TypeId`), consulted by
  `Runner::run_with_options` and `Runner::run_streamed_with_options`.
- **`items.py`** — `RunItemKind::Compaction` (`compaction_item`),
  `ItemHelpers::copy_tool_call_caller`, and `custom_data` on `ToolCallOutput` with
  `RunItem::custom_data()`.
- **`util/_custom_data.py`** — `CustomDataExtractor<C>`, `FunctionToolCustomDataContext`,
  `normalize_custom_data`, `merge_custom_data`, `FunctionTool::with_custom_data_extractor`,
  `ToolContext::set_custom_data`, and `FunctionToolResult::custom_data`. Extractor output is
  merged over whatever the tool body attached and lands on the output item without reaching
  `raw_item`.
- **`agent.py` + `agent_tool_input.py`** — `AgentToolStreamEvent<C> { event, agent, tool_call }`
  for nested `as_tool` streaming; `AgentAsToolOptions::failure_error_function`;
  `build_schema_summary` + `StructuredToolInput { params, summary, json_schema }`, so the
  default structured-input message now has the `## Input Schema Summary:` section Python emits
  when the schema carries descriptions and the full schema is not included.
- **`computer.py`** — `ComputerTool::from_provider(create, dispose)` and
  `ComputerTool::dispose()`; at the end of a run the loop walks the whole handoff graph and
  tears down the computers it created (`dispose_resolved_computers`). The lazily created
  computer lives in a resettable slot instead of a `OnceCell`, but **only a tool built with
  `from_provider` is torn down** — `from_factory` keeps its documented per-tool cache.
  Unlike Python the lifecycle is scoped per *tool*, not per run context, so two concurrent runs
  that share the tool share one computer and the first to finish disposes it.
- **`tool.py` typed configs** — `ShellToolEnvironment` (`Local` / `ContainerAuto` /
  `ContainerReference`) with `ShellToolLocalSkill`, `ShellToolContainerSkill`,
  `ShellToolInlineSkillSource`, `ShellToolContainerNetworkPolicy` and
  `ShellToolContainerNetworkPolicyDomainSecret`; `WebSearchToolImageSettings`;
  `ImageGenerationToolConfig`. Each has `to_value()`, and the tool fields stay `Value`, so an
  unmodelled payload still works (`ShellTool::with_typed_environment` sets the typed form).
- **`FunctionToolResult`** gained `qualified_tool_name` and `custom_data`.

## Follow-up round (after `sha2` became unconditional and the MCP/realtime/models files were freed)

- **`run_internal/prompt_cache_key.py` + `run_grouping.py`** — ported as
  `src/run_internal/prompt_cache_key.rs`: `RunGroupingKind`, `resolve_run_grouping`,
  `resolve_run_grouping_id` (re-exported from the crate root for the sandbox rollout id),
  `PromptCacheKeyResolver` and `model_settings_with_prompt_cache_key` (as
  `apply_prompt_cache_key`). `resolve_settings` now derives one key per run and writes it to
  `ModelSettings::extra_args["prompt_cache_key"]`. Grouping precedence is conversation ->
  session -> trace group -> a generated per-run value; the first three are hashed into
  `agents-sdk:<kind>:<sha256 hex[..32]>`, while the per-run value stays verbatim as
  `agents-sdk:run:<hex>` so unrelated runs never share a cache group. The key is skipped when
  the caller already set `prompt_cache_key` in `extra_args`/`extra_body`, and when the model
  does not support it. It is persisted in `RunState::generated_prompt_cache_key` and
  rehydrated on resume.
- **`Model::supports_default_prompt_cache_key`** — new defaulted trait method
  (`src/models/interface.rs`), implemented by `OpenAIResponsesModel` and
  `OpenAIChatCompletionsModel` as "this client targets the official OpenAI API". Mirrors
  `_supports_default_prompt_cache_key`, so a compatible third-party endpoint never receives
  the field.
- **`MCPConfig.failure_error_function`** — `MCPConfig` is now `MCPConfig<C>` and carries
  `failure_error_function: Option<ToolErrorFn<C>>`, defaulting to
  `default_tool_error_function` like Python. `src/mcp/util.rs` gained
  `to_function_tool_with_failure` and applies the agent's handler to every converted MCP tool;
  `None` re-raises instead. `src/realtime/agent.rs` and `tests/mcp_tests.rs` were updated to
  the generic form, and the config gained builders
  (`with_convert_schemas_to_strict`, `with_include_server_in_tool_names`,
  `with_failure_error_function`, `without_failure_error_function`).
- **`FunctionTool.output_json_schema`** — new field plus
  `FunctionTool::with_output_json_schema`, which rejects a non-object schema and makes it
  strict (`_normalize_function_tool_output_json_schema`). It reaches the wire through
  `FunctionToolSpec::output_json_schema` as `output_schema` on the Responses function tool
  param, and is absent from the Chat Completions param. (`output_type` stays unported: Rust has
  no return-type reflection, so the explicit schema is the only path.)
- **`src/mcp/util.rs:340` left unchanged** (per the "otherwise leave it" instruction): it is a
  tool *listing* failure, and Python routes only tool *execution* errors through
  `get_trace_tool_error`.

## Breaking changes

1. **Tool names are lowercase.** `Agent::new("Billing Agent")` now yields
   `transfer_to_billing_agent` (was `transfer_to_Billing_Agent`), and the default `as_tool`
   name is lowercased too. This changes the wire and the approval key for any capitalized agent
   name. Updated: `src/util.rs`, `src/handoffs.rs:331`, `tests/runner_tests.rs`.
   **Still failing in files this agent must not edit** (only under non-default features, since
   `default = []`):
   - `src/realtime/handoffs.rs:260` — expects `transfer_to_Billing_Agent`, should be
     `transfer_to_billing_agent`.
   - `tests/realtime_tests.rs:198` — `transfer_to_Billing` -> `transfer_to_billing`;
     `:202` and `:1642` — `RealtimeModelToolCallEvent::new("transfer_to_Billing", ...)` ->
     `"transfer_to_billing"`.
   - `tests/sandbox_tests.rs:854`, `:1629` — `"name": "transfer_to_Reviewer"` ->
     `"transfer_to_reviewer"`; `:1700-1702` — `transfer_to_Reviewer` /
     `transfer_to_Engineer` -> `transfer_to_reviewer` / `transfer_to_engineer`
     (only where the agent names are `Reviewer` / `Engineer`).
2. **`RunState` schema moves from `"1.0"` to `"1.21"`** with a completely different envelope.
   New snapshots cannot be read by older builds of this crate; old `"1.0"` snapshots are still
   read.
3. **`ToolErrorFormatterArgs` is now `ToolErrorFormatterArgs<'a, C>`**;
   `RunConfig::with_tool_error_formatter` takes
   `for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Option<String>`.
4. **`OutputGuardrailBlockedMessage`, `OutputGuardrailBlockedMessageArgs` and
   `BlockedMessageFormatter` are generic over `C`**; the args carry `agent: &Arc<Agent<C>>`
   instead of `agent_name: String`.
5. **New enum variants** break exhaustive matches: `RunItemKind::Compaction`,
   `ToolErrorKind::ToolException`.
6. **New public fields** break struct literals: `RunItemKind::ToolCallOutput { custom_data }`,
   `FunctionToolResult { qualified_tool_name, custom_data }`, `ToolContext { tool_namespace,
   agent, run_config }`, `RunState { tool_use_tracker, generated_prompt_cache_key,
   server_conversation }`, `RunResult { agent_tool_invocation }`, `ShellTool`/`ApplyPatchTool`/
   `CustomTool` `{ allowed_callers }`.
7a. **`MCPConfig` is now `MCPConfig<C>`** and no longer derives `PartialEq`/`Eq`; its struct
    literal needs the new `failure_error_function` field (use `MCPConfig::default()` plus the
    builders). `Agent::mcp_config`, `Agent::with_mcp_config`, `RealtimeAgent::mcp_config` and
    `RealtimeAgent::with_mcp_config` take the generic form.
7b. **`FunctionToolSpec` gained `output_json_schema`**, breaking struct literals that do not
    use `..Default::default()`.
7c. **An official-OpenAI model now receives a generated `prompt_cache_key`** in
    `extra_args` when the caller did not set one.
7. **`AgentAsToolOptions::on_stream`** now receives `AgentToolStreamEvent<C>` instead of
   `StreamEvent<C>`, and **`input_builder`** receives `&StructuredToolInput<'_>` instead of
   `(&Value, Option<&Value>)`. `default_tool_input_builder` takes the same options struct.
8. **Server-managed conversations** no longer use an index cursor: deltas are chosen by
   identity and content fingerprint. A tool that returns byte-identical output for the same
   `call_id` twice within one server conversation is now sent once.
9. **Output guardrails + `conversation_id`/`previous_response_id`/`auto_previous_response_id`
   is now a `UserError`** at run start (it used to run and silently leave rejected output in
   the provider's history).
10. **A function-tool batch now cancels its siblings** on the first failure instead of letting
    them finish.
11. **Resuming a `RunState` into a graph with two same-named agents is now an error.**
12. **Tool error span data no longer carries `tool_name`**, only `{"error": ...}`, because the
    shared tracing helper writes that shape.
13. **The invalid-JSON tool error message changed wording** and now omits the payload and the
    parser detail unless `OPENAI_AGENTS_DONT_LOG_MODEL_DATA`/`..._TOOL_DATA` allow them.
14. **`ComputerTool::from_provider`** computers are disposed at the end of every run of any
    agent in the graph that holds the tool.

## Verified decisions (not reversed)

- **Approvals stay keyed by `agent + tool`.** The audit row for `_tool_invocation.py` claims
  Python keys approvals by tool name only. Source says otherwise: `run_state.py:1457-1488`
  serializes `_FunctionToolApprovalKey` entries that carry an `agent`, and schema `1.18`'s
  summary is "binds function-tool approval decisions to their owning agent". Rust's
  `agent \u{1} tool` key matches that. Python's separate agent-agnostic `approvals` bucket is
  read back and consulted as a fallback, so pre-1.18 snapshots are not dropped.

## Not ported

- **`FunctionTool.output_type`** — Rust has no return-type reflection, so a tool cannot infer
  its output schema; `with_output_json_schema` is the explicit equivalent.
- **`AgentAsToolOptions::on_stream_max_pending_events`** — N/A. The Rust callback is awaited
  inline inside the event loop, so there is no pending-event queue to bound.
- **`run_internal/session_persistence.py`: `rewind_session_items`, `_rewind_session_tail_
  suffix`, `wait_for_session_cleanup`, the deferred/held-write machinery
  (`defer_interrupted_session_write`, `held_session_items_for_save`,
  `take_held_session_write`, `resume_pending_session_write`), `admit_pending_input` /
  `commit_server_pending_input`, `persist_session_items_for_guardrail_trip` and
  `_sanitize_openai_conversation_item`** — Rust still writes to the session only *after* a turn
  completes (never before `call_model`), so the partial write these guard against is not
  reachable. The machinery stays absent; it would have to come back with any future pre-turn
  write.
- **Nested-history provenance** (`_nest_handoff_history_with_provenance`,
  `_get_nested_history_owned_items`, `_map_run_item_occurrences`,
  `NestedHistoryOwnedItem*`, `nested_history_owned_session_item_refs`) and the nested-history
  **parser** half (`_extract_nested_history_transcript`, `_split_summary_records`,
  `_parse_summary_line`, `_parse_legacy_typed_item`, `_LEGACY_CONVERSATION_HISTORY_PREAMBLE`)
  — Python-internal occurrence bookkeeping that exists to re-identify run items across
  rewrites. `RunState` emits an empty `nested_history_owned_session_item_refs` list.
  Consequence: `to_input_list(Normalized)` does not try to subtract items a nested handoff
  moved into the summary; it replays the model view only when a handoff **input filter**
  rewrote history (which also rewrites `RunResult::input`).
- **`run_state.py`: `last_processed_response`, `current_response_generated_item_ownership`,
  `pending_session_write`, `terminal_unrecoverable`, `sandbox`, the `ContextSerializer` /
  `ContextDeserializer` hooks and `RunStateValidationErrorFactory`** — emitted as `null`/
  absent. Rust rebuilds the processed response from `model_responses.last()` on resume, so it
  does not need the serialized copy. **Consequence: a Rust-written snapshot of an
  *interrupted* run cannot be resumed by the Python SDK**, which needs
  `last_processed_response` for that; every other direction works.
- **`_run_state_agent_identity.py`'s identity map** (`_build_agent_identity_map`,
  `_agent_identity_signature`, `_handoff_identity_signature`, `_tool_identity_signature`) —
  not ported. Instead, resuming a graph with duplicate agent names is rejected outright, so the
  ambiguity it exists to resolve cannot arise. Structural validation of a changed agent is
  still absent.
- **`_tool_invocation.py` / `_tool_identity.py` / `_function_tool_arguments.py`** — the
  fingerprint/alias layer (`tool_invocation_identity`, `tool_output_identity`,
  `FunctionToolLookupKey` serialization, `get_function_tool_approval_keys` multi-key aliasing,
  `should_allow_bare_name_approval_alias`, `restore_tool_call_routing_identity`,
  `PreparedFunctionArguments`, `arguments_unchanged`) is still absent. Approvals match on
  `(agent, qualified tool name, call_id)`, so they do not survive a namespace rename and a
  mutated-arguments replay is not detected.
- **`context.hosted_mcp_approvals`** — not written and not read. Rust records hosted-MCP
  approvals in the same `agent + "<label>:<tool>"` approval record, which round-trips through
  `function_tool_approvals`.
- **`agent_bindings.py`** (`AgentBindings`, `bind_execution_agent`,
  `model_settings_overrides`) — Rust `RunLoop` has one `current_agent`, so a
  sandbox-prepared clone *is* the identity reported in hooks, spans and
  `RunResult::last_agent`. `resolve_trace_settings` still restores only `trace_id`.
- **`run_internal/run_loop.py`**: `_safe_redacted_persistence_leaf_error` /
  `_RedactedBaseExceptionGroup` (no exception groups in Rust),
  `_persist_stream_input_if_needed` / `_save_resumed_stream_items` as distinct streaming
  persistence paths, `_stream_event_item_occurrence_key` duplicate-event suppression.
- **`run_internal/blocked_output.py`**: `_validate_resumed_session_output_guardrail_safety`
  and `_sever_blocked_output_replay_graph` (the latter is Python object-graph cleanup with no
  Rust analogue).
- **`agent_tool_state.py`**: the scope-id machinery (`get_/set_agent_tool_state_scope`),
  `_tool_call_signature` / `_scoped_tool_call_signature` dedup and `peek_agent_tool_run_result`
  — the behaviour (nested resume checkpoints keyed by tool call id) remains the simplified
  port.
- **`run_config.py`**: `SandboxConcurrencyLimits`, `SandboxArchiveLimits` and the
  `DEFAULT_MAX_*` constants belong to the sandbox agent's scope. `ToolErrorFormatter` and
  `HandoffInputFilter` remain sync-only (Python allows `MaybeAwaitable`).
- **`run_internal/items.py`**: `function_tool_error_output` program-caller JSON wrapping, and
  `drop_orphan_function_calls`'s `pruning_indexes` / `output_pruning_indexes` parameters (the
  Rust port prunes the generated history as a whole).
- **`run_internal/tool_execution.py`**: `build_litellm_json_tool_call`,
  `_uses_programmatic_output_schema`.
- **`run_internal/tool_planning.py`**: `_validate_unresolved_function_calls`,
  `_build_tool_output_index`, `_tool_call_identity` as public symbols (their effect is covered
  by the per-response dedupe and by `drop_orphan_function_calls`).

## Follow-up left open

- **`src/mcp/util.rs:340`** (`"Error fetching MCP tools"`) was **not** converted to
  `record_tool_error_on_span`. It is a tool-*listing* failure, and Python runs only tool
  *execution* errors through `get_trace_tool_error` (`tool.py:1933`,
  `run_internal/tool_execution.py:1849,2119`, `run_internal/tool_actions.py:151,607`); the MCP
  list-tools span error is not redacted there. The helper also needs
  `trace_include_sensitive_data`, which `get_all_function_tools{,_with_reserved}` does not
  receive — threading it through would change a public MCP signature that
  `src/realtime/agent.rs:275` also calls. The MCP invocation path already logs through
  `crate::logger::log_tool_action_error`.

## Tests added (follow-up round)

- `tests/core_prompt_cache_key.rs` — one hashed key per conversation run reused across turns
  and never containing the raw grouping value; session and group runs get their own kinds; an
  ungrouped run keeps its key to itself; a model without support and a caller-supplied key are
  both left alone; the key survives a `RunState` JSON round trip and an approval resume; and
  `resolve_run_grouping_id` precedence.
- `tests/core_tool_output_schema.rs` — a declared output schema is made strict and sent as
  `output_schema` on the Responses param only; a non-object schema is rejected.
- `tests/core_mcp_failure_error_function.rs` — a stub `MCPServer` whose tool always fails:
  the default handler sends the generic message, a custom handler replaces it, and
  `without_failure_error_function` propagates the error.
- Unit tests in `src/run_internal/prompt_cache_key.rs` for grouping precedence, key format and
  the caller-supplied-key guard.

## Tests added

- `tests/core_run_state.rs` — envelope shape, Rust round trip (including `custom_data`),
  reading a Python-shaped snapshot, reading the legacy flat `"1.0"` snapshot, tool-search and
  hosted-MCP interruption items, unknown item types erroring, agent-agnostic approvals, and
  unsupported schema versions.
- `tests/core_server_conversation.rs` — an interrupted turn inside a `conversation_id` run is
  serialized, round-tripped through JSON, approved and resumed; both tool outputs reach the
  server exactly once and neither model-produced `function_call` is replayed.
- `tests/core_tracing_masking.rs` — a `tracing_disabled` run nested in an enabled trace exports
  no trace, no task span, no agent span and no tool span, and the enabled run that follows
  attaches to the outer trace rather than to the mask.
- Unit tests in `src/run_internal/{tool_caller,items,history,oai_conversation}.rs` and
  `src/tool/local.rs` for caller enforcement, orphan pruning and dedupe, ordered nested
  history, server-conversation dedupe, and the typed tool-config wire shapes.

## Interop caveat

No Python environment with `agents` importable was available (`python3 -c 'import agents'`
fails, and the repo's virtualenv path is blocked by a privacy hook), so the envelope was built
by reading `run_state.py`, `usage.py`, `tracing/traces.py` and `tool_use_tracker.py` rather
than by diffing against a snapshot a Python process produced. `tests/core_run_state.rs`
exercises both directions against hand-written envelopes derived from those serializers.
A Python-produced fixture is still worth adding when an environment exists.

## Concerns

1. **Resolved.** `sha2` is now an unconditional dependency, and the prompt-cache-key row is
   ported (see the follow-up round above).
2. **Three sandbox provider tests fail, unrelated to this work.**
   `tests/sandbox_modal_cloudflare_blaxel_tests.rs` —
   `blaxel_persists_and_hydrates_through_a_tar_file`,
   `cloudflare_runs_a_full_session_against_the_worker_protocol` and
   `modal_persists_a_tar_archive_by_default` expect an older `tar --exclude` list than the one
   `src/sandbox/mount_patterns.rs` now produces (`.sandbox-blobfuse-cache`,
   `.sandbox-blobfuse-config`, `.sandbox-mountpoint-env`). That is the sandbox agent's
   in-flight work; `sandbox_tests.rs` (62 tests) passes.
2b. **Cross-agent follow-up is currently zero**, but it was close: the lowercase tool names are
   a wire break that `src/realtime/handoffs.rs`, `tests/realtime_tests.rs` and
   `tests/sandbox_tests.rs` all assert on. Those files were updated by their owners while this
   work ran, and the feature-gated suites were re-run here to confirm (see Verification).
   `src/sandbox/{capabilities/mod.rs,testing.rs}` still fail
   `clippy --features sandbox,encrypt -- -D warnings` on pre-existing `type_complexity`.
3. **Rust -> Python resume of an *interrupted* run is unsupported.** The envelope writes
   `last_processed_response: null`, which the Python runner needs to continue an interrupted
   turn. Rust rebuilds it from `model_responses.last()` and does not need the serialized copy.
   Every other combination works.
4. **Interop was validated against the Python source, not against a Python-produced
   snapshot.** `python3 -c 'import agents'` fails in this environment and the repo's
   virtualenv path is blocked by a privacy hook, so no fixture could be generated.
5. **`src/mcp/util.rs:340` was deliberately not converted** to `record_tool_error_on_span`
   despite the controller's instruction — see **Follow-up left open** for the parity and
   signature reasons. Overrule if the intent was "redact every span error regardless of
   Python's split"; the one-line change then needs a `trace_include_sensitive_data` argument
   threaded through `get_all_function_tools{,_with_reserved}`, which `src/realtime/agent.rs`
   also calls.
