# Phase 02: Tools, agents, handoffs, guardrails, hooks

Files: `src/tool.rs`, `src/tool_context.rs`, `src/run_context.rs`, `src/agent.rs`, `src/handoffs.rs`, `src/guardrail.rs`, `src/tool_guardrails.rs`, `src/lifecycle.rs`, `src/prompts.rs`

- `Tool` enum: Function(Arc<FunctionTool>), WebSearch, FileSearch, CodeInterpreter, ImageGeneration, HostedMCP, Custom(Value passthrough).
- `FunctionTool` builder: typed args, raw JSON args, `needs_approval` (bool|closure), `is_enabled`, failure error fn, timeout (+behaviour), tool guardrails, strict schema.
- `RunContext<C>`: Arc<C>, usage (Mutex), approvals map, tool_input.
- `ToolContext<C>`: tool_name, call_id, arguments.
- `Agent<C>` builder; `Instructions` enum (static / dynamic async fn); `ToolUseBehavior` enum; `clone_with`; `as_tool`.
- `Handoff<C>` + `handoff()` builder with on_handoff / input_type / input_filter / is_enabled; `HandoffInputData`.
- Guardrails: InputGuardrail/OutputGuardrail (closures), results, run_in_parallel; tool guardrails.
- `RunHooks`/`AgentHooks` traits with default no-op methods.

Validation: unit tests for tool schema generation, handoff naming, agent builder defaults.
