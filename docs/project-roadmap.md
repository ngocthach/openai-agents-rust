# Project roadmap

Last updated: 2026-10-29.

## 0.1.0 (released)

- Agents, function tools (typed args, approvals, guardrails, timeouts), hosted tools
- Handoffs (filters, nested history beta), agents as tools (structured input, nested approvals)
- Runner: run / run_sync / run_streamed, max turns, error handlers, hooks, RunState resume
- OpenAI Responses and Chat Completions models, providers, retries, timeouts
- Sessions: in-memory, SQLite, OpenAI Conversations
- Tracing with OpenAI exporter and custom processors
- MCP stdio and Streamable HTTP servers, tool filters, approvals, manager
- Extensions: handoff filters, recommended prompt, DOT visualization, REPL, FakeModel

## 0.2.0 (released)

- Local runtime tools: `ShellTool`, `LocalShellTool`, `ApplyPatchTool` (V4A `apply_diff`,
  `WorkspaceEditor`), `ComputerTool`, `CustomTool`, with approvals and `on_approval`
- `ToolSearchTool`, `tool_namespace`, deferred loading validation, `ProgrammaticToolCallingTool`
- Responses WebSocket transport and `ResponsesWebSocketSession` (feature `websocket`)
- Retry policies (`retry::retry_policies`) with provider advice and replay-safety vetoes
- Chat Completions `reasoning_content` parsing, streaming and DeepSeek replay
- Tool name collision policy, output guardrail blocked-output redaction, agent harness id
- Realtime agents (feature `realtime`), voice pipeline (feature `voice`), sandbox agents with
  unix-local and Docker clients (feature `sandbox`)
- Sessions: advanced SQLite, OpenAI Responses compaction, encrypted, Redis, MongoDB, sqlx, Dapr
- MCP legacy HTTP+SSE transport
- Extensions: tool output trimmer, Codex tool, hosted multi-agent (experimental)

## Not ported

| Area | Reason |
| --- | --- |
| LiteLLM / Any-LLM adapters | Wrap Python libraries; use `OpenAIChatCompletionsModel` with a base URL or a custom `Model` |
| Cloud sandbox providers (E2B, Modal, Daytona, Vercel, Cloudflare, Blaxel) | Vendor SDK integrations; implement `SandboxClient` for a provider |
| Sandbox mounts, PTY sessions, sandbox memory and skills | Large provider-specific surface; PTY needs `unsafe` |
| Realtime SIP model and playback tracker | Telephony-specific |
| `#[function_tool]` macro | Closure-based `FunctionTool::new` covers the same need |
| Retry-after header capture | `AgentsError::ApiError` does not carry response headers; policies use body hints and backoff |

Module-level divergences are listed in `plans/reports/*-port-report.md`.
