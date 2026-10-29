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

## Not yet ported

| Area | Notes |
| --- | --- |
| Realtime agents | WebSocket realtime sessions and audio |
| Voice pipeline | STT / TTS workflows |
| Sandbox agents | Workspace manifests, sandbox clients |
| Local execution tools | `ComputerTool`, `ShellTool`, `LocalShellTool`, `ApplyPatchTool` loops |
| Tool search / programmatic tool calling | `ToolSearchTool`, namespaces, deferred loading beyond the `defer_loading` flag |
| Responses WebSocket transport | `responses_websocket_session` |
| Hosted multi-agent (experimental) | Responses beta |
| Codex tool | Experimental upstream |
| Third-party adapters | LiteLLM, Any-LLM |
| Extra sessions | Redis, SQLAlchemy-equivalent, MongoDB, Dapr, encrypted, compaction |
| MCP HTTP+SSE transport | Deprecated by the MCP project; `rmcp` 3 does not ship a client |
| `#[function_tool]` macro | Closure-based `FunctionTool::new` covers the same need |
