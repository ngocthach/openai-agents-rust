# Phase 05: Tracing, sessions, MCP, extensions, testing, REPL

Files: `src/tracing/*`, `src/memory/*`, `src/mcp/*`, `src/extensions/*`, `src/testing.rs`, `src/repl.rs`

- Tracing: TraceProvider (global, processors list, disabled flag), Trace, Span<SpanData>, span data types (agent, function, generation, response, handoff, guardrail, custom, mcp_tools, task, turn), task-local scope, BatchTraceProcessor (background tokio task, flush, shutdown), BackendSpanExporter (same JSON payload + headers as Python), `set_trace_processors`, `add_trace_processor`, `set_tracing_disabled`, `set_tracing_export_api_key`, `flush_traces`.
- Sessions: trait + InMemorySession + SqliteSession (same schema as Python: agent_sessions, agent_messages) + OpenAIConversationsSession + SessionSettings.
- MCP: `MCPServer` trait; `MCPServerStdio`, `MCPServerStreamableHttp`, `MCPServerSse` on rmcp; tool filters (static/dynamic); tools list caching; conversion to FunctionTool; approval policy; `MCPServerManager`.
- Extensions: handoff_filters::remove_all_tools, handoff_prompt, visualization (DOT string).
- testing::FakeModel scripted responses; repl::run_demo_loop.

Validation: unit tests for exporter payload shape, sqlite round trip, mcp tool conversion (schema), fake model.
