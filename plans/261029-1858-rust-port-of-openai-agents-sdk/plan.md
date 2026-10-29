# Rust port of the OpenAI Agents SDK

Status: in progress (2026-10-29)
Source of truth: `/Users/ngocthach/myGit/openai-agents-python` at 0.24.0 (commit cd26bbbc, 2026-10-27) and https://openai.github.io/openai-agents-python/

## Decisions (made on the user's behalf)

| Topic | Decision | Why |
| --- | --- | --- |
| Crate name | `openai-agents-sdk` (lib `openai_agents_sdk`) | `openai-agents`, `openai-agents-rust`, `openai_agents` are already owned by third parties on crates.io. `openai-agents-sdk` is free and matches the Python package description. |
| Edition / MSRV | edition 2024, rust 1.85+ | Current toolchain is 1.96; async closures and `impl Trait` in traits are stable. |
| Async runtime | tokio | De facto standard; `reqwest` and `rmcp` require it. |
| HTTP / OpenAI client | own thin client on `reqwest` + `serde_json` (Responses API and Chat Completions, SSE streaming) | `async-openai` lags the Responses API and forces its types everywhere. Raw `serde_json::Value` passthrough keeps the crate forward compatible with new Responses item types, exactly like the Python SDK which stores `raw_item`. |
| Function tools | `FunctionTool::new(name, desc, async closure)` with typed args via `schemars::JsonSchema` + `serde::Deserialize`; no proc macro in 0.1 | KISS: a single crate; a `#[function_tool]` macro can be added later without breaking the closure API. |
| Structured output | `Agent::with_output_type::<T>()` where `T: JsonSchema + DeserializeOwned`; strict schema conversion ported from `strict_schema.py` | Mirrors Pydantic behaviour using schemars. |
| Context | `Agent<C>` generic over `C: Send + Sync + 'static`, shared as `Arc<C>` inside `RunContext<C>`; mutable state uses interior mutability | Rust equivalent of the Python "grab bag" context. |
| Tracing | own `tracing` module (Trace/Span/processors) exporting to `https://api.openai.com/v1/traces/ingest` with the same wire format; current trace/span kept in a tokio `task_local` | Keeps parity with OpenAI dashboard; no dependency on the `tracing` crate. |
| Sessions | `Session` trait, `InMemorySession`, `SqliteSession` (feature `sqlite`, rusqlite bundled), `OpenAIConversationsSession` | Covers the built-in Python set that does not require extra services. Redis/SQLAlchemy/Mongo/Dapr are left for follow-up crates. |
| MCP | feature `mcp` on `rmcp` (official Rust MCP SDK): stdio, streamable HTTP, SSE | Only maintained Rust MCP client. |
| Out of scope for 0.1 | Realtime, Voice, Sandbox agents, Codex tool, hosted multi-agent, Any-LLM/LiteLLM adapters, Programmatic Tool Calling execution, ComputerTool/ShellTool/ApplyPatch local execution loops | Each depends on external systems (audio, containers, Codex CLI) and is >20k Python lines. Documented in the roadmap. Hosted tool configs (web search, file search, code interpreter, image generation, hosted MCP) ARE supported because they are pure request payloads. |

## Phases

1. `phase-01-foundation.md` — crate scaffold, errors, usage, model settings, items, agent output schema, strict schema. 
2. `phase-02-tools-agents-handoffs.md` — Tool enum, FunctionTool, hosted tools, Agent builder, handoffs, guardrails, hooks, run context.
3. `phase-03-models.md` — Model/ModelProvider traits, OpenAI HTTP client, Responses model (sync + SSE stream), Chat Completions model (sync + stream), OpenAIProvider, MultiProvider, default models, retries.
4. `phase-04-runner.md` — Runner loop (run / run_sync / run_streamed), tool execution, handoffs, guardrails, approvals + RunState, sessions, error handlers, stream events, results.
5. `phase-05-tracing-sessions-mcp.md` — tracing module + OpenAI exporter, sessions (memory, sqlite, conversations), MCP servers, extensions (handoff filters/prompt, visualization), testing FakeModel, REPL.
6. `phase-06-docs-examples-publish.md` — README, docs, examples, CI, `cargo publish`.

## Acceptance criteria

- `cargo build --all-features`, `cargo clippy --all-features -- -D warnings`, `cargo test --all-features` all pass.
- Examples compile: hello world, tools, handoffs, structured output, streaming, guardrails, sessions, HITL approval, tracing, MCP stdio.
- Live smoke test against the OpenAI API when `OPENAI_API_KEY` is present (responses + chat completions + streaming).
- Published on crates.io as `openai-agents-sdk 0.1.0`.
