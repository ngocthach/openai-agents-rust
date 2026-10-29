# OpenAI Agents SDK for Rust (unofficial)

[![crates.io](https://img.shields.io/crates/v/openai-agents-sdk.svg)](https://crates.io/crates/openai-agents-sdk)
[![docs.rs](https://docs.rs/openai-agents-sdk/badge.svg)](https://docs.rs/openai-agents-sdk)

An unofficial Rust port of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
(Python `openai-agents` 0.24). It keeps the same small set of primitives:

- **Agents**: LLMs configured with instructions, tools, guardrails and handoffs.
- **Agents as tools / Handoffs**: delegate work to other agents.
- **Guardrails**: validate agent input, output and tool calls.
- **Sessions**: conversation memory across runs.
- **Tracing**: traces that show up in the [OpenAI Traces dashboard](https://platform.openai.com/traces).

This crate is not affiliated with or endorsed by OpenAI.

## Installation

```toml
[dependencies]
openai-agents-sdk = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Optional features:

| Feature | Adds |
| --- | --- |
| `sqlite` | `memory::SqliteSession` (same schema as the Python `SQLiteSession`) |
| `mcp` | Local MCP servers over stdio and Streamable HTTP (via [`rmcp`](https://crates.io/crates/rmcp)) |

Set your API key before running agents:

```bash
export OPENAI_API_KEY=sk-...
```

## Hello world

```rust,no_run
use openai_agents_sdk::{Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::<()>::new("Assistant").with_instructions("You are a helpful assistant");
    let result = Runner::run(&agent, "Write a haiku about recursion in programming.").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    openai_agents_sdk::flush_traces_async().await; // deliver the trace before exiting
    Ok(())
}
```

## Tools

Any async closure becomes a tool. Argument types derive `serde::Deserialize` and
`schemars::JsonSchema`; doc comments become parameter descriptions, and the schema is converted to
OpenAI strict mode.

```rust,no_run
use openai_agents_sdk::{Agent, FunctionTool, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to get the weather for.
    city: String,
}

# async fn demo() -> openai_agents_sdk::Result<()> {
let get_weather = FunctionTool::new("get_weather", "Get the weather for a city", |_ctx, a: WeatherArgs| async move {
    Ok(format!("The weather in {} is sunny", a.city))
});
let agent = Agent::<()>::new("Weather agent").with_tool(get_weather);
let result = Runner::run(&agent, "What's the weather in Tokyo?").await?;
# Ok(()) }
```

Hosted OpenAI tools are available too: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`,
`ImageGenerationTool` and `HostedMCPTool`.

## Handoffs and agents as tools

```rust,no_run
use std::sync::Arc;
use openai_agents_sdk::{Agent, Runner, handoff};

# async fn demo() -> openai_agents_sdk::Result<()> {
let history = Arc::new(Agent::<()>::new("History Tutor").with_handoff_description("History questions"));
let math = Agent::<()>::new("Math Tutor").with_instructions("Explain math step by step.");
let triage = Agent::<()>::new("Triage")
    .with_handoff(handoff(history))                                  // hand off the conversation
    .with_tool(math.as_tool("math_expert", "Answers math questions")); // or keep control
let result = Runner::run(&triage, "Who was the first US president?").await?;
println!("Answered by {}", result.last_agent.name);
# Ok(()) }
```

## Structured output

```rust,no_run
# use openai_agents_sdk::{Agent, Runner};
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct CalendarEvent { name: String, date: String, participants: Vec<String> }

# async fn demo() -> openai_agents_sdk::Result<()> {
let agent = Agent::<()>::new("Extractor").with_output_type::<CalendarEvent>();
let event: CalendarEvent = Runner::run(&agent, "Alice and Bob meet Friday").await?.final_output_as()?;
# Ok(()) }
```

## Context

`Agent<C>` is generic over your context type. The context is passed to every tool, hook and
guardrail and is never sent to the model.

```rust,no_run
use openai_agents_sdk::{Agent, FunctionTool, RunOptions, Runner};

struct UserInfo { name: String }

# async fn demo() -> openai_agents_sdk::Result<()> {
let whoami = FunctionTool::<UserInfo>::new_no_args("whoami", "The user's name", |ctx| async move {
    Ok(ctx.context().name.clone())
});
let agent = Agent::<UserInfo>::new("Assistant").with_tool(whoami);
let opts = RunOptions::new(UserInfo { name: "John".into() });
let result = Runner::run_with_options(&agent, "What is my name?", opts).await?;
# Ok(()) }
```

## Streaming

```rust,no_run
use futures::StreamExt;
# use openai_agents_sdk::{Agent, Runner};
# async fn demo() -> openai_agents_sdk::Result<()> {
# let agent = Agent::<()>::new("A");
let mut result = Runner::run_streamed(&agent, "Tell me 5 jokes");
{
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        if let Some(delta) = event?.text_delta() {
            print!("{delta}");
        }
    }
}
println!("\nfinal: {:?}", result.final_output_text());
# Ok(()) }
```

## Human in the loop

```rust,no_run
# use openai_agents_sdk::{Agent, FunctionTool, RunState, Runner};
# async fn demo() -> openai_agents_sdk::Result<()> {
let cancel = FunctionTool::<()>::new_no_args("cancel_order", "Cancel the order", |_c| async { Ok("cancelled") })
    .with_needs_approval(true);
let agent = Agent::<()>::new("Support").with_tool(cancel);
let mut result = Runner::run(&agent, "Cancel my order").await?;
while !result.interruptions.is_empty() {
    let mut state: RunState = result.to_state(); // serializable with to_json()
    for item in state.get_interruptions() {
        state.approve(&item, false); // or state.reject(&item, false, Some("reason".into()))
    }
    result = Runner::run(&agent, state).await?;
}
# Ok(()) }
```

## Sessions

```rust,no_run
use std::sync::Arc;
use openai_agents_sdk::{Agent, InMemorySession, RunOptions, Runner, Session};
# async fn demo() -> openai_agents_sdk::Result<()> {
let agent = Agent::<()>::new("Assistant");
let session: Arc<dyn Session> = Arc::new(InMemorySession::new("conversation_123"));
Runner::run_with_options(&agent, "What city is the Golden Gate Bridge in?", RunOptions::new(()).session(session.clone())).await?;
let r = Runner::run_with_options(&agent, "What state is it in?", RunOptions::new(()).session(session)).await?;
# Ok(()) }
```

Built-in sessions: `InMemorySession`, `SqliteSession` (feature `sqlite`) and
`OpenAIConversationsSession`. Implement the `Session` trait for other stores.

## Tracing

Runs are traced by default and exported to OpenAI in the background. Rust has no automatic flush
at process exit, so call `flush_traces()` (blocking) or `flush_traces_async()` before your program
ends. `Runner::run_sync` flushes for you. Disable tracing with `set_tracing_disabled(true)`,
`RunConfig::with_tracing_disabled(true)` or `OPENAI_AGENTS_DISABLE_TRACING=1`. Group several runs
into one trace with `trace("My workflow").run(async { ... })`, and add your own processors with
`add_trace_processor` or `set_trace_processors`.

## Models

The default provider uses the OpenAI Responses API with the model in `OPENAI_DEFAULT_MODEL`
(default `gpt-5.6-luna`). Use `OpenAIChatCompletionsModel` with `OpenAIClient::with_base_url` for
OpenAI-compatible providers, `set_default_openai_api(OpenAIApi::ChatCompletions)` to switch the
default, `MultiProvider` for prefix routing, or implement the `Model` trait yourself.
`ModelSettings` covers temperature, reasoning, tool choice, retries (`ModelRetrySettings`),
per-attempt timeouts and raw request extras.

## Python to Rust map

| Python | Rust |
| --- | --- |
| `Agent(name=..., instructions=...)` | `Agent::<C>::new(name).with_instructions(...)` |
| `@function_tool` | `FunctionTool::new(name, description, async closure)` |
| `Runner.run / run_sync / run_streamed` | `Runner::run / run_sync / run_streamed` (+ `_with_options`) |
| `RunConfig(...)` | `RunConfig::new().with_...()` |
| `result.final_output` | `result.final_output` (`serde_json::Value`), `final_output_text()`, `final_output_as::<T>()` |
| `handoff(agent, ...)` | `handoff(agent).with_on_handoff(...).with_input_filter(...)` |
| `agent.as_tool(...)` | `agent.as_tool(...)` / `agent.as_tool_with(AgentAsToolOptions)` |
| `@input_guardrail` / `@output_guardrail` | `InputGuardrail::new` / `OutputGuardrail::new` |
| `RunState.from_json` | `RunState::from_json` |
| `SQLiteSession` | `memory::SqliteSession` (feature `sqlite`) |
| `MCPServerStdio` / `MCPServerStreamableHttp` | `mcp::MCPServerStdio` / `mcp::MCPServerStreamableHttp` (feature `mcp`) |
| `agents.testing` | `testing::FakeModel` |

## Not yet ported

Realtime agents, voice pipelines, sandbox agents, the Codex tool, local computer/shell/apply-patch
tool execution, hosted multi-agent, Responses WebSocket transport, third-party adapters (LiteLLM,
Any-LLM), and the MCP HTTP+SSE transport (deprecated upstream). See
[`docs/project-roadmap.md`](docs/project-roadmap.md).

## Examples

See [`examples/`](examples): `hello_world`, `tools`, `handoffs`, `structured_output`,
`stream_text`, `stream_items`, `guardrails`, `human_in_the_loop`, `agents_as_tools`,
`sessions_sqlite`, `chat_completions_provider`, `mcp_stdio`, `custom_trace_processor`.

```bash
cargo run --example hello_world
cargo run --example sessions_sqlite --features sqlite
```

## License

Apache-2.0. This crate ports the MIT-licensed OpenAI Agents SDK for Python; see
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).
