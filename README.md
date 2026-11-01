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
openai-agents-sdk = "0.14"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Optional features:

| Feature | Adds |
| --- | --- |
| `sqlite` | `memory::SqliteSession` and `memory::AdvancedSqliteSession` (same schemas as Python) |
| `mcp` | Local MCP servers over stdio, Streamable HTTP and legacy HTTP+SSE (via [`rmcp`](https://crates.io/crates/rmcp)) |
| `websocket` | Responses API over WebSocket (`OpenAIResponsesWSModel`, `ResponsesWebSocketSession`) and the experimental hosted multi-agent model |
| `realtime` | Realtime agents and sessions over the OpenAI realtime WebSocket API |
| `voice` | Voice pipeline with OpenAI speech-to-text and text-to-speech |
| `sandbox` | Sandbox agents with unix-local and Docker workspaces, mounts, PTY sessions, memory, skills, snapshots, and E2B, Modal, Daytona, Vercel, Cloudflare and Blaxel providers |
| `macros` | `#[function_tool]` attribute macro |
| `encrypt` | `memory::EncryptedSession` (Fernet, wire-compatible with the Python SDK) |
| `redis` / `mongodb` / `sqlx` | `RedisSession`, `MongoDBSession`, `SqlxSession` (PostgreSQL/MySQL; needs Rust 1.94) |
| `provider-auth` | Request signing and OAuth for Bedrock, SageMaker and Vertex AI models in `extensions::models` |

`DaprSession` and `OpenAIResponsesCompactionSession` need no feature.

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

With the `macros` feature, `#[function_tool]` builds the same tool from a plain function and
its doc comment:

```rust,ignore
use openai_agents_sdk::function_tool;

/// Get the weather for a city.
///
/// # Arguments
/// * `city` - The city to get the weather for.
#[function_tool]
async fn get_weather(city: String) -> Result<String, openai_agents_sdk::AgentsError> {
    Ok(format!("The weather in {city} is sunny"))
}

let agent = Agent::<()>::new("Weather agent").with_tool(get_weather_tool::<()>());
```

Hosted OpenAI tools are available too: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`,
`ImageGenerationTool`, `HostedMCPTool`, `ToolSearchTool` and `ProgrammaticToolCallingTool`.
Group function tools with `tool_namespace` and defer them with `with_defer_loading`.

Local runtime tools run on your machine through the runner, with the same approval flow as
function tools: `ShellTool` (your executor, or `ShellTool::with_system_shell`),
`LocalShellTool`, `ApplyPatchTool` (any `editor::ApplyPatchEditor`, e.g. `WorkspaceEditor`),
`ComputerTool` (any `computer::Computer`) and `CustomTool` (raw string input).

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

## Logging

The SDK logs through the [`log`](https://crates.io/crates/log) facade on the `openai.agents`
target, so nothing is printed until a backend is installed. `enable_verbose_stdout_logging()`
installs a built-in one for SDK messages only (the port of Python's call of the same name); any
other backend (`env_logger`, `tracing-subscriber`, ...) works too — just enable `openai.agents` at
debug level. Model and tool payloads stay redacted either way until
`set_dont_log_model_data(false)` / `OPENAI_AGENTS_DONT_LOG_MODEL_DATA=0` allow them.

```rust
// Returns false if another log backend already owns the process.
openai_agents_sdk::enable_verbose_stdout_logging();
```

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
| `@function_tool` | `#[function_tool]` (feature `macros`) or `FunctionTool::new(name, description, async closure)` |
| `LitellmModel` / `AnyLLMModel` | `extensions::models::LitellmModel` / `AnyLLMModel` |
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
| `enable_verbose_stdout_logging()` | `enable_verbose_stdout_logging()` (returns `false` if another `log` backend is already installed) |
| `ShellTool` / `ApplyPatchTool` / `ComputerTool` | `tool::ShellTool` / `tool::ApplyPatchTool` / `tool::ComputerTool` |
| `retry_policies.any(...)` | `retry::retry_policies::any(vec![...])` |
| `RealtimeAgent` / `RealtimeRunner` | `realtime::RealtimeAgent` / `realtime::RealtimeRunner` (feature `realtime`) |
| `VoicePipeline` | `voice::VoicePipeline` (feature `voice`) |
| `SandboxAgent` | `sandbox::SandboxAgent` (feature `sandbox`) |
| `codex_tool(...)` | `extensions::codex::codex_tool(...)` |

## Differences from Python

- LiteLLM and any-llm are Python libraries, so `extensions::models::{LitellmModel, AnyLLMModel}`
  resolve `provider/model` names against a registry of OpenAI-compatible endpoints (Anthropic,
  Gemini, Mistral, Groq, DeepSeek, OpenRouter, xAI, Ollama, a LiteLLM proxy, ...).
- Modal sandboxes drive the official `modal` Python package through a bundled helper, because
  Modal has no public REST API.
- `#[function_tool]` (feature `macros`) generates a `<name>_tool()` constructor instead of
  replacing the function.

Module-level notes live in `plans/reports/*-report.md`.

## Examples

See [`examples/`](examples): `hello_world`, `tools`, `handoffs`, `structured_output`,
`stream_text`, `stream_items`, `guardrails`, `human_in_the_loop`, `agents_as_tools`,
`sessions_sqlite`, `sessions_encrypted`, `chat_completions_provider`, `mcp_stdio`,
`custom_trace_processor`, `shell_and_apply_patch`, `responses_websocket`, `realtime_text`,
`voice_static`, `sandbox_local`, `codex_tool`.

```bash
cargo run --example hello_world
cargo run --example sessions_sqlite --features sqlite
```

## License

Apache-2.0. This crate ports the MIT-licensed OpenAI Agents SDK for Python; see
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).
