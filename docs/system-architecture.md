# System architecture

Last updated: 2026-10-29. Port of `openai-agents` (Python) 0.24.0.

## Layers

| Layer | Modules | Responsibility |
| --- | --- | --- |
| Public primitives | `agent`, `tool`, `handoffs`, `guardrail`, `tool_guardrails`, `lifecycle`, `prompts` | Configure agents, tools, handoffs, guardrails, hooks |
| Run API | `run`, `run_config`, `result`, `run_state`, `stream_events`, `run_error_handlers` | `Runner` entry points, results, resumable state |
| Agent loop | `run_internal` (private) | Turn loop, tool execution, approvals, handoffs, sessions, tracing spans |
| Models | `models::{interface, openai_client, openai_responses, openai_chatcompletions, openai_provider, multi_provider, default_models}` | `Model`/`ModelProvider` traits and OpenAI HTTP implementations |
| Data | `items`, `usage`, `model_settings`, `agent_output`, `strict_schema` | Responses-format items, usage, settings, structured output schemas |
| Memory | `memory` | `Session` trait, in-memory, SQLite (feature `sqlite`), OpenAI Conversations |
| Tracing | `tracing` | Traces/spans, task-local scope, processors, OpenAI exporter |
| MCP | `mcp` (feature `mcp`) | stdio / Streamable HTTP servers on `rmcp`, tool conversion |
| Extensions | `extensions`, `repl`, `testing` | Handoff filters and prompts, DOT visualization, REPL, scripted `FakeModel` |

## Run loop

```
Runner::run* -> run_internal::run_loop
  restore RunState (if resuming) | build input
  trace (reuse current or create) -> task span -> loop:
    agent span (per agent) -> turn span:
      agent start hooks, load session history (turn 1)
      input guardrails (blocking first, parallel with the model call)
      resolve model + settings (model defaults, run overrides, reset tool_choice)
      build model input (prefix + generated items | server-managed delta)
      call_model_input_filter -> model call (retry, timeout, streaming)
      classify output: messages, reasoning, function calls, handoffs, hosted items
      execute function tools (approvals, tool guardrails, hooks, timeouts)
      hosted MCP approvals
      -> Interruption | Handoff (input filter / nested history) | tool_use_behavior | final output | run again
    output guardrails, agent end hooks, session save
```

## Key design decisions

- **Items are JSON.** Input items wrap `serde_json::Value`; output items type the variants the
  runner interprets (message, function call, reasoning) and keep others verbatim. New Responses
  item types flow through without code changes.
- **Context generic `C`.** `Agent<C>`, tools and hooks share `RunContext<C>` (an `Arc<C>` plus
  usage and approval state). Generic types implement `Clone`/`Debug` manually so `C` needs neither.
- **Model trait is context-free.** The runner converts tools and handoffs to `ToolSpec`s before
  calling a model, so custom `Model` implementations do not depend on `C`.
- **Streaming unifies both APIs.** Chat Completions chunks are re-emitted as Responses stream
  events, so the runner only understands one event format.
- **Nested agent tools share the run context.** A nested run that pauses for approval stores its
  serialized `RunState` in the parent context keyed by the outer tool call id; the outer run
  surfaces the interruption and resumes the nested run later.
- **Tracing export runs on a dedicated thread** with its own runtime, so exports survive the
  caller's runtime. `flush_traces()` must be called before process exit.
- **Large futures are boxed** in the loop to keep stack usage low in debug builds.
