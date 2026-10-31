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

## 0.9.0 (released)

- Whole-suite parity measured and burned down: all 12,641 Python test cases classified (7,981 ported, 604
  covered, 3,987 Python-only, 69 not ported); sandbox, runner, MCP and resume divergences fixed

## 0.8.0 (released)

- Completeness pass over the top-level Python test files: scripted model, server conversation tracker,
  `as_tool`, computer actions, extension filters, MCP tool metadata, visualization; typed tool
  arguments are validated against their schema bounds

## 0.7.0 (released)

- Sandbox and runner-tracing parity completed: pause releases the sandbox like Python's, bounded reads
  sanitize failures, the E2B client has Python's timeouts

## 0.6.0 (released)

- Python test-suite parity: the Python SDK's tests were triaged and ported as Rust tests, and the
  divergences they exposed were fixed (runner, `RunState`, approvals, handoffs, guardrails, tracing,
  models, MCP and sessions, realtime, voice, Codex, sandbox and cloud providers)
- Per-case classification (ported, covered, Python-only) lives in `plans/reports/py-test-parity-*.md`

## 0.5.0 (released)

- Independent-audit fixes: handoff failure recovery, in-process archive validation and limits, `extract()`,
  strict `RunState` context, typed realtime reasoning, verbose stdout logging

## 0.4.0 (released)

- Closes the remaining 0.3.0 parity rows: provider adapters (Anthropic, Bedrock, Vertex, Azure, watsonx),
  Python-interoperable interrupted `RunState`, Docker/sandbox core rows, MCP transport options,
  trace state reattach, realtime and cloud provider rows

## 0.3.0 (released)

- Sandbox: mounts (bind, Docker volume, in-container, cloud buckets), PTY sessions, memory,
  skills, snapshots and resumable session state, handoff graph preparation (cycles included)
- Cloud sandbox providers: E2B, Modal, Daytona, Vercel, Cloudflare, Blaxel
- Realtime: SIP model, playback tracker, generation-scoped guardrail interrupts, scripted model
- `EncryptedSession` is Fernet-compatible with Python; compaction suffix replacement
- `LitellmModel` / `AnyLLMModel` over OpenAI-compatible endpoints; `#[function_tool]` macro
- Retry-after / `x-should-retry` headers on API errors; shared log redaction flags
- Runner: Python-compatible `RunState` (schema 1.21), identity-based server conversation
  tracking, programmatic tool calling, `allowed_callers`, prompt cache keys, MCP failure handlers
- Tracing: pluggable `TraceProvider`, disabled-trace masking, exporter `Retry-After`, error
  redaction (`logger` module)
- Sandbox mount entries: S3, S3 Files, GCS, R2, Azure Blob, Box (FUSE, mountpoint, rclone)

## Differences from Python

| Area | Difference |
| --- | --- |
| LiteLLM / any-llm | Registry of OpenAI-compatible endpoints instead of the Python libraries |
| Modal sandbox | Uses the `modal` Python package through a helper (no public REST API) |
| `sqlx` feature | Needs Rust 1.94 (everything else builds on 1.89) |
| `FunctionTool` | `output_type` is `with_output_type::<T>()` (schema derived with `schemars`) |
| Modal selectors | Live `modal.Image`/`modal.Sandbox` objects are replaced by id/tag selectors |
| Docker sandbox | Uses the Docker CLI; no raw-socket exec transport or PTY `preexec_fn` signal reset |
| WebSocket logging | The dependency's DEBUG logs cannot be filtered by a library; wrap your logger in `WebSocketDebugFilter` |

Module-level divergences are listed in `plans/reports/*-report.md`.
