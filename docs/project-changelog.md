# Changelog

## 0.5.0 (2026-10-30)

Closes the gaps found by an independent audit of the public Python surface (1845 names, options and
documented features checked). See `plans/reports/independent-parity-audit-report.md` and the
`audit-fixes-*`, `final-divergences-closure` and `final-sweep` reports.

Added:

- `Handoff::with_failure_error_function`: a failed handoff invocation becomes a tool output and the
  agent runs again instead of ending the run (Python behavior).
- Sandbox archives are validated in process before any member is written (absolute paths, `..`,
  link and device members, zip-slip), with `SandboxArchiveLimits`, a public `extract()` for tar and
  zip, `clone_repo` / `ensure_git_available`; workspace restore no longer shells out to `tar`.
- `custom_data_extractor` on computer, apply-patch and custom tools; typed `RealtimeReasoningConfig`;
  `enable_verbose_stdout_logging()`; `WebSocketDebugFilter`; `RunState` strict context checking and
  `context_override`; `Mount::ephemeral`; Docker concurrent stdin/stdout pumping.

Breaking changes (pre-1.0):

- Per-call approvals on Python-envelope snapshots from schema 1.15 re-raise when the call is not in
  the tool-invocation ledger (Python rule); flat Rust `"1.0"` snapshots keep the old tolerance.
- Archive hydrate refuses symlink members whose target is absolute or leaves the archive; in-root
  absolute links are rebased at capture time, as in Python.
- `RealtimeSessionModelSettings::reasoning` is a `RealtimeReasoningConfig`; `Handoff`,
  `ComputerTool`, `ApplyPatchTool`, `CustomTool`, `SandboxRunConfig` and `FileEntry` gained public
  fields; `Mount` has its own `ephemeral` flag.
- SDK log calls use the `openai.agents` target (`logger::LOG_TARGET`).

## 0.4.0 (2026-10-30)

Closes the parity rows left open in 0.3.0. Per-area detail is in `plans/reports/*-final-parity-report.md`.

Added:

- Providers: native Anthropic Messages, AWS Bedrock and SageMaker (SigV4), Vertex AI (Google OAuth),
  Azure OpenAI, watsonx.ai and more OpenAI-compatible endpoints, model listing, thinking-block
  preservation. Signing needs the new `provider-auth` feature.
- `RunState` interop with Python for interrupted runs (`last_processed_response`), tool-invocation
  ledger with Python-identical digests, Python-generated fixtures in `tests/fixtures/`.
- Sandbox: Docker labels, exposed ports, path-grant validation and resume checks, named volumes,
  snapshot fingerprint fast path, opt-in per-agent sessions, live manifest updates, exposed-port
  resolution for every provider, read-only path grants on raw writes.
- MCP: unlimited retries, isolated-session retry, credential redaction, streamable HTTP options
  (session id, timeouts, auth), message handler, per-server tool policy.
- Tracing: trace state reattach, shutdown guard, per-processor panic isolation, `SpanBuilder`.
- WebSocket keepalive options and replay-safety retry advice; realtime `async_tool_calls`.

Breaking changes (pre-1.0):

- `Agent::get_all_tools` takes `self: &Arc<Self>`; `RunState` has new fields; a reused tool call
  id with a changed identity is a `ModelBehaviorError`.
- `RealtimeToolErrorArgs` is an alias of the core `ToolErrorFormatterArgs`; realtime sessions end
  on transport errors, failed tool tasks and guardrail tripwires; `RealtimeContent` variants carry
  extra fields.
- Chat Completions streams tool-call arguments incrementally; WebSocket transport error texts
  changed; `ModelProvider`/`Model` gained cleanup hooks with defaults.
- Vercel: `AllowAll`/`DenyAll` network policies use the `{mode}` wire shape; Daytona rclone mounts
  no longer add `--allow-other` (matches Python).
- `ModalSandboxClientOptions` gained public fields; `ErrorCode` gained variants.
- Voice STT websocket caps messages at 1 MiB and sends keepalives every 20 s.
- `serde_json` now preserves key order (Python dict order); `reqwest` multipart is always enabled.

## 0.3.0 (2026-10-30)

Completes the port. See the 0.3.0 section of `docs/project-roadmap.md`.

Breaking changes (pre-1.0):

- Errors: `AgentsError::ApiError` has a `headers` field; `AgentsError::Sandbox` is a new variant.
- `EncryptedSession` now uses Fernet (Python-compatible); items written by 0.2.0 are not readable.
- New `Session`, `Model` and `ModelProvider` trait methods have default implementations;
  `ModelProvider` is `#[async_trait]` (impls that override `close()` need the attribute).
- Tool names are lowercased: `Agent::new("Billing Agent")` hands off through
  `transfer_to_billing_agent`, and default `as_tool` names are lowercase.
- `RunState` serializes Python's schema `"1.21"` envelope; older Rust `"1.0"` snapshots are
  still read, but new snapshots cannot be read by 0.2.0.
- Generic over the context type now: `ToolErrorFormatterArgs<'a, C>`, `MCPConfig<C>` (no longer
  `PartialEq`), `OutputGuardrailBlockedMessage` and `BlockedMessageFormatter`.
- `AgentAsToolOptions::on_stream` receives `AgentToolStreamEvent<C>`; `input_builder` receives
  `&StructuredToolInput<'_>`.
- `tracing::TraceProvider` is a trait; the default implementation is `DefaultTraceProvider`.
- New enum variants and public struct fields (`RunItemKind::Compaction`,
  `ToolErrorKind::ToolException`, `FunctionToolSpec::output_json_schema`, `RunState`,
  `ToolContext`, `SandboxError`, `ErrorCode`, `SnapshotSpec`, `Entry` fields); use the builders
  or `..Default::default()`.
- `SandboxRunner::run_streamed*` returns `SandboxStreamedRun<C>`; `Manifest::apply` returns a
  `MaterializationResult`.
- Behavior:
  - Output guardrails combined with server-managed conversations (`conversation_id` /
    `previous_response_id`) are rejected at run start.
  - A failing function tool cancels its sibling calls in the same batch.
  - Server-conversation deltas are tracked by item identity instead of an index.
  - Official OpenAI models receive a generated `prompt_cache_key` unless one is set.
  - Chat Completions sends `tool_choice` without tools and multimodal tool output as content
    parts; an unpaired tool output and colliding `extra_args` keys are errors.
  - Streaming terminal-failure error text follows Python's wording.
  - Tool error span data and invalid-JSON tool errors are redacted unless sensitive data
    logging is allowed.
  - Unix-local sandbox commands use the workspace root as `HOME`.
- Vercel sandbox: snapshot references use Python's `UC_VERCEL_SNAPSHOT_V1` magic (older Rust
  Vercel snapshots do not decode); runtimes map to images (default `node24`) and unknown aliases
  are rejected; `NetworkPolicy::CustomRules` is new; S3 mount flags follow writability, mount
  credentials are validated, and failing mounts report a redacted message.
- Modal sandbox: `ModalSandboxClientOptions` has new public fields (cpu/memory limits, restore
  timeout).

## 0.2.0 (2026-10-29)

Ports the rest of the Python SDK surface. See the 0.2.0 section of `docs/project-roadmap.md`.

Breaking changes (pre-1.0):

- `Tool<C>` has new variants (computer, shell, local shell, apply patch, custom, tool search,
  programmatic tool calling); exhaustive matches need updating.
- `FunctionToolSpec`, `RunConfig` and `ModelRetrySettings` have new public fields; use
  `..Default::default()` or the builders.
- The `sqlite` feature now uses `rusqlite` 0.39 (to share `libsqlite3-sys` with `sqlx`).
- Approvals of namespaced function tools are keyed by `namespace.name`.

## 0.1.0 (2026-10-29)

Initial release: Rust port of the OpenAI Agents SDK (Python 0.24.0) core. See
`docs/project-roadmap.md` for scope and the features not yet ported.
