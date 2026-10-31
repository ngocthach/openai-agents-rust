# Changelog

## 0.7.0 (2026-10-31)

Completes the sandbox and runner-tracing parity work that 0.6.0 left open. All 806 cases of the Python
sandbox suite are now accounted for (651 ported, 36 covered, 119 Python-only, none unported), and all
102 runner-tracing cases are (99 ported, 3 Python-only). Per-file counts are in
`plans/reports/py-test-parity-sandbox-b.md` and `plans/reports/py-test-parity-core4-tracerunner.md`.

Breaking changes (pre-1.0):

- `SandboxSessionGuard::suspend` now releases the sandbox as Python's pause does: it captures the
  snapshot, reads the state, then stops, shuts down and deletes the session. A paused run no longer
  keeps its workspace or backend alive, and the resumed run restores from the snapshot.
- `SandboxSession::read_bounded` now sanitizes failures (only the path, a reason and the provider's
  retryability survive). Backends implement the new `read_bounded_unchecked`; a custom session that
  overrode `read_bounded` should rename the override.
- The default E2B client has a 60 s connect and idle-read timeout, matching the Python SDK, so a
  stalled download is a retryable timeout instead of a hang.

Fixed:

- The rclone config step runs `chmod 0600 <path>` as Python does (the previous spelling failed on macOS).
- The websocket model records the response id by the HTTP client's endpoint, as Python does, instead of
  by the websocket scheme.

## 0.6.0 (2026-10-31)

Parity release driven by the Python SDK's own test suite. The Python tests under
`openai-agents-python/tests/` were triaged case by case; each behavioural case was ported to a Rust
test (about 4,900 new tests across `tests/py_*.rs`), and every divergence a ported test exposed was
fixed in the SDK. Each case is classified as ported, covered by an existing Rust test, or
Python-only (pytest fixtures, asyncio task internals, pydantic and `inspect` introspection, CI and
example scripts), with the reason recorded in `plans/reports/py-test-parity-*.md`.

Not covered by a Rust port: the Python-only cases above, a handful of cases that need Python object
identity or live vendor SDK objects, and the websocket official-endpoint case that needs a connector
seam. Deliberate differences from Python are listed under each area below and in
`docs/project-roadmap.md`. Retry backoff defaults now match Python (0.25 s initial delay, 2.0 s cap).

Credentials in sandbox client options are redacted from `Debug` output, logs, errors and persisted
state, and every persisted manifest entry and mount payload variant round-trips.

Areas: runner, `RunState` interoperability, approvals and call-id reuse, handoffs and nested history,
guardrails and hooks, tracing, models and providers, MCP and sessions, realtime, voice, Codex,
sandbox core and the cloud sandbox providers.

Shell, apply_patch, computer, programmatic tool-calling and collision-policy parity with Python's `test_shell_tool.py`,
`test_shell_call_serialization.py`, `test_apply_patch_tool.py`, `test_local_shell_tool.py`,
`test_computer_tool_lifecycle.py`, `test_strict_schema_oneof.py`, `test_scripted_sandbox.py`,
`test_tool_name_collision_policy.py` and `test_programmatic_tool_calling.py` (see
`plans/reports/py-test-parity-core3-shelltools.md`).

Breaking changes (shelltools round):

- Computers are per run context. `ComputerTool::resolve` and `ComputerTool::to_responses_param` take
  the `RunContext`; each run gets its own computer from a lazy `ComputerTool`, and the run disposes of
  it. `ComputerTool::dispose()` is removed.
- `ComputerTool::from_provider`'s dispose hook may return `Result<(), E: Display>`; a failure is logged
  without the error text unless tool data logging is allowed. Returning `()` still works.
- `ShellTool` configuration errors (local without an executor, hosted with an executor, hosted with
  approval) are raised when the agent resolves its tools, not by the builders.
- `with_output_json_schema` accepts `"type": ["object"]` and stores `"object"`.

Function tools parity with Python's `test_function_tool.py`, `test_tool_identity.py`,
`test_tool_context.py`, `test_tool_custom_data.py`, `test_tool_use_tracker.py`,
`test_custom_tool.py`, and the open rows of `test_function_schema.py` and
`test_function_tool_decorator.py` (see `plans/reports/py-test-parity-core3-functiontools.md`).

Added:

- `FunctionTool::try_new` (typed, fallible) and `FunctionTool::try_new_raw` (raw schema, explicit
  strictness) return `UserError` when a schema cannot be made strict. `FunctionTool::try_with_timeout`
  rejects a zero timeout.
- `ToolContext::result_meta()` and `FunctionToolCustomDataContext::result_meta`: the `_meta` of the
  MCP call result, read by the custom-data extractor as in Python.

Breaking changes (pre-1.0):

- `FunctionTool::new` panics when the argument type's schema cannot be made strict. Python raises
  at definition time; the Rust constructor previously fell back to a non-strict tool silently. Use
  `try_new` to handle the error.
- `FunctionToolCustomDataContext` gained the public field `result_meta`. Code that builds the struct
  with a literal must add the field; the SDK is the only constructor.
- `$ref` fragments are percent-decoded before resolution, as Python's `unquote` does; a `%` escape
  that was taken literally before now decodes.
- `FunctionTool::with_strict_json_schema(true)` re-strictifies from the schema as given, and
  `with_strict_json_schema(false)` restores it, instead of only flipping the flag.
- A typed tool called with a JSON value that is not an object fails with "Invalid JSON input for tool
  X: expected a JSON object".

Deliberate differences from Python:

- A docstring argument description and an explicit `#[function_tool]` `#[arg(desc = "...")]` both
  apply to the same parameter; the explicit `#[arg(desc)]` wins. Python gives the docstring
  precedence over `Annotated` metadata. The Rust order is kept on purpose.
- Typed tool arguments are validated by their `Deserialize` impl; pydantic `Field` constraints
  have no Rust equivalent. Use a validating newtype (`#[serde(try_from = ...)]`) for a bounded value.
- Synchronous `failure_error_function` closures only; Python also accepts an awaitable return.

`RunState` parity with Python's `test_run_state.py` (see `plans/reports/py-test-parity-core2-runstate.md`).

Added:

- `RunState` reads and writes the `sandbox` resume payload of a Python snapshot as opaque JSON, with
  mount credentials, credential-file sources and unsafe endpoints removed on every read and write.
- Nested-handoff ownership (`nested_history_owned_items`) and the generated/session index map are
  serialized and validated with Python's digest, so a resumed run does not replay carried items twice.
- `RunState::try_approve` / `try_reject` (refuse a call that has no canonical invocation identity),
  `to_json_with_tracing_api_key`, `reasoning_item_id_policy`, `tool_input`,
  `ToolApprovalItem::tool_namespace` / `qualified_name`, and `RunItem::tool_lookup_key` /
  `allow_bare_name_alias`.
- Resuming a parked approval turn under output guardrails is refused ("current response boundary
  cannot be proven") when the serialized ownership of the parked response is corrupt or comes from a
  legacy writer; hosted MCP items obey the `allowed_callers` of their server.

Breaking changes (pre-1.0):

- `RunState`, `ProcessedResponseSnapshot` and `RunItem` gained public fields; build them with
  `..Default::default()` / `RunItem::new`.
- Schema, ledger and parse errors use Python's exact messages and never echo the payload.
- `RunState::approve` / `reject` keep their infallible signatures. Python's `approve` / `reject` raise
  for an item that cannot carry the decision; the Rust forms record nothing and return. The fallible
  path with Python's errors is `RunState::try_approve` / `try_reject`.

Deliberate differences from Python:

- A snapshot item without an agent, without a type, or that cannot be read is an error, where Python
  skips it with a log line. A skipped item is history the resumed run would never see.
- Python's object-identity ownership rules (deep copies, aliasing of equal items) have no Rust
  equivalent; items are owned values and the generated and session histories are one list.

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
