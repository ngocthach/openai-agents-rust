# Changelog

## 0.9.1 (2026-11-01)

Corrects the 0.9.0 parity count and ports the last cases. The 0.9.0 entry said 69 cases were not ported;
that came from a ledger merge in which a stale per-area row overwrote a newer one. The true figure was 7,
all in `test_openai_responses_compaction_session.py`, and they are now ported
(`tests/py_core6_final7_compaction.rs`). Measured state: 12,641 Python test cases, 8,036 ported, 604 covered,
4,001 Python-only with a recorded reason, 0 not ported, 0 unaccounted. The library code is unchanged from
0.9.0.

Differences that remain (behaviour, not unported cases): an expired-during-compaction history makes Rust skip
the replacement with a warning where Python raises; a tar nested-path case is skipped on macOS (needs GNU
tar); tar `--no-wildcards` differs; live-session `concurrency_limits` differ.

## 0.9.0 (2026-11-01)

Burns down the measured parity gap. The Python suite's 12,641 test cases are now all classified
(`plans/reports/py-test-parity-ledger.md`): 7,981 ported, 604 covered by existing Rust tests, 3,987
Python-only with a recorded reason, 69 not ported, 0 unaccounted. New tests live in `tests/py_core6_*.rs`;
per-area notes are in `plans/reports/py-test-parity-core6-*.md`. No live model call was made: wire formats
are checked against mock servers only.

Fixes the ported tests exposed:

- Sandbox providers (Blaxel, Cloudflare, E2B, Daytona, Modal, Vercel): workspace symlink escapes are
  refused, credentials no longer end up in a Modal snapshot, and archive and mount credential handling
  matches Python. The local PTY reports an exit code and a live process id from one read.
- Runner: native tools run concurrently with the function-tool batch and race the guardrails; the input
  guardrail is polled before the model call so a guardrail that trips first wins; resumed runs, name
  fallbacks, compaction and session appends match Python; a failed streamed run's `to_state()` returns the
  checkpoint.
- MCP: pagination, retry, session setup, approval caching and resume binding match Python; invalidating the
  tool cache keeps the stale list visible but never serves it.
- Model retry backoff defaults are 0.25 s initial, 2 s max, factor 2, jitter on.

Breaking changes: `RunConfig::tool_post_invoke_wait` (new), `RunState::foreign_owner_call_ids` (new),
`ProcessedResponseSnapshot::mcp_tool_bindings` has a new type, `SessionRunCompletion::omit_reasoning_item_ids`
(new), `RequireApproval::DynamicWithContext` (new), `Session::ignores_item_ids_for_matching` (new),
sandbox backends implement `read_bounded_unchecked` instead of `read_bounded`, Cloudflare resume refuses
credentialed state, E2B native snapshots with skip paths return a tar, and shutdown suppresses a failed
delete.

Known differences and gaps (the "69 not ported" figure first published here was wrong; see 0.9.1): a tar nested-path case is
skipped on macOS (needs GNU tar); tar `--no-wildcards` differs; the live-session `concurrency_limits`
differ. `test_complex_async_tracing` failed once in about 60 loaded runs and has not reproduced.

## 0.8.0 (2026-10-31)

Closes the gaps a completeness check found in the 0.6.0/0.7.0 parity work: about 450 cases of top-level
Python test files had never been assigned to a port, among them the scripted model, the server
conversation tracker, start-hook tool concurrency, computer actions, extension filters, MCP tool
metadata, visualization and `as_tool`. All 174 top-level Python test files are now cited by at least one
Rust port or a recorded Python-only reason (`plans/reports/py-test-parity-core5-leftover*.md`).

Python test-parity fixes (leftover files `test_scripted_model`, `test_scripted_sandbox`,
`test_server_conversation_tracker`, `test_streamed_terminal_output_backfill`,
`test_streaming_tool_call_arguments`, `test_start_hook_tool_concurrency`; see
`plans/reports/py-test-parity-core5-leftover1.md`):

- The scripted test model (`ScriptedModel`) matches Python's streamed events and usage: function-call and
  reasoning payloads are announced as in progress, logprobs are preserved, and `response.completed` usage
  carries the request count and per-request entries. Responders may return other responders, and
  `ScriptedStep::respond_async` / `ScriptedStep::stream_with` are new. Breaking: `ScriptedStep` has two
  new public fields (use `..Default::default()`), `ModelScriptError` display text changed, and scripted
  default usage is one request instead of zero.
- Input items held back by a `call_model_input_filter` are offered again on the next call instead of
  being dropped for good, and server-conversation hydration only marks items the server holds.
- A streamed response whose terminal output is empty is backfilled from the
  `response.output_item.done` items.
- The scripted sandbox session no longer consumes a step on a rejected matcher, reports payload-free
  structured failures, and derives `supports_pty()` and `running()` from its script.

Deliberate difference: the automatic scripted stream rejects fewer item kinds than Python (shell, program
and tool-search items pass through as bare added/done events), because seven streamed tests script them
through it. Python's start-hook concurrency tests mutate `agent.tools` on a shared agent; a Rust `Agent`
is read-only while it runs, so those cases are Python-only and the isolation guarantees are ported.


Python test-parity fixes (leftover files `test_extension_filters`, `test_mcp_tool_metadata`,
`test_visualization` and the completeness sweep; see `plans/reports/py-test-parity-core5-leftover3.md`):

- Typed function-tool arguments are checked against the value bounds in their JSON schema before
  the call (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `minLength`,
  `maxLength`, `pattern`, `minItems`, `maxItems`, `uniqueItems`, `enum`, `const`), through nested
  models and lists. A violation is reported as `Invalid JSON input for tool X`, as Python's pydantic
  validation does; before, serde accepted any value of the right type. Adds a direct `regex`
  dependency for `pattern`.
- `RunErrorDetails` implements `Display` (and `pretty_print`) with Python's `str(details)` summary.
- `extensions::handoff_filters::remove_all_tools` keeps compaction items, as Python does, and the
  module re-exports `nest_handoff_history` and `default_handoff_history_mapper`.
- A persisted `tool_lookup_key` that is not a valid key (missing or empty `name`, unknown `kind`,
  `namespaced` without a namespace) reads as no key instead of being kept verbatim.
- `extensions::visualization` gains `get_all_nodes_with_visited` / `get_all_edges_with_visited`, which
  take the caller's set of already-drawn agent names; a previously drawn agent is skipped but keeps its
  node id reserved.
- `mcp` gains `resolve_mcp_tool_title_value`, `resolve_mcp_tool_description_value` and
  `resolve_mcp_tool_description_for_model_value` for tools that arrive as raw JSON, and
  `collect_mcp_list_tools_metadata` treats an empty `raw_item` as absent.

Python test-parity fixes (leftover files `test_agent_as_tool`, `test_computer_action`,
`test_call_model_input_filter`, `test_invalid_final_output_handler`, `test_agent_config`,
`test_tool_converter`; see `plans/reports/py-test-parity-core5-leftover2.md`):

- `Agent::as_tool` validates the arguments against the tool's object schema (the default
  `{"input": str}` or `parameters`) and drops undeclared fields, reporting `Invalid JSON input for
  tool X`; a plain agent tool no longer inherits the caller's structured tool input; with no
  explicit `run_config` the nested run inherits the caller's.
- `AgentAsToolOptions::on_stream` events go through a bounded queue (`on_stream_max_pending_events`,
  default 1024, `None` for unbounded): a slow handler no longer stalls the nested run, a panicking
  handler is logged and does not fail the call, events emitted before a stream error are still
  delivered, an overflowing backlog cancels the nested run and fails the tool, and the event agent
  follows handoffs. New `Agent::try_as_tool_with` (rejects a zero limit; `as_tool_with` panics) and
  `AgentAsToolOptions::input_items_builder` (a builder that returns input items).
- An empty structured model turn with an `invalid_final_output` handler that declines runs another
  turn instead of failing the run.
- A failing `call_model_input_filter` records `Error in call_model_input_filter` on the turn span; a
  failing computer action records `tool_name: computer` on its span and logs
  `Failed to execute computer action`; the agent span lists the GA `computer` tool name
  (`Tool::trace_name`); the missing-`on_safety_check` warning ends with `Configure
  ComputerTool.on_safety_check to review or reject them.`
- Chat Completions rejects `output_json_schema` tools like the other Responses-only function-tool
  features and names them as Python does (`defer_loading=True`, `namespace.tool`).
- `ModelSettings::from_value` builds settings from a JSON object, rejecting unknown keys at every
  level (`Unknown model settings in retry.backoff: ...`). Breaking: `Reasoning` gains a public
  `extra` map that keeps unmodelled reasoning options (struct literals need `..Default::default()`);
  a partial `retry` / `retry.backoff` object keeps defaults for omitted fields.

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
