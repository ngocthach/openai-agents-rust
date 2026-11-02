# Changelog

## 0.20.0 (2026-11-02)

Closes the largest known gap from 0.19.0. 14 more parity cases ported. Measured state: 12,641 Python test cases,
9,240 ported, 897 covered, 2,504 Python-only, 0 not ported, 0 unaccounted. No live model call was made.


- Function-tool approvals are bound to a capability fingerprint, closing the approval-binding known gap listed under 0.19.0: a
  decision recorded for one tool no longer authorises a different same-named tool on a same-named agent after the
  context, its exported approvals or a run state is reused or restored. The fingerprint is a SHA-256 of the agent
  name and the tool's name, namespace, description, parameter schema and origin
  (`run_context::function_tool_capability_fingerprint`). The runner registers it for every function-tool approval it
  raises, so `RunContext::approve_tool` / `reject_tool` and `RunState::approve` / `reject` on that interruption bind
  automatically, also after `RunState::to_json` / `from_json`. A bound decision applies only while the running tool
  has the same fingerprint; otherwise the call is asked again (sticky and per-call alike). Unbound decisions, including
  every `ApprovalsState` serialized before this change, behave exactly as before.
- New API: `RunContext::approve_tool_for_agent` / `reject_tool_for_agent` (bind a hand-built item to an agent's tool),
  `RunContext::export_approval_capabilities` / `import_approval_capabilities`, the `ApprovalCapability` type and the
  `RunState::approval_capabilities` field. Bound records are stored under their own `ApprovalsState` keys
  (`agent \u{1} tool \u{3} fingerprint`); in a run state envelope they are ordinary `function_tool_approvals`
  entries with an extra `capability` field (ignored by Python), and the capability registry travels under the
  Rust-only `rust` key.
- A closure is not part of the fingerprint: two tools that differ only in their body still share decisions, and so do
  clones of one agent (Python tells them apart by object identity).
- Ports 14 more cases of `test_agent_scoped_approvals.py` (`tests/py_core18_approval_fingerprint.rs`); notes in
  `plans/reports/py-test-parity-core18-approval-fingerprint.md`.

Breaking changes: `RunState` gained a public field, so a struct literal without `..Default::default()` no longer
compiles; an approval the runner raised is now invalidated when the tool's description, schema, namespace or origin changes
between the interruption and the resume.

Known limit: when one owner holds two records for one tool (two fingerprints, or one bound and one unbound), the run
state carries two `function_tool_approvals` entries that a Python reader cannot tell apart and it keeps the last.

## 0.19.0 (2026-11-02)

Re-reads the fourteen files with the most remaining Python-only cases, closes two of the known gaps listed under
0.18.0 and fixes six bugs found while porting. 73 more parity cases moved out of the Python-only bucket (60 ported,
13 covered). Measured state: 12,641 Python test cases, 9,226 ported, 897 covered, 2,518 Python-only, 0 not ported,
0 unaccounted. Notes by area: `plans/reports/py-test-parity-core17-*.md`. No live model call was made.

- Macros: a NumPy-style docstring entry for `*args` or `**kwargs` now reaches its parameter (the stars are
  stripped).
- Redis sessions: with a TTL configured, every write reads the server `TIME` first and refuses a TTL outside Redis's
  expiration range before any script runs, so an out-of-range TTL no longer appends history and then fails at
  `PEXPIRE`. One extra round trip per write when a TTL is set.

- Injected sandbox sessions apply capability-changed path grants: `SandboxSession` gained the defaulted methods
  `live_grants`, `extra_path_grants` and `update_extra_path_grants` (with the `LiveGrants` cell), every built-in
  session holds one, `Manifest::with_extra_path_grants` replaces the grant list, and the unix-local session re-pins
  grants on update while an unchanged alias keeps its first pin.
- Vercel serializes workspace commands, reads, writes, archives and shutdown while S3 mounts are being attached or
  detached; a session without S3 mounts takes no lock.
- `InstrumentedSession` forwards `shutdown_after_failed_close`, so a mounted Vercel session is still shut down when
  an instrumented `aclose` fails.
- A manifest entry path outside the workspace root no longer hides credentialed mounts from validation: it is refused
  as an invalid manifest path, with only the fixed protected-mount sentence when a credentialed mount is present. A
  runner-owned run validates the mount credential boundaries before asking the client for a session.
- A Blaxel read that fails with a non-404 status whose body says "not found" or "no such file" is a
  `WorkspaceReadNotFound`.

Breaking changes: `SandboxSession` has three new defaulted methods (an implementor that wraps another session should
forward them); an invalid entry path now fails runner preparation before the client creates a session.

Known gaps carried forward (all others under 0.18.0 still stand except the two closed here):

- A restored context binds approval decisions by agent name and tool name, so a decision for one tool also
  authorises a different same-named tool on a same-named agent (about 14 cases of
  `test_agent_scoped_approvals.py`). Closing it changes the persisted approval shape; deferred.
- `MCPServerManager::set_connect_timeout(Some(0))` is accepted and refused at the next connect; Python raises at
  assignment.
- A third-party `MountStrategy` implementation is trusted; Python rejects custom strategies.
- `session_state()` serializes the original path grants, not the live ones.
- Post-invoke sibling work is dropped after the wait; Python leaves it running detached.
- The Redis TTL check is a separate round trip, so a TTL exactly at the int64 boundary can still fail if the server
  clock moves between the two commands.
- Read depth: in `test_run_state.py` and `test_run_step_execution.py` cases whose names matched a Rust test were
  not re-read; about 70 hostile-exception functions in `test_mount_security.py` were read for their first 25–40
  lines. Ledger counts have no per-case ids, so the remaining counts are upper bounds.

## 0.18.0 (2026-11-02)

Closes eight of the nine divergences listed as known gaps in 0.17.0 and re-reads the remaining Python-only files
in seven batches. 221 more parity cases moved out of the Python-only bucket (134 ported, 87 covered). Measured
state: 12,641 Python test cases, 9,166 ported, 884 covered, 2,591 Python-only, 0 not ported, 0 unaccounted.
Notes by area: `plans/reports/py-test-parity-core16-*.md`. No live model call was made.

Fixes and additions:

- Hooks and replay: a function or computer call whose end hook failed is refused with `already executed` when
  the same call id returns on a fresh run sharing the context; a streamed run emits `tool_called` before a
  failing `on_llm_end` hook ends the run.
- Sinks: `EventSink::bind` has a default, `CallbackSink::with_session` and `raising_with_session` pass the bound
  session to the callback, and `ChainedSink` forwards the bind.
- Macros: `#[function_tool]` accepts methods taking `&self` or `self` (the receiver type must be `Clone + Send +
  Sync + 'static`) and associated functions written `#[function_tool(associated)]`; a synchronous tool body runs
  on the tokio blocking pool.
- MCP: new `AgentsError::MCPToolCancellationError { message, run_data }` for a cancelled MCP tool call; the
  default MCP HTTP and SSE clients no longer follow redirects (a caller-supplied `http_client` keeps its own
  policy).
- Realtime: connect-time tool lists decide dispatch; `RealtimeRunConfig::with_initial_tools` and
  `with_initial_handoffs` carry live tools and handoffs; a failed `connect` restores the previous model name;
  an interrupt error lists every failure; a transcript received before STT setup finishes is delivered;
  `OpenAIClient::without_query` omits a query parameter from HTTP and websocket URLs.
- Approvals: `ApprovalPolicy::DynamicForAgent` and `evaluate_for_agent` pass the agent name to local-tool policies
  and the sandbox `apply_patch` tool stops evaluating once a decision for the call exists; `try_approve` and
  `try_reject` refuse decisions for ambiguous owners; a restored snapshot onto indistinguishable same-named
  agents asks again instead of applying the saved decision.
- Sandbox: Vercel shutdown unmounts S3 mounts first and retries a failed stop once; unix-local `mkdir_as` and
  `rm_as` check access under `sudo -u` and then act natively; Add File for a run-as user stages the file and
  hard-links it into place; a failing session `start` or capability `check_manifest` on a manifest with
  credentialed mounts returns only the fixed protected-mount sentence; `Filesystem::with_view_image_needs_approval`
  requires approval for `view_image`.
- Sessions: `pop_item` and `clear_session` on the compaction session drop the stored response chain before the
  deletion starts; the voice pipeline aborts and awaits its sibling TTS tasks after a failure;
  `OpenAIVoiceModelProvider` honours the process default client and has `with_organization` and `with_project`.
- Other: the memory-generation and held-write pairing fixes listed in `core16-r16a.md`, settlement of an unpaired
  hosted MCP approval request only when its response is present or its approval is still open, and
  `RunState::try_approve` ambiguity handling.

Breaking changes: `AgentsError` and `ApprovalPolicy` each gained a variant, so exhaustive matches need updating;
`RunState` gained a `#[serde(skip)]` public field, so struct literals need `..Default::default()`; a failed
turn's call ids are refused on replay; after a failed Vercel mount transition the session refuses commands;
the default MCP clients no longer follow redirects; `OpenAIVoiceModelProvider` no longer derives `Default`;
`#[function_tool]` bodies that are synchronous run on the blocking pool.

Known gaps (real differences): a restored `RunContext` carried through `export_approvals`/`import_approvals`
still binds approvals to an agent by name, and `ToolApprovalItem.agent` is a name, so a user-built item cannot
say which same-named agent it approved (fixing it needs a new field on the public struct); Rust awaits a parallel
input guardrail before running a turn's tools where Python runs them concurrently (seven batch-failure cases);
Python re-prepares the sandbox each turn where Rust prepares once per run; capability-changed path grants are
not applied to an injected sandbox session; `OpenAIProvider::get_model` without an API key succeeds where Python
raises; Vercel has no workspace-command serialization while S3 mounts activate; unix-local start with
`workspace_root_ready=false` re-applies the whole manifest where Python re-applies only ephemeral entries;
`is_enabled` callbacks and the hosted MCP approval callback are infallible; `ResponsesWebSocketSession` has no
`run` wrappers; the `docker-removal` feature's privileged Linux paths have not run against a real container and
the feature is excluded from the docs.rs build; many Python-only files were read at name level plus the bodies of
portable-looking candidates (about 30 files of one batch in `core16-r16e.md`); docs.rs builds without the
`mongodb` feature; the websocket crate logs handshake data at TRACE through the `log` crate; the Redis atomic
replace and the Redis Cluster path have only run against mocks.

## 0.17.0 (2026-11-01)

Fallible hooks, Redis Cluster reads, a per-session memory-manager registry and the findings of a case-by-case
re-read of the remaining Python-only files. 203 more parity cases moved out of the Python-only bucket (130
ported, 73 covered). Measured state: 12,641 Python test cases, 9,032 ported, 797 covered, 2,812 Python-only, 0
not ported, 0 unaccounted. Notes by area: `plans/reports/py-test-parity-core14-*.md` and `core15-*.md`. No live
model call was made.

- Sandbox memory: a sandbox session now has one memory generation manager per layout (`Memory::manager_for_session`).
  Capabilities with the same layout and an equal `MemoryGenerateConfig` share it; a different config for the same
  layout, or a layout that reuses another layout's `memories_dir` or `sessions_dir`, fails the run with Python's
  messages. The registration ends when the manager flushes (session close). `MemoryGenerateConfig` gained
  `PartialEq` (model instances compare by identity). Agents that share a session and a layout must share one
  `Memory` configuration.
- Sandbox manifests: custom environment value types can persist. Implement `EnvValue::to_payload` (default `None`,
  still unwritable) and register the type with `sandbox::persisted_registry::register_env_value_type(name, parser)`;
  `Manifest::to_value` then writes the tagged payload (never the resolved value) and `Manifest::from_value` restores
  the same type. Registering `str`, an empty name or a name twice is refused, as Python refuses a duplicate subclass tag.
- Realtime: a `session.created`/`session.updated` event naming a GA output format object (`audio/pcm`, `audio/pcmu`,
  `audio/pcma`) now reaches the playback tracker as `pcm16`, `g711_ulaw`, `g711_alaw`, as Python normalizes it.
- Hooks: every `RunHooks` and `AgentHooks` callback has a fallible twin (`try_on_llm_start`, `try_on_llm_end`,
  `try_on_agent_start`, `try_on_agent_end`, `try_on_handoff`, `try_on_tool_start`, `try_on_tool_end`;
  `try_on_start`/`try_on_end` on `AgentHooks`) returning `Result<()>`. The runner calls the twin and an `Err` fails
  the run where Python's raising hook does. The twin's default calls the infallible callback, so existing hooks
  keep compiling and behaving; override the `try_` method to fail a run. The run-level and agent-level hook of one
  event run together and both always run (run-level failure wins), as Python's `asyncio.gather` does.
  Failure effects: a failed `on_llm_end` leaves the response accepted (resume processes it with no second model
  call and no second end hook), a failed `on_tool_start` leaves the call unproven (a retry refuses it), a failed
  `on_tool_end` keeps the call's output in history and a retry skips the call, and a failed `on_agent_end` withholds
  the final message but keeps the turn's tool calls and outputs.
- Hooks and output guardrails: the final-output end hooks (`on_agent_end`, `AgentHooks::on_end`) now run before the
  output guardrails, as in Python, so a failing end hook means the guardrails never see the output, and a terminal tool
  output behind output guardrails stays private (out of history and of the failure's run data) when the end hook fails.
  A failed resumed turn whose approved tool outputs are still guarded keeps them in the checkpoint but not in the
  error's `run_data.new_items`, so retrying from the checkpoint finishes without running the tool again.
- Retries: response header names (`Retry-After`, `Retry-After-Ms`, `X-Should-Retry`) match case-insensitively when
  the normalized retry delay and the provider's retry advice are read.
- Model settings: deserializing `ModelSettings` rejects a non-positive or non-finite `timeout`, and deserializing
  `ModelRetryBackoffSettings` rejects a negative or non-finite `initial_delay`, `max_delay` or `multiplier`, as
  Python's field validation does. Builder-set and struct-literal values are not checked.
- Sessions: `MongoDBSession` identifies the SDK (`openai-agents`) in the MongoDB connection handshake, both for a
  client built by `from_uri` and for a client passed to `with_client`.
- Sessions: `RedisSession` accepts a Redis Cluster connection (`RedisSession::from_cluster_nodes`,
  `with_cluster_connection`; the `redis` feature now enables the crate's `cluster-async`). As in Python, a cluster
  session reads and pops history and rejects every write (`add_items` with items, `clear_session`, compaction
  replacement) with a user error before anything is sent, because the write scripts span three keys.
- Sandbox state: the persisted provider record also carries the E2B template, metadata, environment and timeouts and
  the Vercel project id, team id, environment and `interactive` flag (`VercelSandboxClientOptions::with_interactive`
  is accepted and enables nothing, as in Python). The state stays credential-free: an environment or metadata entry
  whose name or value looks secret is not recorded, URL values lose userinfo and query, and a template, project or
  team id that looks like a credential is dropped; the resuming client supplies those values and wins per key. The
  recorded template, timeouts, project and team belong to the original session. Old states still read.

Breaking changes: `E2BPersistedOptions`, `VercelPersistedOptions` and `VercelSandboxClientOptions` struct
literals need `..Default::default()`; a resumed session takes the recorded E2B template and timeouts and the
recorded Vercel project and team over the resuming client's own; the final-output end hooks run before the
output guardrails; `OpenAIVoiceModelProvider` no longer derives `Default` (use `new()`); agents that share a
sandbox session and a memory layout must share one `Memory` configuration;
`RealtimePlaybackTracker::audio_format()` reports normalized format names for GA audio formats.

Known gaps (real differences): the `docker-removal` feature's privileged Linux paths are type-checked but have
not run against a real container, and the feature is excluded from the docs.rs build; a restored context is
bound to an agent by name; a function or computer call whose end hook failed is re-executed when the same call
id returns on a fresh run sharing the context, where Python refuses it; streamed `ToolCalled` arrives after
`on_llm_end` where Python emits it before; `CallbackSink` does not receive the bound session; the realtime
interrupt path returns only the first of several failures; `MCPToolCancellationError` has no Rust variant;
`#[function_tool]` is rejected on methods and associated functions; realtime connect-time overrides cannot carry
a tool implementation; a speech-to-text transcript received before session setup is dropped; the `apply_patch`
approval checker cannot stop after a decision lands; Vercel `shutdown` is a single stop request; a dynamic
approval rule is consulted unless the tool opts in with `with_args_have_validators`; about 100 Python-only files
were read at name level plus the bodies of portable-looking candidates rather than case by case; docs.rs builds
without the `mongodb` feature; the websocket crate logs handshake data at TRACE through the `log` crate; the
Redis atomic replace and the Redis Cluster path have only run against mocks.

## 0.16.0 (2026-11-01)

Persists provider options in sandbox session state, bounds E2B and Daytona PTY start, ports Python's Docker
removal service behind an opt-in feature and re-reads the Python-only groups earlier rounds had carried over.
187 more parity cases are ported. Measured state: 12,641 Python test cases, 8,902 ported, 724 covered, 3,015
Python-only, 0 not ported, 0 unaccounted. Notes by area: `plans/reports/py-test-parity-core13-*.md`. No live
model call was made.

- Sessions: `AdvancedSqliteSession` runs appends, pops, usage writes and branch deletions in call order and each
  keeps the branch (and clear generation) selected when it was called, so a concurrent `switch_to_branch` cannot
  redirect an operation already queued. `store_run_usage` logs a failed write (`Failed to store session usage`,
  error text only when model data may be logged) and returns `Ok`, as Python does; a failed `add_items` is
  logged and still returned; creating a branch debug-logs its turn content only when model data may be logged.
- MCP: a request or handshake that outlives the HTTP read timeout reports `Connection timeout.`, and a
  connection refused mid-session reports `Connection lost.`, instead of a bare `Request failed.`; the class is
  read from the `reqwest` error behind the transport failure.
- Sandbox: opt-in feature `docker-removal` ports Python's `DockerRemovalService`. A trusted host process pauses a
  private rootful Linux container, pins the workspace and read-only grant roots through a worker that enters the
  container's mount namespace, authorizes the target against the live bindings and removes it as the requested
  user, with the same request shapes and `docker_removal_*` error reasons. Configure it with
  `DockerSandboxClient::with_removal_service`. The library never escalates: the caller runs the process as root on
  the Docker host and points the service at the `docker-removal-worker` executable (feature-gated binary, one
  worker process per container and a sibling child per removal, since the crate forbids `fork`). With the
  service, `docker create` omits `-w` so the daemon does not create a root-owned workspace. Without the feature
  nothing changes.
- Sandbox state: Vercel runtime, resources, timeouts and network policy, Modal builder version, idle timeout and
  resources, and the E2B MCP shape are stored under `provider_metadata.provider_options` with serde defaults, and
  each client's `resume` layers the record under its own options; old states still read. The persisted E2B MCP
  shape is credential-free on purpose (server names and non-secret options only, URLs without userinfo, query or
  fragment; options whose name looks secret are dropped), so the resuming client supplies secrets; Python
  persists the full config.
- `PtyExecRequest` has `timeout` and `with_timeout`; E2B sends it as the process lifetime and bounds a stalled
  start, and Daytona bounds each start step (an elapsed step is `ExecTimeout` and deletes the half-created
  session). Other providers ignore it.
- A detached session write that fails after a dropped output-guardrail run is logged under the redaction policy
  and, when the run has a `RunCheckpoint`, recorded there as a pending session write so a resume settles it;
  `SessionRunContext::detach()` owns a copy so a context-scoped `EncryptedSession` append is shielded from a
  dropped caller.

Breaking changes: `PtyExecRequest` has a new public field; persisted sandbox states carry a new
`provider_options` entry; `AdvancedSqliteSession::store_run_usage` returns `Ok` after logging a failed write;
MCP failure messages changed (`Connection timeout.`, `Connection lost.`).

Known gaps (real differences): the `docker-removal` feature's privileged Linux paths (entering the container's
mount namespace, dropping to the requested user) are type-checked for Linux but have never run against a real
container and need a smoke test on a Linux Docker 26+ host as root, and the feature is excluded from the docs.rs
build for the same reason; four Docker removal cases (`BaseException` cancellation, broken-pipe `close()` on mock
streams, `ExitStack` registration failure, `KeyboardInterrupt`) stay Python-only; Redis Cluster is not supported
(Python accepts `RedisCluster` for reads and pops); persisted state omits `interactive`, `env`, `metadata`, E2B
`timeouts` and `template`, and Vercel `project_id` and `team_id`; the detached guardrail write never reports to
the dropped caller, and a run without a checkpoint leaves only a log line; Rust cannot detect a custom
`Deserialize`, so a dynamic approval rule is consulted unless the tool opts in with
`with_args_have_validators`; docs.rs builds without the `mongodb` feature; the websocket crate logs handshake
data at TRACE through the `log` crate; the Redis atomic replace has only run against a mock.

## 0.15.0 (2026-11-01)

Reads in full the groups where the round-11 random audit found invalid Python-only reasons (provider and model
extensions, session stores, MCP errors, codex tool, agent-as-tool) and fixes what they exposed. 98 more parity
cases moved out of the Python-only bucket (56 ported, 42 covered). Measured state: 12,641 Python test cases,
8,715 ported, 724 covered, 3,202 Python-only, 0 not ported, 0 unaccounted. Notes by area:
`plans/reports/py-test-parity-core12-*.md`. No live model call was made.

Fixes and additions:

- Agent-as-tool: a nested run's progress is reachable from the outer run's `RunCheckpoint`, an outer call that
  only resumes a nested run is not treated as executed, and a cancelled or dropped run leaves a checkpoint, so a
  retry does not run the committed tool again.
- Sessions: a run dropped while its output guardrails run still writes its turn (from a detached task; an
  immediate cancel of a streamed run writes nothing); blocked tool calls are rebuilt from allowlisted fields so
  `provider_data` no longer reaches the session; reasoning items without a usable id are not written to the
  conversations store; `start_openai_conversations_session` is new; `SqliteSession::clear_session` is one
  transaction; a compaction replacement runs on its own task so a dropped caller cannot leave the store empty;
  `EncryptedSession` forwards the run context (new defaulted `Session::pop_item_validated_with_context`).
- MCP: the `list_tools` retry budget is shared across pages.
- Sandbox: E2B `start` probes the workspace root and only calls the files API when it is missing, `mkdir` of the
  root skips remote validation and `mkdir` without `parents` fails when the parent is missing; Cloudflare bucket
  mounts validate the path in the sandbox and send the resolved path; a transient failure on a Docker session
  with credentialed mounts stays retryable.
- `Agent::model_settings` is private: read it with `model_settings()` and write it with `with_model_settings` or
  `model_settings_mut`, so a write keeps the defaults captured when the agent was created.

Breaking changes: `Agent::model_settings` is no longer a public field; E2B `mkdir` without `parents` no longer
creates missing parents; the E2B start probe changes what a mock sandbox must answer to `test -d`;
`Session` has one more defaulted method.

Known gaps (real differences): the Docker removal service (about 190 cases) is a root host-side helper that is
deliberately not part of this library; provider options (Vercel runtime and network policy, Modal builder
version and resources, E2B mcp) live in the client options, not in the persisted session state; E2B and Daytona
PTY start have no per-call timeout; a dropped guardrail run's detached session write cannot report its own
failure, and a context-scoped `EncryptedSession` append is not shielded from a dropped caller; groups carried
over on earlier reasons without a fresh read include the MCP exception-graph cases, redis-py internals, MCP
manager worker tasks, advanced-SQLite interleavings and codex exec; Rust cannot detect a custom `Deserialize`,
so a dynamic approval rule is consulted unless the tool opts in with `with_args_have_validators`; docs.rs builds
without the `mongodb` feature; the websocket crate logs handshake data at TRACE through the `log` crate; the
Redis atomic replace has only run against a mock.

## 0.14.0 (2026-11-01)

Closes the last documented synchronous-API divergences and adds an independent measurement of the Python-only
bucket. 32 more parity cases moved out of it. Measured state: 12,641 Python test cases, 8,659 ported, 682
covered, 3,300 Python-only, 0 not ported, 0 unaccounted. A seeded random sample of 364 Python-only cases found
94.2% with a valid reason and 3.3% portable (95% interval 1.9-5.7%), which extrapolates to about 100 portable
cases still unported (under 1% of the suite); see `plans/reports/py-test-parity-core11-r11b.md`. No live model
call was made.

Fixes and additions:

- `HandoffInputFilter`, `RunConfig::handoff_input_filter` and `RunConfig::tool_error_formatter` are async and
  fallible. A failing handoff input filter ends the run with its error; a failing run-level formatter logs and
  keeps the default message, as Python does. New builders: `handoff_input_filter`, `try_handoff_input_filter`,
  `async_handoff_input_filter`, `tool_error_formatter_fn`, `try_tool_error_formatter_fn`,
  `async_tool_error_formatter_fn`, `with_input_filter_handler`, `with_handoff_input_filter_handler`,
  `with_tool_error_formatter_handler`; the closure builders keep their shape. The handoff ownership record is a
  `tokio::task_local`, so an async filter that resumes on another thread still works.
- A completed sandbox run writes the `sandbox` section into its `RunState` (not when teardown failed or the
  session is caller-owned); `StreamCompletion` gains `last_agent_id`, so a streamed run that stopped at the second
  of two same-named agents resumes it.
- `FunctionTool::with_args_have_validators(true)` marks a tool whose argument type validates: a dynamic
  `needs_approval` rule is then never consulted and every call pauses first, as in Python.
- E2B `running()` is false until `start` has prepared the workspace root, and `exec` omits the cwd before that,
  so the runner starts a fresh caller-supplied session.

Breaking changes: `HandoffInputFilter`, `RunConfig::handoff_input_filter`, `RunConfig::tool_error_formatter` and
`realtime::RealtimeToolErrorFormatter` are boxed-future types (code that built one with `Arc::new(closure)` must
use the constructors); `StreamCompletion` has a new public field; a completed sandbox run's `RunState` carries a
`sandbox` section; an unstarted E2B session no longer reports `running()`.

Known gaps (real differences): the Docker removal service (about 190 cases) is a root host-side helper that is
deliberately not part of this library; Rust cannot detect a custom `Deserialize`, so a dynamic approval rule is
still consulted unless the tool opts in with `with_args_have_validators`; assigning a field of the public
`agent.model_settings` directly is untracked; E2B `clear_workspace_root_on_resume` preserving nested mounts and
the agent-as-tool resume after a cancelled nested commit are portable but unported; `last_agent_id` is a raw
`Arc` address; docs.rs builds without the `mongodb` feature; the websocket crate logs handshake data at TRACE
through the `log` crate; the Redis atomic replace has only run against a mock.

## 0.13.0 (2026-11-01)

Closes the documented divergences from Python's run-state restore, failure formatters and sandbox resume. 51
more parity cases are ported. Measured state: 12,641 Python test cases, 8,636 ported, 673 covered, 3,332
Python-only, 0 not ported, 0 unaccounted. Notes by area: `plans/reports/py-test-parity-core10-*.md`. No live
model call was made.

Fixes and additions:

- `RunState::from_json` skips, with a redacted warning, an item that has no type or agent, a handoff output
  missing its source or target, a message without id, role or content, and any payload the typed model rejects;
  restore also drops items whose agent is not in the graph. A resume whose parked call lost its tool (disabled
  or removed) carries on: a call nothing waits on is dropped, an undecided approval stays pending, a rejected
  one still answers the model, and an approved one still fails with `Tool X not found in agent Y`. Schema 1.12
  snapshots replay a nested handoff's restored items twice, as Python does.
- Failure and timeout formatters (`tool::ToolErrorFn`) are async and fallible: an `Err` ends the run with that
  error, in the runner, streamed runs, handoff recovery, agent-as-tool, MCP and the realtime session. New
  builders `with_failure_error_handler`, `with_timeout_error_handler`, `failure_error_handler` take a
  `ToolErrorFn` built with `tool::tool_error_fn`, `try_tool_error_fn` or `async_tool_error_fn`; the closure
  builders keep their shape. `Agent::model_settings_mut()` is the in-place settings write.
- A paused sandbox run writes a `sandbox` section into its `RunState` (backend id, current agent key and name,
  session state, per-agent sessions keyed by the identity keys), and a resumed run restores every agent's
  session from it ahead of `SandboxRunConfig::session_state`; snapshots without the section resume as before.
  Per-agent resume needs `SandboxSessionMode::PerAgent`. New API: `SandboxRunner::prepare_from_state`,
  `SandboxStreamedRun::to_state`, `SandboxStreamedRun::sandbox_resume_state`.
- Docker `shutdown` treats an already-removed container as removed, as Python's `delete` does.

Breaking changes: `RunState` gains public fields (`nested_history_lineage_unknown`), so struct literals need
`..Default::default()`; `from_json` no longer errors on unreadable items and a resume no longer errors on a lost
tool that was never approved; `tool::ToolErrorFn` returns a future of `Result<String>` (code that built one with
`Arc::new(closure)`, called one directly or passed a plain closure to
`MCPToolPolicy::with_failure_error_function` must switch to the new constructors); `RunResult::to_state()`
carries the `sandbox` section for a paused sandbox run that uses a client.

Known gaps (real differences): the Docker removal service (about 190 cases) is a root host-side helper that is
deliberately not part of this library; `HandoffInputFilter` and the run-level `tool_error_formatter` on
`RunConfig` are still synchronous; Python writes the `sandbox` section after a completed run too, Rust only for
a suspended one; for two same-named agents in a streamed run the section's current agent is the first of that
name; a dynamic approval rule is still consulted for arguments that match the schema, because Rust cannot detect
a custom `Deserialize`; assigning a field of the public `agent.model_settings` directly is untracked; docs.rs
builds without the `mongodb` feature; the websocket crate logs handshake data at TRACE through the `log` crate;
the Redis atomic replace has only run against a mock.

## 0.12.0 (2026-11-01)

Third pass over the parity ledger's Python-only bucket plus the real feature gaps it exposed. The files whose
Python-only reasons earlier agents had only spot-checked were read case by case, 130 more cases moved out of
the bucket (122 ported, 8 covered) and five gaps were closed. Measured state: 12,641 Python test cases, 8,585
ported, 673 covered, 3,383 Python-only, 0 not ported, 0 unaccounted. Notes by area:
`plans/reports/py-test-parity-core9-*.md`. No live model call was made.

Fixes and additions:

- Approvals are keyed by agent identity, so two agents with the same name no longer share decisions; the
  serialized form keeps the name and adds the identity, and name-only snapshots written before schema 1.7 still
  authorize by name. Duplicate agent names are allowed in a sandbox handoff graph (identity keys `name`,
  `name#2`, ...). A deferred top-level tool's permanent approval no longer authorizes a visible tool with the
  same bare name.
- Pre-approval validation is schema-only: the argument type's `Deserialize`, `deserialize_with` and application
  defaults run only after approval, as Python's validators do. A hosted MCP interruption whose `arguments` were
  removed is asked again.
- Sandbox: pre-stop hooks and `aclose` (`register_pre_stop_hook`, `run_pre_stop_hooks`, `aclose_session`); the
  guard runs the hooks before suspend and close and skips the snapshot after a hook failure.
- Runner: a streamed run cancelled during a compaction notice waits for it and settles the pending write; the
  checkpoint is published before a Session append, so a dropped or cancelled run leaves a recoverable state, and
  a second resume of a checkpoint whose append is running is refused; `memory::rewind_session_items`,
  `wait_for_session_cleanup` and `fingerprint_input_item` are public.
- Generic output schemas have readable names (`Vec<String>`, not `String>`); the default handoff name only
  warns for characters other than whitespace; a hosted MCP approval callback does not run for a request that
  already has a decision and records its answer; `RunState` exposes `SCHEMA_VERSION_SUMMARIES`.

Breaking changes: `RunState` has two more public fields (`session_write_in_progress`,
`legacy_name_owner_snapshot`), so struct literals need `..Default::default()`; `RunCheckpoint::take` can return
a state after a dropped or cancelled run; `ToolApprovalItem.agent` holds the identity key (equal to the name
except for second and later same-named agents); a persisted "always approve" for a deferred top-level tool
written by an earlier version is no longer honoured; a dynamic approval rule is not consulted for calls with an
integer for a float parameter, an absent nested property or a value the schema check cannot confirm (they
pause for manual approval first); the `name()` text of generic output schemas changed; the existing assertion
in `py_core3_runstatectx_hook_identity` now expects `sub#2`.

Known gaps (real differences, not Python-only mechanics): the Docker removal service (148 cases) is a root
host-side helper with no Rust equivalent; Rust cannot detect a custom `Deserialize`, so a dynamic approval rule
is still consulted for arguments that match the schema; schema 1.12 snapshots without an ownership ledger replay
once where Python replays twice; restore keeps or refuses an item whose agent is not in the graph where Python
skips it; a resume whose parked call lost its tool is refused where Python drops it; failure formatters are
synchronous and infallible; per-agent resume of sandbox sessions is not supported (only the starting agent's
session state is accepted); in-place writes to `agent.model_settings` are not tracked; the websocket crate logs
handshake data at TRACE through the `log` crate; the Redis atomic replace has only run against a mock.

## 0.11.1 (2026-11-01)

Documentation build only; the library code is identical to 0.11.0. The 0.11.0 build on docs.rs failed because
the `mongodb` 3.9 dependency does not compile on the nightly compiler docs.rs uses (a compiler async-drop cycle
error inside `mongodb`, not in this crate). docs.rs now builds every feature except `mongodb`, so the
`MongoDBSession` page is missing from docs.rs until the dependency compiles there again; the feature itself is
unchanged and is documented in the README.

## 0.11.0 (2026-11-01)

Second audit pass over the parity ledger's Python-only bucket: every file that still had Python-only cases was
re-read case by case, 441 cases moved out (380 ported, 61 covered), and the divergences they exposed were
fixed. Measured state: 12,641 Python test cases, 8,463 ported, 665 covered, 3,513 Python-only, 0 not ported,
0 unaccounted. Notes by area: `plans/reports/py-test-parity-core8-*.md`. No live model call was made.

Fixes:

- MCP: failed requests read like Python (`Failed to list tools from MCP server '<name>': HTTP error 503`);
  a server-reported 408 earns the isolated-session retry; a zero `max_list_pages`, a negative or non-finite
  backoff cap and a zero lifecycle timeout fail at connect with Python's messages; a failing `initialized`
  notification no longer logs the URL's query secret.
- Sandbox: unix-local `mkdir`, `ls`, `rm`, `mv` and `same_file` pin the validated parent with directory file
  descriptors, so a swapped parent or leaf cannot redirect them; `read_bounded` reads only the prefix;
  sandbox session state, event sinks, suspend and teardown no longer log payloads or raw provider errors for
  credentialed mounts; a failed create reports `WorkspaceArchiveWriteError`; client option payloads accept
  partial field sets; `Manifest` gains `from_value`, `from_untrusted_value` and `to_value`.
- Sessions: `Session` has defaulted `*_with_context` methods and the runner passes the run context to them;
  Redis compaction replaces the history in one atomic script; SQLite write transactions use `BEGIN IMMEDIATE`,
  table claims are atomic and the first-open WAL race is retried; the compaction session drops its response
  chain when a delete fails after committing; the OpenAI session constructors use the process default client.
- Models and realtime: `AnyLLMModel` refuses prompt-managed requests, strips nulls from replayed input and
  flattens object `reasoning`; an infinite `ModelSettings.timeout` means no timeout; concurrent realtime
  `close()` callers share one outcome.
- Tracing and macros: a failing trace processor is logged without data when sensitive-data logging is off;
  `#[function_tool]` doc parsing stops the description at `Returns:`/`Note:`/`# Examples` and reads NumPy and
  Sphinx styles; agents that differ only by `tool_use_behavior` get distinct identity keys.
- E2B `shutdown` never raises and Vercel refuses manifest changes on a stopped session.

Breaking changes: `ModelRetrySettings` and `ModelRetryBackoffSettings` fields are now `Option`, and
`ModelSettings::resolve` merges retry settings field by field; MCP request and connect errors are `User` errors
with new text; `from_json_str` errors for sandbox state have new text; the persisted S3 Files `extra_options`
shape is an ordered object (the array form is still read); realtime and E2B behaviour notes above; the two
OpenAI session `new()` constructors pick up a previously set default client; duplicate agents that differ only
by `tool_use_behavior` may get different identity keys than earlier snapshots. Additive: `Session::*_with_context`,
`SessionRunContext`, `Session::replace_history_atomically`, `SandboxSession::validate_manifest_application`,
`UnixLocalSandboxSession::with_after_pin_hook`.

Known gaps (real feature differences, not Python-only mechanics): the Docker removal service (148 cases) is a
root host-side helper with no Rust equivalent; Rust does not rewind session items on a `conversation_locked`
retry with server-managed conversations; there are no sandbox pre-stop hooks or `aclose`; duplicate agent
names in a handoff graph are rejected in Rust but allowed in Python; the Rust approval store keys approvals by
agent name; a hosted MCP interruption with `arguments` removed is still honoured; the websocket crate logs
handshake data (including the query string) at TRACE through the `log` crate; the Redis atomic replace has only
run against a mock; the manifest config carriers and custom environment value types are not ported; when a
history expires during compaction Rust keeps the live items and skips the replacement where Python's test
drives a failure.

## 0.10.0 (2026-11-01)

Closes the findings of an independent audit of the Python-only bucket in the parity ledger. The audit sampled
40 files and found about 10% of the sampled "cannot exist in Rust" cases hid real behaviour
(`plans/reports/py-test-parity-python-only-audit.md`); the named files are fixed and 47 more Python cases are
now ported. Measured state: 12,641 Python test cases, 8,083 ported, 604 covered, 3,954 Python-only, 0 not
ported, 0 unaccounted. The rest of the Python-only bucket was not re-audited, so some reasons may still be
too generous. Notes by area: `plans/reports/py-test-parity-core7-*.md`.

Fixes:

- Sandbox `local_dir` copy (unix) walks the tree with directory file descriptors, so a source root,
  subdirectory or file swapped for a symlink after validation fails with `LocalDirReadError`
  (`symlink_not_supported` or `path_changed_during_copy`) instead of being followed. This adds a unix-only
  `rustix` dependency under the `sandbox` feature.
- Realtime: every server event is validated, and a malformed or unknown event becomes an `Error` event
  (`Failed to validate server event`); a raw client message that is not a known client event is dropped and
  logged instead of sent. The error names the offending field and type, not the value.
- A bare legacy Vercel provider id resumes, as in Python. Live sandbox sessions honour the run config's
  concurrency limits (`SandboxSession::set_concurrency_limits`, defaulted).
- Shutdown on Cloudflare, Daytona and Modal swallows a failed provider stop or delete, as Python does.

Breaking changes: `SessionSettings` rejects unknown keys when deserialized; `RunState::from_json` rejects a
`pending_input` that is not a list; realtime mock servers and fixtures must send complete events; shutdown no
longer reports a rejected stop or delete on Cloudflare, Daytona and Modal.

Known differences: an expired-during-compaction history makes Rust skip the replacement with a warning
where Python raises; `Manifest` has no serde surface, so the manifest config tests stay Python-only;
`test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval` is classified Python-only without a
port; a tar nested-path case is skipped on macOS (needs GNU tar). The tar `--no-wildcards` difference
reported earlier does not exist: Rust and Python pass the flag in the same places.

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
