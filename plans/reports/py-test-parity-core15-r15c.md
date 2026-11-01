# py-test-parity core15 / r15c: real read of 39 Python-only files

Scope: `scratchpad/r15_2.txt` (39 of 39 files done in this report). Per-file numbers in `py-test-parity-core15-r15c.csv`
(`before - ported - covered = remaining` for every row). Rust tests: `tests/py_core15_r15c_*.rs`.

## Totals (39 files)

| | cases |
|---|--:|
| Python-only before | 405 |
| Moved to ported | 13 |
| Moved to covered | 0 |
| Python-only after | 392 |

Counts are by Python test id (parametrized ids counted). The ledger gives group counts, not case ids, so a
"ported" or "covered" claim is made per Python test read against the Rust test or source it exercises;
the "remaining" figure is the ledger count minus those.

## Per file

| file | before | ported | covered | after | what was read / what remains |
|---|--:|--:|--:|--:|---|
| `test_check_optional_truthiness.py` | 71 | 0 | 0 | 71 | Python AST checker script under .github/scripts (CI tooling for Python source); read the test structure end to end: every case feeds Python source text to the checker, none exercises SDK behaviour |
| `test_function_tool_decorator.py` | 50 | 0 | 0 | 50 | callable-object/`__call__` protocol, `__wrapped__`/`inspect.unwrap` identity, class-scoped annotations, pydantic `ImageGenerationTool` schema, `inspect.signature` kw-only; the invocation, invalid-json, failure-handler, timeout and defer_loading cases are already ported (`py_core3_functiontools_*`) |
| `extensions/sandbox/test_vercel.py` | 35 | 5 | 0 | 30 | ported (py_core15_r15c_vercel_close.rs, source fixes): aclose after a snapshot failure still stops a mounted sandbox, a failed snapshot leaves the session running with its mount re-attached, aclose after a pre-stop hook failure, a closed mounted session refuses exec/start, model-facing tools are refused while a snapshot has the mount detached. Remaining: from_state/constructor two-source trusted mounts, traceback-frame redaction, hostile exception descriptors, asyncio lock/cancellation races, in-SDK stop retry, SDK env lookup |
| `realtime/test_session.py` | 34 | 2 | 0 | 32 | ported (source fixes): final tool names validated after connect-time overrides, close tears down running tool tasks before the transport closes. Not counted as covered although Rust tests exist (concurrent close sharing and cancelled close waiter were already moved out by r8g5; late tool completion, events after close, ambiguous names, failed connect, disabled handoff conflict and handoff re-evaluation could not be shown to still be in this bucket). Remaining: tracked-task/cleanup-task asyncio internals, traceback redaction, live tool/handoff objects inside initial_model_settings (Rust settings carry JSON specs) |
| `extensions/experiemental/codex/test_codex_tool.py` | 26 | 1 | 0 | 25 | ported (source fix): a Codex turn dropped by a failing sibling tool has already persisted its thread id. Remaining: explicit `ThreadIdStore` replaces key derivation/setattr on frozen, slots, pydantic and mapping contexts; dict/kwargs option coercion; circular value; lazy resolver identity; asyncio BaseException/cancellation |
| `realtime/test_testing.py` | 21 | 0 | 0 | 21 | listener-callback semantics of ScriptedRealtimeModel (broadcast quiescing, reentrant delivery, callback cancellation, startup delivery) with no Rust counterpart because transports deliver through channels; Python deepcopy/mutation isolation of mutable dict specs; a matcher that raises |
| `extensions/memory/test_async_sqlite_session.py` | 20 | 2 | 0 | 18 | ported: failed clear and failed pop leave no open write transaction (SqlxSession). Remaining: close()/cancel lifecycle of aiosqlite, injected rollback-failure quarantine, WAL pragma monkeypatch |
| `extensions/experiemental/codex/test_codex_exec_thread.py` | 18 | 0 | 0 | 18 | stream-limit env/explicit/invalid/range, web_search flag mapping and thread run aggregation/failure are tested in exec.rs and codex_tests.rs but were not counted as covered because the earlier report text does not show them in this bucket. Remaining: mapping/kwargs coercion, vendored target triples and path fallback, fake asyncio processes without stdio, env stringification |
| `models/test_any_llm_model.py` | 14 | 0 | 0 | 14 | Rust `AnyLLMModel` posts OpenAI-shaped requests itself; the remaining cases are vendor-SDK internals: provider `aclose()` failure/cancellation/in-flight close (11), Google SDK `http_options` and function-result role normalization (3) |
| `mcp/test_mcp_v2_http.py` | 12 | 0 | 0 | 12 | the portable 5xx cases (a 5xx fails only that request, retried with a budget, handshake 5xx fails connect, tolerated initialized 5xx) were already ported by r8g4 (py_core8_r8g4_mcp_errors.rs); I re-ran them against the Python bodies and found nothing new. Remaining: MCP v2 `server/discover` negotiation, httpx2 hooks and client factory, OAuth sub-request mapping, cancellation |
| `test_hitl_error_scenarios.py` | 12 | 0 | 0 | 12 | hand-built `ProcessedResponse` and private selection/rebuild helpers (`_select_function_tool_runs_for_resume`, `_resolve_interrupted_turn`), falsy `__bool__` pending items (2), dynamically typed `needs_approval='always'` (2), `set_public_agent` clone identity (3; run items carry the agent name), same-named sibling agents by object identity. The racing-decision and resume behaviours reachable through the Runner are ported in py_core2_scopedapprovals_hitl*.rs |
| `memory/test_openai_conversations_session.py` | 9 | 0 | 0 | 9 | asyncio cancel-settle (4), SDK retry layer `max_retries`/`with_options` (3; the Rust client has no SDK retry layer), falsy default client. Batching, lost-response and serialization behaviour is in py_memory_conversations*.rs |
| `test_run_state_pydantic_context.py` | 8 | 0 | 0 | 8 | pydantic BaseModel context defaults, UUID/datetime fields and `arbitrary_types_allowed` serializer requirement |
| `voice/test_testing.py` | 7 | 0 | 0 | 7 | bare-string sugar for scripted steps and queue-time snapshot/freeze of mutable arguments; Rust moves the values in |
| `test_code_change_verification_policy.py` | 6 | 0 | 0 | 6 | assertions on the Python repo's `.agents/skills` markdown and CI scripts |
| `voice/test_openai_model_provider.py` | 6 | 1 | 0 | 5 | ported (py_core15_r15c_voice_provider.rs, source fix): the process default client is used unless an option is explicit, and an explicit api_key/base_url/organization/project builds a client from the environment instead. Remaining: constructor argument conflicts (builder), `shared_http_client` httpx2 type, falsy client |
| `test_agent_runner_sync.py` | 5 | 1 | 0 | 4 | ported: `run_sync` refuses to start inside a running runtime. Remaining: asyncio event-loop policy handling (4) |
| `mcp/test_streamable_http_client_factory.py` | 4 | 0 | 0 | 4 | `patch('agents.mcp.server.streamablehttp_client')` call-argument assertions and a TypedDict annotation check; the caller-supplied HTTP client is tested in py_core7_rtvoicemcp_mcp_transport.rs |
| `realtime/test_app_server_debug.py` | 4 | 0 | 0 | 4 | examples/realtime FastAPI app server debug logging (Python example application) |
| `test_asyncio_stability_command.py` | 4 | 0 | 0 | 4 | Python repo shell script `.github/scripts/run-asyncio-teardown-stability.sh` driving pytest |
| `test_tool_custom_data.py` | 4 | 0 | 0 | 4 | non-JSON object and NaN/inf custom data (`serde_json::Value` cannot hold them); computer and apply_patch custom data are covered in core_audit_fixes_tools.rs |
| `extensions/sandbox/test_vercel_provider_contract.py` | 3 | 0 | 0 | 3 | asyncio cancelled creation, httpx handle lifecycle of an unadopted reconnect client (2); the wire, resume and cleanup cases are ported |
| `models/test_responses_websocket_session.py` | 3 | 0 | 0 | 3 | dictionary run_config normalization and unknown-field rejection (typed config in Rust), keyword override rejection, `hasattr` check |
| `sandbox/test_extract.py` | 3 | 0 | 0 | 3 | `io.IOBase` stream adapters (`zipfile` seekable/chunked non-seekable streams) |
| `test_doc_parsing.py` | 3 | 0 | 0 | 3 | `#[function_tool]` is rejected on methods by design (`does not support methods`); an associated function cannot host the generated args struct; `test_functions_are_ok` only calls fixtures |
| `test_run_internal_approvals.py` | 3 | 0 | 0 | 3 | private `_build_function_tool_call_for_approval_error` on arbitrary dict/object calls; the namespace it keeps never reaches the output item |
| `tracing/test_trace_context.py` | 3 | 0 | 0 | 3 | falsy (`__bool__`) trace and response objects |
| `extensions/sandbox/test_cloudflare.py` | 2 | 0 | 0 | 2 | asyncio cancellation of resume start and private `_before_shutdown` hook |
| `sandbox/capabilities/test_filesystem_capability.py` | 2 | 1 | 0 | 1 | ported (py_core15_r15c_fs_capability.rs): a configurator can give `view_image` a per-call approval rule. Remaining: unbound capability raises (a binding is required by the signature) |
| `sandbox/test_retry.py` | 2 | 0 | 0 | 2 | `__cause__`/`__context__` exception-chain walking helpers, which Rust has no counterpart for |
| `test_pretty_print.py` | 2 | 0 | 0 | 2 | a dataclass output whose `__str__` renders it (streamed and not) |
| `test_tool_guardrails.py` | 2 | 0 | 0 | 2 | invalid guardrail callables (typed in Rust) |
| `models/test_litellm_logging_patch.py` | 1 | 0 | 0 | 1 | patches litellm's own logging module |
| `realtime/test_session_close_iteration.py` | 1 | 0 | 0 | 1 | the `asyncio.CancelledError` variant of readers-finish-before-transport-cleanup; the None and error variants and the two other cases are ported |
| `sandbox/test_pty_output.py` | 1 | 0 | 0 | 1 | monkeypatched `time.monotonic` plus injected poll/settle callbacks |
| `test_agent_tool_input.py` | 1 | 0 | 0 | 1 | `AgentAsToolInput.model_validate` pydantic validation |
| `test_image_generation_tool_reference.py` | 1 | 0 | 0 | 1 | griffe docstring introspection |
| `test_visualization.py` | 1 | 0 | 0 | 1 | monkeypatched `graphviz.Source.render` |
| `voice/test_pipeline_tracing.py` | 1 | 0 | 0 | 1 | a `DefaultTraceProvider` subclass steered by a ContextVar |

## Source bugs fixed

All small and targeted; each has a Rust test in `tests/py_core15_r15c_*.rs`.

- `src/realtime/session.rs` `build_model_settings_from_agent`: duplicate tool/handoff names were validated before the
  connect-time override was merged, so an override that replaces the conflicting tools was still refused. The final
  names are validated (Python `_get_updated_model_settings_from_agent`). `RealtimeSession::close_locked` now awaits the
  aborted guardrail/tool/pump tasks before closing the transport, so a late tool result cannot reach a closed transport
  (Python `test_close_waits_for_background_finalizers_before_model_close`).
- `src/sandbox/session_lifecycle.rs`, `src/sandbox/session.rs`, `src/extensions/sandbox/vercel/mod.rs`: new
  `SandboxSession::shutdown_after_failed_close` (default `false`; Vercel returns `true` when it holds S3 mounts), so
  `aclose` still stops a credential-mounted sandbox after the snapshot or `stop` failed (Python
  `_VercelSandboxSessionWrapper._aclose_impl`). A Vercel session that held S3 mounts and was shut down refuses further
  exec/read/write with `reason = mounted_session_closed` (Python `_ensure_sandbox`). The Vercel session now owns a
  `MountTransition`, so model-facing tools are refused while a snapshot has the mount detached (Docker and unix-local
  already did this).
- `src/extensions/codex/tool.rs`: the thread id is written to the run-context store when `thread.started` arrives, not
  only after the turn, so a Codex turn dropped by a failing sibling tool call can be resumed (Python
  `test_codex_tool_persists_thread_id_for_handled_parallel_cancellation`).
- `src/voice/models/openai_model_provider.rs`: `OpenAIVoiceModelProvider::new()` ignored the process default client.
  It now uses `set_default_openai_client` unless an explicit api key, base URL, organization or project is given, in
  which case a client is built from the environment plus those options (Python `_get_client`). Added
  `with_organization` and `with_project`. The default is resolved when the provider is built, not lazily.

## Breaking changes

- New default method `SandboxSession::shutdown_after_failed_close` (non-breaking for implementors).
- `OpenAIVoiceModelProvider` no longer derives `Default`; `Default`/`new()` are unchanged in behaviour apart from
  honouring the default client. A provider that called `with_api_key` on `new()` now starts from an environment
  client instead of the default client.
- A Vercel session with S3 mounts that was shut down now returns an error from exec/read/write instead of reaching the
  stopped sandbox.

## Honest coverage statement

The ledger gives counts per file, not case ids, so each file was read in the Python source (test names and the bodies of
every case that looked portable) and matched against the Rust tests and source. Bodies I did not read line by line:
`test_check_optional_truthiness.py` (831 lines, all of it feeds Python source to a Python AST checker; structure read),
the cancellation/lock-race tail of `test_vercel.py` and `test_hitl_error_scenarios.py` (private planner helpers), and the
typed-context group of `test_codex_tool.py`. No case is counted as "covered": where a Rust test already exists I could not show from the earlier reports that the case was still in the Python-only bucket, so it is not moved (several of my first candidates were already out: the realtime close-sharing pair by r8g4/r8g5, the MCP 5xx cases by r8g4, whose duplicate tests I deleted).

## Unresolved questions

- `#[function_tool]` is rejected on methods and associated functions by design. Python documents instance methods and
  classmethods; supporting an associated function would need the macro to emit the args struct inside the generated
  constructor body. Left as is (3 cases of `test_doc_parsing.py`).
- Rust `RealtimeSessionModelSettings.tools` carries JSON specs, so a connect-time override cannot carry a tool
  implementation (Python `initial_model_settings["tools"]` can). A tool the override removes from the advertised list
  is still dispatchable because dispatch uses the agent's tools.
- Vercel `shutdown` is a single stop request; Python retries the stop once while detaching mounts. Not ported.

Status: DONE_WITH_CONCERNS

## Verification

`cargo clippy --all-targets -D warnings` is clean with all features and with default features. Passing with `--all-features`: every `py_core15_r15c_*` binary, the Vercel, realtime session, sandbox, codex, voice and sqlx binaries listed above. One unrelated failure: `realtime_tests::the_playback_tracker_decides_where_audio_is_truncated` (expects `audio/pcm`, gets `pcm16`), in the OpenAI realtime audio-format code, which this round did not touch; another agent's in-flight edit.
