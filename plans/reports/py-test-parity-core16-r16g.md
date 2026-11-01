# py-test-parity core16 / r16g

Work list: `r16_6.txt` (all 33 files processed). Rust tests: `tests/py_core16_r16g_*.rs`. CSV: `py-test-parity-core16-r16g.csv` (columns path,moved_to_ported,moved_to_covered,remaining_python_only,read_depth; one row per file; before - ported - covered = remaining).

Counts: `before` is the number of Python test names that no Rust test cited by name and that I read in full; where it differs from the ledger count (python_only) the ledger undercounted names the earlier reports had classified in groups. Each such case ended as ported, covered or valid python-only.

Method: for each file the Python test names were matched against the Rust sources, the body of every uncited case was read, Rust behaviour was checked in source and tests, and each case ended as ported, covered or valid python-only. `read_depth=full` means every python-only body of the file was read.

## Totals

| | cases |
|---|--:|
| Python-only before | 383 |
| Moved to ported | 36 |
| Moved to covered | 42 |
| Python-only after | 305 |

## Per file

| file | before | ported | covered | after | notes |
|---|--:|--:|--:|--:|---|
| tests/sandbox/test_runtime.py | 45 | 5 | 2 | 38 | 5 ported (rejected live manifest update, capability failure redaction [src fix agent.rs], start failure redaction [src fix runtime.rs start_protected], nested handoff history with process_context, start-hook state controls tools); covered 2 (guardrail trip on injected session = r14b_injected_guardrail, call_model_input_filter public agent = py_sandbox_a_runtime_4); resume path-grant rebinding is performed by each client (docker, unix_local) and is covered by py_sandbox_a_runtime_5 and the docker/unix-local tests, so no runtime change was made for it; remaining: deepcopy/clone identity, asyncio cancel/task, monkeypatched prompt resource, private manager/_process_manifest/SystemExit, agent mutation in hooks, per-turn prepare_agent revalidation (Rust prepares once per run) |
| tests/test_function_schema.py | 45 | 2 | 0 | 43 | 2 ported (google Args after Note section without blank line); rest is Python signature/pydantic Field/Annotated/varargs/kwargs/griffe introspection with no Rust equivalent |
| tests/test_start_hook_tool_concurrency.py | 35 | 0 | 0 | 35 | all cases mutate agent.tools on an Agent shared by concurrent runs; Rust hooks receive &Agent so no shared mutable tool list exists (4 isolation cases already ported) |
| tests/sandbox/test_compatibility_guards.py | 27 | 0 | 0 | 27 | __all__ export surface and positional dataclass constructor/field order introspection (6 uncited by name + parametrized provider variants); behaviour guards already ported in py_sandbox_b_compat*.rs |
| tests/test_tool_batch_failure_history.py | 24 | 0 | 0 | 24 | 7 cases (partial history waits for input verdict x6, tool failure cancels pending guardrail) depend on Rust awaiting the input guardrail before running tools (design divergence, unresolved); the rest are monkeypatched asyncio/object identity/aliasing/cyclic python objects/asyncio cancel |
| tests/test_agent_runner_streamed.py | 19 | 4 | 4 | 11 | 4 ported (conversation-locked retry usage entries, approval not streamed, max-turns pending tool events, tool guardrail results survive handoff); covered 4; valid: asyncio cancel/task, dynamic agent.tools mutation, invalid filter types, monkeypatched finalizer |
| tests/extensions/memory/test_dapr_session.py | 14 | 0 | 0 | 14 | all 14 are the close()/ownership/context-manager lifecycle of the Dapr SDK gRPC client (closed operations raise, idempotent close, failed/cancelled cleanup, injected-client no-op, operation queued behind close); the Rust DaprSession is an HTTP session that owns no closable client; ping failure, limits, settings, corruption and metadata cases are covered by py_memory_dapr*.rs |
| tests/sandbox/test_manifest_config.py | 13 | 0 | 0 | 13 | dict-based RunConfig(sandbox=dict)/SandboxRunConfig(manifest=dict)/SandboxAgent(default_manifest=dict) fixture variants and a dict containing typed entry objects; Rust has typed configuration and one untrusted-value boundary (Manifest::from_untrusted_value, ported in py_core8_r8g1_manifest.rs) |
| tests/test_result_cast.py | 11 | 2 | 2 | 7 | 2 ported (wrong-type cast is an error, plain run has no agent tool invocation); covered 2 (agent tool invocation metadata via extractor tests); rest are gc/weakref/release_agents/asdict/pydantic rebuild and the unchecked default cast |
| tests/mcp/test_client_session_retries.py | 14 | 3 | 5 | 6 | 3 ported (failing tool call and prompt requests next to a slow call keep their own outcomes); covered 5 (failed-page retry, repeated cursor, required-parameter validation x3); valid: anyio ClosedResource/exception groups/outer cancellation, serialized-request in-flight accounting (Rust requests are concurrent by design), non-object arguments and non-numeric backoff (typed API). Ledger count was 9; 14 uncited names found |
| tests/test_run_internal_items.py | 16 | 0 | 3 | 13 | 3 covered (programmatic structured tool errors, digest fixed size, default assistant status equivalence); 13 valid: monkeypatched helpers, dynamically typed/broken objects, nested-history ref helpers keyed on object identity/read counts, legacy tool-search type string (ledger count 8; 16 uncited names read) |
| tests/test_trace_processor.py | 20 | 0 | 7 | 13 | 7 covered (retry-after honoured/bounded in tracing_exporter_tests and processors.rs unit tests, string truncation x4, wide-mapping work bound, top-level flush_traces re-export); 13 valid: unpaired surrogates, cyclic/non-JSON objects, int/uuid/nan keys, bytes (Rust spans hold serde_json values), monkeypatched clocks, atexit subprocess exit code, mocked httpx close (ledger count 7; 20 uncited names read) |
| tests/test_test_environment.py | 6 | 0 | 0 | 6 | pytest conftest proxy-environment helper (test infrastructure) |
| tests/utils/test_json.py | 2 | 0 | 0 | 2 | _to_dump_compatible on Python iterators and MappingProxyType; Rust values are serde_json |
| tests/sandbox/test_pty_types.py | 2 | 0 | 0 | 2 | _settle_pty_cleanup preserves asyncio.CancelledError reasons (asyncio cancel internals); the 4 pure PTY helper cases are already ported |
| tests/test_usaspending_setup_db.py | 4 | 0 | 0 | 4 | example script (examples/sandbox/extensions/daytona/usaspending_text2sql) with monkeypatched ssl/urllib: Python-only example/CI |
| tests/sandbox/test_sandboxes_import.py | 3 | 0 | 0 | 3 | Python import-system behaviour (sys.modules, sys.platform patching, __all__) |
| tests/test_image_generation_tool_typing.py | 3 | 0 | 0 | 3 | pyright static canaries, pydantic TypeAdapter schema and dataclass field introspection of ImageGenerationTool |
| tests/models/test_openai_chatcompletions.py | 5 | 0 | 1 | 4 | 1 covered (ModelSettings reach the chat request: py_models_chat_core.rs); valid: __bool__ override on Reasoning, httpx2.Timeout provider-native object, openai SDK client.with_options(max_retries) (Rust has no provider-managed retries), dictionary agent settings (typed) |
| tests/test_tracing_errors.py | 5 | 0 | 2 | 3 | 2 covered (redacted agent-run error detail, py_core3_tracing_errors_streamed.rs); valid: exception whose __str__ raises / BaseException from __str__ / custom span read-write failure (Rust Display is infallible) |
| tests/models/test_openai_retry_helpers.py | 14 | 0 | 9 | 5 | 9 covered by py_models_retry_helpers.rs and py_core15_r15b_retry_headers.rs (retry-after ms/seconds/date, status and error code, shared normalization, network error, websocket unsafe, delay-only advice); valid: httpx/httpx2 Headers objects, response_headers attribute, __cause__ precedence, unsafe_to_replay attribute (dynamic Python exceptions) |
| tests/test_cancel_streaming.py | 6 | 1 | 1 | 4 | 1 ported (timeout around next stream event returns promptly); 1 covered (no error on success); valid: monkeypatched run-loop internals, asyncio task cancellation at terminal wait/cleanup, falsy exception |
| tests/models/test_map.py | 4 | 0 | 1 | 3 | 1 covered (close continues over every provider: py_models_map_defaults.rs; first-failure re-raise is typed away because ModelProvider::close is infallible); valid: __bool__-falsey provider, invalid prefix-mode strings (typed enum), HybridCancellation asyncio |
| tests/test_tool_identity.py | 23 | 15 | 0 | 8 | 15 ported (call/approval qualified name, empty and synthetic namespace collapse, synthetic namespace refusal); valid: dynamic None/non-string/dict-or-object extractors and the lookup-key serialize/deserialize helpers (private; Rust keeps the lookup key only inside approval scope payloads) (ledger count 3; 23 uncited names read) |
| tests/extensions/experimental/hosted_multi_agent/test_model.py | 2 | 0 | 1 | 1 | 1 covered (model built without a client: py_ext_hosted_multi_agent.rs); valid: config accepted as a mapping (typed struct); the other 14 uncited names are covered by differently named tests in py_ext_hosted_multi_agent.rs |
| tests/memory/test_session_helpers.py | 4 | 0 | 0 | 4 | await_session_mutation asyncio cancellation/Future semantics x3 and the examples/memory/custom_session example adapter |
| tests/test_computer_tool_lifecycle.py | 2 | 1 | 0 | 1 | 1 ported (a failing instructions callback leaves no computer unreleased, py_core16_r16g_computer_lifecycle.rs); valid: start hook assigning agent.tools (agent mutation) |
| tests/realtime/test_conversion_helpers.py | 4 | 3 | 0 | 1 | 3 ported (invalid, malformed and typeless raw messages are dropped before the wire: py_core16_r16g_realtime_raw.rs); valid: tool output with call_id None (typed String call id, pydantic validation) |
| tests/realtime/test_session_tool_error_events.py | 1 | 0 | 0 | 1 | the one remaining variant makes the exception's __str__/__repr__ raise AssertionError (Python dunder methods); the redacted event/log cases are ported already |
| tests/sandbox/test_runtime_agent_preparation.py | 3 | 0 | 2 | 1 | 2 covered (base-instructions-only and default-prompt instruction composition: py_sandbox_a_runtime_1.rs, py_sandbox_a_runtime_7.rs); valid: SandboxAgent built from dictionary configuration (typed API) |
| tests/test_call_model_input_filter_unit.py | 3 | 0 | 2 | 1 | 2 covered (sync non-streamed and async streamed filter: py_core3_runstatectx_context.rs); valid: filter returning a non-ModelInputData value (typed) |
| tests/test_runner_model_provider_lifecycle.py | 1 | 0 | 0 | 1 | provider aclose failure while a run fails: Rust ModelProvider::close is infallible, so there is no secondary error to log or suppress (monkeypatched aclose) |
| tests/tracing/test_setup.py | 3 | 0 | 0 | 3 | atexit registration and module-global trace provider bootstrap with monkeypatched atexit/provider classes (Python process machinery) |

## Source fixes

- `src/sandbox/runtime.rs` (`start_protected`): a failing `SandboxSession::start` on a manifest with credentialed mounts surfaces only the fixed protected-mount sentence (injected session, client-created session and per-agent session). Python wraps `BaseSandboxSession.start` with `redact_mount_error_data`.
- `src/sandbox/agent.rs`: a capability `check_manifest` failure on a manifest with credentialed mounts is replaced by the protected-mount sentence (the Rust analogue of a `process_manifest` failure, which cannot occur because `process_manifest` is infallible).

## Breaking changes

None. Errors of the two cases above change message only when the manifest carries credentialed mounts.

A first attempt also rebound persisted path grants in `runtime.rs` before calling `client.resume`. It was removed: the clients already rebind (`docker.rs`, `unix_local.rs`), and `py_sandbox_a_runtime_5::resume_receives_the_trusted_manifest_with_its_mount_entries` requires the client to receive the persisted state unchanged.

## Unresolved questions

- Input guardrail ordering (`test_tool_batch_failure_history.py`, 7 cases: `*_partial_history_waits_for_input_verdict` x6 and `nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail`, plus `selected_tool_error_survives_completed_task_guardrail_race`). Python runs a turn's tools while a parallel input guardrail is pending; Rust awaits the guardrail first. The observable differences are the error that wins when both a guardrail verdict (reject or error) and a tool fail, and whether a tool's effects precede the verdict. Left Python-only; not a reason on the valid list, needs a decision on rewriting `run_turn` concurrency.
- Per-turn sandbox re-preparation (`test_prepare_agent_revalidates_cwd_*`, `test_prepare_agent_rematerializes_cached_tools_when_run_as_changes`): Python's runtime re-runs `prepare_agent` every turn against a possibly restarted session or a mutated `run_as`; Rust prepares once per run (plus the per-turn context rewrite). Treated as agent-mutation/typed-API, but a session stopped mid-run is not restarted by Rust.

## Rust tests added (`tests/py_core16_r16g_*.rs`, 9 files, 21 tests)

`sandbox_runtime` (6), `function_schema` (2), `runner_streamed` (4), `results` (2), `mcp_concurrency` (2), `stream_cancel` (1), `tool_identity` (2), `computer_lifecycle` (1), `realtime_raw` (1).

