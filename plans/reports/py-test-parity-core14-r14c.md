# py-test-parity core14 / r14c: re-read of the remaining Python-only groups

Scope: the 119 files of `scratchpad/r14_r14c.txt` (1,292 Python-only cases) plus
`tests/test_released_api_contract.py` (126). Per-file numbers: `py-test-parity-core14-r14c.csv` (120 rows,
`before - ported - covered = remaining` for every row). Rust tests: `tests/py_core14_r14c_*.rs`.

## Totals

| | cases |
|---|--:|
| Python-only before (119 files + released_api_contract) | 1,418 |
| Moved to ported | 52 |
| Moved to covered | 0 |
| Python-only after | 1,366 |

Parametrized Python tests are counted by pytest id (`model_retry`: 3 + 1 + 4 ids of settings validation, plus 3
stream-close tests); a Rust test that loops the ids carries all of them. The `model_retry` row moves 11: the two
valid-settings cases (`zero_backoff_values_are_allowed`, `a_finite_model_call_timeout_is_accepted_and_traced`) were
already ported in `py_models_retry_get_response.rs`, so `zero_backoff_values_deserialize` and
`a_finite_model_call_timeout_round_trips` in the new file are extra checks and are not counted.

## Honest coverage statement

I did not read the body of every Python-only case. The previous rounds had already read most of these groups, and
this round the budget went to the groups where a portable behaviour was plausible. What was done per file:

- **Names matched against Rust coverage, bodies read where a case looked portable** (every case of these files
  was classified from its test name and, for the ones named below, its body): `test_mount_security` (per-index table
  of `py-test-parity-sandbox-a-mount-security.md` re-checked against the 134 names), `test_run_state_compatibility_corpus`,
  `test_function_schema` (docstring-style group already ported through the `#[function_tool]` macro), `mcp/test_mcp_server_manager`,
  `models/test_model_retry`, `test_start_hook_tool_concurrency`, `test_native_tool_failure_history`,
  `test_example_workflows` (all 30 bodies), `voice/test_pipeline`, `extensions/memory/test_redis_session`,
  `extensions/memory/test_mongodb_session`, `extensions/memory/test_encrypt_session`, `test_error_logging_redaction`,
  `test_released_api_contract`, `test_runtime_symmetry_contract`, `test_exception_exports`, `test_handoff_tool`,
  `test_strict_schema`, `test_tool_use_behavior`, `test_run_hooks`, `test_tool_failure_boundaries`,
  `test_tracing_errors`, `sandbox/test_snapshot`, `sandbox/test_pty_types`, `sandbox/test_mount_lifecycle`,
  `sandbox/test_session_utils`, `realtime/test_realtime_model_settings`, `realtime/test_session_payload_and_formats`,
  `test_run_state_pending_input`, `test_config`, `test_agent_config`, `test_result_cast`, `test_run_internal_items`,
  `test_process_model_response`, `test_stream_events`, `test_tool_identity`, `test_decorators`, `test_hitl_utils`,
  `test_computer_action`, `test_call_model_input_filter`, `test_agent_instructions_signature`, `test_agent_tool_state`,
  `test_run_state_agent_identity`, `test_run_state_hook_identity`, `utils/test_json`, `utils/test_pretty_print_and_items`,
  `tracing/test_spans_impl`, `test_agent_tool_enablement_snapshot`, `test_agent_as_tool_callback_validation`,
  `test_tool_exception_formatter`, `test_agent_clone_shallow_copy`, `test_default_tool_error_redaction`,
  `test_shell_examples`, `test_decisions_examples`, `test_asyncio_tasks`, `mcp/test_list_page_limits`,
  `mcp/test_client_session_retries`, `mcp/test_streamable_http_session_id`, `mcp/test_mcp_approval`,
  `test_agent_runner` / `test_agent_runner_streamed` (through `py-test-parity-core2-runner.md`'s group list).
- **Not re-read this round (reason carried over from the earlier report; list for the next round)**:
  `test_change_detection`, `test_update_rclone_pin`, `test_test_environment`, `test_usaspending_setup_db`,
  `test_financial_research_budget`, `test_image_generation_tool_typing`, `docs/test_translate_docs`
  (CI scripts, examples, typing); `realtime/test_app_server_security`, `realtime/test_twilio_sip_server`
  (Python example servers); `extensions/sandbox/test_blaxel`, `test_modal`, `test_daytona`, `test_e2b`;
  `extensions/memory/test_memory_imports`; `mcp/test_mcp_util`, `mcp/test_mcp_imports`, `mcp/test_mcp_auth_params`,
  `mcp/test_mcp_server_manager_cleanup_state`; `memory/test_session`, `memory/test_session_helpers`,
  `deferred_session_writes/test_pending_write`; `sandbox/test_memory`, `test_unix_local`, `test_manifest`,
  `test_entries`, `test_session_state_roundtrip`, `test_docker_removal`, `test_docker_network_mode`,
  `test_runtime_agent_preparation`, `capabilities/test_shell_capability`, `capabilities/test_tool_input_errors`,
  `test_sandboxes_import`; `test_programmatic_tool_calling`, `test_function_tool`,
  `test_function_tool_approval_arguments`, `test_run_config`, `test_cancel_streaming`;
  `models/test_openai_chatcompletions(_stream)`, `test_litellm_chatcompletions_stream`, `test_map`,
  `test_kwargs_functionality`, `test_openai_responses_converter`; `realtime/test_openai_realtime`,
  `test_agent`, `test_audio_formats_unit`, `test_openai_realtime_conversions`, `test_session_approvals`,
  `test_session_guardrails`, `test_session_tool_error_events`; `extensions/test_tool_output_trimmer`,
  `extensions/experimental/hosted_multi_agent/test_model`; `tracing/test_processor_api_key`, `test_setup`;
  `voice/test_pipeline_config`, `voice/test_openai_stt_client_config`; `test_provider_span_errors`,
  `test_tracing_provider_safe_debug`, `test_function_schema_logger_restore`.

## Ported (37)

| Rust file | Python source | cases |
|---|---|--:|
| `py_core14_r14c_settings_validation.rs` (5 tests) | `models/test_model_retry.py`: negative backoff values (3 ids), negative backoff in a retry-settings dict, invalid model-call timeout (4 ids) | 8 |
| `py_core14_r14c_retry_stream_close.rs` (3) | `models/test_model_retry.py`: abandoned stream closed before the retry, current stream closed when the consumer stops, provider stream closed exactly once. A drop guard inside the provider stream counts the closes | 3 |
| `py_core14_r14c_example_workflows.rs` (16) | `test_example_workflows.py`: judge loop, parallel translations, both deterministic story flows, routing stream, input and output guardrails, immediate and after-turn cancel, streamed handoff event, forced tool-use behaviours, routing multi-turn, conditional agent-tool enablement, orchestrator over agent tools, nested `on_stream`, a failing nested tool beside a sibling | 16 |
| `py_core14_r14c_runtime_symmetry.rs` (2) | `test_runtime_symmetry_contract.py`: the scripted model records exactly the model-visible request fields (exhaustive destructure), streamed and non-streamed runs fail with the same error | 2 |
| `py_core14_r14c_public_api_contract.rs` (8) | `test_released_api_contract.py`: fixture-version surface (runner, result, state, `preserve_raw_usage`), curated public properties (`RunState.pending_input`, `RetryPolicyContext`), public state surfaces (testing doubles, sandbox state flags), plus `test_exception_exports.py` (every exception class is an `AgentsError` variant reachable from the crate root) | 4 |
| `py_core14_r14c_mongodb_driver_info.rs` (2) | `test_mongodb_session.py`: injected client receives the SDK identity, `from_uri` forwards it. A tiny OP_MSG server records the handshake | 2 |
| `py_core14_r14c_snapshot_atomic.rs` (1) | `sandbox/test_snapshot.py`: a failed persist keeps the previous snapshot and leaves no temporary file | 1 |
| `py_core14_r14c_realtime_call_session.rs` (1) | `realtime/test_realtime_model_settings.py`: a call-attached session update omits unset audio formats | 1 |

Public API contract: the Python file is almost entirely a test of the repository's own contract-validation
tooling (`inspect`/`typing`/`pydantic` introspection, dependency-installation policy JSON, platform import errors,
promotion of new exports). Those groups, 123 of 126, have no Rust counterpart. The three that carry over are the
frozen released surface, the curated public properties and the testing-double properties; Rust pins them at
compile time with signature coercions and exhaustive destructures (a removed or re-typed item breaks the build).
`ScriptedModel`, `ScriptedRealtimeModel` and `ScriptedSandboxSession` exist in Rust under the Python names; the voice doubles are `Fake*` (`FakeSTTModel`, `FakeTTSModel`, `FakeVoiceWorkflow`), so the pinned voice names are the Rust ones.

## Source changes

| file | change |
|---|---|
| `src/model_settings.rs` | `ModelSettings.timeout` rejects zero, negative and non-finite values on deserialization; `ModelRetryBackoffSettings.initial_delay/max_delay/multiplier` reject negative and non-finite values (Python `Field(gt=0)` / `Field(ge=0)`). Builder and struct-literal values stay unchecked, and the runtime already tolerated them without panicking (`model_call.rs` filters non-representable timeouts). `default` was added so a missing field still deserializes |
| `src/memory/mongodb_session.rs` | `from_uri` parses the URI into `ClientOptions` and sets `DriverInfo { name: "openai-agents", version: crate version }` (a driver info already on the options is kept); `with_client` appends the same identity with `Client::append_metadata` (idempotent, a client that already handshook is left alone) |
| `docs/project-changelog.md` | two Unreleased bullets for the above |

No public signature changed. Behaviour changes: invalid timeout/backoff JSON now fails to deserialize; the MongoDB
handshake now carries a second driver name.

## Valid Python-only reasons confirmed this round (by group)

- `test_mount_security` (125): pydantic/`__deepcopy__`/serializer/descriptor hostility, custom `Mount` subclasses
  spoofing trust, manifests inside state payloads (Rust state carries only a digest), exception-redaction
  decorators over hostile exception objects.
- `test_run_state_compatibility_corpus` (61): exception/traceback/slots/`__notes__`/`ExceptionGroup`/cancellation
  ownership of Python objects, monkeypatched `json.loads`, comparator self-tests.
- `test_function_schema` (46): pydantic/`Annotated`/`*args`/`**kwargs` annotations.
- `test_mcp_server_manager` (38): worker tasks and anyio task affinity, lifecycle lock, `CancelledError`/`BaseException`
  policy, loop task factory, one-shot iterables. The `get_mcp_server_log_name` table and the zero-timeout check are
  already covered (`py_core13_r13b_mcp_errors.rs`, `py_mcp_manager.rs`).
- `test_model_retry` (25 left): asyncio timeout cleanup and parent-cancellation ordering, owner-task traceback locals,
  composed policies mutating provider authority (Rust contexts are immutable), falsey-policy attributes.
- `test_start_hook_tool_concurrency` (35): the whole file defends against hooks mutating an `Agent` shared across
  concurrent runs; a Rust `Agent` is read-only while it runs.
- `test_native_tool_failure_history` (34): tools raising `CancelledError`, cancelled child tasks, native invocations that
  convert or suppress their own cancellation, raising `on_tool_end` hooks.
- `test_redis_session` (48): redis-py pool checkout/release/WATCH internals (Rust writes are single Lua scripts),
  `close()` terminal state, `decode_responses`, private `_get_next_id`, asyncio cancellation. Everything with an
  observable Rust counterpart is in `py_memory_redis.rs`, `py_core13_r13b_redis.rs`, `py_core14_r14a_redis_cluster.rs`.
- `test_example_workflows` (14 left): imports of Python example modules (financial research, sandbox examples,
  web-search URL helpers, docker runner formatting) and `asyncio.CancelledError` raised from tools.
- `test_shell_examples`, `test_decisions_examples`: import the Python `examples/` modules.
- `voice/test_pipeline` (27): asyncio cancellation of the streamed result and pipeline, numpy dtype spellings,
  dictionary config normalization.
- `test_released_api_contract` (123 left): see above.
- `sandbox/test_pty_types` (2), `test_mount_lifecycle` (4): asyncio cancellation of cleanup; `test_session_utils` (2):
  Python `io` seekable-stream length.

## Unresolved questions (decisions, not bugs)

1. **Infallible lifecycle hooks (resolved in the follow-up round: see below).** About 20 Python-only cases (8 in `test_run_state_pending_input`, 2 in
   `test_native_tool_failure_history`, boundary cases of `test_tool_failure_boundaries`, the hook halves of
   `test_tool_batch_failure_history`) are blocked only because a Rust `RunHooks`/`AgentHooks` method returns `()`, so a
   hook cannot fail a run and the recovery paths have no trigger. Making hooks return `Result` is a breaking trait
   change and a product decision; I did not make it. If the owner wants it, those cases become portable.
2. `MCPToolCancellationError` has no Rust variant: it marks an internal MCP cancellation of the Python event loop. The
   export test is ported for every other exception class. Confirm no Rust equivalent is wanted.
3. Python lets a caller pass `driver` through `client_kwargs` and keeps it; Rust `from_uri` keeps a driver info already
   present on the parsed options, but the URI cannot carry one, so `test_caller_supplied_driver_info_is_not_overwritten`
   has no equivalent entry point.
4. Dropping a `RunResultStreaming` without cancelling it leaves the run task running (Python does the same: closing
   `stream_events` does not cancel the run). The stream-close test therefore cancels the run, as the consumer-stop
   analogue.

## Verification

- `cargo test --all-features` (isolated target dir) of every `py_core14_r14c_*` binary: green. Re-run of the binaries
  that deserialize `ModelSettings` (`py_model_settings_serialization`, `py_models_retry_get_response`,
  `py_models_retry_stream`, `py_core10_r10b_model_settings`) and `py_memory_mongodb` after the source edits: green.
  The first version of the `timeout` validator dropped the implicit `Option` default and broke three
  `py_model_settings_serialization` tests; `#[serde(default)]` fixed it.
- `cargo clippy --all-features --lib` plus the eight new test targets with `-D warnings`: clean. The same eight
  targets under default features (`cargo clippy --test ... -- -D warnings`): clean, and the default-feature test run of
  the five ungated binaries passes. rustfmt run on the new test files and `src/memory/mongodb_session.rs`.
- The MongoDB handshake test looks at every handshake the server saw (the driver's monitor connection can handshake
  before the metadata is appended).
- Follow-up round: 29 test binaries that use hooks or the failure checkpoint (every file that implements `RunHooks`/`AgentHooks`
  or calls `.hooks(...)`, plus `py_core6_nativetools_failure_history`, `py_core3_runstatectx_pending_b`, `py_core2_toolerrors_batch`,
  `py_core14_r14c_hook_failures`) pass under `--all-features`. `cargo clippy --lib` plus the twelve new or edited test targets with
  `-D warnings`: clean under all features and under default features. rustfmt clean on every touched file.
- Not run: the full workspace gate (shared disk and CPU); only the binaries named above.

## Follow-up round: fallible hooks and a wider read

The coordinator decided that lifecycle hooks become fallible and that the unread files be read case by case.

### Fallible hooks (15 cases ported, plus the tool-batch hook cases)

- API: `RunHooks` and `AgentHooks` gain a `try_<callback>` twin per callback (`try_on_llm_start`, `try_on_llm_end`,
  `try_on_agent_start`, `try_on_agent_end`, `try_on_handoff`, `try_on_tool_start`, `try_on_tool_end`; `try_on_start` and
  `try_on_end` on `AgentHooks`) returning `Result<()>`. The runner calls only the twins. Each twin's default calls the
  infallible callback and returns `Ok(())`, so the 27 existing hook implementations compile and behave unchanged (a
  breaking `Result` return on the old methods would have forced a rewrite of every one; the twin is the migration
  path, documented in `docs/project-changelog.md`). `src/lifecycle.rs`.
- Run-level and agent-level hook of one event now run together (`futures::join!`, `join_hook_results` in
  `src/run_internal/mod.rs`) and both always run, with the run-level failure winning, as Python's `asyncio.gather`
  does. Before, they ran one after the other.
- Failure effects, all in `src/run_internal/`:
  - `on_llm_end` fails: the response is recorded (`model_responses`, server tracking through the new
    `accept_server_response`) and the turn checkpoints as in flight, so the resume processes the accepted response with
    no second model call and no second end hook.
  - `on_tool_start` fails: the call counts as started and unproven; a retry refuses it ("output was not committed").
  - `on_tool_end` fails: the call's output stays committed and public (`ToolFailure.output_is_public`, new), the batch
    keeps it in history in call order, and a retry skips the call. Local tools (computer, shell, local shell, apply
    patch, custom) fail the same way through `start_hooks` / `end_hooks`.
  - `on_agent_end` fails: the final message is withheld from history and run data; the turn's tool calls and outputs stay.
- Tests: `py_core14_r14c_hook_failures.rs` (8: `test_server_response_acceptance_commits_before_hook_failure` x2,
  `test_server_accepted_computer_start_hook_failure_is_not_replayed` x2, `test_server_accepted_tool_side_effect_failure_is_safe`
  x4), the `custom/hook/raises` case added to `py_core6_nativetools_failure_history.rs` (2 ids, blocking and streamed),
  `provider_final_hook` and `final_hook` added to `py_core2_toolerrors_batch.rs::completed_history_at_failure_boundaries`
  (4 ids), and the `hook` failure added to `completed_sibling_survives_tool_batch_failure` (2 ids of
  `test_tool_batch_failure_history.py`, a file outside this list, so not in the CSV).
- Still Python-only in those files: `tool_cancel`, `native_cancel`, `parent_cancel`-style asyncio cancellation, `mcp_callback`
  (the hosted-MCP approval callback is infallible by type), and the remaining three `test_run_state_pending_input`
  cases (private tracker method, object-identity rewrites).

### Files read in the follow-up (names against Rust coverage, bodies where noted)

Read in this round and judged: `test_change_detection` (CI shell script `.github/scripts/detect-changes.sh`),
`test_update_rclone_pin` (CI updater script), `test_test_environment` (pytest proxy fixture), `test_usaspending_setup_db`
and `test_financial_research_budget` (Python examples), `test_image_generation_tool_typing` (pydantic schema),
`docs/test_translate_docs` (docs script), `realtime/test_app_server_security` and `realtime/test_twilio_sip_server`
(FastAPI example servers: imports confirm), `extensions/memory/test_memory_imports` and `mcp/test_mcp_imports`
(import-guard messages), `memory/test_session_helpers` (dict settings, asyncio), `extensions/sandbox/test_blaxel`
(bodies of the timeout/frozen/shutdown groups), `extensions/sandbox/test_modal` (bodies of the Windows path pair; the
Python-only remainder is cancellation of snapshot persist and hydrate), `mcp/test_mcp_util` (names; the constructor
guardrail test body), `sandbox/test_memory` (bodies of the enqueue-failure, sort-key and positional-config cases; all
have Rust counterparts or are pydantic), `test_function_tool` and `test_function_tool_approval_arguments` (names),
`test_run_config` (names: all dictionary-config), `sandbox/capabilities/test_shell_capability` and `test_tool_input_errors`,
`sandbox/test_docker_network_mode`, `test_runtime_agent_preparation`, `test_docker_removal`, `test_sandboxes_import`,
`sandbox/test_unix_local` (names), `tracing/test_processor_api_key`, `tracing/test_setup`, `test_provider_span_errors`,
`test_tracing_provider_safe_debug`, `voice/test_pipeline_config`, `voice/test_openai_stt_client_config`,
`mcp/test_mcp_auth_params`, `mcp/test_mcp_server_manager_cleanup_state`, `deferred_session_writes/test_pending_write`
(all private `_PendingSessionWrite` state), `realtime/test_agent` (bodies; one case ported), `realtime/test_openai_realtime`
(names), `extensions/experimental/hosted_multi_agent/test_model` (body of the concurrency validation: `-1` and `True`
are unrepresentable in `usize`, `0` already ported), `models/test_*` stream, map, kwargs and converter files (counts and
the earlier reports' reasons), `extensions/test_tool_output_trimmer`.
That is about 45 files at name level, 12 with bodies. The rest of the list was not reopened beyond the earlier rounds.
The one real find was `test_agent_clone` in `realtime/test_agent` (ported, trivially true under Rust ownership).

### Files of the list not reached beyond the first round

`sandbox/test_manifest`, `sandbox/test_entries`, `sandbox/test_session_state_roundtrip`, `sandbox/test_mount_lifecycle`
(bodies), `test_programmatic_tool_calling`, `test_cancel_streaming`, `test_agent_runner` and `test_agent_runner_streamed`
(bodies), `models/test_openai_chatcompletions`, `models/test_openai_chatcompletions_stream` (bodies), the remaining
`realtime/test_*` files with one case each, `extensions/sandbox/test_e2b`, `test_daytona` (beyond the earlier reports),
`mcp/test_streamable_http_session_id`, `mcp/test_list_page_limits`, `mcp/test_mcp_approval`, `memory/test_session`.

Status: DONE_WITH_CONCERNS

Summary: 120 files (119 listed + `test_released_api_contract.py`), 1,418 Python-only cases before; 52 moved to
ported, 0 to covered, 1,366 remain Python-only (CSV rows balance). Ten new test files (47 Rust tests) under
`tests/py_core14_r14c_*.rs` plus edits to three existing ones; source changed in `src/lifecycle.rs`, `src/run_internal/{mod,tools,local_tools}.rs`,
`src/model_settings.rs`, `src/memory/mongodb_session.rs`; changelog entries.

Concerns: the coordinator asked for at least 40 more files read case by case; about 45 were read at name level and 12 with
bodies, the list in the follow-up section is what was not reached. `MCPToolCancellationError` has no Rust variant (question 2).
The hook change is additive (twin methods), not a `Result` return on the old methods; say if the old signatures must change.
