# Round 15, r15b: python-only audit of 39 files (work list r15_1)

Crate 0.16.0 plus other agents' uncommitted work. Per-file numbers: `py-test-parity-core15-r15b.csv` (before - ported - covered = remaining on every row; "before" is the work-list count, 405 in total).

Totals: 39 files, 405 python-only before, 21 ported, 5 covered, 379 remain.

How each file was read (honest account; the largest files were not read start to finish):

- Read in full: `mcp/test_server_errors.py`, `test_agent_scoped_approvals.py`, `test_tool_batch_failure_history.py`, `sandbox/test_unix_local_user_file_io.py`, `test_source_compat_constructors.py`, `sandbox/test_manifest_config.py`, `sandbox/test_workspace_payloads.py`, `models/test_openai_retry_helpers.py`, `test_run_context_wrapper.py`, `test_computer_tool_lifecycle.py`, `models/test_litellm_extra_body.py`, `tracing/test_traces_impl.py`, `test_agents_logging.py`, `models/test_model_payload_iterators.py`, `codex/test_payloads.py`, `deferred_session_writes/test_compaction.py` (first 340 lines, which hold every python-only case), `sandbox/test_session_sinks.py` (the four python-only tests with their setup).
- Python-only bodies read, rest of the file matched by test name against Rust tests: `test_run_step_execution.py` (hosted MCP matrix, cancel and trace cases, 1100-2000 and 3400-4200), `sandbox/test_docker.py` (every unmatched test), `test_tool_approval_call_id_reuse.py` (every unmatched test), `test_tool_name_collision_policy.py`, `sandbox/test_unix_local_file_io.py`, `test_items_helpers.py`, `tracing/test_import_side_effects.py`, `extensions/memory/test_sqlalchemy_session.py`, `sandbox/test_compatibility_guards.py`, `test_tool_origin.py`, `sandbox/test_docker_removal_worker.py`.
- Name-level only (Python-only reason confirmed against the ledger groups and one sampled body): `extensions/memory/test_dapr_session.py`, `test_asyncio_progress.py` (7 asyncio-internals tests, first 90 lines read), `sandbox/capabilities/test_skills_capability.py`, `test_runner_tracing_disabled.py`, `models/test_anthropic_thinking_blocks.py`, `realtime/test_playback_tracker.py`, `sandbox/capabilities/test_compaction_capability.py`, `test_agent_hooks.py`, `test_tool_output_conversion.py`, `tracing/test_tracing_env_disable.py`.

## Per-file result

| file | before | ported | covered | after | what changed and what remains |
|---|--:|--:|--:|--:|---|
| mcp/test_server_errors.py | 72 | 0 | 0 | 72 | exception-graph, traceback-locals, `caplog` record fields, `patch.object` of private retry methods, constructor validation of NaN/negative/str timeouts (a Rust `Duration` rules them out; a sub-microsecond timeout is ported in `py_core13_r13b_mcp_errors.rs`). The message classes per transport are ported by earlier rounds |
| test_run_step_execution.py | 45 | 1 | 0 | 44 | ported: a failing function tool next to a local shell call keeps the shell output (`mixed_tool_calls_preserve_shell_output_when_function_tool_fails`). Remaining: asyncio cancel/`CancelledError`/settle-turn cases (about 28), hosted MCP matrix built from hand-made `processed_response` and private `collect_manual_mcp_approvals` (the end-to-end halves are `py_core2_approvals_reuse_b.rs::sticky_mcp_*` and `unbindable_*`; `approve_tool` on an incomplete identity is `py_core2_approvals_context.rs`), public-vs-execution agent clone identity, exception objects returned as tool output, `ContextVar` propagation, eager task factory, unhashable argument dicts |
| test_agent_scoped_approvals.py | 40 | 0 | 0 | 40 | Temporal converter, gc/weakref, name-only queries (every Rust approval query names its owning agent), distinct same-named agents or tools told apart by object identity. See concerns: a restored context is bound by agent name |
| test_tool_batch_failure_history.py | 32 | 8 | 0 | 24 | ported: `fresh_guarded_terminal_failure[tool_hook, agent_hook] x streaming` and `approved_failure_diagnostics_preserve_retry_boundary x4` (hooks now fail through `try_` twins; the old "Rust hooks cannot fail" reason was stale). Source fixes below. Remaining: extractor outliving cancellation, `asyncio.wait`/`gather` monkeypatching, opaque `__deepcopy__` identity, cyclic Python outputs, live `ToolOutputText` list/tuple type recovery |
| sandbox/test_compatibility_guards.py | 27 | 0 | 0 | 27 | `__all__` surfaces, dataclass and pydantic field order, positional constructor order |
| sandbox/test_docker.py | 23 | 3 | 0 | 20 | ported: credentialless Docker-volume S3 mount resume (serialized reattach, direct reattach, missing container keeps the volume identity) in `py_core15_r15b_docker_resume.rs`. Remaining: deferred-cleanup asyncio tasks, docker-py exec sockets and length-framed stdin, `READ_PATH_PROBE`, image-ref parsing, `exec_run(workdir=None)` until ready, PTY socket internals, `Mount` subclass trust list, cancellation of resume. The nested read-only grant write is `py_sandbox_b_runtime_helpers.rs` (the real helper script, exit 114) plus `py_sandbox_a_docker_2.rs` |
| sandbox/test_unix_local_user_file_io.py | 19 | 0 | 0 | 19 | every case executes the `sudo -u python3 -c <worker>` body in-process with mocked `os`/`subprocess`; Rust runs user operations as `exec` commands |
| test_source_compat_constructors.py | 17 | 0 | 0 | 17 | Python positional constructor order, dynamically typed `bool` check, legacy keyword aliases |
| extensions/memory/test_dapr_session.py | 14 | 0 | 0 | 14 | `close()`, client ownership, context manager (the Rust session is an HTTP client, nothing to close) |
| sandbox/test_manifest_config.py | 13 | 0 | 0 | 13 | dict-typed manifests coerced from untrusted config; Rust takes a typed `Manifest` |
| test_tool_approval_call_id_reuse.py | 10 | 4 | 0 | 6 | ported: bound call id with missing arguments fails before the error formatter; failed end hook, failed approved agent-tool start hook and failed parallel end hook do not re-execute (`py_core15_r15b_call_id_reuse.rs`). Remaining: asyncio cancelled formatter, `model_dump` signature probing x2, web-search schema drift (pydantic), and two divergences listed in the concerns (computer end-hook failure, streamed events before an `on_llm_end` failure) |
| extensions/memory/test_sqlalchemy_session.py | 9 | 0 | 0 | 9 | engine registry and gc, cross-event-loop engines, fake sessions patching SQLAlchemy internals, cancellation. Concurrent pops and concurrent first access are ported by earlier rounds (`py_memory_sqlx.rs`, `py_core8_r8g3_sqlx_first_access.rs`) |
| sandbox/test_workspace_payloads.py | 9 | 0 | 0 | 9 | adapters for Python file-like objects (`readinto`, text chunks, `Content-Length` attributes); Rust writes take bytes |
| test_asyncio_progress.py | 7 | 0 | 0 | 7 | asyncio task-graph inspection |
| sandbox/test_unix_local_file_io.py | 6 | 0 | 0 | 6 | streaming read handle (no Rust handle API), `os.open` swap injection, file-object copy failure, thread-pool cancellation, manager-private `_create_resources` |
| test_tool_name_collision_policy.py | 6 | 0 | 0 | 6 | asyncio cancellation inside an async formatter (the Rust formatter is synchronous), deep-copy snapshots across an `await` |
| models/test_openai_retry_helpers.py | 5 | 1 | 0 | 4 | ported: header names match case-insensitively (real divergence, fixed in `src/retry.rs`). Remaining: httpx/httpx2 header types, `response_headers` attribute, `__cause__` precedence, `unsafe_to_replay` attribute |
| tracing/test_import_side_effects.py | 5 | 0 | 0 | 5 | subprocess import checks, `atexit` registration |
| sandbox/test_session_sinks.py | 4 | 0 | 0 | 4 | test helper session, cancellation of a finish sink, best-effort task flush, and the callback's bound session argument (a Rust `CallbackSink` receives the event only; see concerns) |
| test_items_helpers.py | 4 | 2 | 0 | 2 | ported: shell output replay strips output-only fields at both levels without mutating; object-valued output replays unchanged. Remaining: pydantic null-refusal validation, `ValidatorIterator` copy |
| tracing/test_traces_impl.py | 4 | 0 | 0 | 4 | `ContextVar` token reset, `__enter__` token logging, a processor raising on trace end |
| models/test_litellm_extra_body.py | 3 | 0 | 0 | 3 | pydantic `__bool__` override, LiteLLM `_skip_mcp_handler` kwarg x2 |
| sandbox/capabilities/test_skills_capability.py | 3 | 0 | 0 | 3 | combining several skill sources or a directory artifact in one constructor call |
| test_computer_tool_lifecycle.py | 3 | 1 | 0 | 2 | ported: lifecycle hooks see the computer tool (`py_core15_r15b_computer_hooks.rs`). Remaining: a start hook that mutates `agent.tools`, provider with a `create` attribute |
| test_run_context_wrapper.py | 3 | 0 | 0 | 3 | private `_to_str_or_none`, raw-dict name and call-id fallbacks, falsy approval item |
| test_tool_origin.py | 3 | 0 | 3 | 0 | covered: `py_core_a_tool_origin.rs::{hosted_mcp_call_carries_the_mcp_origin, run_state_from_json_reads_legacy_snapshot_without_tool_origin, run_state_from_json_reads_legacy_approval_without_tool_origin}`; `py_core15_r15b_tool_origin_legacy.rs` adds the schema 1.5 and 1.6 restores from a real run |
| deferred_session_writes/test_compaction.py | 2 | 0 | 2 | 0 | covered: the entry-settle bookkeeping and recorded store are asserted end to end by `py_deferred_writes_compaction.rs::{a_partial_approval_repark_keeps_the_response_storage_setting, approval_settlement_*}` (first completion deferred with the parked response id and store) |
| extensions/experiemental/codex/test_payloads.py | 2 | 0 | 0 | 2 | mapping protocol and `KeyError` on dataclass items (`as_dict` is ported) |
| models/test_model_payload_iterators.py | 2 | 0 | 0 | 2 | single-use Python iterators materialized by the OpenAI SDK |
| sandbox/test_docker_removal_worker.py | 2 | 0 | 0 | 2 | `ExitStack.callback` patching, `KeyboardInterrupt` in a forked child |
| test_agents_logging.py | 2 | 0 | 0 | 2 | `logging.Handler` objects (replaced stdout, thread-safe handler creation) |
| test_runner_tracing_disabled.py | 2 | 0 | 0 | 2 | `DefaultTraceProvider` subclass with a `ContextVar` token |
| models/test_anthropic_thinking_blocks.py | 1 | 0 | 0 | 1 | positional bool arguments of a Python converter |
| realtime/test_playback_tracker.py | 1 | 0 | 0 | 1 | patches `time.monotonic` of two modules |
| sandbox/capabilities/test_compaction_capability.py | 1 | 0 | 0 | 1 | dynamically typed policy argument |
| test_agent_hooks.py | 1 | 0 | 0 | 1 | falsy hooks object (`__bool__`) |
| test_handoff_failure_error_function.py | 1 | 1 | 0 | 0 | ported: a failing handoff input filter propagates and the failure formatter is not consulted (`py_core15_r15b_handoff_filter.rs`; the old "a Rust filter cannot fail" note was stale, `try_handoff_input_filter` exists). The asyncio cancel case is the `CancelledError` group |
| test_tool_output_conversion.py | 1 | 0 | 0 | 1 | `__str__` override on a user type |
| tracing/test_tracing_env_disable.py | 1 | 0 | 0 | 1 | falsy span object |

## Rust source fixed

- `src/run_internal/mod.rs`
  - `finish_with_output`: the final-output end hooks (`on_agent_end`, `AgentHooks::on_end`) now run before the output guardrails, as Python's `run_final_output_hooks` runs inside the turn step. Before, a guardrail ran first and an `on_agent_end` failure could not keep a guarded terminal tool output private. On an end-hook failure with output guardrails and a terminal tool output the whole turn stays out of history and of `run_data` and the held tool events are dropped; otherwise the tool-output events are released before the error (the `final_hook` boundary expects them).
  - `publish_completed_tools`: a turn resumed from approvals is recognised by the resumed interruptions, not only by approval items still in the turn, so a failed resumed turn keeps its accepted outputs for the retry under a guardrail gate.
  - `run_details`: while a turn is in flight and the output guardrails could still reject its output, the tool outputs it committed are not part of the error's public `new_items` (the checkpoint keeps them).
- `src/retry.rs`: `Retry-After`, `Retry-After-Ms` and `X-Should-Retry` are matched case-insensitively (Python's `header_lookup`).
- `docs/project-changelog.md`: two entries (hook and guardrail order, header case).

## New test files

`tests/py_core15_r15b_run_step.rs` (1), `py_core15_r15b_batch_failures.rs` (2 tests, 8 cases), `py_core15_r15b_docker_resume.rs` (3), `py_core15_r15b_call_id_reuse.rs` (4), `py_core15_r15b_retry_headers.rs` (1), `py_core15_r15b_items.rs` (2), `py_core15_r15b_computer_hooks.rs` (1), `py_core15_r15b_tool_origin_legacy.rs` (2), `py_core15_r15b_handoff_filter.rs` (1).

## Breaking changes

Behavioural, not API: end hooks run before output guardrails (a hook that relied on a guardrail having approved the output now sees it first, which is Python's order), and `RunErrorData.new_items` of a failed resumed guarded turn omits uncommitted-to-public tool outputs.

## Concerns and unresolved questions

1. A restored `RunContext` is bound to an agent by name. Python also refuses to apply a converted context's decision to a distinct same-named agent whose tool of the same name is a different capability (`converted_context_does_not_authorize_same_named_distinct_capability`, `tied_duplicate_owner_restore_requires_reapproval`). Rust would apply it. This is a design difference, not fixed here; r9gaps already added `name#2` keys for snapshots.
2. Computer tool: a computer call whose end hook failed is run again when the same call id comes back on a fresh run that shares the context; Python refuses with "already executed" (`computer_hook_failure_does_not_repeat_side_effect`). The same happens for a function tool whose end hook failed. Rust commits the output before the hook failure (matching Python's function-tool history), so the "uncommitted" ledger does not flag the replay. Not fixed; needs the ledger to distinguish "committed to history" from "completed".
3. Streamed runs emit `ToolCalled` only after `on_llm_end`; Python emits it before (`streamed_validated_items_precede_llm_end_hook_failure`). Fixing it means processing the response before the hook, which changes what a failed `on_llm_end` checkpoints.
4. `CallbackSink` receives only the event; Python passes the bound inner session as a second argument (`callback_sink_receives_bound_inner_session`). Adding it changes the `EventSink` trait, so it was left.
5. Case counts are per Python test function as the work list states them; a few Python parametrizations expand differently in the ledger, so a "ported" count can include cases the ledger already held as ported under another name.

Gates: the nine new binaries pass with `--all-features` and default features; a regression run of 192 related binaries (hooks, guardrails, tool errors, approvals, batch failures, deferred writes, streaming) was 974 passed, 0 failed before the last small edits; clippy and rustfmt results are in the final message.
