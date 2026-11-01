# Core12 r12b: session stores, MCP errors, codex tool, agent-as-tool and runner python-only audit

Crate 0.14.0. Scope: the 35 ledger files with `python_only > 0` under `tests/memory/`, `tests/extensions/memory/`, `tests/mcp/`, `tests/extensions/experiemental/codex/`, `tests/test_agent_as_tool*.py` and `tests/test_agent_runner*.py` (553 cases on the ledger). Ledger counts are the starting point; they were not recounted. Result: 31 cases ported, 0 moved to covered, 522 remain python-only. Per-file numbers: `py-test-parity-core12-r12b.csv` (before - ported - covered = remaining on every row).

## Method and honesty note

Every one of the 35 files was opened and its Python tests listed against the Rust tests (by name and by distinctive literals). The bodies of the cases that looked portable or that earlier audits flagged (r11b's concentration list: session stores and compaction, MCP errors, codex tool, agent-as-tool) were read in full and decided: port, an existing Rust test covers it, or one valid reason.

Read in full: all of `test_agent_as_tool.py` python-only cases, the unmatched cases of `test_agent_runner.py` and `test_agent_runner_streamed.py`, `test_client_session_retries.py`, `test_list_page_limits.py`, `test_mcp_approval.py` (invalid policies), `test_server_errors.py` (head, timeouts, cleanup tail), the manager tests that touch snapshots, one-shot iterables, timeouts and the first lifecycle group, `test_session.py` (sqlite rollback/close group), `test_openai_conversations_session.py`, `test_session_persistence_sanitize.py`, the encrypt wrapper and delegation tests, the compaction restore/generation cases, and the codex cases whose names suggest behaviour (tool name prefix, duplicate names, persisted-thread mismatch, run-context key mode, span budget, thread run failures).

Group-level reason carried over, bodies NOT read case by case this round (status of the report reflects it): the exception-graph and `caplog` cases of `mcp/test_server_errors.py` after the first 130 lines and the cleanup tail, the worker-task, lifecycle-lock and cancellation cases of `mcp/test_mcp_server_manager.py`, `mcp/test_mcp_v2_http.py` (httpx2 internals), `test_streamable_http_client_factory.py`, `test_streamable_http_session_id.py`, `test_mcp_util.py` (13), `extensions/memory/test_redis_session.py` pool/WATCH/release internals, `test_advanced_sqlite_session.py` interleaving group, `test_async_sqlite_session.py`, `test_dapr_session.py`, `test_sqlalchemy_session.py`, `test_mongodb_session.py` (matched by content against `py_memory_mongodb*.rs`, `py_core8_r8g3_mongodb_failures.rs`: batch claim, legacy documents, partial cleanup and generation are already tested), `test_codex_exec_thread.py` (matched against the exec/thread unit tests), `test_agent_runner_sync.py`, `test_session_helpers.py`, `test_session_context_wrapper.py`.

The ledger counts are lower than the number of cases name-matching found already ported or covered under other names, so "covered" is 0 here by design: I only moved cases tied to this round's work.

## Per-file result

| file | before | ported | covered | after | what remains and why |
|---|--:|--:|--:|--:|---|
| test_agent_as_tool.py | 24 | 3 | 0 | 21 | falsey extractor / falsey builder / invalid builder output (Python truthiness, dynamic type, 3); global run registry and scope ids with monkeypatched `Runner.run` (scope for nested ToolContext / wrapper, namespace, ambiguous-identity guard x4, deferred legacy rebuild, legacy hosted sticky x2); `BaseException` from a handler, MCP `CancelledError`, stream-close failure injection x4, sync-vs-async handler |
| test_agent_as_tool_callback_validation.py | 1 | 0 | 0 | 1 | legacy positional `run_config` signature |
| test_agent_runner.py | 32 | 12 | 0 | 20 | hash/equality-impostor dict keys and objects (4; the pydantic `GenericCaller(type="direct")` half of the non-direct-caller case is typed validation), blocked-snapshot helpers with injected cancellation/`KeyboardInterrupt`/`ValueError` and hand-built run states (4), falsy runner / falsy agent / positional `previous_response_id` (3), typed input filters (3), pydantic `model_dump` (3), callback list identity (replace in place, extra rebuilt copy, include-history flag only a server-managed run sets: 3), `execute_approved_tools` with hand-built approvals (5) |
| test_agent_runner_streamed.py | 23 | 4 | 0 | 19 | typed input filters (2), monkeypatched finalizer, private queue / `run_loop_task` / `_stored_exception`, asyncio `CancelledError` from tools and sessions (cancelled sibling x2, session-save cancel / `BaseException`), `agent.tools` mutated from a tool |
| test_agent_runner_sync.py | 5 | 0 | 0 | 5 | asyncio default-loop reuse, creation, replacement, interrupt, async-generator finalisation |
| mcp/test_client_session_retries.py | 16 | 7 | 0 | 9 | non-object `arguments` (typed), anyio mixed exception groups, outer asyncio cancellation x2, serialization of overlapping requests (single-connection pool), non-numeric backoff cap (typed) |
| mcp/test_list_page_limits.py | 7 | 0 | 0 | 7 | integer-only page limit (typed), asyncio cancellation x2, stdio/SSE twins of tests that run on the shared `Core` over HTTP |
| mcp/test_mcp_approval.py | 3 | 0 | 0 | 3 | typo policies; `RequireApproval` is an enum |
| mcp/test_mcp_auth_params.py | 1 | 0 | 0 | 1 | patches `sse_client` |
| mcp/test_mcp_imports.py | 2 | 0 | 0 | 2 | Python import machinery |
| mcp/test_mcp_server_manager.py | 42 | 0 | 0 | 42 | worker tasks and anyio task affinity (Rust connects and cleans up on the caller's task by design, see the `MCPServerManager` doc), lifecycle lock and overlapping cleanup (`&mut self` cannot overlap), `CancelledError`/`BaseException` policy, one-shot iterables, task factory, dynamically typed timeouts |
| mcp/test_mcp_server_manager_cleanup_state.py | 1 | 0 | 0 | 1 | `CancelledError` out of cleanup |
| mcp/test_mcp_util.py | 13 | 0 | 0 | 13 | inner/outer asyncio cancellation x9, legacy call without an agent, live FastMCP |
| mcp/test_mcp_v2_http.py | 12 | 0 | 0 | 12 | httpx2 / MCP v2 client internals (factory, response hook, v1 auth, cache flag), protocol negotiation of the Python v2 server, OAuth sub-request mapping, connect/outer cancellation |
| mcp/test_server_errors.py | 84 | 0 | 0 | 84 | exception-graph shape (`__cause__`, `__context__`, attached httpx request/response, traceback locals, `ExceptionGroup`, redirect history), `caplog` record fields of cleanup, mutated `client_session_timeout_seconds`, monkeypatched private retry/cleanup methods. The user-visible half (message text and credential redaction per operation) is ported by `py_mcp_server_errors.rs` and `py_core8_r8g4_mcp_errors.rs` |
| mcp/test_streamable_http_client_factory.py | 4 | 0 | 0 | 4 | httpx client factories (ssl cert, proxy, retry logic), type annotation |
| mcp/test_streamable_http_session_id.py | 4 | 0 | 0 | 4 | injected `_get_session_id` callbacks; lifecycle is `py_mcp_http_transport::session_id_lifecycle` |
| memory/test_compaction_suffix.py | 2 | 0 | 0 | 2 | cancelled suffix transaction (asyncio) |
| memory/test_openai_conversations_session.py | 12 | 3 | 0 | 9 | settable `session_id` and cancel-settle x4, SDK retry layer (`max_retries`, `with_options`) x3, falsy default client |
| memory/test_openai_responses_compaction_session.py | 12 | 0 | 0 | 12 | asyncio cancel x7, generation counter, duck-typed compaction awareness x3, pydantic warnings kwarg |
| memory/test_session.py | 11 | 0 | 0 | 11 | (the new clear-rollback test strengthens the already-ported write-lock case `py_memory_session::a_failed_clear_session_leaves_the_write_lock_free`, so the count stays) per-thread connection tracking and `close()`, cancelled awaited mutation, rollback-failure eviction and quarantined close (injected connection subclasses), post-commit cancellation x3, dict coercion of settings x2 |
| memory/test_session_context_wrapper.py | 5 | 0 | 0 | 5 | `inspect.signature` opt-in |
| memory/test_session_helpers.py | 2 | 0 | 0 | 2 | repeated asyncio cancellation of `await_session_mutation` |
| memory/test_session_persistence_sanitize.py | 1 | 1 | 0 | 0 | non-object item passes through |
| extensions/memory/test_advanced_sqlite_session.py | 36 | 0 | 0 | 36 | injected `logger` and `caplog` fields, subclass fault injection, connection invalidation, `close()`, gated private workers and queued/async usage-capture interleavings, event-loop reuse, SQL spy of a private cleanup helper |
| extensions/memory/test_async_sqlite_session.py | 21 | 0 | 0 | 21 | `close()`, aiosqlite internals, cancellation |
| extensions/memory/test_dapr_session.py | 14 | 0 | 0 | 14 | client ownership, `close`, context manager |
| extensions/memory/test_encrypt_session.py | 15 | 1 | 0 | 14 | asyncio interleavings of compaction restore, overridden `pop_item` subclasses, negative scan budgets (unsigned), method delegation to arbitrary attributes |
| extensions/memory/test_memory_imports.py | 13 | 0 | 0 | 13 | optional-import machinery (cargo features) |
| extensions/memory/test_mongodb_session.py | 23 | 0 | 0 | 23 | non-string ids, `close`, vendor `driver_info`, asyncio cancellation |
| extensions/memory/test_redis_session.py | 52 | 0 | 0 | 52 | redis-py pool checkout/release/WATCH/cluster internals, client ownership and `close`, `_get_next_id` private API, cancellation |
| extensions/memory/test_sqlalchemy_session.py | 10 | 0 | 0 | 10 | engine registry, dialect fallbacks, cross-loop engines, cancellation |
| codex/test_codex_exec_thread.py | 22 | 0 | 0 | 22 | mapping/kwargs coercion, stdlib monkeypatching, fake asyncio processes and tasks, vendored target triples, env mutation |
| codex/test_codex_tool.py | 26 | 0 | 0 | 26 | run-context `setattr` mechanics on frozen/slots/pydantic contexts, pydantic parameter class, circular value, lazy-resolver identity, asyncio `BaseException`/cancellation. Span budget, resolver and persisted-thread checks have Rust unit tests (`span_budget.rs`, `tool.rs`) |
| codex/test_payloads.py | 2 | 0 | 0 | 2 | `_DictLike` mapping protocol |

## Rust source changes (divergences found by porting)

- `src/agent.rs`: the public `model_settings` field is now private (`pub(crate)`), read with `Agent::model_settings()` and written through `with_model_settings` / `model_settings_mut`, so every write is tracked as Python's in-place assignment is. No in-crate call changed.
- `src/agent.rs`, `src/run_context.rs`, `src/run_internal/mod.rs`, `src/run_internal/tools.rs`: an approved agent-as-tool resume now records the nested run's progress. While the nested run is in flight its own `RunCheckpoint` is reachable from the outer run's checkpoint (`RunContext::begin_nested_resume`), a run that is dropped (non-streamed abort) or cancelled leaves a checkpoint (`RunLoop` `Drop`), and the outer call that only resumes a nested run is not marked executed. A retry from the checkpoint resumes the nested run without running the committed tool again. Port of `record_agent_tool_resume_state`.
- `src/run_internal/mod.rs`: a run whose future is dropped while its final output guardrails run writes the turn to the session from a detached task (Python's cancelled-guardrail path); an immediate cancel of a streamed run still writes nothing.
- `src/run_internal/mod.rs`: a retained blocked function call is rebuilt from allowlisted fields (`blocked_tool_call_payload`); before, a provider's extra payload on the call (`provider_data`) reached the session and the checkpoint.
- `src/memory/openai_conversations_session.rs`: reasoning items with neither a server id nor encrypted content (the SDK placeholder id counts as no id) are not written to a Conversations store; new public `start_openai_conversations_session` (re-exported from `memory`).
- `src/memory/openai_responses_compaction_session.rs`: the clear -> add -> restore replacement of a store without an atomic replace runs on its own task (`await_session_mutation`), so a caller dropped between the clear and the add no longer leaves the store empty; the restore after a failed add still runs. Port of the cancellation cases (`restores_history_when_replacement_add_is_cancelled`); the end state after a dropped caller is the completed replacement, where Python's cancelled add ends in the restored history.
- `src/memory/sqlite_session.rs`: `clear_session` is one transaction; a failure on the sessions row now restores the messages deleted first (it auto-committed them).
- `src/memory/encrypted_session.rs`, `src/memory/mod.rs`: `EncryptedSession` forwards the run context to every operation of the wrapped store (`*_with_context`), new trait method `Session::pop_item_validated_with_context` (defaulted; `SqliteSession` keeps its atomic validated pop).
- `src/mcp/mod.rs`: the retry budget of `list_tools` is shared by all pages (a failing second page after a retried first page now fails, as in Python, instead of retrying again); `backoff_delay` extracted with unit tests for the doubling and capped schedules.

## Breaking changes

- `Agent::model_settings` is no longer a public field. Break list: `tests/py_core2_runconfig_settings.rs` (3 reads, 1 write), `tests/py_models_map_defaults.rs` (2 writes, 1 read), `tests/py_core10_r10b_model_settings.rs` (1 write; the test now asserts the tracked behaviour: a write keeps the defaults captured at construction), `tests/py_sandbox_a_runtime_1.rs`, `tests/py_sandbox_a_runtime_4.rs` (reads through `SandboxAgent::map_agent`), `tests/py_sandbox_b_runtime.rs` (read of a prepared agent). Everything else already used `with_model_settings` / `model_settings_mut`.
- A run whose future is dropped now leaves its checkpoint in a given `RunCheckpoint` (before only a failed run did); the nested-resume exemption means an outer agent-tool call that only resumes a nested run is no longer refused as "already executed".
- `Session` gains a defaulted method; code implementing `Session` is unaffected.

## New tests

`tests/py_core12_r12b_agent_tool.rs` (4: resume survives cancellation, streamed and not; nested hosted MCP sticky decisions end to end; wrapped hosted MCP exact rejection), `_runner_blocked.rs` (4), `_runner_session.rs` (7), `_streamed_cancel.rs` (4), `_mcp.rs` (3), `_memory.rs` (1), `_conversations_start.rs` (1), `_encrypt_context.rs` (1), `_compaction_cancel.rs` (1); lib tests `mcp::backoff_tests` (4).

## Gates

- New and touched binaries pass with `--all-features` and with default features: `py_core12_r12b_*` (9 files, the mcp/sqlite/encrypt ones are feature gated and compile empty by default), `py_core10_r10b_model_settings`, `py_core2_runconfig_settings`, `py_models_map_defaults`, `py_sandbox_a_runtime_1`, `py_sandbox_a_runtime_4`, `py_sandbox_b_runtime`; lib tests 671 (all features) and 215 (default).
- Regression, `--all-features --no-fail-fast`, in batches of 90 with a `cargo clean -p` between: 396 binaries (`py_core*`, `runner*`, `core_*`, `py_mcp*`, `py_memory*`, `sessions*`, `py_deferred*`, `py_codex*`, `codex*`, `py_realtime_session*`, `py_sandbox_a_runtime_[1-7]`, `function_tool*`, `local_tools*`), 0 failures. The full workspace suite was not run.
- `cargo check --workspace --all-targets --keep-going` with all features: no error (the public `model_settings` break is fully absorbed). `cargo clippy --all-features --lib` plus my test targets with `-D warnings`: clean; the same without features: clean. `cargo clippy --all-features --all-targets -D warnings` stops in `tests/py_core12_r12a_vercel.rs` (unused imports), r12a's in-flight file, not mine. rustfmt applied to every file I touched (only my hunks differed).

## Unresolved questions

- `EncryptedSession::add_scoped` with a run context calls `add_items_with_context` directly, without `await_session_mutation` (the context is borrowed from the caller, so the write cannot move to a spawned task); the no-context path keeps the shield. A dropped run on a context-scoped store can leave a partial batch.
- `guardrail_persist` also fires when a streamed result is dropped mid-guardrail and its task is aborted (not only for an aborted non-streamed future); the tests pin the non-streamed abort and the `CancelMode::Immediate` no-write case, not the dropped-stream case.

- `Session::pop_item_validated_with_context` was added so an encrypted session keeps its atomic authenticated pop under a scoped store; confirm the extra trait method is acceptable.
- The detached write after a dropped guardrail run cannot report its own failure; Python's `finally` can raise it. Left as is.
- Not re-proved line by line (see method): the exception-graph, redis-py internals and advanced-SQLite interleaving groups.

Status: DONE_WITH_CONCERNS
Summary: 35 files, 553 ledger cases: 31 ported (agent-as-tool cancel-resume x2, wrapped hosted MCP decision, blocked-output sanitisation 5, conversation-store reasoning/sanitize/stream 7, cancelled-guardrail/recovery-save/omit-policy 5, MCP retry-budget/dropped-connection/backoff 7, conversations start 3, encrypt context 1), 0 moved to covered, 522 remain python-only. Nine source divergences fixed; `Agent::model_settings` is now private.
Concerns: bodies of the exception-graph, redis-py internals, worker-task, httpx2, advanced-SQLite interleaving, aiosqlite, dapr, sqlalchemy and codex exec groups were not re-proved case by case (listed under Method); the full workspace suite was not run (396 related binaries and the lib suites were); the detached write after a dropped guardrail run and the context-scoped encrypted append are not drop-safe in the way Python's shielded paths are.
