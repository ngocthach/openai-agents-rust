# py-test-parity core6: mcpmisc (MCP, memory, deferred writes, utils, gap files)

Scope: the slice assigned to area `mcpmisc` (see the CSV `py-test-parity-core6-mcpmisc.csv`): every
`mcp/` and `memory/` file whose ledger row had `not_ported > 0` or a total that did not equal
`ported + covered + python_only + not_ported`, `mcp/test_mcp_resume_binding.py`, and the three files that
had no ledger classification (`docs/test_translate_docs.py`, `fastapi/test_streaming_context.py`,
`sandbox/test_tar_workspace.py`). Python ground truth was obtained by running the Python suite
(venv in the scratchpad; `pytest --collect-only` gives the exact per-case counts, and the target
tests were run against the Python SDK when a Rust divergence needed a ruling).

## How counts were made

`pytest --collect-only` per file (parametrization expanded; counts equal the ledger `total`).
Each case was then given one bucket by reading the Python test and the Rust test it maps to.
Where a parametrized Python test has a Rust twin that loops the same parameters, each parameter
is counted as ported; where Python parametrizes over something Rust has no equivalent for
(`AsyncSQLiteSession`, the in-memory SQLAlchemy engine, `connect_in_parallel` worker tasks) the
twin cases are python-only with that reason.

## Per-file table

| file | cases | ported | covered | python-only | not ported |
|---|---:|---:|---:|---:|---:|
| mcp/test_client_session_retries.py | 40 | 15 | 3 | 22 | 0 |
| mcp/test_mcp_approval.py | 13 | 9 | 1 | 3 | 0 |
| mcp/test_mcp_resume_binding.py | 61 | 61 | 0 | 0 | 0 |
| mcp/test_mcp_server_manager.py | 88 | 36 | 2 | 50 | 0 |
| mcp/test_mcp_util.py | 135 | 108 | 11 | 16 | 0 |
| mcp/test_message_handler.py | 4 | 0 | 1 | 3 | 0 |
| mcp/test_runner_calls_mcp.py | 24 | 24 | 0 | 0 | 0 |
| mcp/test_server_errors.py | 121 | 22 | 0 | 99 | 0 |
| memory/test_compaction_model_visibility.py | 81 | 81 | 0 | 0 | 0 |
| memory/test_compaction_suffix.py | 46 | 22 | 0 | 24 | 0 |
| memory/test_openai_conversations_session.py | 74 | 33 | 24 | 17 | 0 |
| memory/test_openai_responses_compaction_session.py | 94 | 36 | 35 | 16 | 7 |
| memory/test_session_context_wrapper.py | 15 | 0 | 0 | 15 | 0 |
| memory/test_session_helpers.py | 6 | 3 | 0 | 3 | 0 |
| memory/test_session_limit.py | 15 | 15 | 0 | 0 | 0 |
| memory/test_sqlalchemy_compaction.py | 30 | 12 | 0 | 18 | 0 |
| docs/test_translate_docs.py | 9 | 0 | 0 | 9 | 0 |
| fastapi/test_streaming_context.py | 1 | 0 | 0 | 1 | 0 |
| sandbox/test_tar_workspace.py | 6 | 6 | 0 | 0 | 0 |
| **subtotal** | **863** | **483** | **77** | **296** | **7** |
| resume-agent cases in `test_run_impl_resume_paths.py` (compaction exchange on the checkpoint; retrieval window x2) | 3 | 3 | 0 | 0 | 0 |
| **total** | **866** | **486** | **77** | **296** | **7** |

The ledger counted these 19 files at 863 cases (6,416 ported overall); this area started with
`not_ported` 138 and `unaccounted` 203 in them plus the 16 unclassified gap cases.

## What was ported (new test files)

| file | covers |
|---|---|
| `tests/py_core6_mcpmisc_resume_binding.rs` | all 15 defs / 61 cases of `test_mcp_resume_binding.py` (streaming x restore x listing-change matrices, legacy snapshots 1.14/1.17/1.18/1.19, handoff and local-replacement guards, rejected calls, origin on rejection output) |
| `tests/py_core6_mcpmisc_compaction_visibility.rs` | all 19 defs / 81 cases of `test_compaction_model_visibility.py`; the encrypted/expiry cases let real TTLs lapse instead of moving a mocked clock |
| `tests/py_core6_mcpmisc_compaction_suffix.rs` | the SQLite variant of every suffix case (22) |
| `tests/py_core6_mcpmisc_sqlx_compaction.rs` | `outer=True, reject_insert=True` of the SQLAlchemy atomic-cleanup case |
| `tests/py_core6_mcpmisc_conversations_resume.rs` | the 13 resume-replay cases and the program-item-id case of `test_openai_conversations_session.py` |
| `tests/py_core6_mcpmisc_compaction_session.rs` | `created_by` replay (3), automatic rollback budget (2), pop/clear chain invalidation (4), add/clear waiting for an in-flight compaction (2), orphaned assistant ids (2), forced-marker retention (1) |
| `tests/py_core6_mcpmisc_session_limit.rs` | `test_session_limit.py` for `run`, `run_sync` and `run_streamed` (15); exact input equality for the orphan-output case |
| `tests/py_core6_mcpmisc_mcp_util.rs` | `mcp_annotations` (2 + runner hooks 2), guardrail isolation across servers, disabled-handoff names when prefixing, approval policy that sees the run context |
| `tests/py_core6_mcpmisc_mcp_static_meta.rs` | static `_meta` passed, and merged with the resolver |
| `tests/py_core6_mcpmisc_mcp_retry_setup.rs` | isolated-session setup failure spends retry budget |
| `tests/py_core6_mcpmisc_manager.rs` | lifecycle timeouts in force at call time for the parallel path; strict parallel connect cleanup (2) |
| `tests/py_core6_mcpmisc_tar_workspace.rs` | `shell_tar_exclude_args` (3) and the nested-path archive case (3; runs only with GNU tar, as Python's `skipif` does, so it is skipped on this macOS host) |
| `tests/py_core6_mcpmisc_compaction.rs` | the 3 resume-agent cases: `json_compaction_retry_does_not_promote_filtered_session_history`, `compaction_reload_preserves_session_retrieval_window` (input, auto) |
| `tests/py_core6_mcpmisc_support/mod.rs` | shared fixtures (mock `responses/compact` endpoint, run helpers, session stacks, filters) |

Un-ignored (all 11 `#[ignore]`s in `py_mcp_*.rs` / `py_memory_*.rs`; the "41 ignored" in earlier reports no
longer existed): `py_mcp_resume_binding.rs` (2), `py_mcp_util_schemas.rs` (5), `py_mcp_caching_approval.rs` (3),
`py_memory_conversations_remaining.rs` (1). All pass after the source fixes below.

## Rust bugs fixed

1. **MCP recipient binding was never recorded** (`src/tool.rs`, `src/mcp/util.rs`, `src/run_internal/mod.rs`,
   `src/run_state.rs`, `src/run_state/wire.rs`). A resumed call was matched on the server name only, so a second
   server with the same log name, or a different raw tool behind a re-hashed public name, could run an approved call.
   `FunctionTool` now carries `(server log name, raw tool name, server index)`; the processed-response snapshot
   persists it as `mcp_tool_bindings` (schema 1.18 shape `[name, tool, index]`); resume refuses a missing or different
   binding ("missing or different recipient binding"), an approved MCP call can never become a handoff, a rejected one
   still answers the model with the recipient it was approved for, and the answer keeps its MCP origin even when a local
   tool of the same name appeared.
2. **Strict MCP schemas converted open objects** (`src/strict_schema.rs`, `src/mcp/util.rs`): added Python's
   `_reject_open_objects` mode, so MCP tool schemas that permit undeclared properties (and unvalidated `$ref`s) are served
   as-is and non-strict instead of being silently narrowed.
3. **Tools-list cache refresh ordering** (`src/mcp/mod.rs`): invalidation during an in-flight refresh was lost and the
   last-finished refresh won. Added Python's dirty flag / generation / refresh sequence; `invalidate_tools_cache` keeps the
   last list visible through `cached_tools` but never serves or validates against it.
4. **MCP requests were serialized per server** (`src/mcp/mod.rs`): every request held the connection lock for its whole
   duration, so one slow call queued all others (and the concurrent-refresh cases could not even run). Requests now clone
   the session peer and run concurrently.
5. **`mcp_annotations` accessor** (`src/tool.rs`, `src/mcp/util.rs`): annotations are captured at conversion.
6. **MCP approval policy could not see the run context** (`src/mcp/mod.rs`, `src/mcp/util.rs`): new
   `RequireApproval::DynamicWithContext` / `dynamic_with_context`, evaluated per call with the run's `RunContext`.
7. **Compaction hook saw the wrong history on the automatic path** (`src/memory/openai_responses_compaction_session.rs`):
   the first hook view read everything; it now reads the store's default (limited) view and the exact replaceable window
   is read and re-approved afterwards, as Python does. Stores without an atomic suffix replace now also check tool/approval
   groups and re-approve the bounded snapshot (a pending approval no longer gets compacted away).
8. **Model-exchange matching** (`src/memory/mod.rs`, `src/memory/openai_responses_compaction_session.rs`,
   `src/run_internal/mod.rs`): item digests are key-order independent (`sort_keys`); a stored reasoning item that
   still has its id now matches a run that omitted reasoning ids and is compacted in the form the model saw
   (`SessionRunCompletion::omit_reasoning_item_ids`); a store that strips ids on write can be wrapped
   (`Session::ignores_item_ids_for_matching`, separate from `manages_history_remotely`), and the compaction session forwards it.
9. **A run resumed from a `RunState` compacted automatically** (`src/run_internal/mod.rs`): such a run never read the
   store's history, so like Python it only defers compaction to the next run.
10. **Input persistence order** (`src/run_internal/mod.rs`): the caller's input is stored before the first model call
    (Python's early `save_result_to_session(session, input, [])`), so a failing model or tool call still leaves it in the session.
    This also changes the Conversations request split for long inputs (`[20, 20, 1, 1]`, not `[20, 20, 2]`).
11. **Failed or partial settle of a withheld batch was repeated blindly** (`src/run_internal/mod.rs`,
    `src/run_internal/session_write.rs`): a held batch that settles on a resumed turn now becomes an ordinary recoverable
    write, and a multi-request Conversations append that landed only partly is detected and refused
    ("Cannot reconcile the pending Session write") instead of being replayed.
12. **Compaction owed by a resumed append** (`src/run_internal/session_write.rs`, `src/run_internal/mod.rs`): the recoverable pending write
    now carries `compaction_model_exchange` (`item_digests`, `reasoning_item_id_policy`); a resumed settle restores it, so the
    compaction is authorized by the failed attempt's exchange and no longer promotes hidden history. (Small edits in the resume agent's
    `session_write.rs`: record on write, restore on settle.)
13. **Compaction retrieval window** (`src/memory/openai_responses_compaction_session.rs`): manual calls and deferred-compaction hooks now
    read the store's default (`session_settings.limit`) window; the rollback budget is still checked on the complete history, a failed
    replacement still restores the full history, and the in-flight change check compares the same window.
14. **`shell_tar_exclude_args`** (`src/sandbox/session.rs`): public helper with Python's ordering and `./` anchoring
    (the hosted backends still build their own `--exclude` strings).

## Breaking changes

- `ProcessedResponseSnapshot::mcp_tool_bindings` is now `BTreeMap<String, (String, String, Option<usize>)>` (it was
  `BTreeMap<String, Vec<String>>` and was never populated for local MCP calls).
- `SessionRunCompletion` gained `omit_reasoning_item_ids: bool`; struct-literal constructors must add it or
  use `..Default::default()` (seven literals in `tests/sessions_tests.rs` were updated).
- `RequireApproval` gained the `DynamicWithContext` variant (exhaustive matches must handle it).
- `Session` gained `ignores_item_ids_for_matching` (defaulted).
- Early input persistence changed three older expectations, updated to Python's runner behaviour: `py_core2_runner_extra::the_session_is_written_once_per_turn` (input is its own first batch), `py_core2_runner_session_input::{the_latest_duplicate_output_is_saved_after_its_call, a_replayed_call_stays_before_its_output}`.

## Cross-area bugs deferred / not fixed

- None blocked. Other agents' in-flight edits in `src/run_internal/tools.rs` / `src/run_config.rs` briefly broke the build
  while this area was being finished; nothing of theirs was touched.

## Remaining not ported (7), all in `test_openai_responses_compaction_session.py`

- `test_rollback_does_not_revive_items_expired_during_compaction` (4): Python replaces its pre-request snapshot and relies on
  rollback; Rust re-reads the store after the request and refuses to replace a history that changed in flight (expiry
  included), so the replacement the test drives to failure is never attempted. The no-revival outcome holds by construction, but
  there is no test for it.
- `test_resumed_save_carries_generation_into_compaction`, `test_runner_skips_compaction_after_interleaved_model_wait`,
  `test_runner_skips_compaction_after_interleaved_persist` (3): Python tracks a mutation generation across a run's reads and
  appends; Rust's equivalent safety is the model-exchange match plus the atomic suffix replace, which is covered by the
  visibility and suffix ports, but the generation-specific interleavings were not written.

## Python-only groups (one reason each)

- `AsyncSQLiteSession` twin of every SQLite suffix case (23) and the in-memory SQLAlchemy engine twin (15): Rust has one
  async SQLite store and the pooled sqlx session cannot share an in-memory database.
- Cancellation of `asyncio` tasks (cancelled suffix transaction, cancelled restores, cancelled clear/pop, anyio cancel scopes,
  outer-cancellation preservation, worker-task cancel): Rust cancels by dropping futures.
- `MCPServerManager` worker tasks, lifecycle locks and `suppress_cancelled_error` (anyio same-task rule): the manager takes
  `&mut self`, so overlapping lifecycle calls cannot exist.
- Typed validation of dynamically typed timeouts and backoff caps (`bool`, `str`, NaN, inf, negative, zero): Rust takes
  `Option<Duration>` / `Option<f64>` (a negative or non-finite backoff cap clamps instead of raising; a zero lifecycle
  timeout is an immediate timeout, not a `ValueError`).
- Exception-graph shape of MCP failures (`ExceptionGroup`, `__cause__`, attached httpx request/response, log-record
  attachments, redirect history): Rust errors are strings; credential redaction itself is covered by the ported
  request/connect/cleanup tests.
- Context-aware sessions (`test_session_context_wrapper.py`, 15): Rust `Session` methods take no run-context wrapper and
  there is no signature-introspection opt-in; a scoped store captures its scope at construction.
- `docs/test_translate_docs.py` (9): tests the docs translation script, not the SDK.
- `fastapi/test_streaming_context.py` (1): FastAPI/ASGI streaming through a web framework; the event sequence itself is
  covered by the streaming-event tests.

## Gate status

- All 13 `py_core6_mcpmisc_*` binaries pass (150 tests); verified together with `py_core6_resume_*`, `py_memory_*compaction*`,
  `py_deferred_writes_compaction`, `sessions_tests`, `py_memory_remaining_{a,b}`, `py_memory_conversations_remaining`,
  `py_mcp_resume_binding`, `py_mcp_caching_approval`, `py_mcp_util_schemas`, `py_core2_runner_{extra,session_input}`: no failures.
- `cargo clippy --all-features --lib --test <13 mcpmisc binaries> -- -D warnings` is clean except `src/realtime/debug.rs`
  (`log_model_action_warning` dead code, another agent's in-flight edit; run with `-A dead_code`). The test files carry a file-level
  `allow` for fixture-style lints.
- The full 394-binary all-features suite was not run in one piece (disk budget); a first batched pass over the 269 session/RunState/MCP/
  compaction binaries found only the three expectations above (fixed) plus failures in other agents' in-flight binaries
  (`py_core6_nativetools_failure_history`, `py_core6_modalvercel_modal_pty_hydrate`, `py_sandbox_b_bounded_read`), not touched here.

## Notes for maintainers

- The tar archive case needs GNU tar; on macOS install `gnu-tar` as `gtar` to run it locally.
- The `shell_tar_exclude_args` unit cases compare Python's `shlex.quote` output only for the characters the tests use; `shell_quote`
  differs for `%` and for embedded single quotes (equivalent shell, different spelling).
