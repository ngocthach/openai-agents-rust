# Core20 r20b: python-only re-audit (memory, MCP, models, tracing)

35 files, 96 python-only cases before (ledger counts). 9 moved to ported, 6 moved to covered,
81 remain python-only. Every case body named below was read in the Python source; where a
file has more uncited names than its ledger count (the ledger counts were reconciled after
earlier rounds), all uncited names were read and classified, and the CSV keeps the ledger
arithmetic (`before - ported - covered = remaining`).

## Source change (compaction session: a dropped caller's restore keeps the mutation lock)

`src/memory/openai_responses_compaction_session.rs`. The clear/add/restore of a whole-history
replacement runs detached (`await_session_mutation`), but the session's `mutation_lock` guard
belonged to the caller's future. When the caller was dropped (Python: cancelled) during the
restore, the guard was released at once, so a newer `add_items`/`clear_session`/`pop_item`
through the compaction session ran concurrently with the restore: the restore's clear could
wipe the newer write or the write could land before the restored history. Python's
`_mutation_lock` stays held across the shielded restore
(`test_cancel_restore_waits_for_mutation_lock_before_newer_writes`). Reproduced first
(`a_newer_write_waits_for_the_restore_of_a_dropped_caller` failed: the write finished while the
restore was blocked).

Fix: a second lock, `detached_mutation: Arc<Mutex<()>>`. `replace_history` takes an owned guard
of it and moves it into the detached task, which drops it when the replacement/restore settles.
Every mutation entry point (`run_compaction`, `add_items`, `pop_item`, `clear_session`,
`on_run_completed`) now goes through `lock_mutations()`: `mutation_lock`, then wait for
`detached_mutation`. Lock order is always mutation_lock -> detached_mutation, so no deadlock.
No public API change.

## New tests

- `tests/py_core20_r20b_compaction_cancel.rs` (no feature, 3): restore finishes after its caller
  is dropped; a newer write waits for the restore of a dropped caller (fails without the fix);
  an exception restore drains and releases the lock.
- `tests/py_core20_r20b_suffix_cancel.rs` (`sqlite`, 1): a run dropped while its suffix
  transaction waits on another connection's write lock still commits the replacement, and a
  newer append through the compaction session waits and lands after the summary
  (`original[:-1] + [summary, newer]`). Passed without a source change.
- `tests/py_core20_r20b_mcp.rs` (`mcp`, 5): listing dropped mid-pagination caches nothing
  (tools and prompts, 2 requests each); connection closed during `tools/call` is retried on an
  isolated session (second `initialize`); caller-supplied client with every parameter (client
  default header, `Authorization`, `terminate_on_close=false` sends no DELETE); caller client
  keeps its proxy (absolute-form request lines reach the proxy); Python-equal param defaults.

## Per file

| file | before | ported | covered | remaining | classification per case/group |
|---|---|---|---|---|---|
| deferred_session_writes/test_pending_write.py | 3 | 0 | 2 | 1 | covered: `test_zero_count_final_save_arms_recovery_even_when_deduplicated` = `py_deferred_writes_b.rs::a_failed_settle_is_recovered_when_the_append_fails_atomically` (guardrail agent, failing append, batch reconciled); `test_the_settled_count_matches_what_the_append_actually_wrote` = `py_memory_deferred_writes.rs::resuming_persists_the_parked_call_and_its_output_once` + `py_deferred_writes_e.rs::the_settled_batch_and_the_resolved_turn_land_as_one_ordered_write`. Remaining: `test_settled_held_items_count_toward_the_turn_persisted_count` drives private `save_resumed_turn_items` with a hand-built `RunState` (`_current_step`, empty items); no Rust counterpart function and not reproduced end to end (caveat) |
| extensions/memory/test_memory_imports.py | 13 | 0 | 0 | 13 | both parametrized tests inject a `MetaPathFinder` and check optional-extra ImportError text (Python packaging/import machinery; Rust uses cargo features) |
| mcp/test_client_session_retries.py | 1 | 1 | 0 | 0 | ported `test_streamable_http_retries_closed_resource_on_isolated_session` (server closes the connection). Also read, valid beyond the ledger count: mixed `BaseExceptionGroup` (anyio), outer cancellation during isolated retry (asyncio), non-object `arguments` and non-numeric backoff cap (typed API) |
| mcp/test_list_page_limits.py | 4 | 2 | 0 | 2 | ported `test_cancellation_with_page_budget_preserves_cancellation` x2. Remaining `test_page_limit_requires_an_integer` (`True`, `1.5`): `Option<usize>` |
| mcp/test_mcp_approval.py | 3 | 0 | 0 | 3 | `test_mcp_require_approval_rejects_invalid_fail_open_policies` x3: typo string, typo in mapping, tool in both always and never; `RequireApproval` is an enum and `PerTool` a `HashMap<String,bool>`, none representable |
| mcp/test_mcp_imports.py | 2 | 0 | 0 | 2 | import machinery (`importlib.reload`, broken finder) |
| mcp/test_streamable_http_client_factory.py | 3 | 2 | 0 | 1 | ported: `test_all_parameters_with_custom_factory` (with `test_custom_httpx_client_factory`) and `test_custom_httpx_client_factory_with_proxy`; the ssl-cert/retry-logic factories and default kwargs only assert forwarding, which the same tests and `streamable_http_params_default_like_python` show. Remaining: `test_httpx_client_factory_type_annotation` (typing introspection) |
| mcp/test_streamable_http_session_id.py | 1 | 0 | 0 | 1 | injected `_get_session_id` callbacks (`MagicMock`, changing callback) on a private attribute; the lifecycle is `py_mcp_http_transport.rs::session_id_lifecycle` |
| memory/test_compaction_suffix.py | 2 | 1 | 0 | 1 | ported the `SQLiteSession` variant. Remaining: the `AsyncSQLiteSession` variant (Rust: `SqlxSession`). Python gates `_insert_items` inside the transaction by monkeypatch; the external write-lock gate cannot stand in, because sqlx's deferred transaction fails at once with `database is locked` (code 5) instead of waiting (verified). See concerns |
| memory/test_openai_responses_compaction_session.py | 5 | 3 | 1 | 1 | (ledger count 5; 10 names audited) ported: cancelled again during restore, cancel restore waits for the mutation lock (source fix), exception restore drains when cancelled. Covered: `..._clear_is_cancelled_after_mutation` = `py_memory_responses_compaction_b.rs::history_is_restored_when_clear_fails_after_mutating` (clear mutates then fails: one clear, one add, history back). Remaining/valid: falsy (`__bool__`) default client and decision hook, pydantic `model_dump(warnings=False)`, `is_compaction_aware_session` duck-typing x3 on `MagicMock(spec=Session)` |
| memory/test_session_context_wrapper.py | 5 | 0 | 0 | 5 | `inspect.signature` opt-in rules: legacy `**kwargs`, uninspectable method, partial opt-in, named-parameter rule, all-operations rule; Rust opts in per method by overriding `*_with_context` |
| memory/test_session_helpers.py | 2 | 0 | 2 | 0 | `test_repeated_cancellation_waits_for_mutation` [ok, fail] = `py_memory_limits_helpers.rs::a_session_mutation_runs_to_completion_after_its_caller_is_dropped` (the mutation settles after the caller goes away; the repeated-cancel message is asyncio-only) |
| models/test_agent_registration.py | 3 | 0 | 0 | 3 | dictionary registration config normalisation x2, unknown dict field TypeError (typed harness id); private `_agent_registration` attribute lookup read too |
| models/test_anthropic_thinking_blocks.py | 1 | 0 | 0 | 1 | positional bool arguments of `Converter.items_to_messages` (Python signature) |
| models/test_kwargs_functionality.py | 1 | 0 | 0 | 1 | dictionary `model_settings` variant (the `ModelSettings` variant is ported) |
| models/test_litellm_chatcompletions_stream.py | 3 | 0 | 0 | 3 | provider stream `aclose` failure ignored after/at terminal event, propagated before it, slow close, in-flight close after cancel (asyncio `aclose`; a Rust drop cannot fail or be awaited) |
| models/test_litellm_extra_body.py | 3 | 0 | 0 | 3 | `Reasoning.__bool__` override; `_skip_mcp_handler` x2 is a kwarg of the litellm library call, not sent over HTTP to a proxy |
| models/test_litellm_logging_patch.py | 1 | 0 | 0 | 1 | reloads and patches litellm's own logging module |
| models/test_map.py | 2 | 0 | 0 | 2 | falsy provider (`__bool__`), `HybridCancellation` (CancelledError+Exception); invalid prefix strings (typed enum) and first-failure re-raise (infallible close) read too |
| models/test_model_payload_iterators.py | 2 | 0 | 0 | 2 | single-use Python iterables materialised by the OpenAI SDK (monkeypatched converters) |
| models/test_openai_chatcompletions.py | 4 | 0 | 0 | 4 | falsy `Reasoning`; `httpx2.Timeout` provider-native extra arg; `client.with_options(max_retries=0)` x2 (the Rust client performs no provider-managed retries: `src/extensions/models/mod.rs` docs); dictionary agent settings read too |
| models/test_openai_chatcompletions_stream.py | 4 | 0 | 0 | 4 | dictionary agent settings; in-flight `aclose` after cancel; private `Part`, `_StreamOutputLayout`, `_finish_reasoning_summary_part` helpers (no public path reaches them) |
| models/test_openai_responses_converter.py | 2 | 0 | 1 | 1 | covered: `test_web_search_filters_preserve_openai_forward_compatible_fields` = `py_models_responses_converter.rs::web_search_filters_preserve_forward_compatible_fields` (filters are a `Value`); dict/class filters = `::web_search_filters_preserve_the_provider_payload`. Remaining: positional constructor arguments of `WebSearchTool` |
| models/test_responses_websocket_session.py | 3 | 0 | 0 | 3 | dictionary `run_config` x2; `run`/`run_streamed` injection via monkeypatched `Runner` (Rust callers pass `ws.run_config()`; API difference, no wrapper methods) |
| test_runner_tracing_disabled.py | 2 | 0 | 0 | 2 | `DefaultTraceProvider.create_trace` subclass hook with a `ContextVar` token (Rust `TraceProvider` has no create hook); the opt-out end state is ported in `py_core4_tracerunner_disabled.rs` |
| test_tracing_errors.py | 3 | 0 | 0 | 3 | exception whose `__str__` raises (run, streamed), `BaseException` from `__str__`, failing custom span read/write: Rust `Display` and span recording are infallible |
| test_tracing_provider_safe_debug.py | 1 | 0 | 0 | 1 | Python logging handler with a closed stream |
| tracing/test_import_side_effects.py | 5 | 0 | 0 | 5 | subprocess import checks (no httpx/sqlite3, lazy export, no bootstrap at import) and `atexit` bootstrap counting with module globals reset |
| tracing/test_processor_api_key.py | 1 | 0 | 0 | 1 | replaces `os.environ` with an object that sets the key during lookup; Rust's lookup writes nothing when the env has no key (`processors.rs` resolve), so the race cannot drop a set key, but the env read cannot be hooked |
| tracing/test_reasoning_export.py | 1 | 0 | 0 | 1 | tuple/UserList/UserDict message containers |
| tracing/test_setup.py | 1 | 0 | 0 | 1 | monkeypatched `atexit.register` / provider class bootstrap |
| tracing/test_spans_impl.py | 3 | 0 | 0 | 3 | explicit finish from another context raises `ValueError` (ContextVar token) per span factory; processor raising in `on_span_end` during generator close (Rust processors cannot fail) |
| tracing/test_trace_context.py | 2 | 0 | 0 | 2 | falsy response/trace objects (`__bool__`), falsy current trace warning with monkeypatched provider |
| tracing/test_traces_impl.py | 3 | 0 | 0 | 3 | foreign ContextVar token, `NoOpTrace` double `__enter__` logging, processor raising on trace end |
| tracing/test_tracing_env_disable.py | 1 | 0 | 0 | 1 | falsy (`__bool__`) span subclass as parent |

Pending fallible-callback API: none of these cases depends on fallible `is_enabled` /
`needs_approval`.

## Gates

- `cargo clippy --all-features --lib` + the 3 new test targets `-D warnings`: clean.
- `cargo clippy --lib` (default features) + the 3 new test targets `-D warnings`: clean.
- New binaries, all features: compaction_cancel 3/3, mcp 5/5, suffix_cancel 1/1. Default
  features: compaction_cancel 3/3, the two gated files compile to 0 tests.
- Lib unit tests `openai_responses_compaction` (all features): 10/10.
- Related, all features: py_memory_compaction 8, py_memory_responses_compaction_a 9,
  py_memory_responses_compaction_b 10, py_core6_mcpmisc_compaction_session 9,
  py_core6_final7_compaction 23, py_core6_mcpmisc_compaction_suffix 11,
  py_deferred_writes_compaction 8, py_mcp_pagination, py_mcp_http_transport,
  py_core7_rtvoicemcp_mcp_transport, py_core16_r16c_mcp_redirect, py_memory_limits_helpers,
  py_core8_r8g4_suffix_sqlx, py_memory_deferred_writes: all pass.
- rustfmt (edition 2024) applied to the three new files; the edited source file passes
  `--check`.

## Read-depth caveats

Most case bodies were read with a small extractor that stops at the first dedented line. That
can cut off multi-line strings or parametrize blocks. These files were read directly instead:
`test_import_side_effects.py`, `test_memory_imports.py`, `test_mcp_imports.py`,
`test_streamable_http_session_id.py`, and `test_streamable_http_client_factory.py`
(lines 1-270 and 370-530).

For `test_client_session_retries`, `test_streamable_http_client_factory`,
`test_openai_responses_compaction_session`, `test_openai_chatcompletions_stream`,
`test_litellm_chatcompletions_stream` and `test_map`, there are more uncited names than the
ledger count. The CSV row puts the ledger count onto the groups named here. Which names make up
the ledger's count is an assumption.

## Concerns

- `SqlxSession::replace_compaction_suffix_pruning` runs its transaction in the caller's future,
  so a run dropped mid-replacement rolls the suffix replacement back (consistent history, no
  summary), while Python's `AsyncSQLiteSession` settles it. Not changed: there is no Rust gate
  to test it, and detaching would be an unverified behaviour change. Also observed: a concurrent
  writer holding the write lock makes the sqlx (deferred) suffix transaction fail at once with
  `database is locked` instead of waiting, where `SqliteSession` (IMMEDIATE) waits its busy
  timeout.
- `test_settled_held_items_count_toward_the_turn_persisted_count` stays python-only without an
  end-to-end proof that the approval-only settle path cannot re-append on a later resume.

## Suggested changelog paragraph

> `OpenAIResponsesCompactionSession`: when the caller of a compaction is dropped while the
> replacement or its restore is still running, later `add_items`, `pop_item`, `clear_session`
> and compactions through the session now wait for that replacement to settle, so a newer write
> can no longer be wiped by the restore or land before the restored history.

## Unresolved questions

- Should `SqlxSession`'s suffix replacement be detached so a dropped run still commits it, as
  Python's async SQLite store does?

Status: DONE_WITH_CONCERNS
