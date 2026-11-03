# Python test parity, round 17, r17b (function-tool schema, MCP errors and manager, Redis session)

5 files, 212 python-only cases before (ledger counts). Every test body of every file was read (about 9,500 lines of Python). Result: 7 ported, 2 covered, 203 remain python-only. Two Rust source gaps were fixed on the way (NumPy variadic names in doc comments, Redis TTL horizon).

## Per-file table

| file | before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| tests/test_function_tool_decorator.py | 50 | 0 | 0 | 50 | see "decorator" below |
| tests/test_function_schema.py | 43 | 5 | 1 | 37 | see "schema" below |
| tests/mcp/test_server_errors.py | 39 | 0 | 0 | 39 | see "server errors" below |
| tests/mcp/test_mcp_server_manager.py | 36 | 0 | 0 | 36 | see "manager" below |
| tests/extensions/memory/test_redis_session.py | 44 | 2 | 1 | 41 | see "redis" below |
| total | 212 | 7 | 2 | 203 | |

Earlier reports record python-only counts per file, not per test name, so the exact membership of each "before" set is not recoverable. The moves below are the cases that no earlier report mentions as ported and that I verified were not asserted anywhere in Rust (by reading the Rust tests, not only by name). Where a ported case may overlap an earlier "moved" count it is said so.

## Ported (7), source fixes

- `tests/py_core17_r17b_function_tool_docs.rs` (3): `test_variadic_param_descriptions_preserved` (google, numpy, sphinx). Google and Sphinx already passed; the NumPy form (`*numbers : float`) did not: the doc-comment parser kept the star in the name, so the description never attached to the parameter. Fixed in `macros/src/lib.rs` (`parse_numpy_docs` strips the leading `*`/`**`).
- `tests/py_core17_r17b_constrained_arguments.rs` (1): the `ordinary` rows of `test_annotated_constraints_reject_arguments_before_tool_execution`, end to end through `Runner` with `without_failure_error_function()`: a violating argument fails the run with `ModelBehaviorError` "Invalid JSON input for tool" before the body runs, the valid argument runs it once and `StopOnFirstTool` returns it. Passed without a source change. `py_core15_r15d_tool_constraints.rs` pins the schema-range half through a direct invoke and counted one case for this test, so one of the two `ordinary` rows may be a double count.
- `tests/py_core17_r17b_function_tool_methods.rs` (1): the context rows of `test_class_based_functions` (a `&self` method and an associated function taking `ToolContext` first), next to `py_core16_r16gaps_function_tool_methods.rs` for the plain rows. Passed without a source change.
- `tests/py_core17_r17b_redis_server_clock.rs` (2):
  - `test_add_items_uses_server_absolute_expiration_after_serialization`: the largest TTL whose absolute expiration (server `TIME` plus the TTL) fits an int64 is accepted, and `TIME` is read before the write script. Together with `a_ttl_past_the_server_clock_horizon_is_rejected_before_writing` and `a_compaction_replacement_checks_the_ttl_horizon_before_writing` (not separate Python cases: the second Python rejection test `test_add_items_rejects_unrepresentable_ttl_before_writing` was already ported by `py_memory_redis.rs`). Source fix in `src/memory/redis_session.rs`: Rust only checked that `ttl * 1000` fits an int64, so a TTL between `(i64::MAX - now) / 1000` and `i64::MAX / 1000` passed that check and then failed inside the Lua script at `PEXPIRE`, after the history was already appended (a retry duplicates the history, the exact hazard the Python test names). The three writing paths (add, suffix replacement, whole-history replacement) now read the server clock first and refuse with "ttl is outside Redis's supported expiration range" before sending any script. The mocks in `py_memory_redis.rs` and `py_core8_r8g3_redis_compaction.rs` answer `TIME`.
  - `test_compaction_cancel_after_redis_commit_invalidates_cached_history`: the mock applies the replacement script and holds its reply, the compaction caller is aborted, the reply is released, and the next forced compaction sends the committed history (not the candidates cached before the replacement). Passed without a source change.

## Covered (2)

- `test_schema_with_mapping_raises_strict_mode_error`: `py_core8_r8g6_agent_basics.rs::a_mapping_parameter_cannot_be_made_strict` and `py_core3_functiontools_schema.rs::typed_tool_with_open_map_argument_is_rejected_by_try_new`.
- `test_cancelled_redis_compaction_handoff_remains_resumable`: `py_core9_r9h1_resume_cancel.rs::handoff_resume_after_cancelled_compaction_commits`, which cancels a run whose handoff compaction committed and resumes it; the logic is backend-agnostic (the Redis twin only changes where the replacement lands, and `py_core17_r17b_redis_server_clock.rs` covers the Redis commit-then-cancel half).

## Extra test that moves no count

`tests/py_core17_r17b_mcp_request_failures.rs`: `test_prompt_and_resource_request_errors_hide_url_credentials` for a connection dropped mid-request, for all five prompt/resource operations (earlier ports covered `list_prompts` and `read_resource` only, plus a 503 variant), with tool-data redaction on and off, asserting the exact `Failed to <operation> on MCP server '<name>': Request failed.` text, no URL secret, and no `openai.agents` log record (Python's `caplog` assertion). Passed without a source change. The earlier reports counted these Python cases as covered by the 503 test, so no count moves.

## Remaining python-only, per file

### decorator (50 of 64 collected cases; 14 were already ported)

Read in full. Every remaining case is Python callable or object introspection:

- `@function_tool` on callable objects and instances: async, slotted, `__call__` docstring and class docstring merging, annotated class state ignored, configured form, positional context, `use_docstring_info=False` with a hostile `__doc__`, unknown `__call__` descriptor, class-scoped and inherited-module annotation resolution, invalid generated class names requiring `name_override` (`Café`, 65 characters) (about 15). Rust tools are explicit `FunctionTool` values or macro-generated from a fn item.
- `test_unsupported_callable_shapes_require_explicit_wrappers` (23 shapes: `partial`, `partialmethod`, `staticmethod`, `classmethod`, decorated `__call__`, `update_wrapper`, published `__annotations__`/`__annotate__`, custom/method `__signature__`, local annotations, `singledispatchmethod`, builtins, nested wrappers, keyword-only/variadic/non-first context, generics, `Self`, pydantic generics, PEP 695). `inspect`/typing introspection of Python callables. The three context-position shapes are compile-time in Rust (the macro only treats a first `ToolContext` parameter as context; elsewhere the type must be `Deserialize + JsonSchema`).
- `__wrapped__` exposure and removal (`test_tool_exposes_original_callable_without_mutating_it`, descriptor hidden on subclasses, configured tool, identity for supported shapes, clone paths with `copy`/`deepcopy`/`dataclasses.replace`, `inspect.unwrap` chain, non-decorator tools, replacing the invoker) (8): Python attribute/`functools` machinery.
- A sync function or callable object returning an un-awaited coroutine is passed through (2): dynamic typing.
- `test_callable_object_invokes_the_resolved_call_method` (monkeypatch of a class `__call__`).
- `test_image_generation_tool_parameter_schema` (pydantic model as a tool parameter; the Rust `ImageGenerationToolConfig` keeps `quality` a string so `xhigh`/`max` need no schema).
- `test_function_tool_timeout_arguments_are_keyword_only` (`inspect.signature`).

### schema (43 before, 37 after)

Read in full. Remaining:

- pydantic `BaseModel` attribute shadowing (`model_extra`, `model_fields_set` parameter names) (2).
- `*args` / `**kwargs` signatures (22): `test_varargs_function`, variadic tuple annotations (accept, reject fixed-length, supported alternatives), `**kwargs` dict/scalar, variadic `Field` constraints (schema, per-string, collection defaults, homogeneous tuple, fixed-tuple bypass), `to_call_args` kwargs-key collisions with named, context, positional-only and var-positional parameters. A Rust function has no variadics.
- the `positional` and `keyword` rows of `test_annotated_constraints_reject_arguments_before_tool_execution` (4, variadics).
- `test_default_equality_is_not_used_for_sentinel_comparison` (2, Python default objects with hostile `__eq__`/`__bool__`).
- `AfterValidator` ordering (2), `Annotated`/`Field` description and default precedence across several `Field`s, the docstring and a default `Field` (2): pydantic metadata. The Rust macro has one rule, recorded as an open choice in `py_core3_functiontools_decorator_schema.rs`: an explicit `#[arg(desc = ...)]` beats the doc comment, where Python lets the docstring beat `Annotated` metadata. Not changed (see questions).
- `test_run_context_in_non_first_position_raises_value_error` (compile-time in Rust, see decorator).
- Remaining collected cases are pydantic `Field(...)`/`Annotated` constraint, description-merge and docstring-style variants that earlier rounds ported or that restate a covered Rust behaviour; the case membership cannot be reconstructed per name (listed groups sum to more than 37 for that reason).

### server errors (121 collected, 39 remaining)

Read in full. The groups that have no Rust analogue, with collected case counts (the 39 are drawn from them; earlier rounds counted the other cases of these groups, and all message/lifecycle halves with a Rust analogue, as ported or covered):

- exception-graph shape: `__cause__`/`__context__`, traceback locals, `__notes__`, `BaseExceptionGroup` nesting and mixed groups with cancellation (about 20 cases): Rust errors are flat strings built from the sanitized name; the "no secret in the text" half is asserted in `py_mcp_server_errors.rs`, `py_core8_r8g4_mcp_errors.rs`, `py_core13_r13b_mcp_errors.rs`, `py_core16_r16d_mcp_secrets.rs`.
- raw `httpx` exception propagation for safe URLs (`..._only_map_credentialed_urls`, 10 safe-URL cases): Rust always reduces a transport failure to text.
- `cleanup()` mapped errors and `caplog` records with `exc_info`/`exc_text` (about 38 cases): Rust `cleanup()` never raises or logs a transport failure (it swallows a failing session delete and resets the server, `py_core13_r13b_mcp_errors.rs::cleanup_ends_the_session_once_and_leaves_the_server_reconnectable`, `py_core16_r16d_mcp_secrets.rs::cleanup_swallows_a_failing_session_delete_and_resets_the_server`), so there is nothing to scrub.
- `client_session_timeout_seconds` validation (9): `True`/`"5"` are dynamic types; negative, NaN, infinity, `1e20` and `10**400` cannot be a `Duration`; `5e-7` is rejected by Python and applied by Rust (a `Duration` keeps nanoseconds, `py_core13_r13b_mcp_errors.rs::a_sub_microsecond_client_session_timeout_still_applies`). Re-validation of an attribute mutated after construction (2): server config is fixed after construction.
- asyncio cancellation of paginated lists and of `cleanup` (3).
- list with a tool filter and no context (`run_context and agent are required`, 1): `list_tools()` has no filter parameter.
- `_error_name` private attribute (2): the sanitization itself is `get_mcp_server_log_name` (covered); URL-derived default names do not exist because the Rust constructor takes the name.

### manager (88 collected, 36 remaining)

Read in full. Groups with no Rust analogue (collected counts; the 36 are drawn from them):

- `_ServerWorker` and worker-task tests (about 26): worker task cancellation, queued commands settled on cancel, emergency cleanup bounded by the cleanup timeout, retained/discarded workers after a cancelled caller, lifecycle lock waiting with the `suppress_cancelled_error` policy, serialized overlapping `cleanup_all`/`reconnect`, task factory, cross-task cleanup. Rust connects and cleans up in place (tokio has no same-task requirement), `cleanup_all`/`reconnect` take `&mut self` so they cannot overlap, and there is no `suppress_cancelled_error` option (dropping the future is Rust's cancellation; `py_core13_r13b_mcp_manager.rs` and `py_core16_r16c_mcp_manager.rs` cover what a dropped connect or a hung cleanup leaves behind).
- `CancelledError` / `BaseException` propagation from `connect` (5): asyncio/`BaseException`.
- lifecycle timeout validation (12 of 14): `True`/`"1"` dynamic types; NaN, infinity, negative and `10**400` cannot be a `Duration`; zero is covered (`py_core8_r8g10_mcp_timeout.rs`).
- one-shot iterables (3) and returned lists not aliasing manager state (2): `Vec` arguments and results.

### redis (86 collected, 41 remaining)

Read in full. Remaining groups:

- redis-py connection pool, pipeline and `WATCH` internals (about 30): checkout/release observability for Redis 8.0/8.1, blocking and ordinary pool cancellation, detached/quarantined connections, release listeners, post-commit reset/close/disconnect failures, `WATCH` retry timestamps, response-callback failures, single-connection pool with `WATCH`. Rust writes are one Lua script on a multiplexed connection; there is no pool, pipeline or `WATCH`. The retry-hazard half (a write whose outcome is unknown is never retried) is `py_core13_r13b_redis.rs::a_write_whose_outcome_is_unknown_is_not_retried`.
- `close()` terminal state (closed operations raise, idempotent and concurrent close, failed/cancelled cleanup retried, ping after close, an operation queued behind close) (8): Rust has no `close()`; the connection a session opened is released on drop and use after drop does not compile.
- cancellation shielding of a destructive mutation (`pop`/`clear` wait for the authoritative outcome) (2): asyncio.
- `decode_responses` clients (fakeredis and live) and live-Redis `from_url`/connectivity (4): the Rust connection returns bytes through the `redis` crate; the live variants need a server.
- `clear` re-checking key types after a concurrent replacement (1): a Lua script cannot interleave; the collision half is `py_memory_redis.rs`.
- `_get_next_id` (1): private helper, unused by `add_items`.
- `session_settings={"limit": 5}` dictionary form (1): typed.
- subclass overrides of `add_items`/`clear_session` (cluster `custom_writes`, compaction subclass) (2): a Rust session is a concrete type; the read half is `py_core14_r14a_redis_cluster.rs`.
- `_atomic_replace_items` with an unserializable item (1): an `InputItem` is JSON by construction.

## Source bugs fixed

1. `macros/src/lib.rs` `parse_numpy_docs`: a NumPy `*name` / `**name` entry kept its stars, so its description never reached the parameter.
2. `src/memory/redis_session.rs`: out-of-range absolute TTL expiration rejected before any write (new `ensure_expiration_in_range`, one extra `TIME` round trip per TTL write, as Python's `time()` inside the transaction).

## Breaking changes

None to the public API. A session with a TTL now issues a `TIME` command before each write (add, suffix replacement, whole-history replacement); a TTL whose absolute expiration does not fit an int64 now fails with the same user error before any script runs instead of failing inside the script after the history was appended.

## Unresolved questions

1. `#[arg(desc = ...)]` beats the doc comment in the macro; Python's docstring beats `Annotated` metadata (`test_function_prefers_docstring_descriptions_over_annotated_metadata`, `test_annotated_description_overrides_field_default_description`). Recorded earlier as a design choice; left unchanged.
2. The server-clock check reads `TIME` in a separate round trip, so a TTL exactly at the int64 boundary can still pass the check and fail in Redis if the server clock moves past it in the few milliseconds between the two commands. Python reads the time inside the `WATCH`ed transaction and sends an absolute `PEXPIREAT`, which closes that window; doing the same here needs the three Lua scripts and their mocks to take an absolute expiration.
3. `MCPServerManager::set_connect_timeout(Some(0))` is accepted and rejected at the next connect, where Python raises at assignment and keeps the previous value.

## Gates

- `cargo test --all-features` on the five new test binaries and on `function_tool_macro_tests`, `py_core3_functiontools_decorator_schema`, `py_core8_r8g6_docstrings`, `py_core16_r16gaps_function_tool_methods`, `py_core16_r16g_function_schema`, `py_core5_leftover3_function_schema`, `py_memory_redis`, `py_core8_r8g3_redis_compaction`, `py_core13_r13b_redis`, `py_core14_r14a_redis_cluster` and the macros crate unit test: all pass.
- clippy `-D warnings`: lib plus my test binaries and the two modified mock files, both `--all-features` and default features, clean. A workspace-wide `--all-targets` run currently stops at another agent's in-flight `tests/py_core17_r17a_run_state.rs:149` (E0308), which is not mine.
- rustfmt applied to my files, `src/memory/redis_session.rs` and the two edited mocks; `rustfmt --check` passes on `macros/src/lib.rs` (my edit is format-clean; that crate root was not reformatted).

Status: DONE_WITH_CONCERNS
Summary: 212 python-only cases across 5 files: 7 ported, 2 covered, 203 remain python-only (CSV rows balance: 50, 37, 39, 36, 41). Two source fixes: NumPy `*name` doc entries in the `#[function_tool]` macro, and the Redis session now refuses a TTL whose absolute expiration (server `TIME` plus TTL) does not fit an int64 before any script runs (behaviour change, one extra `TIME` round trip per TTL write, no API change).
Concerns:
- Read depth: every Python test body of the five files was read. The Redis range 2415-3299 (the pool/pipeline internals) was first skimmed through docstrings and asserts and then read in full before this report was finalised; it holds up as redis-py pool, pipeline, release-listener and detached-connection monkeypatching with no Rust analogue.
- The per-name membership of each "before" set is not recoverable from the earlier reports (they give counts per file). The moves are therefore verified against the Rust tests but the python-only residue is described by group, and the group sizes exceed the remaining counts for `test_server_errors.py`, `test_mcp_server_manager.py` and `test_function_schema.py`.
- One ported `ordinary` row of the annotated-constraint test may overlap the case `py_core15_r15d_tool_constraints.rs` already counted.
- The TTL horizon check reads `TIME` in its own round trip, so a TTL exactly at the int64 boundary can still fail inside Redis if the clock moves between the two commands (question 2).
- The workspace-wide `clippy --all-targets` gate is blocked by another agent's in-flight `tests/py_core17_r17a_run_state.rs:149` (E0308), not by these files; clippy `-D warnings` on the lib plus my test binaries is clean on both feature sets.
