# Python test parity, round 16, list r16b (32 files, 317 python-only cases)

Method: for every file the Python test file was opened, its python-only cases were located by
matching Python test names against Rust `fn` names (the earlier reports name groups, not cases),
and the name-unmatched cases were read (body or head) and matched to Rust tests by content.
Where a user-visible behaviour existed, a Rust test was written and any divergence fixed.

Depth note (superseded by the continuation section): the `read_depth` column of the CSV says `full` only for the files where every
python-only body was read. The large files (`test_run_state.py` 144 name-unmatched, `realtime/test_session.py`
77, `test_openai_responses.py` 127, `test_advanced_sqlite_session.py` 101, `test_skills_capability.py` 48)
were read cluster by cluster: bodies of every cluster's representative and of every case that looked
portable, names for the rest. (Superseded: see the continuation section; all rows are now `full`.)

## Totals

| | before | ported | covered | remaining |
|---|---|---|---|---|
| 32 files | 317 | 22 | 8 | 287 |

Ported: 1 (`test_run_state.py`), 1 (`realtime/test_session.py`), 3 (`test_pretty_print_and_items.py`), 1 (`realtime/test_playback_tracker.py`,
three Rust tests), 1 (`sandbox/capabilities/test_filesystem_capability.py`). Covered: 7 in
`test_run_state.py` (statusless model response -> `py_core8_r8g7_run_state_misc.rs`, unknown agent / malformed item /
skipped session item remap / skipped ownership -> `py_core10_r10a_item_skips.rs`, the two unbindable pending
approval cases -> `py_core8_r8g7_run_state_binding.rs`).

## New tests

| file | cases |
|---|---|
| `tests/py_core16_r16b_items_state.rs` | null `text` of an output part reads as `""` (3 Python cases); a `function_call_output` in the original input keeps `name`/`status` through `RunState` (1) |
| `tests/py_core16_r16b_realtime_default_playback.rs` | the transport's own playback estimate never truncates past the received audio; an interrupt and a VAD speech start truncate at the same point; audio before any negotiated format is measured as 24 kHz PCM16 |
| `tests/py_core16_r16b_realtime_namespaced_call.rs` | a running namespaced realtime call keeps its identity: identical redelivery ignored, changed arguments under the same call id fail, tool runs once |
| `tests/py_core16_r16b_sync_tool.rs` | a sync `#[function_tool]` body does not stall a single-threaded runtime and runs on the blocking pool |
| `tests/py_sandbox_b_capabilities.rs` (one test appended) | `Filesystem::with_view_image_needs_approval` |

## Source fixes

1. `macros/src/lib.rs` and `src/lib.rs` (`__private::run_blocking`): a synchronous `#[function_tool]`
   function now runs on the tokio blocking pool (Python: `asyncio.to_thread`), inline when no runtime is
   current. Before, a blocking sync tool froze the whole worker. Behaviour change: the function must be
   `Send + 'static` (it already had to be for the future to be `Send`); panics propagate unchanged.
2. `src/sandbox/capabilities/filesystem.rs`: `Filesystem::with_view_image_needs_approval(bool)`. Python lets
   `configure_tools` set `view_image.needs_approval`; Rust could only require approval for `apply_patch`.
   Only the static form exists: a per-call predicate on `view_image` is not offered (the capability is not
   generic over the context type).

No public API was removed.

## Per-file decisions (remaining python-only reasons, one per group)

| file | before | ported | covered | remaining | reason |
|---|---|---|---|---|---|
| test_run_state.py | 96 | 0 | 8 | 88 | pydantic/dataclass context (de)serializers, `get_interruptions` snapshot hardening of pydantic subclasses (about 35), object-identity routing of detached approvals across nested states, hand-built `ProcessedResponse` tool-object resolution (Rust has no `last_processed_response`), falsy objects, gc/weakref |
| realtime/test_session.py | 32 | 10 | 0 | 22 | asyncio tracked-task / cleanup-task / `__aiter__` cancellation, traceback redaction, live tool/handoff objects in `initial_model_settings` (Rust settings carry JSON specs), private bookkeeping sets. All other name-unmatched cases have Rust counterparts (usage accumulation `py_realtime_session_core.rs::usage_events_accumulate_in_the_session_context`, error/turn events, update_agent, call-id changes in `py_realtime_session_dispatch.rs`, close races in `py_core8_r8g5_realtime_close.rs`) |
| models/test_openai_responses.py | 28 | 1 | 0 | 27 | monkeypatched `websockets.connect`, omit/NOT_GIVEN sentinels, legacy httpx Timeout/URL, httpx client default timeout, event-loop/lock/close-handshake internals. The 99 other name-unmatched cases are renamed ports in `py_models_responses_ws*.rs` / `py_models_responses_http.rs` / `py_models_responses_core.rs` |
| test_function_tool.py | 23 | 0 | 0 | 23 | `**kwargs` tools, `copy.copy` / `dataclasses.replace` / `deepcopy` of tools, pydantic variadic constraints, constructor type validation that Rust types rule out, monkeypatched resolution; the sync-in-thread pair was found portable and fixed (see above) but belonged to the ledger's other groups |
| sandbox/test_unix_local_user_file_io.py | 19 | 0 | 0 | 19 | every case runs the `sudo -u python3 -I -S -c <worker>` body in-process with mocked `os`/`subprocess`; Rust runs user operations as `exec` commands |
| extensions/memory/test_advanced_sqlite_session.py | 14 | 0 | 0 | 14 | `close()`, injected rollback/init failure with connection invalidation, asyncio cancel settlement, `sqlite3` authorizer. The cross-process and unlocked-database cases are ported (`py_core8_r8g3_sqlite_concurrency.rs`, `py_memory_advanced_sqlite_remaining.rs`) |
| extensions/memory/test_memory_imports.py | 13 | 0 | 0 | 13 | Python import-error messages for optional extras (packaging) |
| memory/test_session.py | 11 | 0 | 0 | 11 | per-thread sqlite connection reaping/quarantine, asyncio cancellation, dict coercion of settings (Rust `SessionSettings` is typed and `deny_unknown_fields`, `py_core7_c_memory_settings.rs`) |
| docs/test_translate_docs.py | 9 | 0 | 0 | 9 | a Python docs script (`docs/scripts/translate_docs.py`) |
| realtime/test_realtime_model_settings.py | 8 | 6 | 0 | 2 | live tool/handoff objects with `is_enabled`, asyncio sibling cancellation |
| test_agent_config.py | 7 | 0 | 0 | 7 | dynamically typed constructor validation; the dictionary model-settings cases are ported in `py_core5_leftover2_agent_config.rs` |
| test_agent_tool_state.py | 6 | 0 | 0 | 6 | module-level registry keyed by call object identity with weakref/gc |
| test_tool_use_behavior.py | 6 | 0 | 0 | 6 | private `check_for_final_output_from_tools`; sync/async/callable-object distinctions; the observable contract is in `py_core_a_tool_behavior.rs`, `py_core6_streamedleft_tool_behavior.rs` |
| voice/test_openai_model_provider.py | 5 | 0 | 0 | 5 | constructor with both a client and `api_key`/`base_url` (Rust builders layer options on the client), falsy default client, httpx2 |
| sandbox/test_client_options.py | 4 | 0 | 0 | 4 | pydantic type-discriminator registry |
| test_function_tool_approval_arguments.py | 4 | 0 | 0 | 4 | pydantic validators/aliases/plugins; the approval-before-validation contract is in `py_core2_scopedapprovals_fn_args.rs`, `py_core9_r9h3_validation.rs`, `py_core11_r11a_validator_rule.rs` |
| extensions/sandbox/test_daytona.py | 3 | 0 | 0 | 3 | package re-export/import checks, SDK timeout-type monkeypatch |
| models/test_agent_registration.py | 3 | 0 | 0 | 3 | dictionary normalization of a Python config object (Rust takes the harness id directly) |
| sandbox/capabilities/test_skills_capability.py | 3 | 0 | 0 | 3 | combining several skill sources or directory content in one constructor call (one Rust constructor per source) |
| test_doc_parsing.py | 3 | 0 | 0 | 3 | bound-receiver docstring cases and fixture calls (the style cases are in `py_core8_r8g6_docstrings.rs`) |
| test_run_state_agent_identity.py | 3 | 0 | 0 | 3 | legacy `.agent` attribute handoff shapes, capability identity by Python object config |
| utils/test_pretty_print_and_items.py | 3 | 3 | 0 | 0 | ported, see above |
| models/test_openai_responses_converter.py | 2 | 0 | 0 | 2 | positional-parameter order of a Python class, pydantic extra fields (typed Rust struct; filters ported in `py_models_responses_converter.rs`) |
| sandbox/test_session_state_roundtrip.py | 2 | 0 | 0 | 2 | pydantic subclass registry |
| test_pretty_print.py | 2 | 0 | 0 | 2 | pydantic/dataclass output kinds of the terminal-escape cases (text kind ported in `py_core_b_pretty_print.rs`) |
| voice/test_openai_stt_session_config.py | 2 | 0 | 0 | 2 | dataclass field order / positional subclassing |
| realtime/test_playback_tracker.py | 1 | 1 | 0 | 0 | ported, see above (the `time.monotonic` patch is replaced by real time with short audio) |
| sandbox/capabilities/test_filesystem_capability.py | 1 | 1 | 0 | 0 | ported `view_image` approval; the `configure_tools` callback itself has no Rust counterpart (typed builder instead) |
| test_agent_as_tool_callback_validation.py | 1 | 0 | 0 | 1 | legacy positional `run_config` argument |
| test_function_schema_logger_restore.py | 1 | 0 | 0 | 1 | griffe logger level (Python logging) |
| test_tool_output_conversion.py | 1 | 0 | 0 | 1 | falsy structured model (`__bool__`) |
| voice/test_openai_stt_client_config.py | 1 | 0 | 0 | 1 | openai omit/NOT_GIVEN sentinels |

## Round 16b continuation (coordinator follow-up)

All 32 files are now `read_depth=full` in the CSV. What that means here: for every file the head
(docstring plus first statements) of every name-unmatched Python case was read, and the full body of every
case that touched behaviour Rust could share; cases whose Rust counterpart was found by name or content are
cited. Cases that are pure Python object mechanics (pydantic hooks, `copy.copy`, weakref, asyncio tasks,
module registries) were classified from their heads.

Additional source changes:
- `RealtimeRunConfig::with_initial_tools` / `with_initial_handoffs` (`src/realtime/config.rs`,
  `src/realtime/session.rs`, `src/realtime/openai_model.rs`): live function tools and realtime handoffs that
  replace the agent's at connect time (Python's `initial_model_settings["tools"]` / `["handoffs"]`), filtered by
  enablement, advertised and dispatched, validated for duplicate names after the replacement, and also applied by
  `OpenAIRealtimeSIPModel::build_initial_session_payload` through its `run_config`. Additive only.
- `OpenAIClient::without_query` / `is_query_omitted` (`src/models/openai_client.rs`,
  `src/models/openai_responses_ws.rs`): remove a query parameter from the HTTP URL and the websocket handshake URL
  (Python's `omit` for `extra_query` / `default_query`); additive.

New tests: `tests/py_core16_r16b_realtime_initial_overrides.rs` (11), one test appended to
`tests/py_models_responses_ws.rs` (`omitted_query_removes_inherited_and_request_parameters`).

Counting note: `from_json_preserves_function_call_output_items` was already covered by
`py_core2_runstate_items.rs::from_json_preserves_function_output_original_input`, so it is counted as covered; my
duplicate test in `py_core16_r16b_items_state.rs` stays as an extra check. In `test_run_state.py` the processed-response
serialization, tool-approval item and `deserialize_items_*` families are covered by `py_core2_runstate_items.rs`; the
remaining cases need Python objects (pydantic `get_interruptions` hardening, hand-built `ProcessedResponse`, nested
state registry with gc, falsy objects).

## Unresolved questions

- Per-call predicates for `Filesystem` `view_image` approval were not added.

## Verification

- `cargo test --all-features`: `py_core16_r16b_items_state` (4), `py_core16_r16b_realtime_default_playback` (3),
  `py_sandbox_b_capabilities` (45) pass; `py_core16_r16b_sync_tool` (2) and `py_core16_r16b_realtime_namespaced_call` (1) pass; the other binaries that use `#[function_tool]` (`py_core5_leftover3_function_schema`, `function_tool_macro_tests`, `py_core16_r16g_function_schema`, `py_core3_functiontools_decorator_schema`, `py_core8_r8g6_docstrings`) and the macros crate tests pass after the macro change.
- clippy `-D warnings` with `--all-features` on the three first targets and default features on lib plus the
  two non-feature targets: clean.
