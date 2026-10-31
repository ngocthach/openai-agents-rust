# Python-only classification audit (40-file sample)

Scope: read-only audit of the `python_only` bucket in `plans/reports/py-test-parity-ledger.csv`. Sample: the 40 files in `po_sample.txt` (539 of the 4,001 python-only cases, 13.5%). For each file I read the Python test file and the owning report's python-only reasoning, then picked python-only cases by the report's named reason groups and judged each one. I grepped `tests/*.rs` and read the relevant `src/` code to see whether Rust already asserts the behaviour.

Method limits (read these before using the numbers):
- The ledger only holds per-file counts, so no row says which individual test is python-only. I attributed cases by named group. Where a case fits no named group (voice int16/cross-chunk, Vercel contract remainder) it is marked "not attributable" in the list below. I treat those as portable gaps either way, because Rust has no test for them.
- Cases were chosen to be spread across each file and to include suspicious ones. The sample is not random, so the portable rate is biased upward.
- "Portable" means the stated reason does not hold: Rust could assert the same outcome. It splits into **uncovered** (no Rust test, a real gap) and **covered** (the reason is mislabelled but a Rust test already asserts the outcome).

## Per-file table

| file | cases checked | valid python-only | actually portable (uncovered / covered) |
|---|--:|--:|--:|
| tests/test_decisions_examples.py | 8 | 8 | 0 |
| tests/realtime/test_session.py | 8 | 7 | 1 (0 / 1) |
| tests/test_released_api_contract.py | 8 | 8 | 0 |
| tests/extensions/test_tool_output_trimmer.py | 5 | 4 | 1 (0 / 1) |
| tests/mcp/test_prompt_server.py | 8 | 8 | 0 |
| tests/voice/test_pipeline.py | 8 | 5 | 3 (2 / 1) |
| tests/memory/test_sqlalchemy_compaction.py | 8 | 0 | 8 (0 / 8, in-memory backend twins) |
| tests/test_image_generation_tool_reference.py | 1 | 1 | 0 |
| tests/mcp/test_mcp_auth_params.py | 4 | 1 | 3 (2 / 1) |
| tests/tracing/test_processor_api_key.py | 1 | 1 | 0 |
| tests/sandbox/test_manifest_config.py | 8 | 8 | 0 |
| tests/extensions/sandbox/test_daytona.py | 6 | 6 | 0 |
| tests/memory/test_session.py | 8 | 7 | 1 (0 / 1) |
| tests/test_runner_model_provider_lifecycle.py | 1 | 1 | 0 |
| tests/test_run_serial_tests.py | 6 | 6 | 0 |
| tests/mcp/test_mcp_version_compat.py | 1 | 0 | 1 (1 / 0) |
| tests/sandbox/test_session_utils.py | 2 | 2 | 0 |
| tests/memory/test_session_helpers.py | 3 | 2 | 1 (0 / 1) |
| tests/test_run_state_compatibility_corpus.py | 8 | 5 | 3 (2 / 1) |
| tests/mcp/test_mcp_imports.py | 2 | 2 | 0 |
| tests/models/test_openai_responses_converter.py | 2 | 2 | 0 |
| tests/sandbox/test_pty_output.py | 1 | 1 | 0 |
| tests/mcp/test_mcp_server_manager.py | 8 | 8 | 0 |
| tests/test_repository_workflow_interfaces.py | 8 | 8 | 0 |
| tests/fastapi/test_streaming_context.py | 1 | 0 | 1 (1 / 0) |
| tests/sandbox/test_mounts.py | 2 | 2 | 0 |
| tests/extensions/sandbox/test_vercel_provider_contract.py | 8 | 5 | 3 (3 / 0) |
| tests/realtime/test_app_server_debug.py | 6 | 4 | 2 (2 / 0) |
| tests/test_agents_logging.py | 3 | 3 | 0 |
| tests/realtime/test_openai_realtime_conversions.py | 2 | 1 | 1 (1 / 0) |
| tests/voice/test_pipeline_tracing.py | 1 | 1 | 0 |
| tests/models/test_openai_chatcompletions.py | 3 | 3 | 0 |
| tests/test_code_change_verification_policy.py | 3 | 3 | 0 |
| tests/sandbox/capabilities/test_filesystem_capability.py | 4 | 4 | 0 |
| tests/models/test_any_llm_model.py | 8 | 8 | 0 |
| tests/sandbox/test_bounded_read.py | 2 | 2 | 0 |
| tests/test_items_helpers.py | 5 | 4 | 1 (1 / 0) |
| tests/model_settings/test_serialization.py | 4 | 4 | 0 |
| tests/mcp/test_mcp_server_manager_cleanup_state.py | 1 | 1 | 0 |
| tests/sandbox/test_entries.py | 7 | 2 | 5 (4 / 1) |
| **total (40 files)** | **183** | **148** | **35 (19 uncovered / 16 covered)** |

Whole-file python-only rows (released API contract, repo workflow, serial-test runner, code-change policy, image-generation reference, imports, examples) are fully valid. They test Python reflection, the Makefile and `.github` files, `.agents` skills, or `examples/` code that Rust does not ship.

## Portable cases that no Rust test covers (19)

Each entry gives the Python test, why it is portable, and the minimal Rust assertion.

1. **`tests/sandbox/test_entries.py::test_local_dir_copy_pins_parent_directories_during_open`, `test_local_dir_copy_fallback_rejects_swapped_parent_directory`, `test_local_dir_apply_rejects_source_root_swapped_to_symlink_after_validation` and its `_fallback_` twin (4 cases).**
   - The ledger reason (Python `os.open` / `dir_fd` patching) and the note in `tests/py_sandbox_a_entries_4.rs` lines 17-21 say Rust's `O_NOFOLLOW` covers the same refusal.
   - `src/sandbox/entries.rs::read_host_file_nofollow` (lines 434-443) opens the full path with `O_NOFOLLOW`, which only guards the last component. If an intermediate directory or the source root is swapped for a symlink after validation, the kernel follows it.
   - The covered leaf-swap case, `..._revalidates_swapped_paths_during_open`, is a separate matter: the `symlinked_source_leaf` tests assert that refusal.
   - This is behaviour, and a probable real gap.
   - Port: run a `local_dir` copy through a hook that runs between validation and read, as `py_core6_modelsrt_unix_local_swaps.rs` does for unix-local. Swap `src/sub` for a symlink to an outside dir. Assert `LocalDirReadError`, with reason `symlink_not_supported` or `path_changed_during_copy`, and that nothing is written. Needs a small seam in `copy_host_dir`.
   - Threat model: the attacker needs write access to the host source tree during the copy. The source tree is normally trusted app config, so severity is low to medium. Decide on intent before fixing.

2. **`tests/test_run_state_compatibility_corpus.py::test_v0194_mount_credentials_are_sanitized_and_require_trusted_rebind`.**
   - `tests/fixtures/py_run_state_corpus/security/v1_13_legacy_mount_credentials.json` is copied into the repo but no test references it: grep finds no fixture name or sentinel, and `py_core2_approvals_corpus.rs` only walks `sources["versions"]` and `sources["features"]` (lines 94-117), not the top-level `security` key.
   - `src/run_state/sandbox_payload.rs` implements the sanitiser. `py_core2_runstate_sidecars.rs:162-216` only covers a synthetic payload.
   - Port: `RunState::from_json(fixture)`, re-serialize, assert none of `RUNSTATE_ACCESS_SENTINEL_42`, `RUNSTATE_SECRET_SENTINEL_42`, `RUNSTATE_TOKEN_SENTINEL_42` appears in the output. Assert the `__openai_agents_redacted_mount_authority` marker is present.

3. **`test_run_state_from_string_deep_json_failure_discards_parser_state`.**
   - Port: `RunState::from_json_str("[".repeat(100_000) + "\"S\"" + "]".repeat(100_000))` returns `Err` containing `Failed to parse run state JSON` or `must be an object`, with no panic or stack overflow, and without the sentinel. The existing `json_decode_error_handling` only covers `{ invalid json`.

4. **`tests/realtime/test_openai_realtime_conversions.py::test_try_convert_raw_message_invalid_returns_none`.**
   - Python drops a raw message whose type is not a valid client event (`invalid.event`) and logs an error.
   - `src/realtime/openai_model.rs:720-728` sends any raw message verbatim.
   - Port: after `ready()`, send `RawMessage {"type":"invalid.event"}`. Assert the server receives no frame, or `send_event` returns an error, whichever the intended semantics are. This is a behaviour divergence; settle the intended semantics first.

5. **`tests/realtime/test_app_server_debug.py::test_forwarded_provider_errors_have_payload_free_debug_summaries[malformed=True, dont_log_model_data=True/False]` (2 cases).**
   - The reason given is `examples/` server code. The malformed half is SDK behaviour: a `response.output_audio_transcript.done` event whose `transcript` is a list.
   - Python's pydantic validation emits an Error event with full detail, while SDK logs contain no payload.
   - Rust only validates `response.output_audio.delta` (`openai_model.rs:1559-1572`). It has no handler for the transcript-done event, so the malformed payload passes silently.
   - Port: push that event with `transcript: ["marker"]`. Assert a `RealtimeModelEvent::Error` with context `Failed to validate server event`, and with `set_dont_log_model_data(true)` assert no captured log line contains `marker`. May need a `src` change; a real divergence.

6. **`tests/mcp/test_mcp_auth_params.py::test_sse_with_httpx_client_factory`, `test_sse_with_auth_and_factory`.**
   - Python asserts kwargs on a mocked `sse_client`, but the behaviour is "the caller-supplied HTTP client is used".
   - `MCPServerSseParams.http_client` and `StreamableHttp.http_client` (`src/mcp/sse.rs:131`, `src/mcp/mod.rs:952`) exist, yet no test sets them.
   - Port: build a `reqwest::Client` with default header `x-custom: 1`, pass it as `http_client`, assert the mock server sees the header on SSE and on POST. Add `auth` and assert both headers.

7. **`tests/mcp/test_mcp_version_compat.py::test_stdio_connects_to_legacy_server`.**
   - It is not Python-SDK-specific: it is a stdio server that answers `initialize` with a fixed `protocolVersion` of `2025-06-18`, lists one tool and returns text.
   - Existing Rust mock servers echo the requested version (`mcp_tests.rs:310`).
   - Port: a stdio mock with a fixed older version. Assert `list_tools` returns `legacy_tool` and `call_tool` returns `legacy-result`. Low severity.

8. **`tests/voice/test_pipeline.py::test_streamed_audio_result_odd_length_buffer_int16`, `test_streamed_audio_result_preserves_cross_chunk_sample_boundaries` (not attributable).**
   - `src/voice/result.rs` implements the `pending_byte` carry (lines 387-483) and has an inline unit test for odd lengths.
   - Only the float32 case has an integration test (`py_voice_pipeline.rs:481`).
   - Port: a TTS that yields `[0x01]` then `[0x00]` with dtype `Int16` yields exactly one sample `[1]`. A single `[0x01]` chunk with `Int16` yields `[1]`.

9. **`tests/test_items_helpers.py::test_extract_text_tolerates_none_text_content`.**
   - Python tolerates `text=None` from provider gateways. `MessageContent::try_from` (`src/items.rs:265-270`) defaults a null `text` to `""`, so the behaviour is implemented but unasserted.
   - Port: `MessageContent::try_from(json!({"type":"output_text","text":null}))` then `extract_text` is `None`. Mixed with a real `"hello"` part, in either order, gives `Some("hello")`.

10. **`tests/extensions/sandbox/test_vercel_provider_contract.py`, unported remainder (3, not attributable).** These are wire-level behaviours over the same mock HTTP server as `py_core6_modalvercel_vercel_contract.rs`, not SDK-object patching.
    - `test_installed_provider_does_not_adopt_replacement_session`: the server returns a different session id for the saved name. Assert resume creates a new sandbox, `workspace_root_ready` is false, and the replaced session is untouched.
    - `test_installed_provider_resumes_legacy_id_without_name`: a state with a bare id and no sandbox name resumes with no create and no DELETE. Rust's `split_provider_id` (`vercel/mod.rs:969`) rejects a bare id as `SessionStateInvalid`. This may be intentional; confirm.
    - `test_installed_provider_cleanup_preserves_concurrent_replacement`: while `delete` is blocked in stop, the server swaps the sandbox. Assert the replacement is left running and no DELETE is sent.

11. **`tests/fastapi/test_streaming_context.py::test_streaming_context`.**
    - The ledger notes "GAP: no row in any report", so the row has no recorded reason.
    - The behaviour is portable: start `run_streamed` in one tokio task and consume it in another (axum-style), with tracing on.
    - Port: assert no panic, a single `trace_end`, and the event-kind sequence `agent_updated_stream_event`, 9 raw events, `run_item_stream_event`. Raw events are the response lifecycle: created, in-progress, item added, content part added, text delta, text done, content part done, item done, completed. Check how that maps to Rust's raw-event count.

## Covered portable cases (16): reason mislabelled, no coverage loss

- `test_sqlalchemy_compaction.py` (8 of the sampled 8, about 15 of 18 in the file). These are the `memory` backend parametrization of the same tests.
  - Rust covers the file-backed twin in `py_memory_sqlx_compaction.rs`.
  - "A pooled sqlx session cannot share an in-memory DB" does not hold: `SqlxSession::with_pool` (`src/memory/sqlx_session.rs:345`) takes a caller pool, so two sessions can share one `sqlite::memory:` pool with `max_connections(1)`. The loss is nil, but they are better labelled "covered by file twin", and they inflate python-only by about 15.
- `realtime/test_session.py::test_on_tool_call_task_done_emits_error_event_immediately`: the `Tool call task failed` text is asserted in `py_realtime_session_core.rs:492`.
- `test_tool_output_trimmer.py::test_trimmable_tools_from_string`: covered by `a_single_allowlist_entry_matches_a_whole_tool_name`.
- `test_session.py::test_sqlite_session_file_lock_is_shared_across_instances`: covered by `two_sessions_on_one_file_write_independently`.
- `test_session_helpers.py::test_repeated_cancellation_waits_for_mutation`: covered by `a_session_mutation_runs_to_completion_after_its_caller_is_dropped`.
- `voice/test_pipeline.py::test_voicepipeline_cancelled_consumer_closes_the_session_without_further_tts`: covered by `voicepipeline_dropped_consumer_closes_the_session_without_further_tts`.
- `test_mcp_auth_params.py::test_sse_with_auth`: covered by `mcp_tests.rs:491`.
- `test_run_state_compatibility_corpus.py::test_run_state_from_string_malformed_json_discards_caller_owned_state`: covered by `json_decode_error_handling`.
- `test_entries.py::test_local_dir_copy_revalidates_swapped_paths_during_open`: the leaf case is covered by the `symlinked_source_leaf` tests (`O_NOFOLLOW`).

## Valid python-only reasons confirmed (148)

- **Reflection, repo files and examples:** `importlib`/`inspect` contract sweeps, the Makefile and workflow tests, `.agents` skills, and example scripts and apps.
- **Typed or absent inputs:**
  - `Manifest` has no `Deserialize` at all, so dictionary or `model_dump` manifests are unrepresentable (grep of `src/sandbox/manifest.rs`).
  - Dictionary or positional config, negative `usize`, `bytes`, one-shot iterables.
  - `needs_approval` is an infallible `bool`.
- **Tests of the Python fake itself:** `test_prompt_server.py` asserts the fake server's own string formatting.
- **asyncio, cancel and weakref internals:** `CancelledError` and `anyio` scopes, worker-task and task-factory mechanics, traceback-locals redaction, weakref/gc.
- **Python-library-only objects:**
  - Python-only any-llm Google `contents`; Rust never builds them.
  - The any-llm pydantic `ResponsesParams` sanitiser.
  - `sqlite3` connection pooling and `close()`.
  - Vendor SDK waiters and handles.
- **Documented design divergence:** the Rust HTTP client performs no retries of its own (`src/extensions/models/mod.rs:22-26`), so the `provider_managed_retries` tests have nothing to switch off.

## Overall estimate of the invalid-reason rate

- In the sample, 35 of 183 inspected cases (19.1%) have a reason that does not hold. 19 of those (10.4%) are uncovered behaviour; 16 (8.7%) are mislabelled but already covered, 8 of which are the sqlalchemy fixture twins.
- The sample is biased upward, because I sought out suspicious cases. The whole-file python-only rows (reflection, repo, examples) are about 100% valid and are under-represented.
- Rough population estimate: **about 5-10% uncovered-portable (roughly 200-400 of 4,001)** and **a further 3-5% mislabelled-but-covered**. The uncertainty is wide: the cases cluster within files, and the 95% interval on 10.4% is about 6.8-15.5%. A single-digit percentage of the bucket is hiding real checks.
- The hidden checks cluster in three kinds of reason:
  - "asyncio/drop only";
  - "SDK-object monkeypatching" over what is really a wire behaviour;
  - "typed in Rust", where the null/invalid case is still reachable via JSON.

## Other observations

- The `security/` run-state fixture is shipped but unused (finding 2).
- Documented Rust divergences that make some python-only rows behaviour differences rather than impossibilities: E2B/Cloudflare `shutdown` returns the error where Python swallows and logs it; E2B PTY open has no read timeout (`py-test-parity-core6-cfdaytona.md`).
- `test_app_server_debug.py` is "report counted 3, pytest collects 6", and `test_streaming_context.py` has no report row. The ledger's python-only counts for these were assigned, not derived.
- `test_sqlalchemy_compaction.py` shows 12 + 18 = 30 against a note saying "basis 15". About 3 of its 18 python-only cases are unexplained.

## Unresolved questions

1. Which individual cases are python-only in the ledger? Per-file counts cannot be mapped to tests. Ask the ledger owner for a per-test CSV, or accept group-level attribution.
2. LocalDir parent and source-root swap: is the host source tree in the threat model? If yes, finding 1 is a real fix (`openat`-style walk or a hook). If no, document the leaf-only `O_NOFOLLOW` guarantee and correct the claim in `py_sandbox_a_entries_4.rs`.
3. Raw realtime messages: should Rust reject or drop an unknown-type raw message as Python does? And should it validate malformed non-audio server events (`Failed to validate server event` for every known type)?
4. Vercel: is rejecting a bare legacy provider id intentional, since Python-written states are not meant to resume in Rust?
5. Does `SqlxSession` support `sqlite::memory:` with a single-connection pool, so the in-memory twins can be ported cheaply or reclassified as "covered"?
6. Is Rust's `RunResultStreaming` consumable from a different task than the one that started it, with the tracing context correct (the fastapi case)? I have not run it, since cargo is out of scope.

Status: DONE_WITH_CONCERNS
Summary: 40 files audited, 183 python-only cases checked: 148 valid, 35 (19.1%) with invalid reasons. Of these, 19 (10.4%) are uncovered portable behaviour (14 attributed to python-only groups, 5 unattributable and possibly classed as ported, so 10.4% is not a pure python-only invalid rate) and 16 (8.7%) are mislabelled but covered, 8 of them sqlalchemy twins. Estimated 5-10% of the 4,001 bucket hides uncovered behaviour.
Concerns/Blockers: Per-test attribution is not recoverable from the ledger, so group-level attribution was used. Two likely real gaps came out of the sample (LocalDir parent/source-root swap uses leaf-only `O_NOFOLLOW`; realtime raw-message and malformed-event validation differs from Python) alongside the unused credential fixture; nothing was run, per the read-only constraint.
