# py-test-parity core6: models, realtime, voice, tracing, extensions, sandbox (modelsrt)

Slice: every Python test file under `tests/models`, `tests/realtime`, `tests/voice`, `tests/tracing`, `tests/extensions` (sandbox providers excluded) and `tests/sandbox` (minus `test_tar_workspace.py`) whose ledger row had `not_ported > 0`, unknown counts, or a total that did not equal the sum of its buckets: 77 files, 3,799 cases. Per-file CSV: `py-test-parity-core6-modelsrt.csv`.

Before: 1,839 ported, 129 covered, 747 python-only, 14 not ported, 92 with no per-file split (the five `tests/models` group files), 978 unaccounted. After: **2,619 ported, 163 covered, 1,017 python-only, 0 not ported, 0 unaccounted, 0 assumed.** Every row adds up to its real pytest case count. The follow-up pass read every case that had defaulted to ported (see "Verification of the defaulted cases"); the CSV column `ported_defaulted_unverified` is 0 for every row.

## How the cases were counted

- Case totals are the pytest-collected parametrized ids (`plans/reports`-ledger collect list). The ledger had counted most files per test function, so the unaccounted cases are the extra parametrize ids of tests already classified (gap == sum(ids - 1) in all but 4 files).
- A parametrized Python test is classified as one unit and its extra ids follow it:
  - **ported**: the Rust test of the same name loops the ids (checked by reading the loop bodies for the 10 largest, e.g. `gpt6_tool_request_defaults` 22 ids, `runner_normalizes_nullable_chat_usage`, the mount-security `acknowledgement_*` tests, `incoming_message_size_limit`); where the Rust port is one test per id (`..._allow` / `..._reject` helpers) the extras are ported too.
  - **python-only**: the mechanism cannot exist in Rust: asyncio cancellation/task/event gating, `__traceback__` and hostile-exception redaction, pydantic/dict config input, typed-vs-validated arguments, `omit`/`NOT_GIVEN` sentinels, numpy dtype spellings, hooks that mutate `agent.name`/`agent.tools` in place, Python-library `any-llm` Google objects. `sandbox/test_mount_security.py` uses the per-index table of `py-test-parity-sandbox-a-mount-security.md` (89 python-only / 45 ported, which reproduces its 134 functions exactly).
  - **covered**: `tests/voice/test_utils.py` (one 28-id test covered by `streamed_text_is_spoken_without_losing_or_gluing_words`) and the `test_codex_exec_run_web_search_enabled_flags` pair (inline test `legacy_web_search_toggle_maps_to_config` in `src/extensions/codex/exec.rs`).
- Limits of the method: extras of name-matched tests rest on a loop in the Rust body or a per-id Rust port, spot-checked on the largest tests rather than compared id by id; the cases that had no such evidence were all read individually in the follow-up pass. `sandbox/test_bounded_read.py` (gap 11 of 52 extras), `sandbox/test_runtime.py` and `realtime/test_openai_realtime.py` were partly expanded by earlier reports, so their residue is allocated to ported (bounded_read) or trimmed by one (runtime, realtime openai).
- The five `tests/models` group files (`test_default_models` 40, `test_map` 17, `test_agent_registration` 14, `test_openai_client_utils` 8, `test_generation_metadata` 13): 86 ported, 6 python-only (the three dictionary-config agent-registration tests, `invalid prefix mode` strings, `aclose` first-failure re-raise because `ModelProvider::close` is infallible, and hybrid cancellation).

## Per-file numbers

| file | cases | ported | covered | python-only | not ported |
|---|--:|--:|--:|--:|--:|
| tests/extensions/experiemental/codex/test_codex_exec_thread.py | 55 | 19 | 14 | 22 | 0 |
| tests/extensions/experiemental/codex/test_codex_tool.py | 97 | 50 | 20 | 27 | 0 |
| tests/extensions/experimental/hosted_multi_agent/test_model.py | 32 | 30 | 0 | 2 | 0 |
| tests/extensions/memory/test_advanced_sqlite_session.py | 179 | 131 | 2 | 46 | 0 |
| tests/extensions/memory/test_async_sqlite_session.py | 45 | 3 | 18 | 24 | 0 |
| tests/extensions/memory/test_dapr_session.py | 52 | 38 | 0 | 14 | 0 |
| tests/extensions/memory/test_encrypt_session.py | 66 | 50 | 1 | 15 | 0 |
| tests/extensions/memory/test_memory_imports.py | 13 | 0 | 0 | 13 | 0 |
| tests/extensions/memory/test_mongodb_session.py | 72 | 44 | 0 | 28 | 0 |
| tests/extensions/memory/test_redis_session.py | 110 | 46 | 4 | 60 | 0 |
| tests/extensions/memory/test_sqlalchemy_session.py | 43 | 30 | 0 | 13 | 0 |
| tests/extensions/test_tool_output_trimmer.py | 60 | 56 | 0 | 4 | 0 |
| tests/models/test_agent_registration.py | 14 | 11 | 0 | 3 | 0 |
| tests/models/test_any_llm_model.py | 100 | 57 | 4 | 39 | 0 |
| tests/models/test_chatcompletions_stream_eof.py | 14 | 14 | 0 | 0 | 0 |
| tests/models/test_default_models.py | 40 | 40 | 0 | 0 | 0 |
| tests/models/test_generation_metadata.py | 13 | 13 | 0 | 0 | 0 |
| tests/models/test_gpt6_chat_tool_defaults.py | 22 | 22 | 0 | 0 | 0 |
| tests/models/test_kwargs_functionality.py | 21 | 19 | 0 | 2 | 0 |
| tests/models/test_litellm_chatcompletions_stream.py | 19 | 10 | 0 | 9 | 0 |
| tests/models/test_litellm_extra_body.py | 11 | 7 | 0 | 4 | 0 |
| tests/models/test_litellm_usage_requests.py | 7 | 7 | 0 | 0 | 0 |
| tests/models/test_litellm_user_agent.py | 2 | 2 | 0 | 0 | 0 |
| tests/models/test_map.py | 17 | 14 | 0 | 3 | 0 |
| tests/models/test_model_retry.py | 111 | 68 | 0 | 43 | 0 |
| tests/models/test_openai_chatcompletions.py | 72 | 67 | 0 | 5 | 0 |
| tests/models/test_openai_chatcompletions_converter.py | 61 | 61 | 0 | 0 | 0 |
| tests/models/test_openai_chatcompletions_stream.py | 111 | 104 | 0 | 7 | 0 |
| tests/models/test_openai_client_utils.py | 8 | 8 | 0 | 0 | 0 |
| tests/models/test_openai_responses.py | 204 | 167 | 2 | 35 | 0 |
| tests/models/test_openai_responses_converter.py | 69 | 67 | 0 | 2 | 0 |
| tests/models/test_reasoning_content.py | 15 | 15 | 0 | 0 | 0 |
| tests/realtime/test_agent.py | 7 | 4 | 0 | 3 | 0 |
| tests/realtime/test_audio_formats_unit.py | 16 | 15 | 0 | 1 | 0 |
| tests/realtime/test_openai_realtime.py | 136 | 111 | 2 | 23 | 0 |
| tests/realtime/test_playback_tracker.py | 21 | 19 | 1 | 1 | 0 |
| tests/realtime/test_session.py | 96 | 53 | 7 | 36 | 0 |
| tests/realtime/test_session_approvals.py | 32 | 25 | 4 | 3 | 0 |
| tests/realtime/test_session_close_iteration.py | 5 | 4 | 0 | 1 | 0 |
| tests/realtime/test_session_guardrails.py | 29 | 26 | 2 | 1 | 0 |
| tests/realtime/test_session_tool_error_events.py | 9 | 8 | 0 | 1 | 0 |
| tests/realtime/test_session_tool_output_guardrails.py | 12 | 12 | 0 | 0 | 0 |
| tests/realtime/test_session_tool_outputs.py | 22 | 15 | 0 | 7 | 0 |
| tests/realtime/test_testing.py | 42 | 21 | 0 | 21 | 0 |
| tests/realtime/test_websocket_message_size.py | 12 | 10 | 0 | 2 | 0 |
| tests/sandbox/test_bounded_read.py | 71 | 52 | 17 | 2 | 0 |
| tests/sandbox/test_docker.py | 143 | 115 | 0 | 28 | 0 |
| tests/sandbox/test_entries.py | 48 | 41 | 0 | 7 | 0 |
| tests/sandbox/test_extract.py | 34 | 31 | 0 | 3 | 0 |
| tests/sandbox/test_manifest.py | 25 | 14 | 0 | 11 | 0 |
| tests/sandbox/test_manifest_config.py | 21 | 3 | 0 | 18 | 0 |
| tests/sandbox/test_memory.py | 83 | 61 | 0 | 22 | 0 |
| tests/sandbox/test_mount_security.py | 246 | 111 | 0 | 135 | 0 |
| tests/sandbox/test_mounts.py | 38 | 36 | 0 | 2 | 0 |
| tests/sandbox/test_runtime.py | 227 | 111 | 22 | 94 | 0 |
| tests/sandbox/test_session_state_roundtrip.py | 32 | 15 | 0 | 17 | 0 |
| tests/sandbox/test_snapshot.py | 42 | 32 | 0 | 10 | 0 |
| tests/sandbox/test_tar_utils.py | 75 | 70 | 0 | 5 | 0 |
| tests/sandbox/test_unix_local.py | 49 | 37 | 0 | 12 | 0 |
| tests/sandbox/test_unix_local_create.py | 15 | 1 | 2 | 12 | 0 |
| tests/sandbox/test_unix_local_file_io.py | 45 | 35 | 0 | 10 | 0 |
| tests/sandbox/test_unix_local_special_files.py | 13 | 8 | 0 | 5 | 0 |
| tests/sandbox/test_unix_local_user_file_io.py | 27 | 5 | 0 | 22 | 0 |
| tests/sandbox/test_workspace_paths.py | 72 | 66 | 0 | 6 | 0 |
| tests/tracing/test_reasoning_export.py | 11 | 10 | 0 | 1 | 0 |
| tests/tracing/test_trace_context.py | 13 | 10 | 0 | 3 | 0 |
| tests/tracing/test_traces_impl.py | 14 | 3 | 0 | 11 | 0 |
| tests/tracing/test_tracing_env_disable.py | 23 | 22 | 0 | 1 | 0 |
| tests/voice/test_input.py | 24 | 20 | 1 | 3 | 0 |
| tests/voice/test_openai_model_provider.py | 21 | 14 | 0 | 7 | 0 |
| tests/voice/test_openai_stt.py | 37 | 25 | 0 | 12 | 0 |
| tests/voice/test_openai_stt_eof.py | 12 | 12 | 0 | 0 | 0 |
| tests/voice/test_openai_stt_session_config.py | 11 | 9 | 0 | 2 | 0 |
| tests/voice/test_pipeline.py | 72 | 31 | 9 | 32 | 0 |
| tests/voice/test_pipeline_config.py | 4 | 1 | 0 | 3 | 0 |
| tests/voice/test_pipeline_tracing.py | 26 | 25 | 0 | 1 | 0 |
| tests/voice/test_utils.py | 31 | 0 | 31 | 0 | 0 |
| **total (77 files)** | **3799** | **2619** | **163** | **1017** | **0** |

## New tests (about 140 cases ported, all passing, `--all-features`)

| file | cases | Python tests |
|---|--:|---|
| `tests/py_core6_modelsrt_voice_logging.rs` | 9 | `test_voice_workflow_errors_apply_model_and_tool_logging_policies` (6), `test_voice_on_start_errors_apply_model_and_tool_logging_policies` (3) |
| `tests/py_core6_modelsrt_realtime_logging.rs` | 3 | `test_output_guardrail_failure_follows_both_data_policies` |
| `tests/py_core6_modelsrt_sandbox_memory_logging.rs` | 6 | `test_sandbox_memory_enqueue_failure_follows_both_data_policies` |
| `tests/py_core6_modelsrt_realtime_tool_errors.rs` | 8 | `test_async_tool_failure_events_are_safe` (the 9th, an exception whose `__str__` raises, is python-only) |
| `tests/py_core6_modelsrt_tracing.rs` | 16 | `test_trace_lifecycle_logging_omits_caller_data` (12), `test_disabled_span_logging_respects_data_policy` (2), `test_trace_logs_warning_when_current_trace_exists`, `test_create_trace_for_run_does_not_reattach_after_trace_state_reload` |
| `tests/py_core6_modelsrt_generation_metadata.rs` | 12 | `test_generation_metadata_respects_sensitive_capture` over Chat Completions, `LitellmModel` and `AnyLLMModel`, streamed and not (the earlier port only drove a recording model and never reached the model's span code) |
| `tests/py_core6_modelsrt_encrypt_compaction.rs` | 9 | `test_runner_decrypts_existing_compaction_history_with_ttl_and_limit` (2), `test_encrypted_compaction_replaces_complete_logical_history_after_expiry` (4), `test_runner_recovers_encrypted_resumed_append` (2; the retry runs in the same streamed/non-streamed mode as the failing run). Uses real 2 s TTL sleeps (Python freezes the clock) and a local mock endpoint |
| `tests/py_core6_modelsrt_encrypt_compaction.rs` (deferred candidates) | 1 | `test_runner_defers_compaction_using_decrypted_candidates` |
| `tests/py_core6_modelsrt_unix_local_swaps.rs` | 18 | parent-swap and leaf-swap rows for `mkdir`/`ls`/`rm`/`rmtree`/`mv`/`same_file` (12), creator race, FIFO replacement read/write, user `same_file` on an ungranted path |
| `tests/py_core6_modelsrt_sandbox_limits.rs` | 5 | concurrency and archive limits on resume, persist/hydrate with an unsafe manifest (2), capability-added unsafe authority on a stopped injected session |
| `tests/py_core6_modelsrt_sandbox_provider_defaults.rs` | 1 | streamed row of the provider-defaults test |
| `tests/py_core6_modelsrt_docker_grants.rs` | 14 | five storage mount kinds x create/resume with a read-only host grant, and the four path-grant create/resume rows |
| `tests/py_core6_modelsrt_bounded_prefix.rs` | 18 | every provider x limit prefix read (15), missing file for Blaxel/Cloudflare/Daytona (3) |
| `tests/py_core6_modelsrt_stream_terminal_span.rs` | 8 | LiteLLM and any-llm streamed span populated at `completed` |
| `tests/py_core6_modelsrt_user_agent.rs` | 4 | `User-Agent` default and override on LiteLLM and any-llm |
| `tests/py_core6_modelsrt_mongodb_ids.rs` | 4 | literal session ids that look like query operators |
| `tests/py_core6_modelsrt_realtime_logging.rs` (audio formats) | 6 | unknown audio format diagnostics under either tool flag |

## Rust bugs fixed

1. **Voice failures logged raw error text regardless of the logging policy** (`src/voice/pipeline.rs`, `src/voice/result.rs`). Python applies both `DONT_LOG_MODEL_DATA` and `DONT_LOG_TOOL_DATA` to workflow failures. Rust logged `{error}` unconditionally and used one message for single and multi-turn runs. Now: `Error processing single voice turn` / `Error processing voice turns`, `Voice workflow on_start failed` (warning) and `Error processing voice output` (logged when the consumer receives the error event) go through `log_model_and_tool_action_*`; `Error streaming voice audio` goes through `log_model_action_error`.
2. **Realtime output-guardrail failure leaked the guardrail name and ignored the tool policy** (`src/realtime/session.rs`). It now logs `Output guardrail raised an exception; skipping it` through `log_model_and_tool_action_warning`; the guardrail name appears only when both policies allow the detail.
3. **Sandbox memory enqueue failure logged the raw error** (`src/sandbox/runtime.rs`, both the run and the streamed-run site). Now `Failed to enqueue sandbox memory after run` / `... after streamed run` through `log_model_and_tool_action_warning`.
4. **New public API:** `openai_agents_sdk::tracing::clear_started_trace_ids()` (Python `_clear_started_trace_ids`), needed to prove a resumed run in a fresh process does not reattach to a trace it never started. Additive, not breaking.

## Divergences found while verifying the defaulted cases, and their fixes

- **Compaction policy for tool-output turns** (was deferred): `src/run_internal/mod.rs`, `NextStep::RunAgain` arm now calls `notify_session_completed()` after the turn's save, so a compacting session evaluates its policy against the tool-output turn's response and defers, as Python's per-turn `save_result_to_session` does. Ported as `the_runner_defers_compaction_using_decrypted_candidates`.
- **Unicode identifiers** (was deferred): `valid_identifier` in `src/memory/sqlite_session.rs` accepts Unicode alphanumerics in bare names; the early return in `distinct_non_ascii_identifiers_are_distinct_pairs` is gone and the test asserts.
- **Unix-local path races** (`src/sandbox/sandboxes/unix_local.rs`): `mkdir`, `ls`, `rm`, `mv` and `same_file` ran path-based after validation and never ran the after-validate hook, so a parent swapped for a link was followed. `mkdir`/`ls` now re-resolve after the window; `rm`/`mv`/`same_file` compare the parent's resolution before and after it (a comparison, not grant policy, so exact-file grants keep working). `read_file`/`write_file` open with `O_NONBLOCK` (constants defined locally, no libc) so a FIFO swapped in after the entry check is refused instead of blocking. 12 parent/leaf swap cases plus the FIFO and creator-race cases are ported.
- **Persist/hydrate with an unsafe manifest** (`src/sandbox/events.rs`): `InstrumentedSession::persist_workspace` and `hydrate_workspace` now refuse an unacknowledged mount-scoped credential before any archive work (Python's session wrapper does).
- **Realtime audio-format diagnostics** (`src/realtime/config.rs`): a non-numeric PCM `rate` was silently coerced; it now logs the redacted "Unknown PCM rate" diagnostic as Python does.

## Verification of the defaulted cases

All 203 cases that had defaulted to ported were read against their Python tests and Rust candidates. Outcome: confirmed ported by an existing Rust test (looped or one-per-id): about 75; ported by a new test in this pass: about 60; covered by an inline or neighbouring test: 8; python-only: about 60 (reasons: pydantic/dictionary/typed inputs, `omit` sentinels, pydantic `BaseModel` payload variants, Python-only mutation of provider context, descriptor-pinned `dir_fd` opens, Linux file leases, the sudo user-file worker body, private PTY/tar-writer hooks and fault injection, `RunResultStreaming` private fields, Redis Cluster clients, aiosqlite internals, `HEADERS_OVERRIDE`-style context variables, the voice provider's explicit-client conflict check that Rust's single `with_client` cannot express, live-session `_set_concurrency_limits`). Nothing is left as not ported.

Known limit: the live-session concurrency-limit case is python-only because Rust has no session-level setter; limits travel on the manifest, and `SandboxRunner` does not fold `concurrency_limits` into an injected live session's manifest. That is a behaviour difference (not tested) rather than an unreadable case.

## Behaviour changes to note

- Log message text changed for the three fixes above (no API change). Messages that previously ended in `: {error}` now omit the error when either policy redacts.
- No breaking public API change.

## Gates

- New binaries: 7 pass (`cargo test --all-features`). Regression sweep of the binaries whose sources I touched or that cover the same area (voice, realtime session, tracing, sandbox memory/runtime, encrypted sessions): all pass.
- `cargo clippy --all-features --lib` plus the 7 new test targets: no warning in any file I own. Other agents' warnings remain (`src/mcp/util.rs:15` unused import, `src/run_internal/session_write.rs:119`, `src/extensions/sandbox/vercel/mod.rs:1398`, `src/memory/openai_responses_compaction_session.rs:229`, `src/realtime/debug.rs:29`).
- rustfmt applied to the new test files, `src/voice/result.rs`, `src/voice/pipeline.rs`. `src/sandbox/runtime.rs:945` has a pre-existing format diff that is not in my edit.

## Unresolved

- The realtime dispatch `RuntimeError`/`CancelledError` x stage split (1 ported, 2 python-only) is a judgment call.
- Default-feature check: all 7 new test files compile and pass without `--all-features` (the five feature-gated ones run zero tests there).
