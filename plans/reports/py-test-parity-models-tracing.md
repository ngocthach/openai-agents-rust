# Python test parity: models / tracing / model_settings / tool_output_trimmer / hosted_multi_agent

Date 2026-10-31. Python tests counted with `grep -c "def test_"` per file (parametrize expansions not
counted). Per-helper detail: `py-test-parity-models-tracing-{chatcompletions,responses,ext-models}.md`.
The retry-client helper was stopped by the user; I finished that part, and the per-test split of
`test_model_retry.py` is approximate (marked `~`).

## Totals

| part | python tests | ported | covered | python-only | notes |
| --- | --- | --- | --- | --- | --- |
| chat completions (8 files) | 214 | 203 | 0 | 11 | helper report, 7 new files, 204 Rust tests |
| responses (6 files) | 233 | 191 | 1 | 41 | helper report, 5 new files |
| ext-models: litellm / any-llm / anthropic / gemini / kwargs (13 files) | 129 | 80 | 5 | 44 | helper report, 5 new files, 0 ignored now |
| retry + client + defaults (7 files) | 182 | ~140 | 0 | ~42 | mine, see below |
| tracing (11 files) | 64 | 33 | 0 | 27 | 4 not ported, see below |
| model_settings (1 file) | 21 | 17 | 0 | 4 | pydantic/dict-coercion tests |
| tool_output_trimmer (1 file) | 51 | 47 | 0 | 4 | constructor coercion (bytes/str/frozenset) |
| hosted_multi_agent (2 files) | 23 | 23 | 0 | 0 | 3 ignored + 1 live-API ignored |
| **total** | **917** | **~734** | **6** | **~173** | 4 not ported (tracing) |

Scope check: models 758 = 214 + 233 + 129 + 182; tracing 64; settings 21; trimmer 51; hosted 23.
Sum 917 (the 233 + 214 + 129 + 182 = 758 models tests, + 64 + 21 + 51 + 23 = 159).

## Mine: per-file

| file | #tests | ported | python-only | notes |
| --- | --- | --- | --- | --- |
| tracing/test_import_side_effects.py | 7 | 0 | 7 | import machinery / sqlite3 / httpx import checks |
| tracing/test_logger.py | 1 | 1 | 0 | log target `openai.agents` |
| tracing/test_processor_api_key.py | 7 | 6 | 1 | thread race on `os.environ` monkeypatch |
| tracing/test_reasoning_export.py | 4 | 4 | 0 | `sanitize_for_openai` made `pub` |
| tracing/test_set_api_key_fix.py | 1 | 1 | 0 | |
| tracing/test_setup.py | 5 | 0 | 5 | atexit + module globals; Rust equivalents (`shutdown_guard`, provider reset) tested separately |
| tracing/test_span_ordering.py | 3 | 2 | 1 | `fetch_normalized_spans` helper |
| tracing/test_spans_impl.py | 4 | 0 | 4 | generator close / ContextVar token across tasks |
| tracing/test_traces_impl.py | 8 | 3 | 5 | same ContextVar group, caplog |
| tracing/test_trace_context.py | 13 | 8 | 3 | 3 falsy `__bool__`; 2 NOT PORTED: reload-no-reattach (cache not clearable) and the "Trace already exists" warning assertion |
| tracing/test_tracing_env_disable.py | 11 | 8 | 1 | 1 falsy span; 2 NOT PORTED: log-redaction caplog tests (partly covered by `models_redaction_tests.rs`) |
| model_settings/test_serialization.py | 21 | 17 | 4 | pydantic/dict coercion, field order via dataclass `fields` |
| extensions/test_tool_output_trimmer.py | 51 | 47 | 4 | |
| hosted_multi_agent/test_model.py | 22 | 22 | 0 | 2 ignored (see below) |
| hosted_multi_agent/test_live.py | 1 | 1 | 0 | `#[ignore]`, real key + network |
| test_model_retry.py | 99 | ~65 | ~34 | python-only: asyncio cancellation/cleanup/ContextVar/timed-iterator internals, `_ws_close_invalidated` attr, falsey policy |
| test_openai_retry_helpers.py | 15 | 10 | 5 | httpx/httpx2 header containers, exception attrs |
| test_default_models / test_map / test_agent_registration / test_openai_client_utils / test_generation_metadata | 68 | 63 | 5 | `py_models_map_defaults.rs` (retry-client helper) |

## Rust bugs fixed (my source files)

- `src/tracing/processors.rs`: exporter API key shared across clones; `set_api_key`/`api_key()`; env key
  remembered once found, missing key never cached; empty key clears.
- `src/tracing/processors.rs`: generation usage is folded under `details` at export time (port of
  `_sanitize_generation_usage_for_openai_tracing_api`); `sanitize_for_openai` is `pub`.
- `src/tracing/spans.rs`, `mod.rs`: `no-op` ids disable spans/children; trace-exists warning; creation debug logs.
- `src/tracing/provider.rs`: `force_flush` pins the env disable flag; empty global export key is "unset".
- `src/tracing/span_data.rs`: `Response` span carries `response` and `input` (not exported), as Python.
- `src/usage.rs`: `to_span_usage` is the flat Python shape (`requests`, totals, `input_tokens_details`, `output_tokens_details`).
- `src/models/openai_responses.rs`, `openai_responses_ws.rs`: response span always records usage, records response/input when tracing includes data.
- `src/models/openai_client.rs`: Responses `error` SSE frames are yielded, then the model raises `ModelBehaviorError`; lib test updated, plus a test that `event: error` still fails the stream.
- `src/models/openai_chatcompletions.rs`: thinking-block signatures joined with `\n` in `encrypted_content` (non-stream and stream).
- `src/models/openai_responses_ws.rs`: yield to the reactor before polling a cached socket, so a server close that arrived since the last turn is seen (fixes `reconnects_if_cached_connection_is_closed` under load).
- `src/retry.rs`: `ModelRetryNormalizedError.explicit_fields` / `with_explicit` (provider override can clear `retry_after`/network flags); Python `retry-after` parsing (negative ignored, past date = 0, no 60s cap); nested-then-top-level error code; Python ambiguous-websocket sentence.
- `src/extensions/tool_output_trimmer.rs`: budget underflow at `max_output_chars == len("[Trimmed]")`; `find_recent_boundary` is `pub`.
- `src/extensions/hosted_multi_agent.rs`: run owner captured before the spawned turn task (task-local was lost, so cleanup and the concurrent-run guard never saw the owner); hosted items stay visible in the raw event stream; per-request usage entries and request count on continuations; logical pause counts no request.
- Helper fixes: chat completions converter/stream handler rewrite, content-filter refusal, `reasoning.mode/context` warnings, gpt-6 defaults (`openai_chatcompletions.rs`); `src/extensions/models/mod.rs` reasoning_effort precedence.

## Cross-area issues deferred (not edited)

- `src/run_internal/model_call.rs`: never scopes `provider_managed_retries_disabled` / `websocket_pre_event_retries_disabled` (14 tests ignored in `py_models_retry_stream.rs`); no `replay_unsafe_request` for `ProgrammaticToolCallingTool` (3 ignored); retries `conversation_locked` even with `max_retries=0` (1 ignored in `py_models_retry_get_response.rs`).
- `src/error.rs`: `ModelTimeoutError` message lacks Python's trailing period (1 ignored).
- Tool approval gating before transport (`hosted_multi_agent`): the runner builds requests from tool specs with no approval info, so `tools_with_approval_settings_are_rejected_before_transport` is ignored.
- Streamed runner path ignores `usage.requests` / `request_usage_entries` from `parse_response` (1 ignored in hosted tests).
- `ModelRetryBackoffSettings::default()` is `0.5/5.0/2.0` vs Python `0.25/2.0/2.0`; changing it alters defaults, needs a decision.
- Chat helper section 3: `Usage::from_chat_usage` total fallback + `cache_write_tokens`; `is_openai_api()` base-URL only; official-endpoint content-type validation absent; streamed `raw_usage` is not the provider payload.
- Responses helper: a model turn with empty output loops to `MaxTurnsExceeded` (Python ends the run).

## `#[ignore]`d tests

- `py_models_retry_stream.rs`: 16 (cross-area, run_internal/model_call.rs and error.rs, reasons in attribute)
- `py_models_retry_get_response.rs`: 1 (conversation_locked with `max_retries=0`)
- `py_ext_hosted_multi_agent.rs`: 3 (approval gating, streamed usage entries, live API needing a real key)
- none in chat, responses, ext-models (the five litellm/any-llm ignores were removed after the fixes)

## Shared-file edits

None to `src/lib.rs`. `Cargo.toml` `test-util` dev feature was added by the coordinator.

## Gates

- All my `py_*` test files plus `tracing_*`, `openai_http_tests`, `responses_ws_tests`, `models_*`, `extensions_models_provider_tests` pass with `--all-features`; lib tests for my modules pass (47).
- Clippy `-D warnings` on my test files: clean. Library clippy still fails only on other areas' files (`src/run_internal/agent_identity.rs`, `src/run_internal/mod.rs`, `src/stream_events.rs`).
- Two workspace-run failures fixed: global export key leaked between tests (guard clears it; empty key now means unset); cached-ws-close race (reactor yield).
