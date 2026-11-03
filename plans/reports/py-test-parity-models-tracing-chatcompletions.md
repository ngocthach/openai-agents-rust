# Python test parity: models / chatcompletions

Scope: 8 Python files under `tests/models/` (214 tests, counts from `grep -c "def test_"`).
Rust tests: `tests/py_models_chat_{converter,core,reasoning,stream_handler,stream_model,stream_runner,spans}.rs`.
Rust source changed: `src/models/openai_chatcompletions.rs` only.
Parametrize expansions are notes only, not in totals.

All 7 new files: `cargo test --all-features --test <file>` green (44+42+24+45+35+7+7 = 204 Rust tests); rustfmt'd;
no clippy warnings in my files (see note under section 5: `-D warnings` on the lib currently trips on other agents' in-flight files).

## 1. Per-file table

| file | #tests | ported | covered | python-only | failures found |
| --- | --- | --- | --- | --- | --- |
| test_openai_chatcompletions.py | 48 | 44 (`py_models_chat_core.rs`, `py_models_chat_spans.rs`, `py_models_chat_stream_model.rs::clean_gemini_tool_call_id_removes_thought_suffix`) | 0 | 4: falsy `Reasoning.__bool__` (pydantic); `extra_args={"timeout": httpx.Timeout}` (openai-python request kwarg); 2x `_get_client()`/`with_options(max_retries=0)` (openai-python client internals) | 12 (see section 2) |
| test_openai_chatcompletions_converter.py | 44 | 44 (`py_models_chat_converter.rs`) | 0 | 0 | 8 |
| test_openai_chatcompletions_stream.py | 92 | 86 (`py_models_chat_stream_handler.rs`, `py_models_chat_stream_model.rs`, `py_models_chat_stream_runner.rs`, `py_models_chat_spans.rs`) | 0 | 6: 2x asyncio `aclose`/cancellation shielding; `Part` class; `_StreamOutputLayout` KeyError; `_function_call_item` private helper; `_finish_reasoning_summary_part` private helper | whole handler rewritten (see section 2) |
| test_chatcompletions_stream_eof.py | 4 | 4 (`py_models_chat_stream_runner.rs`; official-endpoint branch exercised on the handler, see section 3) | 0 | 0 | 1 |
| test_gpt6_chat_tool_defaults.py | 1 | 1 (`py_models_chat_core.rs::gpt6_tool_request_defaults`) | 0 | 0 | 1 |
| test_reasoning_content.py | 13 | 13 (`py_models_chat_reasoning.rs`) | 0 | 0 | 3 |
| test_reasoning_content_replay_hook.py | 7 | 6 (`py_models_chat_reasoning.rs`) | 0 | 1: `LitellmModel` + `monkeypatch(litellm.acompletion)`; the Rust LiteLLM adapter reuses the same chat-model hook (`src/extensions/models/mod.rs:1114`), tested via `openai_chatcompletions_hook_can_enable_reasoning_content_replay` | 2 |
| test_deepseek_reasoning_content.py | 5 | 5 (`py_models_chat_reasoning.rs`; 2 LiteLLM round trips rewritten against `OpenAIChatCompletionsModel` + mock server) | 0 | 0 | 0 |
| **Total** | **214** | **203** | **0** | **11** | |

No existing Rust test was cited as COVERED: the existing tests (`openai_http_tests.rs`, `models_thinking_blocks_tests.rs`, ...) check the same area
at coarser granularity, so every behavioral Python test got its own Rust port.

## 2. Rust bugs fixed (all in `src/models/openai_chatcompletions.rs`)

Converter (`items_to_messages*`, content helpers)
- Content-part conversion rewritten as a port of `extract_all_content` / `extract_text_content` (new `pub fn`s):
  `prompt_cache_breakpoint` kept, `image_url.detail` defaults to `auto`, `original` maps to `high`, raw Chat Completions `text`/`image_url` parts
  accepted, `input_audio`/`input_file` validated (`file_data` wins over `file_id`, neither is an error), `video_url` handling.
- `system`/`developer`/assistant easy-input list content now goes through `extract_text_content` instead of being flattened to one string.
- Message classification mirrors `maybe_easy_input_message` / `maybe_input_message` / `maybe_response_output_message`
  (key allow-list, `phase`, id-less session-replay assistant shape, statusless output messages).
- An assistant output message after its turn's `function_call` merges into the pending tool-call assistant message (streamed order
  `[function_call, message]` now equals `[message, function_call]`; refusal merges too) instead of emitting two assistant messages.
- Reasoning items no longer flush the pending assistant message; pending reasoning is preserved across an output-message flush; output-message
  text segments join with `\n`.
- `file_search_call` converts to a `file_search_call` function tool call; `compaction` items rejected; item_reference message matches Python;
  message-shaped or type-less unrecognized items raise `Unhandled item type or structure` (typed foreign items are still skipped, kept for
  `tests/logger_verbose_stdout.rs`).
- Empty `arguments` on a `function_call` becomes `"{}"`.

Non-streaming parse (`parse_chat_completion*`, now `pub`; `parse_chat_completion_as` added)
- Payload with no `choices` raises `ChatCompletion response has no choices`; `finish_reason="content_filter"` with an empty message is
  synthesized into a refusal; `finish_reason="length"` raises only for an empty turn (text/refusal/tool call count, reasoning does not), with
  Python's message; `custom` tool calls skipped, or rejected under strict validation (`get_response` now passes the strict flag).
- `message.reasoning` ignored when `reasoning_content` or `thinking_blocks` exist.
- `provider_data` (`model` = the configured model, `response_id`) now attached to reasoning, message and function-call items, as Python does.
- Usage normalization (`chat_usage`): null counts are 0, a null `total_tokens` stays 0 (no input+output fallback), `cache_write_tokens` kept,
  request counted when usage is absent. Usage is recorded on the generation span before a rejected turn is raised.

Request body (`build_request_body`)
- `reasoning.mode` / `reasoning.context` warn once, or raise under strict validation.
- `gpt-6-sol` / `gpt-6-luna` default `reasoning_effort="none"` when function tools or handoffs are present (explicit effort / `extra_args` win).
- `stream_options` is sent whenever `include_usage` is set explicitly (previously `include_usage=false` dropped it).
- `convert_chat_tool_choice` made `pub`.

Stream handler (rewrite of `StreamState` into a port of `ChatCmplStreamHandler.handle_stream` + `buffer_tool_call_stream`)
- Event sequence now matches Python exactly: `response.created` is lazy, no extra `output_text.done` / `reasoning_summary_text.done` /
  `function_call_arguments.done` events, content-part and refusal indexes (`refusal` before `text` gets index 0/1), one message announcement,
  Python's output-index layout for reasoning / function calls / message (including fallback calls announced at the end).
- Added: multiple-choice / nonzero-index handling (warn once or strict error), `finish_reason` tracking (`content_filter` refusal synthesis,
  `length` error, official-endpoint "stream ended before finish_reason" error), custom tool call skip/strict error, audio error,
  thinking-block validation errors, `response_id`/`model` provider data, Gemini `__thought__` id cleaning and thought signatures,
  url-citation and logprob accumulation rules, tool-call buffering layer with metadata merge and the `finish_reason='tool_calls'` check,
  `x-request-id` carried on the completed response (`_request_id`), stream usage serialized with `cache_write_tokens`, usage `null` when absent,
  span usage/output recorded before the terminal event and on errors.
- New public helpers for transport-free tests: `handle_chat_stream_chunks`, `ChatStreamOptions`, `buffer_tool_call_chunks`,
  `merge_buffered_metadata`. The old inline `StreamState` unit tests were removed (covered by the new integration tests).

## 3. Cross-area bugs deferred (not edited)

1. `tests/openai_http_tests.rs:377-381` (`chat_completions_stream_tool_call_arguments_incrementally`) asserts a
   `response.function_call_arguments.done` event. Python's Chat Completions stream never emits it and the ported event-order tests forbid it.
   Minimal fix: delete that `find(...)`/`assert_eq!(done["arguments"], ...)` block (the final arguments are already asserted via `output_item.done`).
   This test currently FAILS because of the intended parity fix.
2. `src/usage.rs:168` `Usage::from_chat_usage` falls back to `input+output` when `total_tokens` is 0/null and never fills `cache_write_tokens`;
   Python uses `total_tokens or 0` and keeps `cache_write_tokens`. Worked around locally (`chat_usage`); same total fallback exists in
   `Usage::from_responses_usage` (`src/usage.rs:127`), which the stream path still goes through in the runner (only matters for a null streamed total).
3. `src/models/openai_client.rs:274` `is_openai_api()` is base-URL only (`https://api.openai.com`), so the official-endpoint stream rule
   (`require_finish_reason`) cannot be reached by a local mock. Tested at handler level instead. Suggested hook: a client option forcing
   `is_openai_api()` true (test-only), or accept `http://127.0.0.1` via a builder flag.
4. Python `ChatCmplHelpers`/`_validate_official_openai_input_content_types` (reject non-`input_*` parts on the official endpoint) is not
   implemented in Rust; no Python test in these 8 files asserts it, so it is a latent gap only.
5. Streamed `preserve_raw_usage`: Python attaches the provider's raw chat usage (`_agents_sdk_raw_usage`); the Rust completed event carries
   Responses-shaped usage, so `ModelResponse.raw_usage` on streamed chat runs is not the provider payload. Needs a key on the completed
   `response` plus a read in the runner (`src/run*.rs`, not mine). Not asserted by Python tests except the private-attribute part (classified in
   `stream_handler_keeps_empty_choice_usage_chunks`, usage assertion only).

## 4. Shared-file edits needed

None. (`pub mod openai_chatcompletions` already exists in `src/models/mod.rs`; no `lib.rs` / `Cargo.toml` change.)

## 5. `#[ignore]`d tests

None.

Notes: `cargo clippy ... -- -D warnings` on a test target fails today only because of warnings in other agents' lib files
(`src/extensions/models/mod.rs:1016`, `src/models/openai_responses_ws.rs:553`, `src/sandbox/*`); plain clippy shows zero warnings for
`src/models/openai_chatcompletions.rs` and the 7 new test files. `/Users/ngocthach/.cache/oars-build/tp-models` was ~1.4 GB at last check.
Regression check on neighbouring suites after my src changes: `models_thinking_blocks_tests`, `models_redaction_tests`,
`extensions_models_provider_tests`, `core_prompt_cache_key`, `core_model_run_lifecycle`, `logger_verbose_stdout` pass;
`openai_http_tests` has the single failure in section 3.1.

## 6. Totals

| | tests |
| --- | --- |
| Python tests | 214 |
| Ported | 203 |
| Covered | 0 |
| Python-only | 11 |
| Rust tests added | 204 across 7 files |
