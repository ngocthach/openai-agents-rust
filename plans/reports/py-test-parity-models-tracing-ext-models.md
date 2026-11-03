# py-test-parity: ext-models (LiteLLM / any-llm / Anthropic / Gemini)

STATUS: complete. Gates (shared build dir, --all-features): `cargo test --test <file>` green for all
five files; `cargo clippy --test <file> -- -D warnings` clean for all five; files rustfmt'd.
Build dir peaked ~1.3 GB (limit 4 GB).

New files: tests/py_ext_models_{ordering,litellm,any_llm,thinking,stream}.rs
Rust tests: ordering 10, litellm 24, any_llm 25, thinking 14, stream 7 = 80 (75 pass, 5 `#[ignore]`).
Src edit: src/extensions/models/mod.rs only.

## 1. Per-file table

| file | #tests | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_any_llm_model.py | 58 | 25 | 4: forced_responses_errors -> mod.rs `tests::resolves_providers`; split_does_not_duplicate -> mod.rs `tests::pairs_tool_calls_with_their_results` + py_ext_models_ordering.rs::split_does_not_duplicate_content_or_thinking; responses_stream_rejects_failed_terminal_events + responses_stream_rejects_error_event -> openai_http_tests.rs::responses_stream_terminal_events_become_errors | 29: any-llm private provider API / module import teardown / google-native hooks (no Rust counterpart, see mod.rs doc) / pydantic payload+reasoning-object normalizers / trace-exporter redaction / litellm kwarg nesting / prompt fail-fast (deliberate divergence, see 3) / provider-stream aclose+cancel+close-failure (11) / span-timing before completed (3) | 1 (content_filter refusal; ignored) |
| test_litellm_chatcompletions_stream.py | 16 | 7 | 0 | 9: provider-stream aclose/cancel/close-failure (8); span populated before completed (1) | 2 (content_filter refusal x2; ignored) |
| test_litellm_content_filter.py | 3 | 3 | 0 | 0 | 1 (ignored) |
| test_litellm_extra_body.py | 8 | 5 | 0 | 3: falsy `__bool__` Reasoning (pydantic); `_skip_mcp_handler` litellm kwarg (litellm library plumbing); metadata kwarg vs extra_body.metadata separation (one HTTP body in Rust) | 1 (reasoning_effort precedence; fixed, see 2) |
| test_litellm_logging_patch.py | 1 | 0 | 0 | 1: patches litellm library serializer via env var | 0 |
| test_litellm_logprobs.py | 4 | 4 | 0 | 0 | 0 |
| test_litellm_usage_requests.py | 2 | 2 | 0 | 0 | 0 |
| test_litellm_user_agent.py | 1 | 1 | 0 | 0 | 0 |
| test_anthropic_thinking_blocks.py | 11 | 10 | 0 | 1: positional-argument signature compatibility | 1 (encrypted_content signature join; ignored) |
| test_extended_thinking_message_order.py | 10 | 10 | 0 | 0 | 0 |
| test_gemini_thought_signatures.py | 3 | 2 | 1: models_thinking_blocks_tests.rs::gemini_thought_signatures_survive_a_round_trip | 0 | 0 |
| test_gemini_thought_signatures_stream.py | 2 | 2 | 0 | 0 | 0 |
| test_kwargs_functionality.py | 10 | 9 | 0 | 1: litellm `max_retries=0` kwarg leak check (Rust client does no retries of its own, see mod.rs doc) | 0 |
| **TOTAL** | **129** | **80** | **5** | **44** | **5 ignored + 1 fixed** |

80 + 5 + 44 = 129, and per-file sums equal `grep -c "def test_"`.

Group reason for litellm-library cases: asserts litellm-library call plumbing; Rust routes
provider/model to an OpenAI-compatible endpoint instead (see src/extensions/models/mod.rs doc).
Where a litellm test asserted the outgoing request or the parsed result, it was ported against a
local mock server.

Parametrize expansions (note only, not in totals): any_llm nullable usage x12 and user agent x2,
parallel_tool_calls x6 / x18, reasoning summary x3, litellm usage x6, kwargs parallel x6 x2. The
ported tests loop over the same parameter sets inside one test each. Python-only `override_ua`
(HEADERS_OVERRIDE contextvar) has no Rust counterpart; the default user agent is asserted.

## 2. Rust bugs fixed

- src/extensions/models/mod.rs: `LitellmModel` resolves `reasoning_effort` in LiteLLM's order
  (`reasoning.effort` > `extra_body` > `extra_args`) and strips the losing copies. Before, an
  `extra_body` value overrode `reasoning`, and effort in both `reasoning` and `extra_args` failed as
  a duplicate keyword. Found by reading the code against the Python tests before the first run.
- src/extensions/models/mod.rs: logs one warning (target `openai.agents`) that `Reasoning.summary`
  is not forwarded on the LiteLLM chat path.

## 3. Cross-area bugs deferred (not my files)

- src/models/openai_chatcompletions.rs, non-stream parse near ~1146 (beside the
  `finish_reason == "length"` check) and stream finalize near ~1544-1560: Python turns
  `finish_reason == "content_filter"` with empty output into a `ResponseOutputRefusal` message
  (non-empty text); streaming emits output_item.added, content_part.added, refusal.delta,
  content_part.done, output_item.done with content_index 0 and output_index after any reasoning
  item. Rust returns zero output. Minimal fix: non-stream, if output is empty and finish_reason is
  `content_filter`, push a message with a `refusal` part; stream, track `saw_content_filter` and at
  finish with no text/tool calls/refusal synthesize the refusal events. Responses that carry
  content stay untouched (those tests pass today).
- src/models/openai_chatcompletions.rs ~1586 and ~2645 (`item["encrypted_content"] = json!(sig)`
  from `last_signature`): Python joins every thinking-block signature with `\n`
  (`"TestSignature123\nTestSignature456"`); Rust keeps only the last. Minimal fix: collect all
  signatures and join with `\n`. The replay side (~1333) already splits on `\n`, so the round trip
  stays consistent.
- Deliberate divergence, not changed: Python `AnyLLMModel` raises `UserError` on any `prompt`;
  Rust forwards `prompt` on the Responses path and warn-and-ignores it on the chat path.

## 4. Shared-file edits needed

None. No `pub mod` / `pub use` / dependency lines; tests use only existing public API.

## 5. `#[ignore]`d tests (all blocked on section 3, none need network/credentials)

- py_ext_models_litellm.rs::content_filter_finish_reason_surfaces_refusal
- py_ext_models_any_llm.rs::chat_path_surfaces_content_filter_refusal
- py_ext_models_stream.rs::stream_response_synthesizes_refusal_on_content_filter
- py_ext_models_stream.rs::stream_response_content_filter_refusal_after_reasoning
- py_ext_models_thinking.rs::anthropic_thinking_blocks_with_tool_calls

Remove each `#[ignore]` once the matching fix lands; the assertions already encode the Python
expectation.

## 6. Totals

| ported | covered | python-only | total | rust tests | ignored |
|---|---|---|---|---|---|
| 80 | 5 | 44 | 129 | 80 | 5 |
