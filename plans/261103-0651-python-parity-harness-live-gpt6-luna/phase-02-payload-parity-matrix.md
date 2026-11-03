---
phase: 2
title: Payload parity matrix
status: completed
priority: P1
dependencies:
  - 1
---

# Phase 2: Payload parity matrix

## Overview

One scenario per feature. Each scenario is three files (`py.py`, `responses.json`, a Rust
test fn) and produces a verdict. No API spend. Output: a matrix in
`plans/reports/parity-capture-matrix-261103-0651-python-vs-rust-report.md` with one row per
scenario and every diff classified.

## Requirements

- Functional: all "must" scenarios below have both sides and a verdict; "should" scenarios
  are done unless blocked by a missing feature (then recorded as a gap).
- Non-functional: each scenario runs in < 10 s; scenarios are independent (own path, own
  script).

## Architecture

Diff classification, applied to every non-empty diff:

| Class | Meaning | Action |
| --- | --- | --- |
| expected | SDK identity (`user-agent`, `x-stainless-*`), key order | waiver with reason |
| rust-bug | Rust sends something different or extra/missing | phase 3 fix + test |
| python-only | Python-specific payload (e.g. pydantic-generated titles) with no semantic effect | waiver with reason, note in report |
| rust-ahead | Rust sends a field Python lacks and the field is valid for the API | waiver + note; keep |

## Scenario matrix

Must (in this order; the first two are the most likely divergence):

| # | scenario | proves | Python reference | Rust reference |
| --- | --- | --- | --- | --- |
| 1 | `default_model` | default `gpt-5.6-luna`, default settings, `instructions`, `input` | `examples/basic/hello_world.py` | `examples/hello_world.rs` |
| 2 | `explicit_gpt6_luna` | `model: gpt-6-luna` adds no implicit `reasoning`/`text.verbosity` | `default_models.py` | `src/models/default_models.rs` |
| 3 | `model_settings` | temperature, top_p, max_tokens, tool_choice, parallel_tool_calls, truncation, reasoning{effort,summary}, verbosity, metadata, store, include, prompt_cache_key, service_tier, extra_body/headers/query | `integration_tests/openai/test_model_settings.py` | `src/model_settings.rs` |
| 4 | `function_tools` | strict tool schema (`additionalProperties: false`, all `required`), scripted call → 2nd request has `function_call` + `function_call_output` | `examples/basic/tools.py` | `examples/tools.rs` |
| 5 | `structured_output` | `text.format` json_schema strict; `non_strict` variant | `examples/basic/non_strict_output_type.py` | `examples/structured_output.rs` |
| 6 | `handoffs` | `transfer_to_*` tool schema, recommended prompt prefix, scripted handoff → 2nd request to target with input filter | `examples/handoffs/` | `examples/handoffs.rs` |
| 7 | `agents_as_tools` | `as_tool` schema; inner run request; outer 2nd request | `examples/agent_patterns/agents_as_tools.py` | `examples/agents_as_tools.rs` |
| 8 | `guardrails` | two input sub-cases with the same flag on both sides — `run_in_parallel=True` (concurrent with the first model call) and blocking — plus an output tripwire; verdict = request count per sub-case and the error raised | `examples/agent_patterns/input_guardrails.py`, `test_guardrails.py` | `examples/guardrails.rs` |
| 9 | `human_in_the_loop` | needs-approval tool → interruption, no 2nd request; approve + resume → 2nd request with `function_call_output`; reject → rejection output | `integration_tests/openai/test_approval_resume.py` | `examples/human_in_the_loop.rs` |
| 10 | `sessions` | SQLite session, two turns: 2nd `input` = history + new; `conversation_id` and `previous_response_id` variants | `examples/memory/`, `examples/basic/previous_response_id.py` | `examples/sessions_sqlite.rs` |
| 11 | `streaming` | `stream: true` body equals non-stream body; event sequence each SDK emits from the same SSE written to `<sdk>.events.json` and diffed. Shared vocabulary: raw events use the API `type` string; run items use the Python names (`message_output_item`, `tool_call_item`, `tool_call_output_item`, `handoff_call_item`, `handoff_output_item`, `reasoning_item`); agent updates as `agent_updated:<name>`; the Rust writer maps its enum variants to these names | `examples/basic/stream_text.py`, `stream_items.py` | `examples/stream_text.rs`, `stream_items.rs` |

Should:

| # | scenario | proves | Python reference | Rust reference |
| --- | --- | --- | --- | --- |
| 12 | `hosted_tools` | web_search (filters, user_location), file_search (vector_store_ids, max_num_results), code_interpreter, image_generation, hosted MCP (server_url, require_approval) | `examples/tools/*.py` | `src/tool/hosted*.rs` |
| 13 | `inputs_media` | remote image URL, local image base64, remote PDF, local file | `examples/basic/{remote_image,local_image,remote_pdf,local_file}.py` | item builders |
| 14 | `prompts_dynamic` | `prompt {id, version, variables}`; dynamic instructions fn | `examples/basic/prompt_template.py`, `dynamic_system_prompt.py` | `Agent::with_prompt`, instructions fn |
| 15 | `chat_completions` | messages mapping (system/user/tool), tools format, strict, stream chunks | `integration_tests/openai/test_chat_completions.py` | `examples/chat_completions_provider.rs` |
| 16 | `retry` | 429 then 200: both retry once; request count equal | `examples/basic/retry.py`, `test_retry.py` | client retry |
| 17 | `tracing_export` | `/v1/traces/ingest` payload: trace + agent/generation/function/handoff spans, after id/timestamp normalization | `test_tracing.py` | `src/tracing/processors.rs` |
| 18 | `reasoning_items` | scripted reasoning item + message; 2nd request resends the reasoning item | `examples/reasoning_content/` | input_id resend (0.23.0) |

## Related Code Files

- Create: `parity/scenarios/<scenario>/{py.py,responses.json}` for #2–#18; test fns in
  `tests/parity_capture.rs`; `plans/reports/parity-capture-matrix-261103-0651-python-vs-rust-report.md`
- Modify: `parity/waivers.json`, `parity/capture_server.py` (chat completions, 429, traces
  ingest, SSE for function calls — add when the scenario needs it)

## Implementation Steps

1. For each scenario, in matrix order: write `responses.json` first (what the "model"
   answers, turn by turn), then `py.py` from the Python reference, then the Rust test fn
   from the Rust reference, keeping prompts, tool names, descriptions and argument names
   byte-identical.
2. Run `bash parity/run.sh capture <scenario>`; read `parity/out/<scenario>/diff.txt`.
3. Classify every diff line with the table above; add waivers only for `expected`,
   `python-only`, `rust-ahead`, each with a one-line reason; log `rust-bug` rows in the
   report for phase 3.
4. For `streaming`, both scenario scripts also write the event-type list; `compare.py`
   diffs `<sdk>.events.json` when present.
5. For `guardrails`, `human_in_the_loop`, `retry`: the verdict includes request *count*
   per path family, not only bodies.
6. After all scenarios: `bash parity/run.sh capture all` must end with every row `OK` or
   `DIFF` with every diff classified; write the report.

## Success Criteria

- [ ] 11 must scenarios have both sides and a verdict; should scenarios done or recorded as
      gaps with the reason.
- [ ] `plans/reports/parity-capture-matrix-261103-0651-python-vs-rust-report.md` lists scenario, request count
      per SDK, diff class per diff, waiver reasons, and the rust-bug list for phase 3.
- [ ] `parity/waivers.json` has a reason on every entry; no blanket `*` waivers except
      `user-agent` / `x-stainless-*`.
- [ ] Gate script passes with the full `tests/parity_capture.rs`.

## Risk Assessment

- Pydantic emits `title` fields in tool/output schemas; schemars emits `$schema`/`title`
  differently → decide once (strip both in the normalizer only if the API ignores them;
  otherwise it is a real diff — the Rust strict-schema port already drops them).
- Handoff prompt prefix text must be byte-identical; a one-character drift is a real bug.
- `extra_body` merge order and `include` ordering may differ harmlessly → sort `include`
  in the normalizer with a note.
- Scenario count is large; keep each file under ~60 lines and copy prompts from the
  references instead of inventing new ones.
