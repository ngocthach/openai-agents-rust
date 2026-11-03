# Payload parity matrix: Python SDK 0.24.0 vs Rust port — capture mode

Date: 2026-11-03. Plan: `plans/261103-0651-python-parity-harness-live-gpt6-luna/` (phase 2).
Harness: `parity/` (see `docs/parity-harness.md`). No API call made; both SDKs ran against the
local capture server through `OPENAI_BASE_URL` with the dummy key `sk-parity-dummy`.

Reproduce: `bash parity/run.sh setup && bash parity/run.sh capture all`.

## Verdicts

18 scenarios, 18 `OK` after classification. Request counts are per SDK and identical on
both sides in every row.

| # | scenario | requests | diffs found → class → handling |
| --- | --- | --- | --- |
| 1 | `default_model` | 1 | `include: []`, `tools: []` sent by Python, omitted by Rust → python-only (empty ≡ absent for the API) → normalizer rule `EMPTY_MEANS_ABSENT` |
| 2 | `explicit_gpt6_luna` | 1 | same as 1; no implicit `reasoning`/`text.verbosity` on either side for `gpt-6-luna` |
| 3 | `model_settings` | 1 | none: all 20 settings, `?q=1` query and `x-parity` header identical |
| 4 | `function_tools` | 2 | pydantic `title` annotations in the tool schema → python-only (JSON-Schema annotation) → normalizer drops `title` inside schema subtrees; tool result object: Python `str(result)` vs Rust JSON → language-inherent → scenario waiver `/body/input/*/output` |
| 5 | `structured_output` | 1 | none (strict `text.format` schema identical) |
| 6 | `handoffs` | 2 | handoff output `{"assistant": "History Tutor"}` spacing → python-only → normalizer canonicalizes JSON carried in strings |
| 7 | `agents_as_tools` | 3 | **rust-bug**: Python's default `as_tool` schema carries `"description": "Default input schema for agent-as-tool calls."` → fixed in `src/agent.rs` |
| 8 | `guardrails` | 4 | none; parallel input guardrail: both SDKs send the main-agent request alongside the guardrail agent's; blocking: guardrail request only; output guardrail: main request only; same tripwire errors |
| 9 | `human_in_the_loop` | 4 | none; approve → `function_call_output` with the tool result; reject → `Tool execution was not approved.` on both |
| 10 | `sessions` | 6 | none; SQLite history (3 turns), `previous_response_id` (2 turns), `conversation_id` (1) |
| 11 | `streaming` | 2 | none; `stream: true` bodies identical; 18-event sequence identical in the shared vocabulary (raw API events, `tool_called`, `tool_output`, `message_output_created`, `agent_updated`) |
| 12 | `hosted_tools` | 1 | **rust-bug**: Rust added `code_interpreter_call.outputs` to `include`; upstream never has (checked `git log -S`) → removed in `src/tool.rs` |
| 13 | `inputs_media` | 1 | none (remote/inline image, remote/inline PDF parts pass through identically) |
| 14 | `prompts_dynamic` | 2 | none (`prompt {id, version, variables}`; dynamic instructions) |
| 15 | `chat_completions` | 2 | none (messages mapping, tools format, tool-result message) |
| 16 | `retry` | 2 | none (429 then success; one SDK-level retry on both, client-level retries off in Python) |
| 17 | `tracing_export` | 3 | span order inside the ingest batch differs → expected (a batch is a set, linked by ids) → normalizer sorts `data` by object/type/name; trace + response, function, agent and custom spans identical |
| 18 | `reasoning_items` | 2 | none (the reasoning item is resent with the call and its output) |

## Waivers (`parity/waivers.json`)

| scope | pointer | reason |
| --- | --- | --- |
| `*` | `/headers/user-agent` | SDK identity |
| `*` | `/headers/x-stainless-*` | Python `openai` client build metadata |
| `function_tools` | `/body/input/*/output` | object tool result: Python `str()` (pydantic repr) vs Rust JSON; opaque string to the API |

Normalizer rules (documented in `docs/parity-harness.md`): transport headers dropped; ids and
`created_at`/`started_at`/`ended_at` masked; `required`/`include` sorted; empty top-level
`tools`/`include` ≡ absent; `title` dropped inside schema subtrees; JSON inside strings
canonicalized; trace batches sorted.

## Rust fixes (phase 3)

| scenario | change | test |
| --- | --- | --- |
| 7 | default agent-as-tool parameter schema gains Python's description (`src/agent.rs`) | `tests/py_parity_request_shapes.rs::the_default_agent_tool_schema_matches_python` |
| 12 | no `include` entry for the code interpreter tool (`src/tool.rs`) | `tests/py_parity_request_shapes.rs::only_file_and_web_search_add_include_entries` |

Known divergences kept on purpose (not reopened): rebuilt-agent resume runs the given graph's
tools; a JSON-restored state cannot fall back to the original agent.

## Harness notes

- Scripted replies can be matched to the requesting agent (`"when": {"instructions_contains"}`)
  so concurrent requests (parallel guardrails, agent tools) get the right answer regardless of
  arrival order; `compare.json` `{"order_free": true}` pairs such records by path + instructions.
- Streamed requests receive the scripted object as SSE (`response.created` … `response.completed`,
  text and function-call argument deltas) built by the server; Chat Completions replies stream
  as chunks.
- One `cargo test` process per scenario keeps `OPENAI_BASE_URL` and the tracing switch
  process-local on the Rust side.

## Unresolved questions

None for capture mode. Live behaviour (phase 4) is the next step.
