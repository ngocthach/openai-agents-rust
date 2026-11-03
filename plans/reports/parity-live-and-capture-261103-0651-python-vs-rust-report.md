# Python SDK 0.24.0 vs Rust port: parity harness results (capture + live `gpt-6-luna`)

Date: 2026-11-03. Plan: `plans/261103-0651-python-parity-harness-live-gpt6-luna/`.
Harness usage: `docs/parity-harness.md`. Capture-mode detail (18 scenarios, every diff
classified): `plans/reports/parity-capture-matrix-261103-0651-python-vs-rust-report.md`.

Reproduce:

```sh
bash parity/run.sh setup
bash parity/run.sh capture all            # no API call
bash parity/run.sh live all               # real key from .env, PARITY_MODEL=gpt-6-luna
```

## Summary

| stage | scenarios | result |
| --- | --- | --- |
| capture (request payloads) | 18 | 18 identical after normalization; 2 Rust fixes |
| live (`gpt-6-luna`, behaviour shape) | 12 | 12 match across SDKs; 1 Rust fix |

Rust changes (all in 0.24.3):

| found by | change | regression test |
| --- | --- | --- |
| capture `agents_as_tools` | default agent-as-tool schema carries Python's description; input builders get the summary built from it | `tests/py_parity_request_shapes.rs`, `tests/py_core3_guardrails_agent_setup.rs` |
| capture `hosted_tools` | no `code_interpreter_call.outputs` in `include` (upstream never sends it) | `tests/py_parity_request_shapes.rs` |
| live `session_two_turns` | `RunResult::input` after a client-managed session run is the prepared input (history + new message), so `to_input_list()` carries the conversation | `tests/py_parity_session_result_input.rs` |

## Live matrix (`gpt-6-luna`, `max_output_tokens` 2000, one pass of 12 + 4 re-runs)

| scenario | requests py/rs | outcome on both SDKs |
| --- | --- | --- |
| hello | 1/1 | haiku; `reasoning_item` + `message_output_item` |
| tools_two_turn | 2/2 | `get_weather(city="Tokyo")` called once, then the answer |
| structured_output | 1/1 | `CalendarEvent` parsed; participants `[alice, bob]` |
| handoff | 2/2 | `Triage->Spanish Assistant`; last agent `Spanish Assistant` |
| streaming | 2/2 | `agent_updated`, `tool_called`, `tool_output`, `message_output_created`; raw stream starts `response.created`, ends `response.completed` |
| approval_resume | 2/2 | interruptions `[1, 0]`, tool ran once after the JSON round trip |
| session_two_turns | 1/1 (2nd turn) | answer names the stored fact; 4 conversation items in the session and in `to_input_list()` |
| model_settings_reasoning | 1/1 | `reasoning.effort=low`, `summary=auto`, `verbosity=low` accepted; no reasoning item returned |
| agents_as_tools | 3/3 | `translate_to_spanish` called once (inner run), then the answer |
| web_search | 1/1 | web search call(s) made, 1 citation annotation in the answer |
| guardrail_tripwire | guardrail only | `InputGuardrailTripwireTriggered` on both (blocking guardrail) |
| unsupported_param | 0/0 | HTTP 400 on both: `Unsupported parameter: 'temperature' is not supported with this model.` |

Model facts learned: `gpt-6-luna` rejects `temperature`; it returns a `reasoning_item` before the
message in some runs and not others, at the default settings — the item flipped sides between
re-runs of `handoff` and `streaming`, so the comparison ignores reasoning items
(`new_item_types_stable`, `stream_item_events_stable`, `session_items_stable`). With
`max_output_tokens` 400 the haiku run ended `incomplete` on both SDKs with the same terminal
event error (`response.incomplete`, `max_output_tokens`), identical handling; the cap was
raised to 2000.

Spend for the whole phase (dry runs, the first pass, 4 re-runs, the final pass): about 55
calls; the final pass alone used 16 requests per SDK, 10,015 / 18,050 input tokens and
1,354 / 1,119 output tokens (py / rs; web search dominates the input side). Dashboard total
to be read by the user; the per-pass estimate (≈70 calls) was not exceeded.

## Remaining differences (kept, documented)

- An object returned by a function tool reaches the model as Python's `str()` of it (pydantic
  repr) in Python and as JSON here (waived in capture `function_tools`).
- Rebuilt-agent resume runs the given graph's tools; a JSON-restored state cannot fall back to
  the original agent (0.24.2 entries).
- `RunState::original_input` (what `to_state()` stores) stays the caller's new input; Python
  stores the prepared input there too. Rust's resume path reads that field as the input to
  persist, so changing it needs its own design; only `RunResult::input` changed.

## Unresolved questions

- Actual price of `gpt-6-luna`: read the usage dashboard for 2026-11-03.
