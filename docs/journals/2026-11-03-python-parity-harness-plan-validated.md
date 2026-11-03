# 2026-11-03 — Python parity harness plan written and validated

## What happened

The port reached 0.24.2 with every upstream test case ported, covered or classed
Python-only, but nothing had yet compared what the two SDKs actually *send* to the API,
and no run had touched a real model. The user asked for a plan to test the full feature
set against the Python SDK, using the key in `.env`, on `gpt-6-luna`, with the request
payloads compared before any real call.

Plan: `plans/261103-0651-python-parity-harness-live-gpt6-luna/` (five phases).

## Key decisions

- One local capture server behind `OPENAI_BASE_URL` for both SDKs; scripted answers so
  the second-turn requests are comparable; normalized diff with reasoned waivers. No
  tokens spent until every diff is classified and Rust bugs are fixed.
- Rust scenarios are env-gated tests run one `cargo test` process per scenario, so the
  process-global env (`OPENAI_BASE_URL`, tracing switch) stays independent and the real
  default-provider path is exercised.
- Python SDK installed by `uv` into `~/.cache/oars-build/py-venv`; the user's checkout
  stays untouched and the hook-blocked `.venv` token never appears in a command.
- Real key: phase 4 only. One names-only `GET /v1/models` during planning confirmed the
  account lists `gpt-6-luna` (and `gpt-5.6-luna`, both SDKs' default).
- Live comparison is on shape (tools called, handoff target, schema parse, usage, event
  types) with a shared item-type vocabulary, never on free text. Budget ≈70 calls.
- Confirmed live divergences are fixed in Rust unless upstream is the bug.
- Release only if `src/` changes; same gated protocol as 0.19–0.24.

## Validation

Seven interview questions, all recommended options accepted. Verification checked 52
cited paths/symbols; one failure (`src/tool/strict_schema.rs` → `src/strict_schema.rs`)
corrected. Whole-plan sweep: zero unresolved contradictions.

## Lessons

- A plan that cites file paths from memory needs a grep pass before the interview; the
  one wrong path was the only module not opened during the session.
- The first-scenario success criterion was initially "diff is empty", which would have
  made phase 1 fail on a legitimate finding (Python sends `"stream": false` explicitly).
  Phase gates should assert the pipeline works, not that nothing was found.
