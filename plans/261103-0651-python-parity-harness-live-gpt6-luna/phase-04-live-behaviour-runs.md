---
phase: 4
title: Live behaviour runs
status: completed
priority: P2
dependencies:
  - 3
---

# Phase 4: Live behaviour runs

## Overview

Run a curated subset against the real API with `gpt-6-luna` in both SDKs and compare
behaviour shape. Uses `OPENAI_API_KEY` from `.env` (loaded by the runner, never echoed).

## Requirements

- Functional: each live scenario writes `parity/out/live/<scenario>/<sdk>.json` with the
  shape fields below; `parity/compare_live.py` applies the scenario's `expect.json`.
- Non-functional: bounded spend (`max_output_tokens` 300, ≤ ~70 calls total, `web_search`
  once per SDK); every call authenticated with the real key only inside this phase;
  `parity/out/` stays gitignored.

## Architecture

Shape record per run (both SDKs write the same JSON keys):

```json
{"scenario": "...", "sdk": "py|rs", "model": "gpt-6-luna",
 "final_output": "...", "final_output_parsed": {...} | null,
 "last_agent": "...", "new_item_types": ["message_output", "tool_call", ...],
 "tool_calls": [{"name": "...", "arguments": {...}}],
 "handoffs": ["from->to"], "interruptions": [1, 0],
 "usage": {"requests": n, "input_tokens": n, "output_tokens": n},
 "response_ids": n, "stream_event_types": [...], "error": null | {"type": "...", "status": 400},
 "elapsed_ms": n}
```

`new_item_types` and `stream_event_types` use the shared vocabulary from capture scenario
11 (Python item names, API event `type` strings, `agent_updated:<name>`); the Rust writer
maps its variants to those names so the comparison is on behaviour, not naming.

`expect.json` per scenario lists assertions evaluated on each SDK's record *and* across
SDKs: `tool_calls[*].name` equal, `last_agent` equal, `new_item_types` equal, `usage.requests`
equal, `error.type` equal, `final_output_parsed` non-null on both, etc. Free text is never
compared.

Runner: `bash parity/run.sh live <scenario|all>` → loads `.env` (`set -a; . ./.env; set +a`),
exports `PARITY_MODEL=${PARITY_MODEL:-gpt-6-luna}`, `OPENAI_AGENTS_PARITY_LIVE=1`, runs the
Python script then `cargo test --test parity_live <scenario>`, then `compare_live.py`.
Tracing stays enabled in live mode so traces appear in the dashboard for both SDKs
(`trace_id` recorded, not compared).

## Live scenario list

| # | scenario | assertions (both SDKs, then across) |
| --- | --- | --- |
| L1 | `hello` | `final_output` non-empty; `usage.requests == 1`; `usage.output_tokens > 0`; response model echoes `gpt-6-luna` |
| L2 | `tools_two_turn` | `get_weather(city)` called with a parseable city; `usage.requests == 2`; tool-call names equal across SDKs |
| L3 | `structured_output` | `final_output_parsed` matches schema (calendar event: name, date, participants); non-null on both |
| L4 | `handoff` | `last_agent` is the language agent both sides; `handoffs == ["Triage->Spanish"]` |
| L5 | `streaming` | `stream_event_types` starts `response.created`, ends `response.completed`; run-item kinds equal across SDKs |
| L6 | `approval_resume` | `interruptions == [1, 0]`; tool ran once after approval; `usage.requests == 2` |
| L7 | `session_two_turns` | second-turn `usage.requests == 1`; `to_input_list()` length equal across SDKs; final output references the first turn's fact (assert the fact token appears) |
| L8 | `model_settings_reasoning` | `reasoning.effort: low`, `verbosity: low` accepted (no error) on both; a reasoning item present if the API returns one on both |
| L9 | `agents_as_tools` | nested tool name called on both; `usage.requests == 3` |
| L10 | `web_search` | one call per SDK; an annotation/citation present on both |
| L11 | `guardrail_tripwire` | `error.type == "InputGuardrailTripwireTriggered"` (Rust equivalent variant) on both; `usage.requests` equal |
| L12 | `unsupported_param` | `temperature` on `gpt-6-luna`: same outcome on both (both 200, or both 400 with equal `error.status`) — documents the model's parameter rules |

Add a scenario only if it tests behaviour the capture matrix cannot (model-dependent).

## Related Code Files

- Create: `tests/parity_live.rs`; `parity/scenarios/live/<scenario>/{py.py,expect.json}`;
  `parity/compare_live.py`; `parity/shape.py` (Python record writer) and a Rust `shape`
  helper inside `tests/parity_live.rs`
- Modify: `parity/run.sh` (live subcommand), `docs/parity-harness.md`

## Implementation Steps

1. Dry run L1 with `PARITY_MODEL=gpt-5.6-luna` first (known-good default) to validate the
   plumbing, then switch to `gpt-6-luna`.
2. Implement L1–L6, run, fix harness issues; then L7–L12.
3. After the first full pass, read the OpenAI usage dashboard total and record the spend in
   the report; stop and report if a single pass exceeds the estimate by 3×.
4. Any cross-SDK mismatch: re-run that scenario twice to separate model variance from SDK
   divergence; only a 3/3 mismatch counts as a divergence.
5. Record every divergence with both JSON records (they hold no secrets, but they stay
   under `parity/out/`, which is gitignored; copy the relevant fields into the report).
<!-- Updated: Validation Session 1 - divergence policy -->
6. Divergence policy (validated): a confirmed 3/3 divergence is a Rust change — fix it the
   phase-3 way (cause from the Python source, fix, test, gate) and re-run the scenario.
   Keep the Rust behaviour only when the Python side contradicts its own docs or an
   upstream test; record those as known gaps with the upstream line reference.

## Success Criteria

- [ ] All 12 scenarios produce records on both SDKs; `compare_live.py` prints a verdict per
      scenario.
- [ ] Cross-SDK assertions pass for L1–L9 (after any fixes); L10–L12 pass or are recorded
      with the model's actual response.
- [ ] Every confirmed divergence is either fixed with a test or recorded as an upstream
      bug with the Python line reference.
- [ ] Spend recorded; no key in any file under version control (`git grep -c "sk-"` only
      dummy values).

## Risk Assessment

- `gpt-6-luna` may reject parameters or behave differently from `gpt-5.6-luna` → L12 and
  the dry run make that explicit instead of failing mid-matrix.
- Rate limits → the runner runs scenarios sequentially; Rust tests use `--test-threads=1`
  in live mode.
- Model variance in tool-call arguments → compare names and argument keys, not values,
  except the city in L2 (prompt pins it).
- Live runs post traces to the dashboard with real data of these prompts only.
