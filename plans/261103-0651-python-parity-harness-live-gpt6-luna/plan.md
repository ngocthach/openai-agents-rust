---
title: 'Python SDK parity harness: payload diff, then live gpt-6-luna'
description: >-
  Compare the Rust port against the Python SDK feature by feature: first the
  exact request each SDK sends (no tokens spent), then a curated set of live
  runs on gpt-6-luna.
status: completed
priority: P2
branch: feat/rust-port-of-agents-sdk
tags:
  - parity
  - testing
  - live
blockedBy: []
blocks: []
created: '2026-11-02T23:56:39.208Z'
createdBy: 'ck:plan'
source: skill
---

# Python SDK parity harness: payload diff, then live gpt-6-luna

## Overview

Goal: prove, feature by feature, that `openai-agents-sdk` (Rust, 0.24.2) behaves like
`openai-agents-python` 0.24.0 (`/Users/ngocthach/myGit/openai-agents-python`, cd26bbbc).

Two stages, in this order:

1. **Payload parity (no API spend).** Both SDKs point `OPENAI_BASE_URL` at one local capture
   server. The server records every request body and answers from a per-scenario script, so
   both SDKs see identical model output and their *next* requests are comparable too. The
   normalized request bodies are diffed. The "final prompt" is this body: `model`,
   `instructions`, `input[]`, `tools[]`, `text.format`, `reasoning`, settings, headers.
2. **Live behaviour on `gpt-6-luna` (real key from `.env`).** A curated subset runs against
   the real API in both SDKs. Model text is non-deterministic, so the comparison is on
   shape: which tools were called, whether the handoff happened, whether output parses
   into the schema, usage, event-type sequences.

Stage 2 does not start until every stage-1 diff is classified (expected / Rust bug fixed /
waived with a reason).

Verified before planning: `.env` holds `OPENAI_API_KEY` (gitignored); a names-only
`GET /v1/models` with it lists `gpt-6-luna` and `gpt-5.6-luna` (both SDKs' default).
Neither SDK's default-settings table matches `gpt-6-luna`, so no implicit reasoning
settings should appear for it in either; `gpt-5.6-luna` maps to effort `none` in both.

## Scope challenge (decided, per the standing "choose for me" instruction)

- Existing code reused: Rust env-gated live tests (`OPENAI_AGENTS_DOCKER_LIVE` pattern),
  `OPENAI_BASE_URL` in both SDKs, `with_endpoint` on the Rust trace exporter and
  `BackendSpanExporter(endpoint=...)` in Python, `OPENAI_AGENTS_DISABLE_TRACING` in both,
  `openai_agents_sdk::testing::response_to_stream_events` as the SSE spec, the Rust examples
  that already mirror Python examples, upstream `integration_tests/openai/*` as the live
  scenario list, the existing gate script and release protocol.
- Minimum change set: one Python capture server, one normalizer/diff script, one runner
  script, two Rust test files, one Python scenario per feature. No new crate, no workspace
  change, no new Rust dependency.
- Complexity: ~25 new files, almost all small scenario files; one new directory `parity/`.
- Selected: **HOLD** scope; mode **fast** (both repos are local and already scouted; no
  researcher agents needed). Validation/red-team offered at handoff.

## Decisions

| Topic | Decision | Why |
| --- | --- | --- |
| Capture server | `parity/capture_server.py`, Python stdlib `http.server`, one process for both SDKs | No new Rust deps; Python runs anyway. |
| Routing | base URL per SDK and scenario: `http://127.0.0.1:PORT/<scenario>/<sdk>/v1` | Both clients append `responses` / `chat/completions` / `traces/ingest` to the base. `cargo test` runs scenarios concurrently; the path keeps records apart. |
| Records | `parity/out/<scenario>/<sdk>.jsonl` (one request per line); `parity/out/` gitignored | Live mode stores real responses; keep them out of git. |
| Scripted answers | `parity/scenarios/<scenario>/responses.json`: ordered list of Responses objects (or `{"status": 429}`); server serves entry *n* to the *n*-th request of each SDK; `stream=true` requests get the same object as SSE | Same model output for both → second-turn requests comparable. |
| Secrets | Capture mode uses `OPENAI_API_KEY=sk-parity-dummy`; live mode loads `.env` with `set -a; . ./.env; set +a` inside the runner script only; the server records `authorization: present`, never the value; nothing echoes the key | Earlier dummy-only rule now scoped: the real key is used only in phase 4 and the one-time models check. |
| Rust scenarios | `tests/parity_capture.rs` and `tests/parity_live.rs`; one `#[tokio::test]` per scenario, named like the scenario dir; each returns early unless `OPENAI_AGENTS_PARITY_CAPTURE=1` / `OPENAI_AGENTS_PARITY_LIVE=1` is set. The runner starts **one `cargo test --test parity_capture <scenario>` process per scenario** with `OPENAI_BASE_URL=http://127.0.0.1:PORT/<scenario>/rs/v1` set for that process; tests use the default provider (no injected client) | Env is process-global (`OPENAI_BASE_URL`, `OPENAI_AGENTS_DISABLE_TRACING` is read once); one process per scenario keeps them independent, compiles once, and exercises the real `OpenAIProvider` + `get_default_model` path the matrix is meant to prove. Matches the Docker live-test gate pattern; no workspace change. |
| Python scenarios | `parity/scenarios/<scenario>/py.py`, run with `uv run --project <py-repo> --python 3.13`, `UV_PROJECT_ENVIRONMENT=$HOME/.cache/oars-build/py-venv` | Keeps the user's checkout untouched; the scout-block hook rejects commands containing `.venv`. |
| Normalizer | `parity/compare.py`: sort keys; drop `created_at`, ids matching `^(resp|msg|fc|call|rs|ws|trace|span)_`, `user-agent`, `x-stainless-*`; sort `required` arrays; everything else is a real diff; `parity/waivers.json` holds accepted diffs with reasons; exit 1 on unwaived diffs | Only semantic differences surface. |
| Tracing | `OPENAI_AGENTS_DISABLE_TRACING=1` for every scenario except `tracing_export`, which points both exporters at `<base>/traces/ingest` | Otherwise Python posts traces to the real API with the dummy key. |
| Live model | `PARITY_MODEL` env, default `gpt-6-luna`; `max_output_tokens` 300; ≤ 12 scenarios × 2 SDKs × ≤ 3 calls ≈ 70 calls; `web_search` once per SDK | "Some test cases", bounded spend. Price of `gpt-6-luna` unknown → check the usage dashboard after the first run. |
| Runner | `parity/run.sh {capture|live} [scenario|all]`, absolute paths for both repos, `CARGO_TARGET_DIR=$HOME/.cache/oars-build/parity` | Shell hook blocks commands naming the build dir; the script owns it. |
| Markdown | Usage doc `docs/parity-harness.md`; results in `plans/reports/` | Repo rule: markdown only in `docs/` or `plans/`. |

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Harness setup](./phase-01-harness-setup.md) | Completed |
| 2 | [Payload parity matrix](./phase-02-payload-parity-matrix.md) | Completed |
| 3 | [Fix divergences](./phase-03-fix-divergences.md) | Completed |
| 4 | [Live behaviour runs](./phase-04-live-behaviour-runs.md) | Completed |
| 5 | [Report and release](./phase-05-report-and-release.md) | Completed |

Phases run in order. Phase 4 is gated on phase 3's sign-off.

## Acceptance criteria

- Every scenario in the phase-2 matrix has a Python and a Rust side and a diff verdict.
- Zero unwaived payload diffs; every waiver names the reason.
- Live runs on `gpt-6-luna` pass their shape assertions in both SDKs, or the difference is
  recorded as a known gap in `docs/project-changelog.md`.
- The gate script passes; no secret in git (`git grep sk-` finds only dummy keys).
- If Rust code changed: new version published to crates.io, tagged locally, docs.rs green.

## Dependencies

- Builds on `plans/261029-1858-rust-port-of-openai-agents-sdk/` (implemented; no blocking
  relationship, nothing there changes).
- External: `uv` (installed), Python 3.13 (installed via uv), the OpenAI key in `.env`,
  network access to `api.openai.com` for phase 4 only.

## Out of scope

- Realtime, voice, sandbox, Codex and MCP-stdio parity (need audio, containers, extra
  processes); the existing Docker live tests cover sandbox.
- Responses WebSocket transport (needs a WS capture server).
- Pixel-exact text equality in live runs.

## Validation Log

### Session 1 — 2026-11-03
**Trigger:** `/ck:plan validate` chosen at the post-plan handoff (fast mode ran no gates).
**Questions asked:** 7

### Verification Results
- **Tier:** Full (5 phases)
- **Claims checked:** 52 (24 Python paths, 19 Rust paths, 9 symbols/env names)
- **Verified:** 51 | **Failed:** 1 | **Unverified:** 0

#### Failures
1. [Fact Checker] `src/tool/strict_schema.rs` — not found; actual `src/strict_schema.rs`
   (`ensure_strict_json_schema`). Corrected in phase 3 after Q1.

Verified symbols: `OpenAIClient::with_base_url` (`src/models/openai_client.rs:142`),
`response_to_stream_events` (`src/testing.rs:171`), `Agent::with_prompt` (`src/agent.rs:439`),
`Agent::as_tool` (`src/agent.rs:1269`), `RunResult::to_input_list` (`src/result.rs:160`),
`run_in_parallel` on both guardrail types (`src/guardrail.rs:84`, Python `guardrail.py:100,217`),
`RECOMMENDED_PROMPT_PREFIX` (`src/extensions/handoff_prompt.rs:5`),
`InputGuardrailTripwireTriggered` (`src/error.rs:121`), Python item-type literals
(`items.py:176,306,389,499`), no existing use of `OPENAI_AGENTS_PARITY_*` / `PARITY_MODEL`.

#### Questions & Answers

1. **[Assumptions]** Verification found the plan cites `src/tool/strict_schema.rs`, but the
   Rust strict-schema module is `src/strict_schema.rs`. Correct the plan to the real path?
   - Options: Yes, use src/strict_schema.rs | Leave as is
   - **Answer:** Yes, use `src/strict_schema.rs`
   - **Rationale:** phase 3 names candidate files for fixes; a wrong path costs a rediscovery.
2. **[Risks]** Phase 4 budgets the live runs at about 70 calls to gpt-6-luna (12 scenarios × 2
   SDKs × up to 3 calls, 300 output tokens each) and stops if a pass costs 3× the estimate.
   gpt-6-luna's price is unknown until the first pass. Is that spend policy acceptable?
   - Options: Yes, run the full pass | Dry run first on gpt-5.6-luna | Smaller set: L1–L6 only
   - **Answer:** Yes, run the full pass
   - **Rationale:** sets the spend ceiling; the L1 dry run on the default model stays as a
     plumbing check, not an approval gate.
3. **[Tradeoffs]** The real OPENAI_API_KEY from .env is used only in phase 4 (live runs);
   phases 1–3 use a dummy key against the local capture server. Keep it that way?
   - Options: Phase 4 only | Allow earlier spot checks
   - **Answer:** Phase 4 only
   - **Rationale:** no real request leaves the machine before the payload matrix is clean.
4. **[Scope]** Phase 2 lists 11 'must' scenarios and 7 'should'. How much of the matrix
   should this plan cover?
   - Options: All 18 | Must-set only (11) | Must-set now, should-set after live runs
   - **Answer:** All 18
   - **Rationale:** full feature coverage; the should-set costs no API spend.
5. **[Tradeoffs]** When a live run on gpt-6-luna shows Python and Rust behaving differently
   (confirmed 3/3 re-runs) and the Rust behaviour is defensible, what should phase 4/5 do?
   - Options: Match Python unless upstream is a bug | Record as known gap, no code change |
     Decide per case at the end
   - **Answer:** Match Python unless upstream is a bug
   - **Rationale:** parity is the goal; live divergences loop back through phase 3's fix +
     test + gate flow; only upstream bugs (contradicting its docs or tests) are kept as gaps.
6. **[Scope]** Phase 5 releases a new crate version (0.24.3 patch, or 0.25.0 if a public
   contract changes) only if phase 3 changed code under src/. Is that the right release policy?
   - Options: Yes, release if src/ changed | No release in this plan | Always release
   - **Answer:** Yes, release if src/ changed
   - **Rationale:** same gated protocol as 0.19–0.24; harness-only commits land without a bump.
7. **[Architecture]** Phase 1 installs the Python SDK with uv into
   `~/.cache/oars-build/py-venv` (UV_PROJECT_ENVIRONMENT), leaving the Python checkout
   untouched. OK?
   - Options: Yes, cache dir | Use a .venv inside the Python checkout
   - **Answer:** Yes, cache dir
   - **Rationale:** keeps the checkout clean and the blocked `.venv` token out of commands.

#### Confirmed Decisions
- Strict-schema path: `src/strict_schema.rs`.
- Live spend: one full 12-scenario pass, ≤3 re-runs per mismatch, stop at 3× estimate.
- Real key: phase 4 only.
- Matrix: all 18 capture scenarios.
- Live divergence policy: fix Rust to match Python unless upstream is the bug.
- Release: only if `src/` changed; patch vs minor by contract impact.
- Python env: uv cache dir, checkout untouched.

#### Action Items
- [x] Phase 3: fix the strict-schema path.
- [x] Phase 4: add the divergence policy to the steps (loop back to phase 3 flow).
- [x] Phase 5: state that live-run fixes are part of the release decision.

#### Impact on Phases
- Phase 3: candidate file list corrected.
- Phase 4: step 5 now routes confirmed divergences to a fix, not only a record.
- Phase 5: release trigger covers phase-3 and phase-4 code changes.

### Whole-Plan Consistency Sweep
- Files reread: plan.md, phase-01-harness-setup.md, phase-02-payload-parity-matrix.md,
  phase-03-fix-divergences.md, phase-04-live-behaviour-runs.md, phase-05-report-and-release.md
- Decision deltas checked: 7
- Reconciled stale references: 3 (strict-schema path; phase-4 divergence handling;
  phase-5 release trigger)
- Unresolved contradictions: 0
