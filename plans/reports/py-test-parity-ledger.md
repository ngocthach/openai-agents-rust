# Python test suite vs Rust port: reconciled parity ledger

> **Round 8 update (0.11.0).** After a case-by-case re-audit of every file that still had Python-only cases, 441 more moved out of that bucket (380 to ported, 61 to covered). Measured whole-suite state: **12,641 cases in 425 files = 8,463 ported (66.9%) + 665 covered (5.3%) + 3,513 python-only (27.8%) + 0 not ported + 0 unaccounted.** Ported + covered = 9,128 (72.2%). The 3,513 python-only cases keep one reason per group in `plans/reports/py-test-parity-core8-*.md`; where an agent accepted an earlier reason after a spot check instead of reading every case (notably `r8g7`, 823 cases, and part of `r8g5`) the report says so, so some of those reasons may still be too generous. Reasons that are real feature gaps, not Python-only mechanics, are listed in the changelog. The tables below are the pre-round-6 snapshot kept for history.


Generated read-only from `/Users/ngocthach/myGit/openai-agents-python/tests/` (424 `.py` files), the 50 `plans/reports/py-test-parity-*.md` reports and `tests/*.rs`. Companion data: `py-test-parity-ledger.csv` (one row per Python file plus one GROUP row).

## 1. Headline totals

The Python suite has **12,638 test cases in 392 files** (392 test files; 32 helper modules with zero tests are listed in section 9). Counted: 12,515 by `pytest --collect-only` in a venv with every optional extra installed, plus 123 by AST for `tests/test_run_config.py`, which pytest cannot collect (a parametrize id calls a `__len__` that raises).

What the reports let us say about those cases, using each report's own bucket numbers (latest report wins; see section 2):

| bucket | cases | share of the suite |
|---|--:|--:|
| ported (a Rust test asserts the same scenario) | 6,416 | 50.8% |
| covered (an existing Rust test asserts it) | 456 | 3.6% |
| **ported + covered** | **6,872** | **54.4%** |
| python-only (cannot exist in Rust) | 3,493 | 27.6% |
| not ported | 1,050 | 8.3% |
| classified but split not recorded (`unclassified`) | 5 | 0.0% |
| **sum of report buckets** | **11,420** | 90.4% |
| not in any report bucket: parametrize cases the reports did not expand, net of over-counts (section 4) | 1,202 | 9.5% |
| memo, overlaps the rows above: cases in files with no usable per-file split (3 gap files with no row = 16, 5 grouped models files = 92 by pytest, of which 68 are in the sum of report buckets via the group row) | 108 | 0.9% |
| check: 11,420 + 1,202 + 16 (the 3 gap files; the 5 grouped files' remainder is inside the unaccounted figure) = 12,638 | | |

**The one number.** **6,872 of 12,638 cases (54.4%) are ported or covered** (a lower bound, as are the python-only and not-ported counts: the residual below belongs to all four buckets), 3,493 (27.6%) are python-only, 1,050 (8.3%) are explicitly not ported, and 1,223 (9.7%) are unaccounted, unclassified or unknown (the 1,202 net unaccounted + 16 unclassified gap-file cases + 5 classified-without-split). On only the cases the reports classified (11,420): ported+covered 60.2%, python-only 30.6%, not ported 9.2%.

Estimate (labelled as such): most unaccounted cases are parametrized expansions of defs that the reports counted once and the reports state follow the def's bucket. Allocating the 1,204 positive unaccounted cases pro rata to each file's own bucket mix (files with no classified bucket excluded: 1) gives about ported 7,112, covered 524, python-only 3,853, not ported 1,130, i.e. ported+covered about 60.4%, not ported about 8.9% of 12,638. This is not a measured number.

Files: 43 files still have at least one not-ported case; **3 files are real gaps** (no row in any report and no Rust citation: `tests/docs/test_translate_docs.py`, `tests/fastapi/test_streaming_context.py`, `tests/sandbox/test_tar_workspace.py`; 16 cases). Separately, 5 `tests/models/*` files have a row and a citation but no per-file split (one group row; 92 cases by pytest, 68 by the report), so their buckets are `unknown`.
18 `#[ignore]` attributes remain in `tests/*.rs` (10 files, listed in section 5); the ported column includes the Rust tests the reports counted as ported-but-ignored (79 at report time, many since un-ignored).

## 2. Method

1. Python counts: `pytest tests --collect-only -q --continue-on-collection-errors` in the supplied venv, after installing the test extras (numpy, websockets, litellm, docker, pyyaml, aiohttp, twilio, sqlalchemy, greenlet, aiosqlite, asyncpg, redis, fakeredis, graphviz, cryptography, pymongo, dapr, grpcio, fastapi, inline-snapshot, pytest-asyncio, pytest-mock, pytest-xdist). 391 files via pytest (12,515 cases); `tests/test_run_config.py` via AST with parametrize expansion (123). A cross-check AST pass disagreed with pytest on 21 files (non-literal parametrize, fixture params), so reports that used AST or `grep def test_` undercount those files; the CSV `method` column records the method used by this ledger.
2. Reports: all tables with a Python file in the first column were normalised (about 1,200 table rows over 50 reports). For each file the latest report (by file mtime) that covers the whole file wins. Multi-round files were combined by hand and are marked `source_reports`: `test_max_turns` (core-b + core2-runstep), `test_trace_processor` (core3-tracing + core4-tracerunner), `test_programmatic_tool_calling` (shelltools + core4), the four function-tool files (core3-functiontools + core5-leftover3, split inferred), `test_advanced_sqlite_session` / `test_encrypt_session` (fork D table + D2 remainder pass), `test_mongodb_session` / `test_dapr_session` (fork E table + E2), `test_sqlalchemy_compaction` (E2 supersedes C2), and the memory-core files (C2/C4/C5/C6).
3. Bucket mapping: ported = ported + ported-but-ignored + cross-area tests that exist but are `#[ignore]`d; not ported = unported + cross-area unportable + deferred + open + derived remainder (stated total minus buckets where the report notes list the rest as NOT PORTED). Whole-file python-only rows (ported = covered = not ported = 0) take the pytest count for every such file (9 files, 150 cases beyond the report's def counts: docker_removal x3, twilio x2, app_server x2, memory_imports, spans_impl; test_update_rclone_pin and test_code_change_verification_runner likewise).
4. Basis caveat: each report counts in its own unit (defs, or defs with parametrize expanded). `report_total` in the CSV is the sum of that row's buckets, `unaccounted` = pytest total - report_total (negative = the report over-counted). Files in def basis therefore carry unaccounted cases.
5. Verification: for every file with ported/covered > 0 the ledger checks that a `tests/**/*.rs` file cites the Python file name (strict match on `<name>.py`) and counts that file's `#[test]` functions; report-named Rust files were checked for existence; per-file results are in the CSV `verification` column.

## 3. Per-area table

| area | files | cases (pytest) | report-basis | ported | covered | python-only | not ported | unclassified | unaccounted (net) | unknown | ported+covered of cases |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| core | 126 | 4,449 | 4,452 | 2,638 | 185 | 1,232 | 397 | 0 | -3 | 0 | 63.5% |
| models | 35 | 1,038 | 779 | 629 | 6 | 144 | 0 | 0 | 259 | 92 | 61.2% |
| tracing | 26 | 487 | 462 | 263 | 2 | 193 | 4 | 0 | 25 | 0 | 54.4% |
| mcp | 23 | 617 | 556 | 269 | 31 | 189 | 67 | 0 | 61 | 0 | 48.6% |
| memory | 27 | 1,196 | 994 | 554 | 100 | 265 | 70 | 5 | 202 | 0 | 54.7% |
| realtime | 31 | 664 | 571 | 353 | 23 | 195 | 0 | 0 | 93 | 0 | 56.6% |
| voice | 15 | 282 | 166 | 94 | 18 | 54 | 0 | 0 | 116 | 0 | 39.7% |
| sandbox | 74 | 3,043 | 2,630 | 1,416 | 58 | 644 | 512 | 0 | 407 | 6 | 48.4% |
| extensions | 8 | 329 | 287 | 200 | 33 | 54 | 0 | 0 | 42 | 0 | 70.8% |
| other | 27 | 533 | 523 | 0 | 0 | 523 | 0 | 0 | 0 | 10 | 0.0% |
| **total** | 392 | 12,638 | 11,420 | 6,416 | 456 | 3,493 | 1,050 | 5 | 1,202 | 108 | 54.4% |

Area rules: `tracing` also holds the top-level tracing tests; `extensions` = codex, hosted multi-agent, tool-output trimmer, handoff filters, visualization; `sandbox` includes `extensions/sandbox` provider tests; `other` = example, release and repo-CI script tests plus docs and fastapi. In the `models` row the 5 grouped files appear as the GROUP row (68 report-basis) with `unknown` showing their 92 pytest cases. `sandbox` unknown = `test_tar_workspace` (6), `other` unknown = docs (9) + fastapi (1).

## 4. Where the unaccounted cases are

Net 1,202 = 1,205 cases (incl. the 24 left over from the grouped models row) that exist in pytest but in no report bucket, minus 3 where a report counted more than pytest collects. Largest: sandbox 407, models 259, memory 202, voice 116, realtime 93. Per file see CSV `unaccounted`. Over-counts (report total above pytest): test_tool_guardrails.py -3.

## 5. Gaps, unknowns and unverified claims

### Files with no classification (real gaps)

| file | cases | state |
|---|--:|---|
| `tests/docs/test_translate_docs.py` | 9 | no row in any report, no Rust citation (tests the docs translation script; probably python-only but never classified) |
| `tests/fastapi/test_streaming_context.py` | 1 | no row, no citation (FastAPI streaming under a different async context; never classified) |
| `tests/sandbox/test_tar_workspace.py` | 6 | no row, no citation, and `src/` has no `shell_tar_exclude_args` equivalent (never classified) |
| `tests/models/{test_default_models,test_map,test_agent_registration,test_openai_client_utils,test_generation_metadata}.py` | 92 | cited by `tests/py_models_map_defaults.rs`, but reported only as one group row (`models-tracing:46`: 68 tests, 63 ported, 5 python-only); per-file split unknown |

### Claims that could not be fully verified

- `tests/test_output_guardrail_cancellation.py` (1): core3-guardrails-hooks counts it ported; no Rust test cites it and core5-leftover3 later says python-only (CancelledError handler). Conflict, unverified.
- Cited only by a report-named Rust file, not by name in any `tests/*.rs` (existence of the named file or test confirmed; per-case mapping not): `mcp/test_message_handler` (mcp_tests.rs), `realtime/test_openai_realtime_sip_model`, `test_agents_logging` (logger_verbose_stdout.rs), `test_extra_headers`, `test_prompt_cache_key` (core_prompt_cache_key.rs), `test_run_state_agent_identity` (py_core_a_run_loop.rs), `test_runner_guardrail_resume` (runner_tests.rs, weak), `test_tool_metadata` (core_audit_fixes_tools.rs + local_tools_tests.rs, weak). `test_oaiconv_resume_response_id` and `test_apply_diff_helpers` are verified in `src/` unit tests.
- Rust test counts cited in reports drifted for 6 files (py_core_a_run_loop 8 vs 9, py_core3_functiontools_schema 31 vs 34, py_deferred_writes_a/b/d/e 10/1/4/5 vs 11/3/13/13): the deferred-writes files grew because their `#[ignore]`d tests were un-ignored; no ported claim is contradicted.
- Approximate inputs: the 9 `tests/extensions/sandbox/*` rows come from `sandbox-providers` (`~` estimates, no per-test ledger); the function-tool split of core5-leftover3's 13 closed rows is inferred; `models/test_model_retry.py` (about 65 ported / about 34 python-only of 99, the report's own `~` figures) and `models/test_openai_retry_helpers.py` carry approximate splits; `test_openai_responses_compaction_session` buckets sum to 77 against 73 defs by the report's own admission.
- Cross-area ignored tests: 18 `#[ignore]` attributes remain: py_mcp_util_schemas 5, py_mcp_caching_approval 3, py_mcp_resume_binding 2, py_core_a_stack 2 (measurement helpers), py_sandbox_a_runtime_6 1, py_models_retry_stream 1, py_memory_conversations_remaining 1, py_ext_hosted_multi_agent 1 (live network), py_core3_shelltools_local_shell 1, py_core3_shelltools_collision_policy 1. The ledger counts the Python cases they port as ported because the reports do.

## 6. Remaining not-ported cases

| file | cases | not ported | reason (source) |
|---|--:|--:|---|
| `extensions/sandbox/test_blaxel.py` | 251 | 199 | not yet classified or ported (approximate): PTY, mounts, tar persist/hydrate error paths, preview ports (sandbox-providers:20) |
| `test_run_impl_resume_paths.py` | 198 | 183 | 183: resume-path groups (callback-retry history, session append recovery, pending-write validation, legacy approval fallback, handoff append recovery, terminal markers) left for a follow-up file py_core2_runstep_resume_b.rs that does not exist; checked not stale: py_core2_runstep_resume.rs still has only 3 tests (core2-runstep:25) |
| `extensions/sandbox/test_vercel.py` | 130 | 98 | not yet classified or ported (approximate): mostly S3 mount lifecycle, partly covered by sandbox_e2b_daytona_vercel_tests but not itemised (sandbox-providers:17) |
| `extensions/sandbox/test_modal.py` | 119 | 96 | not yet classified or ported (approximate): snapshot persistence, PTY, tar chunking (sandbox-providers:19) |
| `mcp/test_mcp_resume_binding.py` | 61 | 57 | 57 cross-area: RunState::from_json takes no agent (cases unportable until it does) (mcp-memory:36) |
| `extensions/sandbox/test_cloudflare.py` | 96 | 51 | not yet classified or ported (approximate): resume/mount lifecycle, PTY websocket, redaction (sandbox-providers:16) |
| `test_native_tool_failure_history.py` | 72 | 50 | 50 design gaps (native/hosted concurrency), deferred (core2-toolerrors:27) |
| `extensions/sandbox/test_e2b.py` | 80 | 35 | not yet classified or ported (approximate): PTY and mount remount ordering (sandbox-providers:14) |
| `test_agent_runner_streamed.py` | 208 | 34 | 34 deferred cross-area: 26 corrupted/legacy ownership snapshot validation missing in run_state wire, 8 chat-completions adapter conversation policy (models area) (core2-runner:19) |
| `extensions/sandbox/test_daytona.py` | 72 | 30 | not yet classified or ported (approximate) (sandbox-providers:15) |
| `memory/test_compaction_model_visibility.py` | 81 | 19 | 19 cross-area: compaction exchange ignores call_model_input_filter (core-a round 2 un-ignored 2 tests; split not restated) (mcp-memory-fork-c2-memory-remainder:25) |
| `test_strict_schema.py` | 69 | 18 | 18 (core-a, never restated) (core-a:114) |
| `test_error_logging_redaction.py` | 170 | 17 | 17 real divergences deferred (parallel input-guardrail / native concurrency design gaps) (core2-toolerrors:22) |
| `memory/test_openai_conversations_session.py` | 74 | 15 | 15: resume-replay cases need pending_session_write / RunState restore against an agent (13), no Program item type (1), pop-without-id divergence (1) (mcp-memory-fork-c6-conversations-session) |
| `test_run_step_execution.py` | 111 | 14 | 14: hosted-MCP approval identity/scope matrix (core2-runstep:24) |
| `memory/test_openai_responses_compaction_session.py` | 94 | 10 | 10: deferred compaction under pending approval, interleaved write races (cross-area, 7) + 3 unported (created_by replay, rollback budget via Runner, response-chain private fields) (mcp-memory-fork-c4-responses-compaction) |
| `memory/test_session_context_wrapper.py` | 15 | 10 | 10 cross-area: Rust sessions take no RunContext wrapper (mcp-memory-fork-c2-memory-remainder:23) |
| `test_hitl_error_scenarios.py` | 88 | 10 | 10 bug deferred (rebuild from pending approvals, changed approval while policy waits, shell validation) (core2-scopedapprovals:20) |
| `test_run_internal_items.py` | 54 | 10 | 10: programmatic structured tool errors (2), nested-history owned-item refs (6), fingerprint metadata (1), legacy tool-search output (1) (core2-runstep:21) |
| `extensions/memory/test_encrypt_session.py` | 66 | 9 | 9 cross-area: EncryptedSession cannot wrap an OpenAIResponsesCompactionSession outermost (needs a compaction-session hook) (mcp-memory-fork-d-sqlite-encrypt:8) |
| `test_pretty_print.py` | 16 | 9 | 9 remainder: core-b NOT YET PORTED, never restated (core-b:129) |
| `test_tool_batch_failure_history.py` | 64 | 9 | 9 design gaps (parallel guardrail / native concurrency), deferred (core2-toolerrors:23) |
| `test_items_helpers.py` | 40 | 8 | 8 remainder: core-b lists the rest as NOT YET PORTED, never restated (core-b:128) |
| `mcp/test_mcp_util.py` | 135 | 7 | 5 cross-area (no reject-open-objects strict-schema mode, ignored tests) + 2 unported (mcp-memory:21) |
| `test_tool_use_behavior.py` | 17 | 7 | 7 (core-a, never restated) (core-a:122) |
| `test_stream_events.py` | 14 | 5 | 5: ordering helpers in a private streaming module (core-a, never restated) (core-a:111) |
| `test_tool_guardrails.py` | 22 | 5 | 5 (core-a, never restated; report over-counts 25 cases vs pytest 22) (core-a:119) |
| `memory/test_compaction_suffix.py` | 46 | 4 | 4 cross-area: EncryptedSession over compaction ordering, call_model_input_filter (mcp-memory-fork-c2-memory-remainder:22) |
| `test_agent_scoped_approvals.py` | 104 | 4 | 4 bug deferred (converted-context approved invocation) (core2-scopedapprovals:19) |
| `test_tool_output_conversion.py` | 27 | 4 | 4 remainder: core-b NOT YET PORTED, never restated (core-b:130) |
| `extensions/sandbox/test_vercel_provider_contract.py` | 29 | 3 | not yet classified (approximate) (sandbox-providers:18) |
| `mcp/test_runner_calls_mcp.py` | 24 | 2 | 2 cross-area (hosted-MCP/runner) (mcp-memory:24) |
| `memory/test_sqlalchemy_compaction.py` | 30 | 2 | 2 cross-area: outer-encrypted prefix cleanup (1 test, ignored) (mcp-memory-fork-e-remote-stores) |
| `test_local_shell_tool.py` | 8 | 2 | 2 cross-area cases (ported tests ignored) (core3-shelltools:21) |
| `test_result_cast.py` | 16 | 2 | 2 remainder: core-b NOT YET PORTED, never restated (core-b:131) |
| `test_tool_approval_call_id_reuse.py` | 130 | 2 | 2: legacy changed-pending-call and one 1:1 mapping gap (core2-approvals:17) |
| `test_usage.py` | 23 | 2 | 2 (core-a, never restated) (core-a:127) |
| `tracing/test_trace_context.py` | 13 | 2 | 2: reload-no-reattach (cache not clearable) and the "Trace already exists" warning assertion (models-tracing:38) |
| `tracing/test_tracing_env_disable.py` | 23 | 2 | 2: log-redaction caplog tests (partly covered by models_redaction_tests.rs) (models-tracing:39) |
| `extensions/memory/test_advanced_sqlite_session.py` | 179 | 1 | 1 cross-area case left after the D2 remainder pass (documented divergence on stale-usage races, quoted custom session table test is ported but ignored) (mcp-memory-fork-d-sqlite-encrypt:7) |
| `mcp/test_mcp_approval.py` | 13 | 1 | 1 cross-area: approved MCP call not bound to its offering server (mcp-memory:23) |
| `test_process_model_response.py` | 29 | 1 | 1: local MCP function tool title on the run item (needs FakeMCPServer) (core2-runstep:23) |
| `test_run_state.py` | 469 | 1 | 1 legacy hosted-MCP case (core2-runstate:10) |

Total not ported 1,050 in 43 files. 512 of them are the sandbox provider files that the report marks 'not yet classified' (they may include cases already covered by `sandbox_e2b_daytona_vercel_tests` / `sandbox_modal_cloudflare_blaxel_tests`); the report says so itself.

## 7. Discrepancies between reports, and between reports and the tree

- Report totals vs pytest: 82 files have report-basis totals different from pytest (CSV `unaccounted`). Causes: reports using `grep def test_` without parametrize expansion (codex, realtime, voice, sandbox-a, models, tracing), AST with literal-only parametrize (mcp, memory), and fixture parametrization (sandbox, memory).
- Self-inconsistent report totals: sandbox-a 585+22+337 = 944 vs 943 stated (its own note); sandbox-providers python-only 102 vs 100 and not-yet-classified 512 vs 510; mcp-memory fork-b table rows sum to 235 defs vs 239; fork-c4 groups sum to 75-77 vs 73 defs and 98 vs 94 expanded; fork-c6 `test_session` groups 39 vs 40 defs; mcp-memory C3 total row (56/121) vs per-file rows (68/160 including test_compaction).
- core-b says 17 cases remain in the four 'finish' files (items_helpers, pretty_print, tool_output_conversion, result_cast); its own rows leave 23 (8+9+4+2). The ledger uses 23. No later report restates them; their Rust files (py_core_b_items_helpers 24, pretty_print 7, tool_output_result 12 tests) did not grow.
- core5-leftover3's sweep says no behavioural top-level file is left without a Rust port, yet core-a still lists not-ported remainders for test_stream_events (5), test_strict_schema (18), test_tool_use_behavior (7), test_usage (2), test_tool_guardrails (5), and core-b for the four files above; py_core_a_* test counts match core-a, so those remainders are current, not stale.
- test_output_guardrail_cancellation: ported (core3-guardrails-hooks) vs python-only (core5-leftover3).
- test_tool_choice_reset: core-a row has ported 7 for 6 cases; test_tool_guardrails: core-a counts 25 cases, pytest collects 22; test_runtime: sandbox-a combined row 165 defs (96+22+48 = 166).
- test_update_rclone_pin (13 vs pytest 19) and test_code_change_verification_runner (16 vs 19): core-b counted by AST; whole-file python-only so the ledger uses the pytest counts.
- test_streamed_terminal_output_backfill: report counts 5 with 2 'python-only' that are nested helper functions, not tests; ledger counts 3 ported of 3.
- test_agent_runner pytest count: core-a 240 vs core2-runner 232 (the AST count includes @function_tool-decorated test_tool_* helpers); ledger uses 232. test_run_state: 470 vs 469 (nested helper); corpus: 84 (core-a) vs 123 (current).
- Rust files named in reports that do not exist: core-b's 11 planned `py_core_b_*.rs` files (superseded by core2-core5 rounds), `py_core2_runstep_resume_b.rs` (183 test_run_impl_resume_paths cases are listed as not ported against it), `py_mcp_resources_prompts.rs` (report itself says it does not exist), `py_scratch_dbg.rs` (stray file, absent), `py_sandbox_a_docker_1..6` (range shorthand; files py_sandbox_a_docker_* exist). All other named Rust files exist.
- Stale ignore counts: mcp-memory reported 41 ignored tests and core-a round 2 un-ignored the run_internal ones; only 18 `#[ignore]` attributes remain. The deferred-writes ports counted as 47 cross-area (c3) and 12+18 (c5) are no longer ignored, so they stay in ported.
- Table vs addenda: fork-d's table says 22+1 unported, its 'Remainder pass (fork D2)' ports them all; fork-e's table says 4 unported, E2 ports them; ledger applies both. test_compaction_model_visibility: c2 says 1 green of 3 variants, core-a round 2 says its 2 ignored tests were un-ignored; split not restated, ledger keeps c2's 1 ported and flags possibly stale.
- Collision of Python file names (test_session, test_compaction, test_max_turns, test_testing, test_tracing, test_apply_patch_tool, test_model) was resolved by matching report counts and area to the right path; Rust citations for them are filtered by path/area.

## 8. Coverage of reports

All 50 `py-test-parity-*.md` reports were read for table rows; every one is used as a source or superseded: sources used in the CSV are listed per file. The older non-`py-test-parity` reports (e.g. `sandbox-providers-final-parity-report.md`, `sessions-memory-final-parity-report.md`) were not parsed, as instructed; `sandbox-providers-final-parity-report.md` says nothing was ported in its pass.

## 9. Helper modules with zero tests

`__init__.py`, `_fake_workspace_paths.py`, `conftest.py`, `deferred_session_writes/__init__.py`, `deferred_session_writes/helpers.py`, `fastapi/__init__.py`, `fastapi/streaming_app.py`, `fixtures/run_state/generate_corpus.py`, `mcp/__init__.py`, `mcp/helpers.py`, `mcp/model_compat.py`, `mcp/servers/legacy.py`, `mcp/servers/paginated.py`, `model_test_helpers.py`, `models/__init__.py`, `realtime/__init__.py`, `realtime/session_test_support.py`, `sandbox/__init__.py`, `sandbox/_apply_patch_test_session.py`, `sandbox/_docker_removal_helpers.py`, `sandbox/_filesystem_test_session.py`, `sandbox/integration___init__.py`, `sandbox/integration__helpers.py`, `sandbox/integration_test_model.py`, `test_responses.py`, `testing_processor.py`, `utils/factories.py`, `utils/hitl.py`, `utils/simple_session.py`, `voice/__init__.py`, `voice/helpers.py`, `voice/pipeline_test_models.py`

`tests/test_responses.py` and `tests/sandbox/integration_tests/test_model.py` carry a `test_` prefix but contain no tests (helper functions only; core-a also lists test_responses as a helper).

Status: DONE_WITH_CONCERNS
Summary: 12,638 Python test cases; ported 6,416, covered 456, python-only 3,493, not ported 1,050, unclassified 5, unaccounted 1,202, unknown 108; ported+covered 54.4% (lower bound).
Concerns: reports count in different units (defs vs expanded), leaving a net unaccounted block; 3 files never classified; sandbox provider rows are approximate; some not-ported remainders are stale-prone.
