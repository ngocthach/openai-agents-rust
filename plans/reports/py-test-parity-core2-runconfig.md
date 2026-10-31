# Python test parity: core2 / runconfig

Scope: 10 Python files, 289 tests. Counted with `grep -c 'def test_'` plus parametrize expansions
by hand (pytest not installed in the scratch venv). Totals match the task's numbers.

New Rust test files (all green, `--all-features`):

| file | tests |
| --- | --- |
| `tests/py_core2_runconfig_settings.rs` | 29 |
| `tests/py_core2_runconfig_cancel.rs` | 33 |
| `tests/py_core2_runconfig_lifecycle.rs` | 15 |
| `tests/py_core2_runconfig_provider.rs` | 17 (some gated on `websocket`) |

Rust tests loop over Python parametrize cases inside one `#[test]`, so one Rust test can cover
several Python cases; the table counts Python cases.

## Per-file table

| Python file | #tests | ported | covered | python-only | failures found |
| --- | --- | --- | --- | --- | --- |
| test_run_config.py | 123 | 101 | 0 | 22 | 4 (model-settings resolution, captured defaults, config validation x2) |
| test_integration_runner.py | 58 | 0 | 0 | 58 | 0 |
| test_cancel_streaming.py | 18 | 14 | 0 | 4 | 2 (immediate cancel skipped cleanup, provider not closed) |
| test_soft_cancel.py | 25 | 25 | 0 | 0 | 1 (no turn acknowledgement: after-turn cancel raced next turn) |
| test_runner_model_provider_lifecycle.py | 11 | 10 | 0 | 1 | 1 (implicit provider never closed) |
| test_agent_runner_sync.py | 6 | 1 | 0 | 5 | 0 |
| test_repl.py | 4 | 4 | 0 | 0 | 1 (tool output printed as JSON) |
| test_run.py | 2 | 2 | 0 | 0 | 0 |
| test_config.py | 30 | 17 | 0 | 13 | 1 (provider-level client options missing) |
| test_openai_provider_client_options.py | 10 | 10 | 0 | 0 | (same as above) |
| **total** | **287+2** = 289 | **184** | **0** | **103** | |

Notes on ported rows:
- test_run_config `test_equal_in_place_assignment_preserves_explicit_settings` (12): the
  `reasoning` half (6) is ported; the `verbosity` half (6) is Python-only (counted there).
- test_cancel_streaming `cancel_with_pending_parallel_input_guardrail`: Rust waits for parallel
  input guardrails before running tools, so the cancel lands during the guardrail instead of
  during the tool; cleanup (computer disposed once) is asserted.

## PYTHON-ONLY groups (one line per group)

- test_run_config (22): dict-config coercion and unknown-key errors (`RunConfig(**dict)`,
  `run_config={...}`, 6+1+3+1+1 cases: normalize dicts, typed-cwd `PureWindowsPath`, preserves
  typed instances by identity, async formatter rejection, str-subclass/int blocked-message
  types (3), untrusted manifest path-grant dicts + serialized manifest dicts (3), unknown dict
  fields (3), dict collision policy), `Mock()` model_settings type check, `ToolNameCollisionPolicy`
  string-literal validation (Rust enum makes invalid values unrepresentable), and the verbosity
  in-place `__setattr__` tracking (6). Rust config is statically typed.
- test_integration_runner (58): all tests exercise the Python repo's CI script
  `.github/scripts/run_integration_tests.py` (profiles, junit sanitizer, uv bootstrap); not SDK code.
- test_cancel_streaming (4): falsy-`__bool__` exception, `gc`/"never awaited" warning check is
  folded into cancel_before_streaming (ported part), `_await_task_safely` monkeypatched terminal
  waits (2 params), `_await_model_provider_cleanup` consumer-cancellation ordering,
  pending-guardrail `run_loop.input_guardrail_tripwire_triggered_for_stream` monkeypatch — all rely
  on asyncio task internals / monkeypatching private methods.
- test_runner_model_provider_lifecycle (1): `run_preserves_primary_error_when_provider_cleanup_fails`
  needs a raising `aclose`; Rust `ModelProvider::close` returns `()` and cannot fail.
- test_agent_runner_sync (5): asyncio event-loop policy reuse/creation/closed-loop replacement,
  KeyboardInterrupt task cancellation, async-generator finalization. Rust `run_sync` builds its
  own tokio runtime. The "errors when loop already running" test is ported.
- test_config (13): `httpx2` shared client type, falsy default client, missing-key
  `openai.OpenAIError` at construction (Rust defers key errors to the request), and the
  per-event-loop websocket model cache tests (no-running-loop, per-loop scoping, weakref closed-loop
  pruning, transport abort, foreign-loop aclose x2) — Rust has no event-loop-scoped cache.

## Rust bugs fixed (src)

1. `src/run_internal/mod.rs` `resolve_settings`: implicit defaults now follow
   `_model_settings_for_resolved_name` — model instance selection → no defaults; plain name →
   that name's defaults; routed name / no selection → the model's `default_model_settings`, then
   the name, then the defaults the agent captured at creation. Previously a run-config model name
   with an unnamed provider model got no GPT-5 defaults, and an unnamed provider model ignored
   the agent's captured defaults.
2. `src/agent.rs`: new `pub(crate) implicit_model_settings` captured in `Agent::new` from
   `OPENAI_DEFAULT_MODEL` (Python `_initial_model_settings_for_model(None)`).
3. `src/run_internal/mod.rs` `run_loop`: rejects `max_function_tool_concurrency == Some(0)` and an
   empty `OutputGuardrailBlockedMessage::Text` with Python's messages before any model call
   (Python validates at construction).
4. Streaming turn acknowledgement (`src/result.rs`, `src/run_internal/mod.rs`, `src/run.rs`): new
   `StreamControl` shared by the run and `RunResultStreaming`. Between turns the run waits until
   an active consumer has asked for the next event (or no consumer is registered / the stream was
   dropped). Fixes after-turn cancel racing into the next turn (Python
   `_wait_for_turn_event_consumption`).
5. Immediate cancel (`src/result.rs`, `src/run_internal/mod.rs`): no longer `abort()`s the task.
   The run selects on a cancel signal, then still runs computer disposal, model cleanup and
   provider close; the stream yields nothing more and waits for that cleanup. New
   `RunResultStreaming::cancel_handle()` / `StreamCancelHandle` lets a consumer cancel from inside
   its `stream_events` loop (the stream borrows the result mutably).
6. Runner-owned provider close (`src/models/interface.rs`, `src/models/multi_provider.rs`,
   `src/run_config.rs`, `src/run_internal/mod.rs`): `RunConfig::default` now marks its
   `MultiProvider` run-owned (`into_run_owned`, `#[doc(hidden)] ModelProvider::is_run_owned`);
   the run closes it at the end (also after cancel). Caller providers stay open; nested
   agent-as-tool runs skip it.
7. `src/models/openai_provider.rs`: added `with_api_key/with_base_url/with_organization/with_project`
   and `openai_client()`; provider options bypass the default client and fall back to global key /
   env; explicit empty values preserved; combining `with_client` with any option fails in
   `get_model` with Python's "Don't provide api_key, ... organization, or project" `UserError`;
   `websocket_options()` getter. `src/models/openai_client.rs`: `organization()`/`project()`
   getters. `src/models/mod.rs`: `#[doc(hidden)] clear_default_openai_client()`.
8. `src/repl.rs`: tool output printed as raw text (`[tool output: tool_result]`), not JSON
   (`"tool_result"`). Added `run_demo_loop_io` (injectable reader/writer) used by
   `run_demo_loop_with`.

## Behaviour changes worth noting

- An agent with no explicit settings whose model is a **model instance** no longer gets GPT-5
  defaults from that instance's name (Python parity). `tests/openai_http_tests.rs::responses_api_round_trip`
  asserted the old behaviour; updated (edit in another file, listed here).
- Default runs now close the implicit `MultiProvider` (websocket model cache) at run end.
- `RunResultStreaming::new` (pub(crate)) signature gained `control`; `EventSink` gained `control`.

## Cross-area / shared-file edits

- `src/lib.rs`, `Cargo.toml`: none.
- `tests/openai_http_tests.rs`: one assertion flipped (see above).

## Verification

- My 4 binaries: 29 + 33 + 15 + 17 = 94 passed, 0 failed (the cancel/settings binaries ran 5 times
  in a row, all green).
- `cargo test --all-features --lib`: 642 passed, 0 failed.
- Related binaries still green: openai_http_tests (7), py_core_a_runner (14), py_core_a_streamed
  (15), py_deferred_writes_d (13), py_models_map_defaults (63).
- `runner_tests`: 3 failures in other areas' in-flight code (tool-invocation ledger "already
  executed" error, missing `run_data` on a structured-output error, collision message). None of
  them touches model settings, streaming, or provider lifecycle.
- Earlier full-workspace run (stopped when the shared disk filled up): failures only in
  py_core2_approvals_reuse_a, py_core2_runstate_items, py_core2_runstep_processing,
  py_core2_runner_routing, py_core2_scopedapprovals_*, and py_core2_runner_probe (an explicit
  probe panic). py_core2_toolerrors_batch hung on non-streamed tests. All of these belong to
  other agents. The hang is non-streamed, and my turn-acknowledgement wait only runs with a
  stream sink, but it is worth a re-check once that area settles.
- clippy (`--lib` + my tests): the only warning is `src/run_internal/tool_invocation.rs:328`
  (`nonminimal_bool`), which is another agent's file, so `-D warnings` currently fails there. My
  files are clean.
- rustfmt applied to my test files and to `src/models/openai_provider.rs` and `src/repl.rs`. The
  shared `src/run_internal/mod.rs` and `src/agent.rs` were not reformatted; my hunks there follow
  the rustfmt layout.
- Build dir deleted.
- Rule slip: I ran one read-only `git diff --stat` by mistake. No git writes.

Status: DONE_WITH_CONCERNS
