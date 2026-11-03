# Python test parity: realtime session (slice B)

Counted by grep `def test_`; parametrize NOT expanded (each parametrized def is one row). Ledger: 290 rows total
(208 rows in the 11 session files below, plus 82 rows for realtime files ported earlier: agent, audio formats,
item parsing, model events, playback tracker, tracing, twilio/app-server demos).

## Per-file table (session slice)

| Python file | # | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_session.py | 85 | 51 | 7 | 27 | 5 Rust gaps (fixed, see below) |
| test_session_approvals.py | 24 | 18 | 4 | 2 | 2 ordering gaps (fixed) |
| test_session_guardrails.py | 27 | 23 | 2 | 2 | 1 (fixed: duplicate guardrail ran twice) |
| test_session_history.py | 22 | 20 | 1 | 1 | 0 |
| test_session_tool_outputs.py | 13 | 7 | 0 | 6 | 0 |
| test_session_exceptions.py | 7 | 5 | 0 | 2 | 0 |
| test_session_tool_output_guardrails.py | 6 | 6 | 0 | 0 | 0 |
| test_session_payload_and_formats.py | 4 | 0 | 0 | 4 | 0 |
| test_session_close_iteration.py | 3 | 3 | 0 | 0 | 0 |
| test_session_tool_error_events.py | 1 | 1 | 0 | 0 | 1 divergence (deferred) |
| test_realtime_model_settings.py | 16 | 6 | 1 | 9 | 0 |
| **session slice total** | **208** | **140** | **15** | **53** | |

Ledger totals (all 290 rows): PORTED 179, COVERED 18, PYTHON-ONLY 93, TODO 0.

Rust test files (all green, `--all-features`): `tests/py_realtime_session_core.rs`, `..._dispatch.rs`,
`..._approvals.rs` (20), `..._guardrails.rs` (34), `..._history.rs` (31), `..._tool_outputs.rs` (7),
`..._settings.rs` (8).

## PYTHON-ONLY groups
- asyncio task cancellation / done-callbacks / private task sets and bookkeeping fields (`_tool_call_tasks`, `_cleanup_task`, ...).
- `__traceback__` / `capture_locals` redaction checks, `caplog`, `monkeypatch` of module logger / `_debug`.
- Mock objects of arbitrary type in `get_all_tools`, pydantic/dataclass serializer fallbacks, duck-typed websocket helpers.
- Tools/handoffs passed through `initial_model_settings` (Python carries live objects; Rust settings carry wire specs, tools and handoffs always come from the agent).
- examples/realtime FastAPI demo servers (earlier rows).

## Rust bugs fixed (src/realtime/session.rs)
1. `EndOfStream` now ends the event stream after queued events drain (previously `recv` hung).
2. Tool/handoff dispatch re-evaluates `is_enabled` at call time (Python rechecks); a disabled one answers "Tool X not found".
3. `close()` can be retried after `model.close()` fails (idempotent once it succeeds), serializes concurrent callers, and no longer aborts the task that called it (a tool closing its own session).
4. Empty tool call id is rejected before dispatch ("Realtime tool calls require a non-empty string call ID.").
5. Approval ordering: a standing (always) decision is consulted before the dynamic `needs_approval` rule, and is rechecked after the rule and after pre-approval input guardrails run.
6. (fork) An output guardrail listed on both agent and run config ran twice; now deduped by name.

## Deferred / divergences (not fixed)
- **Tool failure error text**: Python emits `RealtimeError{"message": "Tool call task failed"}` for any failed tool task (redaction); Rust emits the detailed error text. `tests/realtime_tests.rs::a_tool_guardrail_tripwire_ends_the_session` (not editable here) asserts the detailed text. Minimal fix: in `session.rs` `handle_tool_call` use a fixed message for the `end_with_error` event and update that test.
- Sticky approvals are keyed by agent *name*; Python scopes them to the agent object, so two distinct same-named agents share decisions in Rust (`RunContext` approvals state, outside this slice).
- Concurrent `close()` callers do not share a single failure (`AgentsError` is not `Clone`); the second waits and retries.
- `OutputGuardrail` exposes no identity, so the dedupe in fix 6 is by name; a `ptr_eq` accessor in `src/guardrail.rs` would be exact.
- `src/realtime/testing.rs` had a duplicated `is_connected` (another agent's in-flight edit) reported by fork F1.

## Shared-file edits
None to `src/lib.rs` / `Cargo.toml`.

> **Update:** the open items about namespaced/deferred realtime tools and the "Tool call task failed" wording were resolved afterwards; see "Decisions applied after review" in `py-test-parity-realtime-voice-codex.md`.
