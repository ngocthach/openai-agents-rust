# Py test parity: realtime testing / handoffs / runner (slice C)

Counted by grep `def test_`; parametrize not expanded (4 parametrized tests in test_testing.py count once).

| file | # | ported | covered | python-only | failures |
|---|---|---|---|---|---|
| test_testing.py | 38 | 20 | 0 | 18 | 3 (all fixed, below) |
| test_realtime_handoffs.py | 19 | 13 | 0 | 6 | 0 |
| test_runner.py | 8 | 8 | 0 | 0 | 0 |
| total | 65 | 41 | 0 | 24 | 3 |

Tests: tests/py_realtime_testing.rs (28 tests: testing + runner), tests/py_realtime_handoffs.rs (13). All green; realtime_tests.rs still green.

## Python-only groups
- Listener fan-out, delivery worker, reentrancy, asyncio cancellation, queued-delivery-after-close (16 in test_testing): Rust transport exposes one event channel, no listener callbacks.
- `__deepcopy__` snapshot/failure, matcher mutation, matcher raising, static-script mutation (5-ish): Rust owns/clones values, matchers take `&event` and cannot raise.
- Handoffs: asyncio.gather sibling cancel (Rust `is_enabled` returns bool, evaluated sequentially), isinstance/typing, inspect.signature param counts, non-callable on_handoff, input_type-without-callback, async `__call__` objects: unrepresentable under Rust types.
- test_runner uses Mock-patched RealtimeSession; ported by observing the model connect call instead (Rust runner connects before returning).

## Bugs fixed (src/realtime/testing.rs)
1. UnexpectedSend message embedded `{event:?}` of the expected event, leaking payloads (Python redacts). Now "the scripted event".
2. Duplicate connect said "connected."; now "Already connected" (Python text). Added connect-time rejection of connect events + connect error together (Python raises at construction), message aligned.
3. Step message aligned ("cannot define both emit events and an error"); unconsumed message "scripted Realtime step(s)". Added `is_connected()` / `is_closed()` accessors on ScriptedRealtimeModel (additive public API).

## Deferred cross-area bugs
None found.

## Gates
rustfmt on my 3 files done; clippy --all-features --all-targets currently fails only in others' files (extensions/models/mod.rs:1016, models/openai_responses_ws.rs:553/556, sandbox/dependencies.rs:188, sandbox/sandboxes/unix_local.rs:1149); nothing reported in realtime or my tests. No lib.rs/Cargo.toml edits.
