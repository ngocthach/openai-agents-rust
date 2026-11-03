# Python test parity, core round 5: leftover1

Python tests: `/Users/ngocthach/myGit/openai-agents-python/tests/`. Rust tests: `tests/py_core5_leftover1_*.rs`.
Counting: `def test_` plus parametrize expansion (matches the earlier reports' totals). For
`test_streamed_terminal_output_backfill.py` the raw grep gives 5, but 2 of those are nested helper
functions named `test_tool` inside real tests, so the file has 3 test cases.

## Per-file totals

| Python file | total | ported | covered | python-only | not ported | Rust file |
|---|---|---|---|---|---|---|
| `test_scripted_model.py` | 97 | 67 | 1 | 29 | 0 | `py_core5_leftover1_scripted_model.rs` (44 tests) |
| `test_scripted_sandbox.py` | 36 | 9 | 2 | 25 | 0 | `py_core5_leftover1_scripted_sandbox.rs` (8 tests); the 2 covered are in `py_core3_shelltools_scripted_sandbox.rs` |
| `test_server_conversation_tracker.py` | 24 | 19 | 0 | 5 | 0 | `py_core5_leftover1_server_conversation.rs` (16 tests) |
| `test_streamed_terminal_output_backfill.py` | 5 | 3 | 0 | 2 (nested `test_tool` helpers, not tests) | 0 | `py_core5_leftover1_streamed_output.rs` |
| `test_streaming_tool_call_arguments.py` | 4 | 4 | 0 | 0 | 0 | `py_core5_leftover1_streamed_output.rs` (7 tests with the file above) |
| `test_start_hook_tool_concurrency.py` | 39 | 4 | 0 | 35 | 0 | `py_core5_leftover1_start_hook_concurrency.rs` (2 tests) |
| **total** | **205** | **106** | **3** | **96** | **0** | 77 Rust tests |

## Rust divergences fixed (source)

`src/testing.rs` (ScriptedModel):
- A responder may return another responder (loop, each hop validated); was rejected.
- Async responders (`ScriptedStep::respond_async`) and lazily built exact streams
  (`ScriptedStep::stream_with`, new `async_responder` / `stream_factory` fields, both count toward
  the "one outcome" conflict check). An `Err` item fails the call part-way.
- A steps usage that is absent or all-zero falls back to the default usage, then to one counted
  request (was zero requests).
- Retry advice is forgotten when a responder fails with an error of the same message.
- The generation span stays open while a stream is drained (a mid-stream error is recorded on it,
  dropping the stream finishes it) and is guarded by `ModelSpanGuard`, so a call dropped by a
  timeout records `Model call timed out after N seconds.` on the span.
- `apply_patch_call` output is normalized like Python: non-empty call id, item id defaults to the
  call id and must be non-empty when given, status `in_progress`/`completed` (default completed).
- Automatic streaming rejects `custom_tool_call`, `code_interpreter_call`, `mcp_call` and
  `image_generation_call` with `Automatic streaming does not support <type>`. Python rejects every
  kind except message, function call, reasoning and apply_patch_call; Rust keeps passing other raw
  items (shell, program, tool search, web search) through as a bare added/done pair, because 7
  streamed tests in `py_core2_toolerrors_batch`, `py_core2_runstep_processing`,
  `py_core3_shelltools_shell_tool` and `py_core4_programmatic_*` script those items through the
  automatic stream (cross-area; narrowing it would need those tests moved to exact streams).
- Stream event fidelity: `output_item.added` for a function call has empty arguments and
  `in_progress`, for reasoning no `encrypted_content`, for apply_patch `in_progress`; text
  logprobs travel on the delta/done/part events; `response.completed` usage carries `requests`,
  `request_usage_entries` and token details, so a streamed run reports the same usage as a
  non-streamed one.
- Messages follow Python: `Scripted model step #N ...`, `Unexpected (non-)streaming model call #N:
  no scripted steps remain.` (`tests/testing_model_tests.rs` asserted the old wording; updated).

`src/sandbox/testing.rs` (ScriptedSandboxSession):
- A rejected matcher no longer consumes the step (a retry with matching arguments works).
- Failures are payload-free (no call arguments in message or context) and structured through
  `SandboxError::context`: `call_index`, `actual_method`, `expected_method`, `remaining_steps`,
  `method`, `pending_methods`.
- `supports_pty()` is derived from the script (either PTY method turns the capability on);
  `running()` reflects `start`/`stop`/`shutdown`.

`src/run_internal/oai_conversation.rs` (server conversation tracker):
- `remaining_initial_input`: an initial item that a `call_model_input_filter` held back is offered
  again on the next call until a call carries it or the server produces it (was dropped for
  good). `rewind_input` added.
- `hydrate_from_state` marks only items the server can be identified as holding (id, tool call or
  output, staged input, tool search), as Python's `should_mark`; a bare message without an id
  stays unsent.

`src/run_internal/model_call.rs`:
- A streamed response whose terminal `response.completed` output is empty is backfilled from the
  `response.output_item.done` items, so tools run (Python: `streamed_response_output`).

## Shared-file edits

- `src/lib.rs`: one `#[doc(hidden)] pub use run_internal::oai_conversation::ServerConversationTracker;`
  (the tracker is only reachable internally; its contract is tested directly).
- `src/run_internal/mod.rs`: `mod oai_conversation;` became `pub(crate) mod oai_conversation;`.
- `tests/testing_model_tests.rs`: one assertion string updated for the new unexpected-call wording.
- `ServerConversationTracker` methods went from `pub(crate)` to `pub` (type is hidden from docs).

## Breaking changes (unreleased tree)

- `ScriptedStep` gained two public fields (`async_responder`, `stream_factory`); struct literals
  must use `..Default::default()` (all in-repo uses already do).
- `ModelScriptError` display text changed (see above).
- `ScriptedModel` default usage is one request instead of zero when no usage is scripted.

## Python-only groups (reason)

- `test_scripted_model.py` (29): snapshot/aliasing and detached-history checks (5: Python passes
  mutable objects by reference, Rust moves the step in and records owned clones), dict step specs
  and unknown keys (1), mistyped step fields (8 of the 10 envelope cases, unrepresentable in the
  typed struct; the 2 reachable reasons are ported), frozen event sequence and static/shorthand
  snapshots at queue time (5), `contextvars` span scope (1), `aclose()` failure handling (3), user
  `__str__`/`BaseException` failure modes (4), and mutation of event payloads after emission (2).
  One more case (redaction without rendering the exception) is covered by
  `redacts_generation_span_errors_from_run_config`.
- `test_scripted_sandbox.py` (25): `hasattr`/`dir`/type-hint introspection (2+), `io.BytesIO` /
  `io.StringIO` result and argument snapshots and unsupported-stream rejection (12), dict
  step-spec validation (8), `deepcopy` failure (1), matcher exceptions (1), static-result
  snapshot (1). The Rust builder makes the invalid steps unrepresentable and a matcher returns
  `bool`.
- `test_server_conversation_tracker.py` (5): monkeypatched `id()` and address-reuse cases (3,
  Rust has no object identity: fingerprints decide), and the reasoning-id policy constructor
  argument (2): the Rust tracker has no policy, the runner strips ids after the tracker has
  selected items (covered by `py_core2_runner_routing` reasoning-policy tests).
- `test_start_hook_tool_concurrency.py` (35): see below.
- `test_streamed_terminal_output_backfill.py` (2): nested helper functions, not tests.

### `test_start_hook_tool_concurrency.py`

The file tests one hazard: start hooks, handoff callbacks and enablement checks assign
`agent.tools` on an `Agent` shared by concurrent runs, and the SDK rejects every overlapping run
with "concurrent runs". A Rust `Agent` is read-only while it runs (hooks receive `&Agent`); a run
that needs another tool set builds a per-run replacement with `RunOptions::prepare_agent`, so no
shared mutable list exists. The 35 cases (shared-list replace and in-place append x run/agent
hooks x streamed = 8, MCP discovery mutation 2, failed/cancelled hook 4, suspended-hook mutation
2, restore-after-replace 2, `run_sync` threads 1, model wait 2, resumed agent 2, sequential
guardrail 2, handoff-target registration 10) are therefore Python-only. The 4 that survive are
the isolation guarantees and are ported: context-driven enablement stays per run (2) and a
per-run tool replacement never reaches an overlapping run nor the shared agent (2).

## Gates

- Full `cargo test --all-features --no-fail-fast`: 6338 passed before the final narrowing of the
  automatic-stream rejection; the 8 failures it exposed (all in other areas' tests that stream
  shell/program/tool-search items, plus one raw-usage payload assertion) were fixed in
  `src/testing.rs` and those binaries re-run green. `cargo test --lib`: 651 passed.
- `cargo clippy --all-features --lib` and my 5 test targets (+ `testing_model_tests`) with
  `-D warnings`: clean. rustfmt run on the files I created (`src/testing.rs` was formatted whole).

## Unresolved

- `tests/py_model_settings_serialization.rs` does not compile in the current tree (another agent's
  in-flight `Reasoning.extra` change); not touched.
