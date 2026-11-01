# Python test parity, round 9, r9h1 (top-level run-state and resume files)

14 files, 275 python-only cases before. Every group was read in the Python test and matched against the Rust tests. Result: 15 ported, 0 covered, 260 remain python-only. Two Rust source gaps were fixed on the way (cancelled-run recovery, schema summaries).

## Per-file table

| file | python_only before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| test_run_state.py | 150 | 1 | 0 | 149 | see "test_run_state.py groups" below |
| test_native_tool_failure_history.py | 34 | 0 | 0 | 34 | asyncio cancellation semantics (tool raising `CancelledError` 16, guardrail verdict from a cancelled child task 4, a native invocation that converts or suppresses its own cancellation 12) and an `on_tool_end` hook that raises (2: Rust hooks return `()`) |
| test_run_impl_resume_paths.py | 24 | 13 | 0 | 11 | post-tool callbacks that raise: `end_hook` variants 4 (Rust hooks return `()`), nested-run callback failure 4 (an extractor returns `Option<Value>`, it cannot raise), cancelled callback 1 (`CancelledError`); monkeypatched private run-loop functions 2 (`resolve_interrupted_turn` short circuit, `run_single_turn` injecting two equal outputs) |
| test_live_example.py | 16 | 0 | 0 | 16 | tests of `examples/live/app` (a FastAPI relay), a Python example with no Rust counterpart |
| test_handoff_history_duplication.py | 13 | 1 | 0 | 12 | object identity of copied `result.input` / `new_items` (input copy 2, copy and reorder 2, unmarked equal input 1, rebuilt `HandoffInputData` 1, exact `RunItem` identity 1, per-occurrence copy lineage 1), private helper `_get_nested_history_owned_items` 1, dataclass field introspection 1, schema 1.12 layout 2 (see "Divergences") |
| test_human_in_the_loop_server_example.py | 13 | 0 | 0 | 13 | tests of `examples/agent_patterns/human_in_the_loop_server.py`, a Python example |
| test_redis_session_example.py | 8 | 0 | 0 | 8 | tests of `examples/memory/redis_session_example.py` (monkeypatched `RedisSession.from_url`, capsys), a Python example |
| test_handoff_failure_error_function.py | 4 | 0 | 0 | 4 | an async formatter (the Rust `failure_error_function` is a synchronous closure returning text) and failures outside the recovery boundary: `CancelledError` from `on_handoff`, a raising formatter, a raising input filter (Rust closures return plain values) |
| test_items_helpers.py | 4 | 0 | 0 | 4 | gc / weakref retention of agents (2), pydantic `model_validate` rejection of a `None` refusal (1, a test of the openai models) and of `ValidatorIterator` copies (1) |
| test_run_context_wrapper.py | 3 | 0 | 0 | 3 | `str()` that raises (1), `_resolve_tool_name` / `_resolve_call_id` fallbacks for a `ToolApprovalItem` built without a tool name or from a raw dict (1, Rust's `tool_name` and `raw_item` are required typed fields), a falsy `ToolApprovalItem` subclass (1) |
| test_pretty_print.py | 2 | 0 | 0 | 2 | a dataclass output whose `__str__` renders it (streamed and not) |
| test_process_model_response.py | 2 | 0 | 0 | 2 | falsy `ShellTool` / `ApplyPatchTool` subclasses (`__bool__`) |
| test_function_schema_logger_restore.py | 1 | 0 | 0 | 1 | griffe logging level restore |
| test_provider_span_errors.py | 1 | 0 | 0 | 1 | monkeypatching the private `attach_error_to_span` to raise |
| **total** | **275** | **15** | **0** | **260** | |

## Ported (15)

`tests/py_core9_r9h1_resume_cancel.rs` (3 tests, 7 cases):

- `handoff_resume_after_cancelled_compaction_commits` (4: JSON round trip x streamed retry). A store that commits the compaction replacement and holds its acknowledgement stands in for Python's paused SQLite worker; the streamed run is cancelled with `CancelMode::Immediate`; the checkpoint carries no pending write, the resume finishes the handoff once and compacts once.
- `cancelled_compaction_append_preserves_committed_and_surviving_writes` (1). Passed without a source change.
- `resumed_session_append_cancellation_retains_recoverable_state` (2: non-streamed with the future aborted, streamed with an immediate cancel). While the Session holds the resumed batch's append, a second resume of the same checkpoint is refused with "pending Session write is already in progress"; after the first run is dropped the checkpoint stays claimed until the append lands (a third resume in that window is refused too), then the checkpoint it published reconciles with the append and the approved tool runs once.

`tests/py_core9_r9h1_resume_stream.rs` (3 tests, 6 cases):

- `resumed_stream_waits_for_event_consumption_before_continuing` (2: run again, handoff). Passed without a source change once the consumer was attached before the resumed tool finished (the resumed tool is gated on the consumer's first event, as the Python test waits for its first agent-updated event).
- `resumed_run_again_saves_the_next_turn_items` (1, streamed and not): the next turn's call and output reach the session after a resumed approval, which is the observable half of `test_resumed_run_again_resets_persisted_count`.
- `resumed_interruption_passes_server_managed_conversation_flag` (3: conversation id, previous response id, auto chaining): a resumed server-managed snapshot refuses a handoff input filter when the parked handoff runs.

`tests/py_core9_r9h1_handoff_history.rs`: `removing_one_owned_new_item_keeps_the_others_owned` (1, streamed and not), the public-API half of `test_nested_history_session_resolution_keeps_surviving_owned_items`. The file also holds `first_nested_handoff_after_restore_replays_tool_search_items_once`, which pins a recorded divergence and moves no count.

`tests/py_core9_r9h1_run_state.rs`: `supported_schema_versions_have_non_empty_summaries` (1), backed by the new public `SCHEMA_VERSION_SUMMARIES`.

## test_run_state.py groups (149 remain)

Each group was read; none has a public Rust surface, or Rust differs by a documented design choice.

- Context serialization (9): Python dataclass / pydantic / arbitrary-object contexts, `context_serializer`, `context_deserializer`, omitted-context metadata. Rust stores the application context as JSON on `RunState::context`; the strict read and write halves are already covered by `core_python_run_state.rs`.
- `get_interruptions` snapshot hardening (about 35): `__deepcopy__` hooks, pydantic subtype `model_serializer`, typed extras, storage descriptors, class properties, non-finite and cyclic mapping data, nested aliases. Values are cloned; none of these shapes exist in a `serde_json::Value` / typed `FunctionCall`.
- Approval identity with uncopyable or unsafe siblings (about 12): Python deepcopy failures.
- Nested agent-tool state surviving dropped functions (about 6). Python's `_deserialize_processed_response` drops a parked function whose tool was disabled or removed and re-homes nested state by position. Probed: Rust keys nested state by call id, and a resume whose parked call lost its tool is refused ("Tool early not found in agent probe") instead of dropped, so the scenarios cannot arise. Recorded under Divergences.
- Private `_deserialize_processed_response` / `_deserialize_items` / `_lookup_function_name` / `_serialize_item` helpers with live tools, agents and hand-built `ProcessedResponse` objects (about 40). A Rust resume re-processes the parked response with `process_model_response` (ported), so there is no snapshot-to-live-tool deserializer.
- Python-object output coercion (datetime, bytes, pydantic and dataclass outputs, guardrail `output_info` objects; about 10).
- `last_processed_response.new_items` merged into `generated_items` (about 8): Rust always holds those items in `generated_items`. The Python merge dedupes by id and relies on a marker for anonymous items, so a Rust merge would duplicate them on every round trip.
- Object identity and falsy objects (copied generated items, repeated references, `__bool__` subclasses; about 12), `None` state fields (`approve` / `to_json` with no context or agent; 4).
- Sandbox registry monkeypatching (`importlib`, reserved mount registration; 2), `resolve_resumed_context` object identity (1), the private `_AgentToolResumeCheckpoint` class (1).
- `test_current_schema_missing_ledger_rejects_malformed_current_authority`: the malformed current call is a `model_construct` object without `arguments`; Rust's `FunctionCall.arguments` is a required `String`, so that input cannot be built (an empty string is a different, valid call).
- `_deserialize_items` skips an item with an unknown agent, a missing agent, a missing type or a malformed payload (about 14). Deliberate Rust difference, pinned by `py_core2_runstate_sidecars::rust_refuses_*`: a skipped item is history the resumed run never sees, so Rust refuses the snapshot (missing agent / type) or keeps the item (unknown agent). Probed this round: an item with an unknown agent, a raw item without `type` (so `deserialize_items_fallback_union_type`, which feeds one with a `type`, does not cover Python's typeless skip) or a malformed message is kept; an item missing its agent or its `type` is refused; Python skips all of them.

## Source changes

- `src/run_internal/mod.rs`
  - The completion notice (`notify_session`) runs on its own task whose handle stays on the loop. A streamed run cancelled during the notice now waits for it and, when it succeeded, settles the pending write, as Python's shielded compaction does; before, a cancelled compaction left an acknowledged record behind and the resume compacted the history a second time.
  - `RunLoop` gains `write_in_progress` and `published_checkpoint`; the run takes back a checkpoint it published once it finishes successfully.
- `src/run_internal/session_write.rs`: `settle_recoverable_write` claims a shared in-progress flag (a second settle of the same checkpoint is refused with Python's message; the claim travels inside the spawned append, so a dropped caller keeps the checkpoint claimed until the Session finishes writing, as Python's `finally` does) and publishes the failure checkpoint into the caller's `RunCheckpoint` before handing the batch to the Session. A caller that drops the run (aborts the task) while the Session holds the append still gets a checkpoint that carries the write, instead of nothing.
- `src/run_state.rs`: `RunState::session_write_in_progress` (`#[serde(skip)]`, shared by clones) and the `SessionWriteInProgress` type; `src/run_state/wire.rs` initialises it and adds the public `SCHEMA_VERSION_SUMMARIES` (22 entries, text copied from Python).

## Breaking changes

- `RunState` has one more public field; struct literals need `..Default::default()`.
- `RunCheckpoint::take` can now return a state after a run that was dropped or cancelled, not only after one that failed, and a run that finishes successfully leaves the slot empty.
- `SCHEMA_VERSION_SUMMARIES` is new public API (additive).

## Divergences recorded, not changed

- Schema 1.12 snapshots without an ownership ledger (`first_nested_handoff_after_restore_uses_explicit_occurrence_lineage[schema_1_12]`, 2 cases): Python replays the forwarded and the session copy (two occurrences) because its `_session_items` and `_generated_items` are separate lists that only the missing index map ties together. Rust keeps one list, can prove the lineage, and replays once. Replicating a duplicate seemed wrong; the test pins Rust's behaviour.
- Equal copies of `result.input` (Python: a deep copy loses the identity-based ownership, two occurrences; Rust: value equality, one). Rust's `InputItem` is a plain value.
- `HandoffInputData` rebuilt by hand keeps its ownership in Rust (the ledger is not attached to the container).
- Failure formatters cannot re-raise or be async in Rust (`ToolErrorFn` returns `String`).
- A resume whose parked tool call lost its tool (disabled or removed) is refused with "Tool X not found in agent Y" in Rust; Python drops the function and carries on with the retained ones.

## Unresolved questions

- Should `RunState::from_json` / resume skip items whose agent is not in the graph (Python) rather than keep them? Left as the documented Rust choice; porting it needs the ownership indexes remapped, as Python does with source indexes.
- Should the Rust handoff `failure_error_function` become fallible (`Result<String>`) so a formatter can re-raise? That would change a public type shared with function tools.

## Gates

- New binaries: `py_core9_r9h1_resume_cancel` (3), `py_core9_r9h1_resume_stream` (3), `py_core9_r9h1_handoff_history` (2), `py_core9_r9h1_run_state` (1), all green with `--all-features` and default features.
- Regression, `--all-features --no-fail-fast`: `py_core6_*`, `py_core2_*`, `py_core3_runstatectx*`, `py_core4_*`, `py_core9_*`, `py_memory_*`, `py_deferred_writes_*`, `core_python_run_state`, `runner_tests`, lib tests: 171 binaries green, one red: `py_core6_resume_name_fallback::resolve_interrupted_turn_only_uses_name_fallback_for_legacy_approval_agents` (schema 1.6 approve / reject expect a name-fallback execution and get a re-asked approval). It is not caused by this work: it concerns same-named agents with legacy approvals and passes through none of the changed code (no session, no checkpoint). The likely mechanism is r9gaps' identity-keyed approvals: a schema 1.6 / 1.7 snapshot whose interruption lost its identity now resolves its owner to the first same-named agent, so a decision on the second agent's item authorizes nothing and the run asks again. `py_core9_r9gaps_approval_identity` is green on the same tree.
- `cargo clippy --all-features --lib --test <4 files> -- -D warnings` and the same with default features: clean. `rustfmt --edition 2024` applied to the four new test files. `rustfmt --check` still reports a diff in `src/run_state/wire.rs` (`hosted_pending_without_arguments`, another agent's function); my `SCHEMA_VERSION_SUMMARIES` block is formatted.

Status: DONE_WITH_CONCERNS
Summary: 275 cases read; 15 ported (cancelled compaction and append 7, resumed stream and persistence 6, handoff ownership 1, schema summaries 1), 0 covered, 260 remain python-only (asyncio / hooks that raise / examples / pydantic / object identity / private helpers, plus the divergences above).
Concerns: one failing regression test from another agent's in-flight change; `rustfmt --check` diff in another agent's function in `wire.rs`; the unknown-agent item skip and the schema 1.12 duplicate remain deliberate divergences.
