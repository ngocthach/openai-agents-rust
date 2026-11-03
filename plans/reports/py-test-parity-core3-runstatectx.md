# Python test parity: runstatectx (core3)

Scope: 11 Python files under `openai-agents-python/tests/` (138 tests, counted as `def test_` plus parametrize expansion; the pending-input file is 25 functions / 54 cases). Reference run: the Python scope passes 138/138 in `scratchpad/pyenv`.

## Per-file table

| Python file | total | ported | covered | python-only | deferred (failing, not fixed) | failures found (fixed) | Rust file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_source_compat_constructors.py | 20 | 3 | 0 | 17 | 0 | 0 | py_core3_runstatectx_compat.rs |
| test_run_state_hook_identity.py | 15 | 13 | 0 | 2 | 0 | 8 | py_core3_runstatectx_hook_identity.rs |
| test_run_context_wrapper.py | 14 | 11 | 0 | 3 | 0 | 0 | py_core3_runstatectx_context.rs |
| test_run_state_aliases.py | 10 | 10 | 0 | 0 | 0 | 4 | py_core3_runstatectx_aliases.rs |
| test_run_state_pydantic_context.py | 8 | 0 | 0 | 8 | 0 | 0 | none |
| test_runtime_symmetry_contract.py | 6 | 4 | 0 | 2 | 0 | 2 | py_core3_runstatectx_symmetry.rs |
| test_run_internal_error_handlers.py | 5 | 3 | 0 | 2 | 0 | 0 | py_core3_runstatectx_error_handlers.rs |
| test_call_model_input_filter_unit.py | 3 | 2 | 0 | 1 | 0 | 0 | py_core3_runstatectx_context.rs |
| test_hitl_session_scenario.py | 2 | 2 | 0 | 0 | 0 | 0 | py_core3_runstatectx_hitl.rs |
| test_streaming_logging.py | 1 | 0 | 0 | 1 | 0 | 0 | none |
| test_run_state_pending_input.py | 54 | 43 | 0 | 11 | 0 | see fix list | py_core3_runstatectx_pending_a.rs, py_core3_runstatectx_pending_b.rs |
| **Total** | **138** | **91** | **0** | **47** | **0** | | |

"covered" is 0: the core-a triage listed 12 pending-input cases as covered by `core_python_run_state.rs`, but no existing test matches them one-to-one, so every case was ported instead.

## Python-only (one reason per group)

- Positional constructors (`RunConfig`, `ModelSettings`, `ToolContext`, `RunResult`, `RunResultStreaming`, `RunErrorDetails`, `FunctionTool` guardrail slots) and keyword aliases (`_run_impl_task`, `run_loop_task`): Rust uses named fields and builders, with one field name per concept.
- `ToolExecutionConfig` non-bool rejection, `ToolContext` v0.7 positional form, `AgentHookContext`: a typed Rust field cannot hold a non-bool, and `AgentHookContext` has no Rust counterpart.
- Pydantic context (8 cases): Rust does not capture the application context into `RunState` (documented in `run_state.rs`); pydantic `default_factory`, `model_validate` and `arbitrary_types_allowed` are Python internals.
- `test_run_state_hook_identity`: the `hook_type` and `hook_category` discriminators depend on Python class names and dataclass-ness. Rust records only hook presence.
- `test_runtime_symmetry_contract`: scripted-model `__dataclass_fields__` inspection (2 parametrized cases).
- `test_run_internal_error_handlers`: monkeypatched output schema and `str(object())` fallback, and unserializable (set) output. A Rust `Value` cannot hold these.
- `test_call_model_input_filter_unit`: invalid filter return type. The Rust closure return type is checked at compile time.
- `test_run_context_wrapper`: `_to_str_or_none` with a raising `__str__`, raw-dict tool-name and call-id fallbacks, the `"unknown_tool"` fallback, and the falsy `ToolApprovalItem`. Rust items are typed.
- `test_streaming_logging`: Python emits a resume debug record with per-item attributes. Rust emits no resume-time item log, so the leak path does not exist; the record contract has no Rust analogue.
- `test_run_state_pending_input` (11 cases):
  - `test_server_tracker_distinguishes_identical_input_occurrences_after_restore`: calls a private tracker method directly.
  - `test_server_filter_reconstructed_pending_rewrite_is_rejected` (2): Python associates by object identity; Rust filter inputs are owned copies, so an in-place rewrite and a rebuilt item are indistinguishable. The in-place case is ported and accepted.
  - Hook-failure cases (8): `test_server_response_acceptance_commits_before_hook_failure` (2), `test_server_accepted_computer_start_hook_failure_is_not_replayed` (2), `test_server_accepted_tool_side_effect_failure_is_safe` (4). Rust lifecycle hooks return `()`, so no hook can fail a run; the recovery paths have no Rust trigger.

## Documented differences (kept on purpose; the tests assert the Rust behaviour)

1. Identical late input is not resent (`identical_late_input_is_not_resent_by_fingerprint`, blocking and streamed). Server-managed delta selection is by content fingerprint (`ServerConversationTracker::needs_send`). This is deliberate, so a retry or a resumed state can neither resend nor skip an item. An identical later occurrence of input the server already holds is therefore not sent again. Python selects by object identity and sends it. Both occurrences stay in the run's history.
2. Staged input is admitted at turn start (`server_managed_staged_input_is_admitted_at_turn_start_and_sent_once_on_retry`). Rust moves staged input into the run history before the model call. A failed request therefore leaves it admitted, not pending, and the retry sends it exactly once. Python keeps it pending until a model call accepts it.

## Fixed in this round (previously deferred)

- Omitted staged input survives a streamed after-turn cancel (`server_filter_omission_survives_streamed_cancel_and_round_trip`). Staged input the server never received is returned to `pending_input` when the streamed run is cancelled after its turn, and the admitted copy is removed so the resume admits it once. The tracker's `was_sent` check uses the same fingerprint. Restoring a snapshot no longer marks admitted input as sent, which had made the resume drop it.
- Accepted response not re-requested (`accepted_response_refused_on_retry_without_model_call`, blocking and streamed). When a server-managed response is refused after acceptance (reused completed call id, or processing failure), the snapshot is marked unrecoverable. The retry is refused before any model call with the Rust message "This RunState ended at an unrecoverable boundary". Python's message is "accepted model response could not be processed"; the behaviour (no second model call, no lost output) matches.

## Rust source fixes (all landed)

- `src/run_internal/mod.rs`
  - Cancel path returns undelivered staged input to `pending_input`; response refused after acceptance marks the snapshot unrecoverable; `previous_response_id` restored only for server-managed runs.
  - Resume guard: Python's identity rule replaces the Rust "same-named agents" refusal. Identity-less names resolve to the first agent with that name; refuse only hook-indistinguishable duplicates (`has_ambiguous_hook_identities`) and legacy (pre-1.21) ambiguity (`has_legacy_hook_identity_ambiguity`).
  - Snapshot builder: writes `pending_input`, and sets `hook_identity_ambiguous` and `tool_result_may_end_run`.
  - Staged input guardrails are always awaited before the model call, as Python's `admit_pending_input` does.
  - Admitted staged input is saved to the session before the model call (not server-managed), so a failed request keeps it.
  - Resume refuses staged input behind a parked approval that may end the run.
  - Restores `previous_response_id` from history only for server-managed runs (it was set for every run, which broke later non-server retries).
  - Removed the now-unused `VecDeque` import.
- `src/run_internal/agent_identity.rs`: `has_ambiguous_hook_identities`, `has_legacy_hook_identity_ambiguity`, `interruption_may_end_run`.
- `src/run_state.rs`: new `#[serde(skip)] pub` fields `legacy_hook_identity_snapshot`, `hook_identity_ambiguous`, `tool_result_may_end_run`. `from_json` maps the Python `async_` field name to `async` and sets the legacy flag. `add_input` messages now match Python ("Cannot add input to a terminal RunState", "...no remaining model turns") and refuse when the parked approval may end the run. Python ignores unknown envelope keys, so the new key is safe to read there.
- `src/run_internal/oai_conversation.rs`: `was_sent`; hydration no longer marks admitted input as sent.
- `src/run_state/wire.rs`: `to_wire` refuses an ambiguous snapshot; additive `tool_result_may_end_run` key (written only when true); `rename_python_async_alias` helper.

Behaviour change to note: a name-only current agent in a graph with same-named agents now resumes on the first agent with that name (Python's rule), where Rust previously refused.

## Shared-file edits (outside this area; please review)

- `src/testing.rs`: `FakeModel` stream `response.completed` usage now carries `input_tokens_details` and `output_tokens_details`. Without it, streamed usage silently dropped cache and reasoning tokens, which the symmetry contract checks.
- `tests/py_core_a_run_loop.rs`: `a_duplicate_agent_snapshot_without_identity_is_refused` now asserts Python's behaviour (the identity-less name resolves, and no "same-named agents" refusal). This reverses a core-a decision, based on `run_state.py` `_resolve_agent_from_data`.

## Gates

- `cargo test` (own binaries, all green): context 13, hook_identity 13, aliases 10, symmetry 4, hitl 2, pending_a 16, pending_b 27, compat 3, error_handlers 3. Related suites green: core_python_run_state, core_run_state, py_core_a_run_loop, py_core2_runstate_*, py_core2_approvals_context, core_trace_resume, runner_tests, `--lib` (651).
- `cargo clippy --all-features --test <my files> -D warnings`: clean.
- `cargo clippy --all-features --lib -D warnings`: clean (re-checked in this round).
- Full `cargo test --workspace --all-features`: not completed. Other agents' test files (`py_core3_handoffs_tool.rs`, `py_core3_tracing_*`, `py_core3_functiontools_*`) were mid-edit and failed to compile during this run.
- rustfmt applied to my test files only.
- Build dir `$HOME/.cache/oars-out/c3-runstatectx` must be deleted at the end (`cargo clean -p` brought it to 1.3 GB).

Status: DONE_WITH_CONCERNS
Summary: 138 Python cases classified: 91 ported and passing, 47 python-only, 0 deferred. Two documented differences (fingerprint delta selection; staged input admitted at turn start). Rust bugs fixed across the run-state, identity, streaming-usage and server-managed code.
Concerns/Blockers: full `cargo test --workspace` not completed (other agents' in-flight test files). Two shared-file edits (`testing.rs`, `py_core_a_run_loop.rs`) need review.
