# Python-only re-audit, round 19, agent r19c

Scope: part 1 (port what is portable) for nine sandbox/core files, part 2 (read-only check of
the tooling/example bucket) for thirteen files. New tests: `tests/py_core19_r19c_sandbox_memory.rs`,
`tests/py_core19_r19c_unix_local.rs`, `tests/py_core19_r19c_public_surface.rs`.

## Per-file result

| file | python_only before | ported | covered | valid after | reason per remaining group |
|---|---|---|---|---|---|
| sandbox/test_manifest_config.py | 13 | 0 | 0 | 13 | the `RunConfig(sandbox={...})` and `SandboxAgent(default_manifest={...})` dictionary carriers (Rust `RunConfig` has no sandbox field, both Rust carriers take a typed `Manifest`; the one untrusted-JSON boundary, `Manifest::from_untrusted_value`, is ported in `py_core8_r8g1_manifest.rs`), and the "dictionary holding typed `Dir`/`LocalFile` objects" case x3 (a JSON value cannot hold typed entries: dynamically typed argument) |
| sandbox/test_memory.py | 9 | 5 (10 at case level) | 0 | 4 | `subclass_uses_declared_model_settings_types` (pydantic subclass), `model_settings_field_types_describe_normalized_values` (`get_type_hints`), `instructions_requires_bound_session` (Rust renders the session-dependent fragment through `bound_instructions(&CapabilityBinding, ..)`, so it cannot be called unbound), `flush_propagates_worker_base_exception[cancelled_error]` (asyncio `CancelledError` raised by the worker task) |
| sandbox/test_workspace_payloads.py | 9 | 0 | 0 | 9 | `coerce_write_payload` adapts Python file-like objects (`read()` returning `None`/`bytearray`/`str`, `readinto`/`seek`/`tell`, `length`/`headers["Content-Length"]` attributes); Rust writes take `&[u8]` and have no stream adapter |
| sandbox/test_unix_local.py | 8 | 1 | 0 | 7 | start/resume/hydrate cancellation x3 and TTY start cancellation (asyncio task cancellation with `Path.mkdir`/`validate_tarfile`/`safe_extract_tarfile`/`os.openpty` monkeypatched), TTY fd-close task ownership (`asyncio.to_thread` patched, private `_fd_close_tasks`), `rebase_uses_archived_topology_when_workspace_changes` x2 (mutates the workspace from a patched `TarFile.addfile` mid-capture; Rust has no capture hook) |
| sandbox/test_unix_local_create.py | 6 | 0 | 0 | 6 | `failed_create_..._never_deletes_a_replacement[replace_target=True]` x3 (swaps the target from a patched `shutil.copyfileobj` mid-write) and `create_close_failure_reports_incomplete_contents` x3 (patched `os.fdopen` close). By construction Rust keeps a replacement: the owner path never unlinks after the exclusive open (`unix_local.rs::write_new_file`), the run-as path only links a fully written staging file |
| test_asyncio_tasks.py | 12 | 0 | 0 | 12 | `gather_with_cancel` / `run_producer_consumer` structured-concurrency helpers and coroutine `close()` semantics (incl. `Agent.get_all_tools` coroutine close): asyncio internals, no Rust counterpart |
| test_asyncio_progress.py | 7 | 0 | 0 | 7 | `get_function_tool_task_progress_deadline` inspects asyncio task graphs (timer handles, shield, gather, `call_later`) |
| test_source_compat_constructors.py | 17 | 0 | 0 | 17 | positional-argument binding of `RunConfig`/`ModelSettings`/`FunctionTool`/`ToolContext`/`RunResult(Streaming)`/`AgentHookContext`/`RunErrorDetails`, legacy `_run_impl_task` keyword, non-bool `pre_approval` value: Rust fields are named and typed, so mis-binding cannot compile |
| test_run_state_pydantic_context.py | 8 | 0 | 0 | 8 | pydantic `model_dump`/`model_validate`/`default_factory`/`arbitrary_types_allowed` context capture; Rust never serializes the application context itself (`RunState::with_context`, `ContextRestoreOptions`, documented in `run_state.rs`) |
| test_released_api_contract.py | 123 | 9 | 0 | 114 | see part 2 |
| test_check_optional_truthiness.py | 71 | 0 | 0 | 71 | `.github/scripts/check_optional_truthiness.py` AST checker over synthetic sources |
| test_change_detection.py | 62 | 0 | 0 | 62 | `.github/scripts/detect-changes.sh` and workflow YAML |
| test_integration_runner.py | 58 | 0 | 0 | 58 | `.github/scripts/run_integration_tests.py` profiles, credential classes, JUnit sanitizer |
| test_release_automation.py | 56 | 0 | 0 | 56 | `.github/scripts/release_automation.py` and release workflows |
| realtime/test_twilio_example.py | 37 | 0 | 0 | 37 | `examples/realtime/twilio` FastAPI server: webhook signatures, body limits, websocket lifecycle |
| test_dapr_session_example.py | 33 | 0 | 0 | 33 | `examples/memory/dapr_session_example.py` setup helper (docker, component YAML, credentials) |
| test_run_examples_script.py | 33 | 0 | 0 | 33 | `examples/run_examples.py` runner selection, skips, redis/temporal prerequisites |
| test_shell_examples.py | 21 | 0 | 0 | 21 | `examples/tools/shell*.py` approval prompts, subprocess env scrubbing, auto-runner tags |
| test_decisions_examples.py | 15 | 0 | 0 | 15 | `examples/decisions/*` with a mocked `client.decisions.create` vendor API |
| test_live_example.py | 16 | 0 | 0 | 16 | `examples/live/app` delegation handler and relay server |
| test_human_in_the_loop_server_example.py | 13 | 0 | 0 | 13 | `examples/agent_patterns/human_in_the_loop_server.py` `ApprovalServer` (owner binding, decision ids, replay) |
| realtime/test_twilio_sip_server.py | 14 | 0 | 0 | 14 | `examples/realtime/twilio_sip/server.py` call tracking and observer logging |
| **total (22 files)** | **641** | **15** | **0** | **626** | |

## What was ported

- `py_core19_r19c_sandbox_memory.rs` (test_memory.py):
  - `test_streaming_memory_payload_uses_result_input_for_server_managed_conversation` x3: a
    streamed sandbox run with `conversation_id`, `previous_response_id` or
    `auto_previous_response_id` records the turn's own input in its rollout segment. Passes.
  - `test_memory_generate_config_normalizes_dictionary_model_settings`,
    `..._preserves_forward_compatible_reasoning_settings` x2, `..._rejects_invalid_model_settings`
    x2 through `ModelSettings::from_value` into `MemoryGenerateConfig`. Pass.
  - `test_sandbox_memory_input_preserves_empty_session_delta` and
    `test_streaming_memory_payload_preserves_empty_input_override`: **`#[ignore]`d, real
    divergence** (below).
- `py_core19_r19c_unix_local.rs` (test_unix_local.py):
  `test_client_delete_keeps_workspace_removal_off_the_event_loop`. On a single-threaded runtime
  an observer task sees the workspace partly removed while `delete` is in flight, which only
  happens if the removal runs off the runtime thread (`tokio::fs::remove_dir_all`). Passes.
- `py_core19_r19c_public_surface.rs` (test_released_api_contract.py): the shape the
  `WebSearchToolImageSettings` cases freeze (`test_web_search_image_settings_policy_freezes_fields`
  x3, `test_new_top_level_typed_dict_requires_complete_policy` x6): a top-level export with two
  optional fields, `max_results` (integer) and `caption` (boolean). The type was only reachable as
  `openai_agents_sdk::tool::WebSearchToolImageSettings`; it is now also re-exported from the
  crate root (`src/lib.rs`, additive). The validator's drift detection those cases also exercise is
  repository tooling.

test_memory.py ledger basis: the ledger says 9; matching every test name against the Rust tests
finds 14 uncited pytest cases (the dictionary/forward-compat/invalid model-settings groups and the
server-managed streaming group are parametrized, and earlier rounds counted per definition). This
round ports 10 of the 14 (8 passing, 2 ignored) and leaves 4 valid; the CSV keeps the ledger's
`before` = 9, so ported = 9 - 4 = 5. The other uncited names (`accepts_renamed_limit_field`,
`preserves_typed/disabled_model_settings`, `preserves_positional_arguments_and_default_turn_limit`,
`phase_two_enforces_configured_turn_limit`) are already counted ported under adapted names in
`py_sandbox_a_memory_1.rs` (`memory_generate_config_accepts_renamed_limit_field_and_keeps_default_turn_limit`,
`memory_generate_config_preserves_typed_and_disabled_model_settings`) and `py_sandbox_a_memory_2.rs`
(`sandbox_memory_consolidation_turn_budget_on_session_close`, `sandbox_memory_turn_exhaustion_releases_runner_owned_session`).

## Divergence found, not fixed (owner handoff)

A sandbox run resumed from a `RunState` with an SDK session records the paused run's original
input again in its memory rollout segment. Python records an empty `input` for that segment
(`_sandbox_memory_input` returns the session delta, which is `[]` on resume:
`run.py` sets `session_input_items_for_persistence = []` for a resumed state with a session; the
streamed path sets `_original_input_for_persistence = []`). Rust feeds `result.input`
(`SandboxRunner::run_to_outcome` -> `manager.enqueue_result`) and `completion.input`
(the streamed completion hook -> `build_rollout_payload`) straight through, both in
`src/sandbox/runtime.rs`, which another agent owns. Effect: the turn is duplicated in the rollout
the phase-1 model extracts from. Suggested fix: in both paths, when the input is
`RunInput::State`, an SDK session is set and none of `conversation_id` / `previous_response_id` /
`auto_previous_response_id` is set, record `input: []`. The two tests in
`py_core19_r19c_sandbox_memory.rs` carry the `#[ignore]` reason and pass once fixed (drop the
attribute).

## Part 2 findings

- test_released_api_contract.py: only four cases read the real SDK rather than synthetic modules:
  `fixture_matches_installed_version`, `repository_release_policy_declares_v020_contract_surfaces`,
  `repository_release_policy_declares_public_state_surfaces` (the three ported in
  `py_core14_r14c_public_api_contract.rs`) and the nine `WebSearchToolImageSettings` cases ported
  now. `released_opaque_sentinel_default` (a private Python sentinel object),
  `voice_testing_start_sentinel`, `release_contract_promotion_rejects_missing_voice_concrete_state_policy`
  and `repository_release_policy_declares_public_optional_modules` (numpy extra, docs directives)
  read SDK objects only to exercise the contract tooling or Python packaging. Every other case
  drives `integration_tests/_contract_*` against `SimpleNamespace` modules.
- No case in the other twelve files tests SDK behaviour on its own. SDK behaviour they touch
  incidentally, and its Rust coverage: decisions (a sequential input guardrail tripwire means no
  model call: `py_core_b_guardrails.rs::blocking_guardrail_prevents_agent_execution`; a failing
  `needs_approval` evaluator: the Rust evaluator returns `bool`, so it cannot fail),
  human-in-the-loop server (approve/reject one call and resume, `failure_error_function=None`
  re-raising: covered by the HITL and tool-error ports), shell examples (per-call approval does
  not authorize the next batch: the scoped-approval ports in `py_core2_scopedapprovals_hitl*.rs`).

## Source changes

- `src/lib.rs`: `WebSearchToolImageSettings` re-exported from the crate root (additive, not
  rustfmt-ed).
- No other source change. No breaking change.

## Gates

- `cargo test --all-features` over 188 binaries (every test file that uses
  `openai_agents_sdk::sandbox`, plus the three new files), in five batches: 2173 passed, 1 failed,
  4 ignored (the 2 new divergence tests among them). The failure,
  `py_core6_modalvercel_modal_pty_hydrate::pty_keeps_the_session_live_until_the_delayed_tail_arrives`,
  is a timing test; rerun alone it passes (13/13). Not related to this change.
- New binaries: `py_core19_r19c_sandbox_memory` 4 passed / 2 ignored, `py_core19_r19c_unix_local`
  1 passed, `py_core19_r19c_public_surface` 2 passed.
- `cargo clippy --lib` + the three new test files, `-D warnings`: clean with `--all-features` and
  with default features (the default build compiles the feature-gated files to nothing).
- `rustfmt --edition 2024` on the three new test files; `src/lib.rs` not formatted.

## Suggested changelog paragraph

Root export and memory-rollout audit: `WebSearchToolImageSettings` is now re-exported from the
crate root, as Python exports it from `agents`. New parity tests cover the rollout input a streamed
server-managed sandbox run records, dictionary model settings for memory generation, and the
off-thread workspace removal of `UnixLocalSandboxClient::delete`. Two ignored tests document a
remaining divergence: a sandbox run resumed with an SDK session re-records its original input in
the memory rollout, where Python records none.

## Unresolved questions

- The resumed-run memory input divergence needs the owner of `src/sandbox/runtime.rs`.
- Run-as (bound-user) Add File stages and hard-links, so a failed write leaves no partial target
  and reports `workspace_write_type_error`, where Python's in-place worker leaves the partial file
  and reports "Inspect the destination". Rust's behaviour is stricter (nothing to inspect). The
  ledger's 6 for this file are read here as the three `replace_target=True` and three
  close-failure cases; the `replace_target=False` bound-user case is not ported anywhere, so the
  file's earlier ported/covered tally (7/2) may hold one case too many.

## Read depth

- Bodies read in full: test_manifest_config.py, test_workspace_payloads.py,
  test_unix_local_create.py, test_asyncio_tasks.py, test_source_compat_constructors.py,
  test_run_state_pydantic_context.py, test_shell_examples.py, test_decisions_examples.py,
  test_human_in_the_loop_server_example.py; test_memory.py lines 440-545, 805-935, 1670-1715,
  1855-1862 (every uncited case); test_unix_local.py the seven uncited bodies;
  test_asyncio_progress.py lines 1-80 in full, 76-282 skimmed for the inspected constructs.
- test_released_api_contract.py: every test name listed; bodies read for the cases that import
  real SDK objects (located by grep) and about 15 tooling cases sampled; the rest classified by
  name plus the fact that their modules are `SimpleNamespace`s.
- Name list plus header/fixture read (loader of the `.github/scripts` file or `examples.*` import),
  not every body: test_check_optional_truthiness.py, test_change_detection.py,
  test_integration_runner.py, test_release_automation.py, test_run_examples_script.py,
  test_dapr_session_example.py, test_live_example.py (header and two bodies),
  realtime/test_twilio_example.py, realtime/test_twilio_sip_server.py.
