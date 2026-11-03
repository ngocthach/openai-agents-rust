# Core11 r11b: independent sample of the remaining Python-only bucket

Crate 0.13.0. Question: of the 3,332 cases the ledger still files as `python_only`, how many carry an invalid reason (portable behaviour, or a Rust test already exists)?

## Method

- Frame: every ledger file with `python_only > 0`, minus the CI-script / example / release files (`test_release_*`, `test_run_examples_script`, `test_live_example`, `test_integration_runner`, `test_repository_workflow_interfaces`, `test_*_example.py`): 241 files, 3,028 cases. The excluded 14 files hold 304 cases and were only spot checked.
- Draw (seed 20261101, Python `random.Random`): files drawn with probability proportional to `python_only`, without replacement, until at least 60 files and about 400 cases were held; at most 7 cases per file (all of them when the file has fewer). Result: 65 files, 402 targeted cases.
- The ledger has counts, not case ids. A case is identified from (a) Python tests with no mention in any Rust source or doc comment, ranked by how few of their string literals occur in the Rust tests, (b) the group text of the r8, r9 and r10 reports, and (c) a read of the Python test. Picks that turned out to be ported already under another name (34) were discarded as outside the bucket; replacement picks were not drawn, so the in-bucket sample is 364 cases, not 402. For about ten files (test_extract, realtime/test_session, tool_batch_failure_history, trimmer, docker group, tool_use_behavior and a few more) the pool heuristic picked the wrong tests, so the real python-only group was identified from the reports and re-drawn from it (`sample2.json` order in the scratchpad).
- Every in-bucket case was judged against the valid-reason list: valid, covered (a Rust test exists, the ledger count is stale), or portable (invalid reason).

## Result

| | cases | share of 364 |
|---|---|---|
| valid python-only | 343 | 94.2% |
| already covered by a Rust test (stale count) | 9 | 2.5% |
| portable (invalid reason) | 12 | 3.3% |

- Strict invalid-reason rate (portable only): 12/364 = 3.3%, Wilson 95% interval 1.9% to 5.7%; mean-of-files with a file-level bootstrap 2.9%, 95% [1.1%, 5.2%].
- Broad rate (portable plus stale covered): 21/364 = 5.8%, Wilson 3.8% to 8.7%; mean-of-files 5.1%, 95% [2.5%, 8.1%]. Counting the 34 discarded already-ported picks as misfiled too would give 55/398 = 13.8%, an upper bound that assumes every discarded pick was inside the bucket.
- Extrapolated to the 3,028-case frame: about 100 portable cases (57 to 173), about 175 portable-or-covered cases (115 to 262). The excluded 304-case group is repository CI scripts and Python example apps; the spot check (7 files, about 20 test names: release automation, integration runner, workflow-interface checks, examples runner, live example, twilio example, dapr example) found no portable case.
- The strict rate is under 5%, the broad rate over it with a lower interval bound under 5%. Because the broad rate and the intervals straddle the threshold and the failures concentrate, the audit was continued on the concentrating group (below).

## Where invalid reasons concentrate

| group (sampled cases) | invalid / sampled |
|---|---|
| Provider and model extension tests: e2b 3/7, any_llm 3/7, codex tool 1/6, vercel 2/7 (both stale), modal 0/7, blaxel 0/6 | 9/40 (22%) |
| Sandbox session-state payloads (session_state_roundtrip) | 3/7 |
| Session stores and compaction (memory/test_session, responses compaction, advanced sqlite, encrypt) | 3/18 |
| MCP server errors | 2/6 (both stale) |
| Core runner (streamed runner 1/6, agent_as_tool 1/7) | 2/13 |
| Zero in about 250 cases: CI-script-like files, examples, pydantic/typing introspection, asyncio internals, gc/identity, Docker removal service, run-state private helpers, test-helper self-tests | 0 |

The cause is consistent: those files test provider adapters through Python SDK fakes, and the earlier audits grouped them under "SDK objects" without reading the Rust adapter, which exposes the same behaviour over HTTP. The r8g3 report already said of E2B `workspace_root_ready` that it "looks portable with extra harness work".

## Per-file table (sampled files)

`python_only` is the ledger count; `judged` is the in-bucket cases read.

| # | file | python_only | judged | valid | covered | portable | reason / note |
|---|---|---|---|---|---|---|---|
| 1 | `test_released_api_contract.py` | 126 | 7 | 7 | 0 | 0 | CI script release contract validator |
| 2 | `memory/test_session.py` | 13 | 7 | 5 | 1 | 1 | asyncio/threads/dyn typing V; resolve C; P=settings ctor |
| 3 | `test_run_state_compatibility_corpus.py` | 62 | 7 | 6 | 1 | 0 | python exception/traceback redaction internals |
| 4 | `sandbox/test_session_state_roundtrip.py` | 10 | 7 | 4 | 1 | 2 | pydantic registry V; P=malformed json+non-mapping manifest redaction |
| 5 | `test_start_hook_tool_concurrency.py` | 35 | 7 | 7 | 0 | 0 | shared mutable Agent.tools hazard impossible (Rust &Agent) |
| 6 | `sandbox/test_extract.py` | 3 | 3 | 3 | 0 | 0 | io.IOBase stream adapters (real group) |
| 7 | `test_trace_processor.py` | 14 | 7 | 7 | 0 | 0 | non-JSON values (real group) |
| 8 | `realtime/test_twilio_sip_server.py` | 14 | 6 | 6 | 0 | 0 | Python example server module |
| 9 | `sandbox/test_docker_removal.py` | 64 | 7 | 7 | 0 | 0 | docker removal service (owner decision)  |
| 10 | `test_code_change_verification_runner.py` | 19 | 7 | 7 | 0 | 0 | CI script |
| 11 | `mcp/test_server_errors.py` | 86 | 6 | 4 | 2 | 0 | final |
| 12 | `sandbox/test_memory.py` | 16 | 7 | 7 | 0 | 0 | pydantic cfg/dataclass/private helpers/registry |
| 13 | `realtime/test_session.py` | 34 | 7 | 7 | 0 | 0 | asyncio task/close internals (real group) |
| 14 | `test_tool_batch_failure_history.py` | 32 | 7 | 7 | 0 | 0 | REVISED: approved_failure_diagnostics needs a failing lifecycle hook; Rust hooks return () so no trigger; others identity/cyclic/asyncio |
| 15 | `test_items_helpers.py` | 4 | 4 | 4 | 0 | 0 | gc/weakref, pydantic |
| 16 | `sandbox/test_retry.py` | 2 | 2 | 2 | 0 | 0 | exception chain walking |
| 17 | `extensions/sandbox/test_vercel.py` | 42 | 7 | 5 | 2 | 0 | asyncio lock, from_state ctor private; C dynamic mount, exec timeout |
| 18 | `sandbox/test_docker.py` | 23 | 7 | 7 | 0 | 0 | docker-py sockets/spool/pty sockets/asyncio cancel/probe script (group by keyword; 5 pool picks already ported dropped) |
| 19 | `voice/test_openai_model_provider.py` | 7 | 5 | 4 | 1 | 0 | dyn ctor validation, httpx type; wire tests already ported dropped |
| 20 | `test_check_optional_truthiness.py` | 71 | 7 | 7 | 0 | 0 | CI script (AST linter for Python source) |
| 21 | `extensions/sandbox/test_modal.py` | 41 | 7 | 7 | 0 | 0 | modal SDK/pydantic/asyncio/env |
| 22 | `sandbox/test_mount_security.py` | 130 | 7 | 7 | 0 | 0 | hostile pydantic/SystemExit/decorator internals/subclass |
| 23 | `test_run_state_pending_input.py` | 11 | 4 | 4 | 0 | 0 | private tracker/identity/hook failure (hooks return ()); 5 picks already ported dropped, 3 group members added |
| 24 | `models/test_any_llm_model.py` | 24 | 7 | 4 | 0 | 3 | close failure/cancel V; P x3 any-llm cache_write default, prompt_cache_retention unset, chat content_filter refusal |
| 25 | `memory/test_openai_responses_compaction_session.py` | 13 | 4 | 3 | 0 | 1 | duck-typing/asyncio V; P strip_orphaned empty; 3 ported dropped |
| 26 | `extensions/memory/test_advanced_sqlite_session.py` | 36 | 4 | 4 | 0 | 0 | close/asyncio-threads/SQL spy; 3 ported dropped |
| 27 | `sandbox/test_compatibility_guards.py` | 27 | 6 | 6 | 0 | 0 | __all__/dataclass field order/inspect |
| 28 | `test_scripted_model.py` | 29 | 7 | 7 | 0 | 0 | tests of Python test-helper ScriptedModel: close(), aliasing, __str__, dict keys, KeyboardInterrupt |
| 29 | `test_example_workflows.py` | 30 | 7 | 7 | 0 | 0 | Python examples/ scripts |
| 30 | `test_run_state.py` | 134 | 7 | 7 | 0 | 0 | private _deserialize_*/get_interruptions deepcopy/type-system/vacuous line-coverage (r9h1 groups) |
| 31 | `test_source_compat_constructors.py` | 17 | 7 | 7 | 0 | 0 | positional ctor binding / legacy kwargs |
| 32 | `test_run_step_execution.py` | 45 | 6 | 6 | 0 | 0 | CancelledError/BaseException/hooks return ()/hand-built processed responses; completed-mcp replay already ported |
| 33 | `test_agent_runner_streamed.py` | 24 | 6 | 5 | 0 | 1 | private queue/run_loop_task/asyncio/logging V; P streamed omit-policy strips stored reasoning ids; session-level id strip covered |
| 34 | `realtime/test_app_server_security.py` | 29 | 7 | 7 | 0 | 0 | examples/realtime/app FastAPI demo server |
| 35 | `extensions/experiemental/codex/test_codex_tool.py` | 27 | 6 | 5 | 0 | 1 | python object contexts/asyncio/private resolver; P codex persist_session reuse thread; thread_id from input covered |
| 36 | `test_handoff_history_duplication.py` | 10 | 7 | 7 | 0 | 0 | object identity / private helpers / dataclass introspection (r9h1) |
| 37 | `test_programmatic_tool_calling.py` | 7 | 4 | 4 | 0 | 0 | return-annotation/typing inference (r9h3) |
| 38 | `test_function_tool_decorator.py` | 50 | 7 | 7 | 0 | 0 | callable objects/__wrapped__/inspect (r9h2 read case by case) |
| 39 | `sandbox/test_runtime.py` | 54 | 7 | 7 | 0 | 0 | SandboxSession wrapper API, deepcopy, hooks mutating Agent in place, run_loop_task; 1 weak (SandboxAgent.as_tool under plain Agent parent: Rust uses SandboxRunner) |
| 40 | `sandbox/test_docker_removal_worker.py` | 58 | 7 | 7 | 0 | 0 | docker removal in-container worker (ctypes/descriptors) + owner decision |
| 41 | `realtime/test_openai_realtime.py` | 17 | 7 | 7 | 0 | 0 | asyncio/monkeypatched private _emit_event/falsy pydantic/caplog redaction |
| 42 | `realtime/test_app_server_debug.py` | 4 | 4 | 4 | 0 | 0 | examples/realtime app server debug logging |
| 43 | `test_server_conversation_tracker.py` | 5 | 5 | 5 | 0 | 0 | id() identity (3) + tracker policy ctor arg (2; policy applied by runner, covered at runner level) weak |
| 44 | `test_agent_instructions_signature.py` | 5 | 5 | 5 | 0 | 0 | dynamic arity/type validation |
| 45 | `test_agent_runner.py` | 32 | 7 | 7 | 0 | 0 | dynamic type, equality impostors, pydantic warnings, BaseException monkeypatch, asyncio |
| 46 | `extensions/sandbox/test_e2b.py` | 19 | 7 | 4 | 1 | 2 | e2b: V httpcore x2, remount-by-unmount design, stale-flag mkdir; C normalize_path (py_core6_cfdaytona_paths); P running() gate PORTED+source fix; P clear-root-on-resume unported |
| 47 | `test_image_generation_tool_typing.py` | 3 | 3 | 3 | 0 | 0 | pyright canaries/pydantic TypeAdapter |
| 48 | `sandbox/test_dependencies.py` | 2 | 1 | 1 | 0 | 0 | asyncio cancellation private; eager retry already covered dropped |
| 49 | `realtime/test_playback_tracker.py` | 1 | 0 | 0 | 0 | 0 | pick already ported (py_realtime_transport) dropped; no in-bucket case |
| 50 | `extensions/test_tool_output_trimmer.py` | 4 | 4 | 4 | 0 | 0 | unsigned args/dynamic bytes allowlist (group from r8g3) |
| 51 | `realtime/test_session_guardrails.py` | 1 | 1 | 1 | 0 | 0 | callable without __name__ + caplog |
| 52 | `test_native_tool_failure_history.py` | 34 | 7 | 7 | 0 | 0 | asyncio CancelledError/hooks return () (r9h1 read all 34) |
| 53 | `sandbox/test_docker_removal_client.py` | 14 | 4 | 4 | 0 | 0 | docker removal service (owner decision) |
| 54 | `memory/test_session_context_wrapper.py` | 5 | 5 | 5 | 0 | 0 | inspect.signature opt-in (r9gaps: all 5) |
| 55 | `extensions/sandbox/test_blaxel.py` | 40 | 6 | 6 | 0 | 0 | private PTY entries/import guards/pydantic ge=1; 1 weak (supports_pty without url: Rust has no optional url); exec timeout covered dropped |
| 56 | `extensions/memory/test_encrypt_session.py` | 15 | 3 | 3 | 0 | 0 | asyncio interleavings/inherited subclass pop; 4 picks already covered in py_memory_encrypt dropped |
| 57 | `test_agent_as_tool.py` | 24 | 7 | 6 | 0 | 1 | run registry/scope ids/BaseException/stream close/identity V; +1 P-hard below + P-HARD agent_as_tool resume survives cancellation after nested output commit (needs checkpoint publication for nested agent-tool state; not attempted) |
| 58 | `sandbox/test_session_sinks.py` | 4 | 4 | 4 | 0 | 0 | python test helper/asyncio; callback bound-session param weak |
| 59 | `test_function_schema.py` | 46 | 7 | 7 | 0 | 0 | pydantic/typing Annotated/*args annotations |
| 60 | `tracing/test_traces_impl.py` | 4 | 4 | 4 | 0 | 0 | ContextVar tokens/logging/generator close |
| 61 | `test_tool_name_collision_policy.py` | 6 | 6 | 6 | 0 | 0 | deepcopy snapshots / asyncio cancellation (4 read, 2 same names pattern) |
| 62 | `test_function_tool.py` | 23 | 5 | 5 | 0 | 0 | asyncio to_thread/**kwargs/dynamic types; 2 covered dropped |
| 63 | `test_error_logging_redaction.py` | 83 | 6 | 6 | 0 | 0 | logging record attrs/exception chains/BaseException groups/warnings module; 1 covered dropped |
| 64 | `test_tool_use_behavior.py` | 6 | 2 | 2 | 0 | 0 | typed-API/private empty-list helper; 3 custom behavior cases covered in py_core_a_tool_behavior dropped |
| 65 | `test_agent_scoped_approvals.py` | 40 | 6 | 6 | 0 | 0 | Temporal converter/gc/object-identity name-only queries (r9gaps) |

Totals over the sampled files: 1,910 ledger cases; moved to ported 13 (10 sampled cases plus 3 further E2B cases found while continuing the audit); moved to covered 9; 1,888 remain. See the CSV.

## Portable cases and what was done

Ported (13), tests in `tests/py_core11_r11b_*.rs` unless noted:

- `memory/test_session.py::test_session_settings_constructor`: `py_core11_r11b_misc.rs` (SqliteSession keeps the settings it was built with).
- `sandbox/test_session_state_roundtrip.py` malformed-JSON and non-mapping-manifest redaction (2): `py_core11_r11b_session_state.rs`.
- `models/test_any_llm_model.py` (3): `py_core11_r11b_any_llm.rs` (Responses usage without `cache_write_tokens`, unset `prompt_cache_retention` stays off the request, chat `content_filter` becomes a refusal). No Rust divergence.
- `memory/test_openai_responses_compaction_session.py::TestStripOrphanedAssistantIds::test_noop_when_empty`: assertion added to the unit test `orphaned_assistant_ids_are_stripped` in `src/memory/openai_responses_compaction_session.rs`.
- `test_agent_runner_streamed.py::test_streamed_omit_policy_strips_reasoning_ids_already_stored_in_the_session`: `py_core11_r11b_misc.rs`. Passes; no divergence.
- `codex/test_codex_tool.py::test_codex_tool_persists_session`: `py_core11_r11b_codex_session.rs` (second call resumes the thread; without `persist_session` it starts fresh). Passes.
- `extensions/sandbox/test_e2b.py` (4): `test_e2b_running_requires_workspace_root_ready` and `test_e2b_exec_omits_cwd_until_workspace_ready` (both needed a source fix, below; tests in `tests/py_sandbox_e2b.rs`), and `persist_workspace_remounts_mounts_after_snapshot` plus `persist_workspace_remounts_prior_mounts_after_unmount_failure`: `py_core11_r11b_e2b_mounts.rs`, through `persist_session_snapshot`, which is where Rust detaches provider mounts.

Covered, moved (9): `test_session_settings_resolve`, `test_run_state_restores_an_exact_context_wrapper...` (`py_core8_r8g8_runstate_identity`), `test_exposed_ports_are_normalized`, `paginated_list_failure_does_not_retain_opaque_cursor` and `error_name_sanitizes_url_derived_names` (`py_core8_r8g4_mcp_errors`, `py_mcp_server_errors`), vercel `s3_dynamic_mount_is_rejected_before_materialization` and `mount_command_timeout_includes_output_collection` (`py_core8_r8g3_vercel_live_manifest`, `py_core6_modalvercel_vercel_mounts`), voice provider `explicit_options_override_default_client`, e2b `normalize_path_rejects_symlink_escape` (`py_core6_cfdaytona_paths`).

Not ported, still portable (2):

- `test_e2b_clear_workspace_root_on_resume_preserves_nested_mounts` (Rust `start` restores or applies the ephemeral entries but never clears the old root first; unverified whether a reattached sandbox can hold stale files).
- `test_agent_as_tool_resume_survives_cancellation_after_nested_output_commit` (2 cases): needs the nested agent-tool progress published into a caller-visible checkpoint on cancel, the mechanism r9h1 built for sessions; not attempted.

Found while continuing on the provider group, not ported: E2B `start_skips_files_api_when_workspace_root_exists` (Rust always calls the filesystem MakeDir first; low value), Daytona `persist_workspace_remounts_*` (2, same generic flow as the E2B pair; needs the Daytona harness), and Cloudflare/Daytona/Blaxel name-level reads of the remaining cases found only stale or SDK-object cases.

## Source changes

`src/extensions/sandbox/e2b/mod.rs`:

- `running()` now returns false until `start` has prepared the workspace root, as Python's `workspace_root_ready` gate. Before, a created-but-unstarted session reported running whenever the platform said so; `SandboxRunner` and `ensure_session` skip `start()` for a running caller-supplied session, so a fresh E2B session handed to the runner was never started (manifest unapplied). The workspace JSONL sink also holds its buffer until the session is usable.
- `exec` with no explicit cwd runs without one before `start`, instead of passing a root that may not exist yet.

Breaking behaviour: an E2B session that was created but not started now reports `running() == false`. `tests/py_sandbox_e2b.rs` was updated accordingly (the two running tests and the root-and-environment exec test start the session first; two tests added). No other crate or public signature changed.

Rejected change, with evidence: wrapping Daytona/Blaxel/Cloudflare/E2B `persist_workspace` in `with_ephemeral_mounts_removed`. Provider mount strategies return `supports_native_snapshot_detach() == false`, so `persist_session_snapshot` and `restore_session_snapshot` already detach and re-attach them; an extra wrapper detached twice and broke seven existing Cloudflare tests and one Daytona test. Reverted.

## Gates

- New binaries: `py_core11_r11b_any_llm` (3), `_misc` (2), `_session_state` (2), `_codex_session` (2), `_e2b_mounts` (2): all pass with `--all-features`; the five compile and pass with default features.
- Touched or related binaries pass: `py_sandbox_e2b` (45), `py_core6_cfdaytona_e2b`, `py_core6_cfdaytona_paths`, `py_core6_cfdaytona_cloudflare`, `py_core6_cfdaytona_daytona`, `sandbox_e2b_daytona_vercel_tests`, `sandbox_modal_cloudflare_blaxel_tests`, `py_core6_blaxel_*`, `py_sandbox_{b_bounded_read,b_bounded_read_providers,b_compat_providers,b_credentials,cloudflare,daytona,blaxel}`, lib compaction tests.
- `cargo clippy --all-features --all-targets -- -D warnings` and `cargo clippy --all-targets -- -D warnings`: clean. rustfmt applied to my files. The full workspace suite was not run.

## Unresolved questions

1. E2B: through `persist_session_snapshot`, a native snapshot that fails with HTTP 500 and a mount that then fails to re-attach produced an event sequence with no detach or attach recorded before the error (a third mount-transition case, `keeps_remounting_and_raises_remount_error_first`, was dropped for that reason). Not investigated; it may be a real gap in how a failed native snapshot interacts with the detach helper.
2. Should the remaining provider files (vercel 40, modal 41, blaxel 40 beyond the sample; any_llm 14 and codex 21 beyond name-level) be read in full next round? The sample puts roughly one case in four of this group at stake.
3. The 34 discarded picks suggest the ledger `python_only` counts for some files still include cases already ported (compare `test_playback_tracker`, whose single python-only case is the ported `interrupt_clamps_truncate_to_received_audio`); a ledger reconciliation by test id would give an exact figure.

Status: DONE_WITH_CONCERNS
Summary: 65-file PPS sample, 364 judged cases: 94.2% valid, 3.3% portable (95% 1.9-5.7%), 5.8% portable-or-stale (3.8-8.7%); about 100 portable cases extrapolated to the 3,028-case frame (about 175 counting stale ones). Ported 13 cases (3 went through a source fix in the E2B session), moved 9 to covered, 1,888 of 1,910 remain in the sampled files.
Concerns: two portable cases left unported; the continued audit of provider files was name-level for any_llm, codex, daytona and cloudflare; full workspace suite not run.
