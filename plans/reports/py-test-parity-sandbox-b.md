# Python test parity: sandbox-b (tests/sandbox capabilities, sinks, utils, runtime helpers, skills)

Status: DONE_WITH_CONCERNS. 600 of 806 Python tests ported, 28 covered, 118 Python-only, 60 not ported (the 8 sticky-approval rows are routed to the core agent and not counted here). Rust tests in 25 `tests/py_sandbox_b_*.rs` files, all green; whole sandbox suite (lib + `sandbox_*` + `py_sandbox_*` + `py_core_a_stack`, `--all-features`) 1985 passed, 0 failed. `cargo clippy --all-features -- -D warnings` is clean on the lib and every `py_sandbox_b_*` target.

## Counting method

`def test_` plus `@pytest.mark.parametrize` expansion computed with the stdlib `ast` (static list literals; product of all parametrize decorators on a function and its class). The venv has no pytest, so `--collect-only` was not available. Corrections: one parametrize list uses a starred comprehension (`test_instructions_keep_multi_line_frontmatter_descriptions`, 2 modes x 45 cases, AST saw 30 of 45 per mode), counted as 90, and `test_bounded_provider_read_preserves_retry_policy` has a non-literal list, counted as 1. Total 806 tests in 39 files (0-test helper modules excluded). The first report revision said 776; the +30 is the corrected starred list.

Buckets: ported = translated to a Rust test (a Rust test may loop over many Python parameter rows); covered = another Rust test asserts the same code path; python-only = cannot exist in Rust; not ported = see "Remaining".

## Per-file table

| file | #tests | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| capabilities/test_apply_patch_preflight.py | 1 | 1 | 0 | 0 | 0 |
| capabilities/test_apply_patch_tool.py | 51 | 40 | 0 | 3 | 8 |
| capabilities/test_compaction_capability.py | 7 | 6 | 0 | 1 | 0 |
| capabilities/test_filesystem_capability.py | 7 | 5 | 0 | 2 | 0 |
| capabilities/test_shell_capability.py | 40 | 37 | 0 | 3 | 0 |
| capabilities/test_skills_capability.py | 184 | 175 | 0 | 9 | 0 |
| capabilities/test_tool_input_errors.py | 19 | 17 | 0 | 2 | 0 |
| capabilities/test_view_image_tool.py | 13 | 13 | 0 | 0 | 0 |
| integration_tests/test_runner_pause_resume.py | 1 | 0 | 0 | 0 | 1 |
| test_apply_patch.py | 41 | 41 | 0 | 0 | 0 |
| test_bounded_read.py | 60 | 25 | 17 | 1 | 17 |
| test_client_options.py | 13 | 0 | 0 | 13 | 0 |
| test_compaction.py | 24 | 24 | 0 | 0 | 0 |
| test_compatibility_guards.py | 56 | 9 | 0 | 27 | 20 |
| test_docker_network_mode.py | 20 | 4 | 10 | 4 | 2 |
| test_errors.py | 5 | 5 | 0 | 0 | 0 |
| test_exposed_ports.py | 6 | 6 | 0 | 0 | 0 |
| test_external_provider_example.py | 8 | 0 | 0 | 8 | 0 |
| test_helper_startup.py | 8 | 8 | 0 | 0 | 0 |
| test_mount_lifecycle.py | 8 | 4 | 0 | 4 | 0 |
| test_parse_utils.py | 13 | 13 | 0 | 0 | 0 |
| test_posix_tool_paths.py | 4 | 3 | 0 | 0 | 1 |
| test_pty_output.py | 30 | 29 | 0 | 1 | 0 |
| test_pty_types.py | 6 | 4 | 0 | 2 | 0 |
| test_recursive_removal_compatibility.py | 17 | 16 | 0 | 0 | 1 |
| test_remote_mount_policy.py | 4 | 4 | 0 | 0 | 0 |
| test_retry.py | 7 | 5 | 0 | 2 | 0 |
| test_run_cwd.py | 5 | 5 | 0 | 0 | 0 |
| test_runtime_agent_preparation.py | 12 | 7 | 1 | 4 | 0 |
| test_runtime_helpers.py | 8 | 8 | 0 | 0 | 0 |
| test_sandboxes_import.py | 3 | 0 | 0 | 3 | 0 |
| test_session_manager.py | 10 | 5 | 0 | 3 | 2 |
| test_session_sinks.py | 16 | 8 | 0 | 5 | 3 |
| test_session_utils.py | 21 | 19 | 0 | 2 | 0 |
| test_token_truncation.py | 15 | 15 | 0 | 0 | 0 |
| test_types.py | 2 | 2 | 0 | 0 | 0 |
| test_view_image_content_validation.py | 8 | 8 | 0 | 0 | 0 |
| test_workspace_jsonl_sink.py | 44 | 29 | 0 | 10 | 5 |
| test_workspace_payloads.py | 9 | 0 | 0 | 9 | 0 |
| TOTAL | 806 | 600 | 28 | 118 | 60 |

Rust test files: py_sandbox_b_{misc, token_truncation, capabilities, apply_patch, apply_patch_tool, view_image, pty, session_utils, runtime, runtime_helpers, helper_startup, sinks, sink_traces, jsonl_sink, compat, removal, mount_lifecycle, run_cwd, skills, bounded_read} plus shared fake `py_sandbox_b_support/mod.rs`.

## Follow-ups from the coordinator, done

- `py_sandbox_b_helper_startup::runner_snapshot_does_not_execute_workspace_startup_files`: gave the run a writable `SandboxPathGrant` for the temp dir holding `outside.txt`. Python's test is Linux-CI only (it has no macOS skip and its approved command writes outside the workspace), so the grant, not a skip, is the fix; the startup-file assertions are unchanged.
- sandbox-a's final changes (`write_new_file` overwrites, `manifest_digest` mount topology, `describe()` tree, PTY exit `-signum`, stop/shutdown spans): every `py_sandbox_b_*` test passes against them unchanged. My editor does not rely on `write_new_file` refusing an existing file through the trait default: the UnixLocal override is exclusive, and the exec-only default probes with a read first, so create-over-existing is still refused (`create_rejects_an_existing_file*` pass).

## Python-only reasons (one line per group)

- capabilities: bound/unbound mutable `Capability` objects, `configure_tools` callbacks that mutate or replace tool-set members, pydantic `model_validate` of policies, combining several skill sources in one pydantic model (Rust has one constructor per source, so a combination cannot be written), directory content as skill body (Rust skill content is bytes), custom lazy sources returning an arbitrary `dict`: Rust capabilities are stateless values bound per run.
- tool_input_errors `validation`/`cancel` failure kinds: pydantic `ValidationError` and `asyncio.CancelledError` as injected exceptions. `stops_when_approval_resolves`: an `asyncio.Event` interleaves a context approval with a running async checker.
- compat guards "export surface / dataclass field order / positional options order": Python `__all__`, `inspect.signature`, dataclass ordering; client_options (13), external_provider_example (8), sandboxes_import (3), workspace_payloads (9): pydantic polymorphic option registry, example script, import paths, `io.IOBase` write coercion.
- retry exception-chain helpers, pty `_settle_pty_cleanup`, injectable monotonic clock, `best_effort_stream_len`, mount-lifecycle cancellation shielding (3), sink `best_effort` task flushing and exec-span cancellation: asyncio/`__cause__` internals; a dropped Rust future just stops.
- sinks: caplog and `_debug.DONT_LOG_TOOL_DATA` log-redaction assertions, `on_error` raise/log/ignore on an infallible Rust `EventSink`, session callback arguments, header snapshotting.
- provider bounded reads that assert vendor SDK objects (E2B `files.read` stream, Modal descriptors, Vercel mount context around `iter_file`, Daytona `download_file_without_preload_content`): no Rust counterpart; the wire behaviour (prefix, hang up, no bytes in errors) is ported.

## Rust bugs found and fixed (src/**)

Round 1 (items 1-17 as in the first revision, abbreviated): shell `exec_command` always exposed with one-shot fallback; PTY refuses a missing cwd; byte-mode truncation and public truncation helpers; strict `try_parse_ls_la` (+ `ListingEntry`, `Permissions::try_from_ls_field`); per-code default `retryable`; `Permissions`/`Group` `Hash`; UnixLocal exposed ports; **archive command ran through a login shell, so a model-written `.profile` executed during snapshots (security)**; **a completed run never snapshotted (only suspend did)**; `SandboxEditor` rewrite (workspace-relative `normalize_operation`, Python messages, exclusive create, staged/aliased `move_to`, `same_file`, `mv -fT`); read-error classification with the existence probe; persist skip paths; `view_image` messages and bounded read; leading-backslash anchor; capability error text and run-cwd instruction block; `InstrumentedSession` forwarding + per-op and per-sink payload policy; `WorkspaceJsonlSink` delivery semantics (`try_emit`, budgets, lifecycle flush, `with_ephemeral`).

Round 2:

18. **Recursive removal could delete a protected read-only grant through a symlinked ancestor** (`workspace/link/tree` where `link` points at the grant's parent): the path policy judged the spelling only. `UnixLocalSandboxSession::rm` now re-checks the resolved tree against the pinned grants (workspace root still wins, a writable grant nested in a read-only one stays removable). `unix_local.rs`. Found by `test_recursive_remove_preserves_nested_read_only_grant[ancestor]`.
19. **Grant precedence picked the wrong grant**: `WorkspacePathPolicy::matching_grant` kept the last of equals (Python's `max` keeps the first), and UnixLocal's `authorize_real` took the first prefix match instead of the most specific, so a writable grant nested in a read-only one was refused. `workspace_paths.rs`, `unix_local.rs`.
20. Snapshot detach/restore did not follow Python's `with_ephemeral_mounts_removed`: restored in forward order, skipped restoring already-detached mounts after a failed detach, never shut the session down after a failed restore, and lost the operation error. New public `with_ephemeral_mounts_removed`, used by `persist_session_snapshot`. `session.rs`.
21. **Skills capability rewritten to Python's behaviour** (`skills.rs`): hand-written front-matter reader (`parse_frontmatter`, block scalars, chomping/indent indicators, nested fields, comments, CRLF), name/description from front matter with `No description provided.`, Python's instruction text (`(file: path)`, run-scoped paths, lazy sections, both how-to variants), `try_process_manifest` with Python's collision/overlap errors, new `Capability::check_manifest` hook so `process_manifest` errors surface at prepare time, `load_skill` returning `{status, skill_name, path}` (`already_loaded`, unavailable/not-found/ambiguous errors, scoped absolute path), runtime metadata read from the workspace as the run-as user, host sources confined to cwd or a grant. Breaking, see below.
22. Sandbox operations opened no spans: `InstrumentedSession` now wraps every operation in `sandbox.<op>` (error type/code/retryability on span data and error), and events drop `trace_id` under a disabled parent span. `events.rs`.
23. Bounded provider reads streamed whole files: E2B, Daytona, Vercel and Blaxel now stop at the prefix and hang up (`sandbox::http::read_body_prefix`, `bounded_read_zero`); a mid-stream failure is a transport error carrying no payload. Modal (helper call returns base64 of the whole file) keeps the default.

## Breaking changes / behaviour changes

- Round 1 list still applies (`normalize_operation` workspace-relative, Python apply_patch messages, `exec_command` on non-PTY sessions, preset `retryable`, `SandboxSessionEvent` serde-skip fields, `mv -fT`, snapshot on completed run).
- `SkillMetadata` is now `{name, description, path}` (dropped `deferred`, `compatibility`); `LazySkillSource` is now `list_skill_metadata(skills_path, grants)` (sync) + `load_skill(name, session, skills_path, user) -> LoadedSkill`; `Skills::load_skill` takes `(session, scope, user, name)`; `load_skill` tool output is the JSON record, not prose; skill instruction layout changed; a lazy source outside the cwd needs an extra path grant (as in Python).
- `Capability` gained `check_manifest` (default accepts).
- Grants: when two grants cover a path the most specific wins and the first declared wins a tie.

## Cross-area findings (not edited)

- Native `apply_patch_call` and the custom `apply_patch` tool are both named `apply_patch`, and Rust keys approvals by agent + tool name, so a sticky native approval would also approve the custom tool. Python scopes sticky approvals by tool-invocation type (`test_multi_operation_checker_ignores_native_sticky_approval`, 8 of 16 rows not ported). Fix is in `src/run_context.rs` / `src/run_internal/local_tools.rs`, outside sandbox.
- Exec-only backends (default `SandboxSession::rm`) validate the spelling only and do not resolve symlinks in the sandbox as Python's remote path does through the installed resolver helper; the `remote` rows of the recursive-removal tests (9) are not ported for that reason. Needs the helper wired into the default `rm`/`validate_recursive_remove`.
- clippy `-D warnings` over the whole workspace fails only in other areas (earlier list); `src/sandbox`, the lib and all `py_sandbox_b_*` targets are clean.

## Decisions needed (user)

1. Filesystem capability exposes native `apply_patch` by default; Python exposes only the custom tool. Flipping the default (`Filesystem::with_custom_apply_patch`) is breaking for recorded `apply_patch_call` conversations. (The default snapshot decision from the first report is moot: sandbox-a wired the default local snapshot.)
2. Python's `LocalDirLazySkillSource(source=LocalDir(...))` carries entry metadata (permissions, group applied on load); the Rust source takes a bare path. Supporting it means giving the source an `Entry` (2 rows not ported).

## Remaining (not ported) with reason

- test_bounded_read.py (31): Modal/Blaxel/Cloudflare/Docker/E2B-SDK specifics, retry-policy classification per provider (`preserves_retry_policy`, daytona timeout, vercel/e2b primary-failure-during-close, blaxel status policy, cancellation closes response/descriptor). Need per-provider mock sessions; the shared reader is tested.
- test_skills_capability.py (6): LocalDir entry metadata on load (2), run-as user on load (1), probe-span error scoping (2), non-not-found read error propagation (1).
- test_compatibility_guards.py (20): per-backend options/state JSON round trips for the cloud providers and their mount strategy type strings (6 + 6 + remaining discriminators); Rust pins the strategy strings in each provider's source but only core + E2B are asserted here.
- test_recursive_removal_compatibility.py (10): remote backend (9, see cross-area), live Docker authority (1).
- test_tool_input_errors.py (9): Python formats per-field "Invalid tool arguments for X" hints; Rust raises a tool failure the runner turns into a generic message. Behaviour differs by design; the safety guarantees are ported.
- test_apply_patch_tool.py (8): native-sticky half of the matrix (cross-area).
- test_workspace_jsonl_sink.py (5): mounted-outbox rebind with S3 mounts (needs a mounted-session double), legacy-stream variants that patch `read`.
- test_session_manager.py (2), test_session_sinks.py (3), test_docker_network_mode.py (2), integration pause/resume (1), posix memory-layout probe (1).
- Vacuous on macOS: the two `mv` fallback tests return early without GNU `mv` (Python skips); the fingerprint nested-name test is Linux-only like Python.

## Notes for reviewers

- `exec_command` with a missing `workdir`: Python returns a model-visible exit-code-1 body; Rust raises `exec_nonzero`. Divergence, not a defect.
- `write_new_file` default probes with a read before mkdir+write; UnixLocal overrides with an exclusive create.
- `AgentsError::Sandbox` carries only code+message, so Python assertions on `error.context[...]` are checked on message text where the error crosses that boundary.
- Trace-processor tests are one test per binary (`py_sandbox_b_sink_traces`) because processors are process-global.
- Build dir `/Users/ngocthach/.cache/oars-build/tp-sbxB` deleted at the end.

Status: DONE_WITH_CONCERNS
Summary: 562/806 ported, 28 covered, 118 python-only, 98 remaining; 23 Rust bugs fixed (security: login-shell startup files, snapshot on completion, recursive remove through a symlinked ancestor); skills rewritten to Python behaviour.
Concerns/Blockers: sticky-approval scoping and exec-only symlink-aware removal are outside sandbox; provider-specific bounded-read rows and 6 skills rows remain.


## Round 3 (coordinator decisions and follow-ups)

Behaviour changes (read these as breaking):

- `Filesystem` now defaults to Python's tools: `[view_image, apply_patch]` with `apply_patch` as the freeform custom tool (`SandboxApplyPatchTool`). The native `apply_patch_call` tool is opt-in through `Filesystem::with_native_apply_patch()` / `with_custom_apply_patch(false)`. `tests/sandbox_tests.rs` (a `native_defaults()` helper) and `tests/py_sandbox_a_runtime_4.rs` (two calls) were updated to opt in; `sandbox::agent::tests::prepares_an_agent_with_capability_tools_and_instructions` now expects `view_image` before `apply_patch` (this was the failing unit test, caused by the Python tool order, not by skills or `check_manifest`).
- `LocalDirLazySkillSource` carries entry metadata like Python's `LocalDir`: `LocalDirLazySkillSource::from_entry`, `entry()`; skills are materialized with the entry's permissions/group. `Capability::check_manifest` (default `Ok`) is new; `LazySkillSource::load_skill` reads through the new `SandboxSession::read_file_expecting`, which opens an expected-failure span.
- A bounded read that fails mid-stream (`sandbox::http::read_body_prefix`) is now `WorkspaceArchiveReadError` (retryable) instead of `ExecTransportError`, as in Python.
- The default `SandboxSession::rm` / `rm_as` call the new `SandboxSession::validate_remote_path_access(path, for_write, recursive_remove)` whenever a read-only grant exists: the installed resolver helper walks symlinks inside the sandbox (exit 111/113/114 mapped as in Python's remote path), so an alias cannot route a delete into protected data. Without a read-only grant nothing changes.
- `exec_command`, `write_stdin` and `view_image` answer invalid arguments with Python's one-line feedback ("Invalid tool arguments for <fields>. Check the tool schema and try again.", fields in declaration order, never the value) through `capabilities/tool_input_errors.rs`; malformed JSON gives "Invalid tool arguments.". They no longer raise a run error for bad input. Invalid workdir / path text was already Python's.
- Docker gained `read_bounded` (trusted `/bin/sh`, `PATH=/usr/bin:/bin`, no encoding, exit 44 not found, timeout 30 s). Modal gained `read_bounded` through a new helper op that opens the descriptor, reads in bounded 100 MiB requests, closes it (5 s limit), and reports provider failures as a reason plus retryability only.

Core hand-offs finished:

- `SandboxRunner` (`run_with_options`, `run_to_outcome`, `run_streamed_with_options`) now prepares through `RunOptions::prepare_agent`, after the first turn's blocking input guardrails, so rejected input never creates a session; streamed memory uses `StreamCompletion::last_agent_name`. Preparation runs on its own task (trace scope carried over with `ScopeSnapshot`) so the run keeps its stack margin: `py_core_a_stack` minimum stack sandbox 640 KiB (was 896), sandbox streamed 448 KiB. A resumed run (`RunInput::State`) is still prepared up front, because the saved state names an agent that exists only in the prepared handoff graph. A streamed run waits for preparation (or for the run to end without it) before returning the handle, so `session()` and `take_guard()` still work immediately. `tests/py_sandbox_b_deferred_prepare.rs` pins rejected-input (plain and streamed: 0 creates, 0 deletes) and accepted-input (1 create, 1 delete). `tests/py_sandbox_a_runtime_4.rs::a_tripped_guardrail_releases_the_runner_owned_session` had pinned the old up-front behaviour and now asserts Python's (0 sessions created).

New tests: `py_sandbox_b_bounded_read` (+4: HTTP retry policy 503/429/403, not-found, failed stream), `py_sandbox_b_bounded_read_docker_modal` (6), `py_sandbox_b_tool_input_errors` (2), `py_sandbox_b_removal` (remote backend rows, leaf alias), `py_sandbox_b_deferred_prepare` (4).

## Remaining (60 rows)

- bounded reads (17): per-provider rows that need vendor SDK objects (Cloudflare/E2B/Vercel descriptor close rows are covered by the wire-level tests, Daytona acquisition/body timeout retryability, Blaxel True/None policy), plus Cloudflare retry-policy rows.
- cloud-provider compat guards (20): per-provider options/session-state JSON round trips and mount-strategy type strings (modal, daytona, cloudflare, blaxel, vercel).
- recursive removal (1): the live-Docker authority row (needs a Docker daemon).
- apply_patch tool (8), session manager (2), session sinks (3), workspace JSONL sink mounted-outbox rebind (5), docker network (2), pause/resume integration (1), posix memory-layout probe (1).
- sticky-approval rows (8): routed to the core agent.

Shared-file edits: none (`src/lib.rs` and `Cargo.toml` untouched). Build dir `/Users/ngocthach/.cache/oars-build/tp-sbxB` removed.

Status: DONE_WITH_CONCERNS
Summary: Filesystem default and LocalDir entry metadata done; skills, tool-argument hints, recursive-removal remote rows, Docker/Modal bounded reads and retry policy ported; SandboxRunner now prepares after input guardrails via `prepare_agent`; the failing agent unit test is fixed. Whole area 1985 passed, clippy clean.
Concerns/Blockers: 60 rows remain unported (compat guards 20, bounded-read provider rows 17, small leftovers); Daytona timeout and Blaxel policy rows not done.
