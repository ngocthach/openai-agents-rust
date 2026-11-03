# Python test parity: sandbox-b (tests/sandbox capabilities, sinks, utils, runtime helpers, skills)

Status: DONE. 651 of 806 Python test rows ported, 36 covered by other Rust tests, 119 python-only, 0 not ported (806 total). 375 Rust test functions in 34 `tests/py_sandbox_b_*.rs` files. Whole sandbox area (lib + `sandbox_*` + `py_sandbox_*` + `py_core_a_stack` + `py_core_a_sticky_approvals`, `--all-features`): 2054 passed, 0 failed, 2 ignored (the two diagnostic stack probes). `cargo clippy --all-features -- -D warnings` is clean on the lib and every `py_sandbox_b_*` target.

## Counting method

`def test_` plus `@pytest.mark.parametrize` expansion computed with the stdlib `ast` (static list literals; product of all parametrize decorators on a function and its class). The venv has no pytest, so `--collect-only` was not available. Corrections: one parametrize list uses a starred comprehension (`test_instructions_keep_multi_line_frontmatter_descriptions`, 2 modes x 45 cases, AST saw 30 of 45 per mode), counted as 90, and `test_bounded_provider_read_preserves_retry_policy` has a non-literal list, counted as 1. Total 806 tests in 39 files (0-test helper modules excluded). The first report revision said 776; the +30 is the corrected starred list.

Buckets: ported = translated to a Rust test (a Rust test may loop over many Python parameter rows); covered = another Rust test asserts the same code path; python-only = cannot exist in Rust; not ported = see "Remaining".

## Per-file table

| file | #tests | ported | covered | python-only | not ported |
|---|---|---|---|---|---|
| capabilities/test_apply_patch_preflight.py | 1 | 1 | 0 | 0 | 0 |
| capabilities/test_apply_patch_tool.py | 51 | 40 | 8 | 3 | 0 |
| capabilities/test_compaction_capability.py | 7 | 6 | 0 | 1 | 0 |
| capabilities/test_filesystem_capability.py | 7 | 5 | 0 | 2 | 0 |
| capabilities/test_shell_capability.py | 40 | 37 | 0 | 3 | 0 |
| capabilities/test_skills_capability.py | 184 | 175 | 0 | 9 | 0 |
| capabilities/test_tool_input_errors.py | 19 | 17 | 0 | 2 | 0 |
| capabilities/test_view_image_tool.py | 13 | 13 | 0 | 0 | 0 |
| integration_tests/test_runner_pause_resume.py | 1 | 1 | 0 | 0 | 0 |
| test_apply_patch.py | 41 | 41 | 0 | 0 | 0 |
| test_bounded_read.py | 60 | 41 | 17 | 2 | 0 |
| test_client_options.py | 13 | 0 | 0 | 13 | 0 |
| test_compaction.py | 24 | 24 | 0 | 0 | 0 |
| test_compatibility_guards.py | 56 | 29 | 0 | 27 | 0 |
| test_docker_network_mode.py | 20 | 6 | 10 | 4 | 0 |
| test_errors.py | 5 | 5 | 0 | 0 | 0 |
| test_exposed_ports.py | 6 | 6 | 0 | 0 | 0 |
| test_external_provider_example.py | 8 | 0 | 0 | 8 | 0 |
| test_helper_startup.py | 8 | 8 | 0 | 0 | 0 |
| test_mount_lifecycle.py | 8 | 4 | 0 | 4 | 0 |
| test_parse_utils.py | 13 | 13 | 0 | 0 | 0 |
| test_posix_tool_paths.py | 4 | 4 | 0 | 0 | 0 |
| test_pty_output.py | 30 | 29 | 0 | 1 | 0 |
| test_pty_types.py | 6 | 4 | 0 | 2 | 0 |
| test_recursive_removal_compatibility.py | 17 | 17 | 0 | 0 | 0 |
| test_remote_mount_policy.py | 4 | 4 | 0 | 0 | 0 |
| test_retry.py | 7 | 5 | 0 | 2 | 0 |
| test_run_cwd.py | 5 | 5 | 0 | 0 | 0 |
| test_runtime_agent_preparation.py | 12 | 7 | 1 | 4 | 0 |
| test_runtime_helpers.py | 8 | 8 | 0 | 0 | 0 |
| test_sandboxes_import.py | 3 | 0 | 0 | 3 | 0 |
| test_session_manager.py | 10 | 7 | 0 | 3 | 0 |
| test_session_sinks.py | 16 | 11 | 0 | 5 | 0 |
| test_session_utils.py | 21 | 19 | 0 | 2 | 0 |
| test_token_truncation.py | 15 | 15 | 0 | 0 | 0 |
| test_types.py | 2 | 2 | 0 | 0 | 0 |
| test_view_image_content_validation.py | 8 | 8 | 0 | 0 | 0 |
| test_workspace_jsonl_sink.py | 44 | 34 | 0 | 10 | 0 |
| test_workspace_payloads.py | 9 | 0 | 0 | 9 | 0 |
| TOTAL | 806 | 651 | 36 | 119 | 0 |

### Rust test files and the Python rows they cover

Test function counts are `#[test]`/`#[tokio::test]` per file (375 across 34 files, plus the shared fake `py_sandbox_b_support/mod.rs`).

| Rust file | tests | Python file(s) |
|---|---|---|
| py_sandbox_b_apply_patch.rs | 41 | test_apply_patch.py (+ `_apply_patch_test_session.py` fixture) |
| py_sandbox_b_apply_patch_tool.rs | 24 | capabilities/test_apply_patch_tool.py, test_apply_patch_preflight.py (the other half of the native-sticky matrix is `py_core_a_sticky_approvals.rs`, counted as covered) |
| py_sandbox_b_bounded_read.rs | 9 | test_bounded_read.py (E2B, Daytona, Vercel prefix and hang-up, retry policy by status, missing file) |
| py_sandbox_b_bounded_read_docker_modal.rs | 7 | test_bounded_read.py (Docker trusted utilities, Modal descriptor close, large bounded requests, retry policy, cancellation) |
| py_sandbox_b_bounded_read_providers.rs | 11 | test_bounded_read.py (Cloudflare SSE and wire limit, Daytona timeouts and status, E2B stream failure, body timeout and close, Vercel primary failure during close, Blaxel status policy, cancellation closes the response, retry policy) |
| py_sandbox_b_capabilities.rs | 44 | capabilities/test_shell_capability.py, test_filesystem_capability.py, test_compaction_capability.py |
| py_sandbox_b_compat.rs | 8 | test_compatibility_guards.py (core discriminators, session-state round trip, run-state shape), test_docker_network_mode.py (state rows) |
| py_sandbox_b_compat_providers.rs | 19 | test_compatibility_guards.py (provider options, session state, mount strategy type strings, every manifest entry, pattern, strategy, env value and snapshot kind round trip) |
| py_sandbox_b_credentials.rs | 6 | no Python row: pins the coordinator decision that credentials appear only on explicit serialization (Debug, errors, logs, session state, RunState) |
| py_sandbox_b_deferred_prepare.rs | 4 | no Python row of mine: pins `SandboxRunner` preparing after blocking input guardrails (Python `test_runner_*guardrail_trip*` rows are sandbox-a's) |
| py_sandbox_b_docker_live.rs | 3 | test_docker_network_mode.py (create omits network mode, resume reapplies it), test_recursive_removal_compatibility.py (live Docker authority, real daemon) |
| py_sandbox_b_helper_startup.rs | 2 | test_helper_startup.py |
| py_sandbox_b_jsonl_sink.rs | 22 | test_workspace_jsonl_sink.py (incl. mounted-outbox rebind, bounded-read retryability) |
| py_sandbox_b_misc.rs | 25 | test_types.py, test_errors.py, test_compaction.py, test_remote_mount_policy.py, test_exposed_ports.py, test_retry.py, test_posix_tool_paths.py |
| py_sandbox_b_mount_lifecycle.rs | 5 | test_mount_lifecycle.py |
| py_sandbox_b_pause_resume.rs | 1 | integration_tests/test_runner_pause_resume.py (full fixture: 9 entry types, mock tools, patch, escape, archive, PTY, pause, resume, tree) |
| py_sandbox_b_posix.rs | 1 | test_posix_tool_paths.py (memory layout probe) |
| py_sandbox_b_pty.rs | 20 | test_pty_types.py, test_pty_output.py, test_parse_utils.py |
| py_sandbox_b_removal.rs | 6 | test_recursive_removal_compatibility.py (local and exec-only backends, leaf alias, as-user, default-remote restore) |
| py_sandbox_b_run_cwd.rs | 3 | test_run_cwd.py |
| py_sandbox_b_runtime.rs | 6 | test_runtime_agent_preparation.py (rows sandbox-a does not cover) |
| py_sandbox_b_runtime_helpers.rs | 8 | test_runtime_helpers.py |
| py_sandbox_b_session_manager.rs | 3 | test_session_manager.py (snapshot spec choice and fallback) |
| py_sandbox_b_session_read_span.rs | 1 | test_session_sinks.py (expected read span error is call-scoped; one test per binary, trace processors are global) |
| py_sandbox_b_session_sink_failure.rs | 1 | test_session_sinks.py (expected read span records finish sink failure; one per binary) |
| py_sandbox_b_session_sinks.rs | 2 | test_session_sinks.py (apply_manifest keeps write instrumentation; forwarding of helpers records no event) |
| py_sandbox_b_session_utils.rs | 15 | test_session_utils.py |
| py_sandbox_b_sink_traces.rs | 1 | test_session_sinks.py (trace-dependent rows; one test per binary) |
| py_sandbox_b_sinks.rs | 11 | test_session_manager.py (payload policies, chained sinks), test_session_sinks.py |
| py_sandbox_b_skill_traces.rs | 1 | capabilities/test_skills_capability.py (probe span rows; one per binary) |
| py_sandbox_b_skills.rs | 33 | capabilities/test_skills_capability.py, skill row of test_run_cwd.py |
| py_sandbox_b_token_truncation.rs | 15 | test_token_truncation.py |
| py_sandbox_b_tool_input_errors.rs | 2 | capabilities/test_tool_input_errors.py (plain and streamed runs, backend failures) |
| py_sandbox_b_view_image.rs | 15 | capabilities/test_view_image_tool.py, test_view_image_content_validation.py |

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

## Cross-area findings

- Native `apply_patch_call` and the custom `apply_patch` tool used one approval key; the core agent now scopes sticky approvals by invocation type and ported the native-sticky matrix (`py_core_a_sticky_approvals.rs`). The 8 rows of `test_multi_operation_checker_ignores_native_sticky_approval` that this area did not port are therefore counted as covered.
- The default `SandboxSession::rm` / `rm_as` now resolve symlinks inside the sandbox through the installed resolver helper (see Round 3), so the exec-only (remote) rows of the recursive-removal tests are ported.
- clippy `-D warnings` over the whole workspace is not part of this area's gate; `src/sandbox`, the lib and all `py_sandbox_b_*` targets are clean.

## Decisions made by the coordinator (applied)

1. `Filesystem` defaults to Python's tools (`view_image`, then the custom `apply_patch`); native `apply_patch_call` is opt-in.
2. `LocalDirLazySkillSource` carries entry metadata like Python's `LocalDir`.
3. Client-option credentials serialize only when a caller serializes explicitly (as Python's `model_dump`); they are redacted from `Debug`, errors, logs, persisted session state and persisted RunState, with a test per surface (`py_sandbox_b_credentials.rs`).
4. The persisted registry covers every entry, mount pattern, mount strategy, env value and snapshot kind, with a round-trip test per variant (`py_sandbox_b_compat_providers.rs`).

## Notes for reviewers

- `exec_command` with a missing `workdir`: Python returns a model-visible exit-code-1 body; Rust raises `exec_nonzero`. Divergence, not a defect.
- `write_new_file` default probes with a read before mkdir+write; UnixLocal overrides with an exclusive create.
- `AgentsError::Sandbox` carries only code+message, so Python assertions on `error.context[...]` are checked on message text where the error crosses that boundary.
- Trace-processor tests are one test per binary (`py_sandbox_b_sink_traces`) because processors are process-global.
- Build dir `/Users/ngocthach/.cache/oars-build/tp-sbxB` deleted at the end.



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

## Round 4 (reconciliation with the tree)

Rows added since the previous revision (all green): P1 provider compat guards and docker rows (27 rows: client options, session state, mount strategies, core registry, docker network, live Docker removal, posix probe); P2 bounded reads (Cloudflare SSE and wire limit, Daytona timeouts and status, E2B stream and body timeout, Vercel close, Blaxel status, cancellation, retry policy); P3 skills, streamed tool-input rows, removal as-user and default-remote restore, the full pause/resume integration row, the mounted-outbox rebind; P4 session manager and session sinks. In this pass: bounded-read sanitization, the E2B body timeout, jsonl bounded-read retryability (4 rows), the mounted rebind, and the pause/resume semantics.

Rust source changes in this pass:

- `SandboxSession::read_bounded` is now a provided method that sanitizes: a missing file is `WorkspaceReadNotFound`; any other failure becomes `WorkspaceArchiveReadError` whose context holds only `path` and `reason` (`bounded_read_failed` or `bounded_read_wire_limit`) with the provider's retryability, dropping provider text, context and cause, as Python's `read_bounded` does over `_read_bounded`. Backends implement the new `read_bounded_unchecked` (Docker, Modal, E2B, Daytona, Vercel, Blaxel, Cloudflare renamed; `InstrumentedSession` forwards both). **Breaking** for custom sessions that overrode `read_bounded`: rename to `read_bounded_unchecked` to get the sanitizing wrapper.
- E2B's default HTTP client now has a 60 s connect and idle-read timeout (Python's SDK default request timeout), so a stalled download is a retryable timeout instead of a hang.
- `SandboxSessionGuard::suspend` now matches Python's pause: capture the snapshot, read the session state, then stop, shut down and delete the session. A paused run no longer leaves the workspace or backend alive; the resumed run restores from the snapshot (`restore_session_snapshot`). **Breaking**: the three assertions in `tests/sandbox_tests.rs` that pinned "a suspended session keeps its workspace" now assert Python's behaviour. Callers that relied on reattaching to a live workspace must resume from the returned `SandboxSessionState`.
- The rclone config step runs `chmod 0600 <path>` as Python does (the previous `chmod 600 -- <path>` failed on macOS, so every rclone mount failed there); `tests/py_sandbox_a_entries_3.rs` pinned the old spelling and was updated.
- Added `Entry::type_discriminator()` and `ConfiguredMountStrategy::mount_type()` (additive).

## Rows not ported

None. Every one of the 806 Python rows is ported, covered or python-only.

Python-only (119) is kept to rows that cannot exist in Rust, listed by reason above (pydantic validation and export surfaces, asyncio cancellation and `__cause__` internals, `caplog` assertions, vendor-SDK objects with no Rust counterpart, Python module imports and example scripts). Environment-gated: the live-Docker removal row runs against the real daemon and prints a skip reason only if `docker info` fails; here the daemon was up (`docker info` succeeded) and the test ran. The two `mv` fallback tests return early without GNU `mv` (Python skips); the fingerprint nested-name test is Linux-only like Python. The pause/resume row was run on macOS only.

Shared-file edits: none to `src/lib.rs` or `Cargo.toml`. Edits to other areas' tests: `tests/sandbox_tests.rs` and `tests/py_sandbox_a_runtime_4.rs` (earlier rounds), `tests/py_sandbox_a_entries_3.rs` (chmod spelling, this pass). Output dir `/Users/ngocthach/.cache/oars-build/tp-sbxB` deleted at the end.

Status: DONE
Summary: 806 Python rows: 651 ported, 36 covered, 119 python-only, 0 not ported. 375 Rust tests in 34 `py_sandbox_b_*` files. Area run 2054 passed, 0 failed, 2 ignored; clippy -D warnings clean on the lib and all my targets.
Concerns/Blockers: none blocking. Behaviour changes to review: the suspend semantics (pause now releases the sandbox), the sanitizing `read_bounded` / new `read_bounded_unchecked` hook, E2B default client timeouts.
