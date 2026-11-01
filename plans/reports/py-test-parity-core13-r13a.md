# Core13 r13a: provider options in session state, PTY start timeouts, observable dropped writes

Crate 0.15.0. Scope: the three gaps left by the core12 reports (`py-test-parity-core12-r12a.md` and `-r12b.md`, concerns sections).

## 1. Provider options in the persisted session state

`SandboxSessionState` has no per-provider subclasses in Rust. The provider fields Python's state classes carry are written into one `provider_metadata` entry (`provider_options`, helpers `SandboxSessionState::with_provider_options` / `provider_options`), as a typed record per provider with serde defaults:

| provider | record | fields | missing-field default |
|---|---|---|---|
| Vercel | `VercelPersistedOptions` | `runtime`, `resources`, `timeout_ms`, `snapshot_expiration_ms`, `network_policy` (Python's payload shape: `"allow-all"`, `"deny-all"`, `{allow, subnets}`) | unset |
| Modal | `ModalPersistedOptions` | `image_builder_version`, `idle_timeout_secs`, `cpu`, `cpu_limit`, `memory_mb`, `memory_limit_mb` | builder `2025.06`, rest unset (Python's defaults) |
| E2B | `E2BPersistedOptions` | `mcp`, non-secret shape only (see below) | none |

Each client's `resume` now layers the record over its own options (`with_persisted_options`), so a replacement sandbox, a native-snapshot hydrate and the session state the replacement hands out keep the original's runtime, limits, builder version and MCP servers. A state without the record (an older payload) reads and uses the resuming client's options.

Tests: `py_core13_r13a_vercel_state.rs` (6), `py_core13_r13a_modal_state.rs` (5), `py_core13_r13a_e2b_state_pty.rs` (state part: 8, including the native-snapshot restore). They re-point the Python state-payload cases (`test_vercel_create_passes_provider_options`, `..._serialized_session_state_...`, `..._resume_falls_back_to_recreate...`, `test_installed_vercel_preserves_released_policy_and_state`, the three `test_modal_deserialize_session_state_defaults_missing_*`, `test_modal_sandbox_create_uses_custom_image_builder_version`/`..._passes_resources`, `test_e2b_deserialize_session_state_defaults_missing_mcp`, `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`) at the real state. Earlier rounds had ported most of them against the client options, so only `test_installed_vercel_preserves_released_policy_and_state` leaves the Python-only bucket here.

Not persisted: Python's `interactive`, `env`/`base_envs`, `metadata`, E2B `timeouts`, `template`, Vercel `project_id`/`team_id` (Rust has no counterpart or keeps credentials/ids out of the state). The record is nested under `provider_metadata.provider_options`, not at the payload's top level, so a Python payload is not readable as is (the Rust state was never wire-compatible: it has its own `$schemaVersion`).

Deliberate difference from Python (coordinator decision): the state stays credential-free. Python persists the whole E2B `mcp` mapping; Rust persists server names and non-secret options only. An option whose name looks secret (`key`, `token`, `secret`, `password`, `auth`, `credential`, `cookie`, `bearer`, `signature`, `session`) is dropped, and a URL value loses userinfo, query and fragment. On resume the client's own MCP config is layered over the recorded shape (client wins per key; servers only the client knows are kept), so secrets must come from the resuming client. A client with no MCP config starts the servers without their secrets. Tests assert no secret string appears in the serialized state, before or after a resume.

## 2. E2B and Daytona PTY start timeout

`PtyExecRequest` gains `timeout: Option<Duration>` (`with_timeout`). Python: `pty_exec_start(..., timeout=)`.

- E2B: the timeout is the process lifetime envd is told to enforce (`Connect-Timeout-Ms`; none = the 24 h unbounded sentinel, zero = 1 s, as `_coerce_exec_timeout`) and also bounds a start request that stalls (`ExecTimeout`, nothing registered).
- Daytona: each provider call of the start (toolbox session create, command start, terminal connect including the handshake) is bounded by the timeout (zero = 1 ms); an elapsed step is `ExecTimeout`, the half-created toolbox session is deleted, nothing stays registered.

Tests: `py_core13_r13a_e2b_state_pty.rs` (PTY part: 2), `py_core13_r13a_daytona_pty_timeout.rs` (5). Ported cases: `test_e2b_pty_start_maps_timeout_failures`, `test_pty_start_maps_sdk_timeout_failures`. Not asserted: the `httpcore` context fields (`reason=stream_read_timeout`, `provider_error`), which name Python exception classes.

## 3. Dropped output-guardrail write and context-scoped encrypted append

- `EncryptedSession`: a context-scoped append now runs under `await_session_mutation` like the context-free one. `SessionRunContext::detach()` makes an owned copy (`OwnedSessionRunContext`, shares the run's context, usage and approvals) so the write can move to its own task. `SessionRunContext` keeps its public API; only its private field changed. Test: `py_core13_r13a_encrypted_shield.rs` (2): a caller dropped between items does not leave a partial batch, the store sees the right tenant, a live caller still sees the failure.
- Detached write after a dropped output-guardrail run: before appending, the task reads the history tail digests; on failure it logs `Failed to save the turn of a run that was dropped during its output guardrails to the session` under the model/tool data redaction policy (error text only when both policies allow it) and, if the run was given a `RunCheckpoint`, records the batch there as `pending_session_write` (with the tail digests as `before`). A resume from that checkpoint reconciles it: unchanged tail repeats the append, `tail + batch` accepts a lost acknowledgement, anything else is refused. The checkpoint is published before the write starts, so the failure handler cannot lose the race with the `Drop` that built it. Test: `py_core13_r13a_dropped_write.rs` (2), including serialization and a resume against the recovered session.

What stays impossible, precisely:
- The dropped caller never receives the error (a dropped future has no one to return it to). Python's shielded `finally` raises it into the awaiting coroutine.
- A run without a `RunCheckpoint`, or whose checkpoint was already `take()`n, leaves only the log line.
- If the tail pre-read itself fails the record has `before = null`; the resume then reads the tail first and cannot detect an append that half-landed on a non-atomic store.
- No Tokio runtime at drop time, or the runtime shutting down, loses the write (unchanged).
- A caller dropped before the shielded future is first polled never starts the write (same as Python before `shield` is entered).

## Rust source changes

`src/sandbox/state.rs`, `src/sandbox/pty.rs`, `src/extensions/sandbox/{vercel/mod.rs,modal/sandbox.rs,modal/mod.rs,e2b/mod.rs,daytona.rs}`, `src/memory/{mod.rs,encrypted_session.rs}`, `src/run.rs`, `src/run_internal/{mod.rs,session_write.rs}`.

## Breaking changes

- `PtyExecRequest` has a new public field `timeout`: code that builds it with a struct literal (not `new`) must add it.
- A persisted session state now carries `provider_metadata.provider_options`; consumers that compared states for equality against states built elsewhere see a new entry. Old payloads still deserialize.
- E2B: the persisted state carries only the non-secret shape of the MCP config; secrets come from the resuming client's options (differs from Python, which persists them).

## Gates

- New binaries (`--all-features`): `py_core13_r13a_vercel_state` 6, `_modal_state` 5, `_e2b_state_pty` 10, `_daytona_pty_timeout` 5, `_encrypted_shield` 2, `_dropped_write` 2 passed. Default features: `_dropped_write` 2 passed (`_encrypted_shield` is gated on `encrypt`).
- Regression, `--all-features`, 0 failures: the provider and session binaries (`py_core11_r11b_*`, `py_core12_r12a_{e2b,modal,vercel}`, `py_core12_r12b_*` encrypt/runner, `py_core6_{cfdaytona,modalvercel}_*`, `py_core8_r8g3_*`, `py_memory_encrypt*`, `py_sandbox_{daytona,e2b,modal,vercel}`, `py_sandbox_b_pty`, `sandbox_*_tests`, `py_core6_blaxel_pty`, `py_core6_cfdaytona_cloudflare`, `sandbox_pty_signal_defaults`, `py_core2_runner_guardrail_sessions`), the `RunCheckpoint` users (`py_core2_runner_streamed`, `py_deferred_writes_e`, `py_core3_runstatectx_pending_b`, `py_core2_runstate_boundary`, `py_core6_resume_*`, `py_core8_r8g9_deferred`, `py_core2_toolerrors_batch`, `py_core12_r12b_agent_tool`), the `run_context::<T>()` users (`mcp_tests`, `py_mcp_tool_filtering`, `py_core8_r8g10_session_context`, `py_core9_r9gaps_rewind`), and the lib unit tests (671 passed). The full workspace suite was not run.
- clippy, my files: no warning in either feature set (`cargo clippy --all-features --all-targets` and the default run, read for my paths). `cargo clippy --all-features --all-targets -D warnings` as a whole currently stops on three lints in `src/sandbox/sandboxes/docker_removal.rs` (`too_many_arguments`, `collapsible_if`) and `docker_removal_worker.rs` (`unnecessary_cast`): untracked files of another agent's in-flight work, not mine. The default-feature `-D warnings` run is clean.
- rustfmt applied to the test files; the touched source files are rustfmt-clean.

## Unresolved questions

- Secret detection for E2B MCP options is by option name and URL shape; an unusually named secret option would still be persisted.
- `PtyExecRequest::timeout` is honoured by E2B and Daytona only; Modal, Cloudflare, Blaxel, Docker and unix_local start the process without a bound of their own (the field doc says so). Python's other providers take `timeout` too, and were not part of this task.
- Daytona bounds each start step by the full timeout, as Python does, so the whole start can take up to three timeouts.

Status: DONE_WITH_CONCERNS
Summary: 3 gaps closed; 30 new tests in 6 binaries; 3 Python-only cases moved to ported (`test_installed_vercel_preserves_released_policy_and_state`, `test_e2b_pty_start_maps_timeout_failures`, `test_pty_start_maps_sdk_timeout_failures`), 0 to covered ; the other state-payload cases listed in section 1 were already ported or covered against the client options and now assert the real state.
Concerns: the items under "What stays impossible"; E2B secret detection is name-based.
