# Core12 r12a: provider, session-state and any-llm Python-only cases

Crate 0.14.0. Scope: every Python-only case of the sandbox provider tests (E2B, Vercel, Modal, Blaxel, Daytona, Cloudflare), the sandbox mount, snapshot and session-state tests, and `tests/models/test_any_llm_model.py` (the ledger has no `tests/extensions/models/` rows with Python-only cases; the any-llm tests live under `tests/models/`).

## Method

- The ledger gives counts, not case ids. For each file the Python test names were matched against all Rust sources (with and without the `test_`/`any_llm_` prefixes) and the unmatched ones were read in full (body, helpers and the Rust code they exercise). The unmatched lists are larger than the ledger counts for E2B (22 vs 14), Cloudflare (10 vs 5), Modal (44 vs 41) and any-llm (23 vs 21): earlier rounds had already moved some of them under other names. Row arithmetic below follows the ledger count.
- A case ended as ported (new Rust test), covered (existing Rust test, cited), or valid Python-only (reason below).

## Per-file table

| file | before | ported | covered | remaining | remaining reasons |
|---|---|---|---|---|---|
| `extensions/sandbox/test_blaxel.py` | 40 | 0 | 12 | 28 | covered: read-not-found shapes (`py_sandbox_blaxel.rs::blaxel_read_404_is_a_not_found`), 408/504 timeout (`py_core6_blaxel_session.rs::exec_maps_408_and_504_to_a_timeout`), hardlink and device tar members (`py_core6_blaxel_archive.rs`), `supports_pty` needs the key (`supports_pty_needs_an_api_key`), PTY finalize/collect (`py_core6_blaxel_pty.rs`), `_FakeFs` test-double self-tests. Remaining: SDK import guards/`aiohttp` presence probes, pydantic `ge=1`, dynamically typed payloads (`WorkspaceWriteTypeError`), `delete(wrong type)`, drives-API-absent on the vendor SDK, asyncio cancellation during PTY cleanup |
| `extensions/sandbox/test_cloudflare.py` | 5 | 3 | 0 | 2 | ported: bucket mount/unmount/read/write path validated by the in-sandbox resolver (source fix). Remaining: asyncio cancellation of resume start (3 cases, one of them overlapping the ledger count), private `_before_shutdown`/`_runtime_helpers` hooks. Hydrate-around-mounts and terminal-delete observability are covered by `py_core6_cfdaytona_cloudflare.rs::resume_start_unmounts_before_hydrate_when_sandbox_is_running` and `resume_reapply_surfaces_terminal_delete_failure` |
| `extensions/sandbox/test_daytona.py` | 6 | 2 | 0 | 4 | ported: PTY teardown bounded when logs stall. Remaining: package re-export/import checks (2), SDK timeout-type monkeypatch, asyncio waiter task internals |
| `extensions/sandbox/test_e2b.py` | 14 | 5 | 3 | 6 | ported: `start` skips the files API when the root exists, `mkdir` of the root without a command, `mkdir` parent probe, remount error winning over an archive failure, clear-on-resume keeps nested mounts. Covered: `defaults_missing_mcp` (Rust state carries no `mcp`; `py_core6_cfdaytona_e2b.rs::e2b_resume_*`), the two `normalize_path` symlink cases (`py_core6_cfdaytona_paths.rs`). Remaining: package/SDK-class import resolution, positional options ctor, PTY waiter/exit-visible/prune asyncio internals, `httpcore` exception class names, PTY per-call `timeout` argument (`PtyExecRequest` has none) |
| `extensions/sandbox/test_modal.py` | 41 | 6 | 11 | 24 | ported: options payload defaults (3), unset image builder version stays off the create request, named secret without environment, failed remount after a directory snapshot. Covered: named-secret and strategy round-trips, `bucket_config_*`, reconnect/restore resume cases (`py_core6_modalvercel_modal_persist.rs`), nested bucket clear-on-resume (generic, `py_sandbox_a_runtime_2.rs`). Remaining: SDK `_call_modal` timeout-argument spies (3), Windows-host path normalization, asyncio cancellation of snapshot persist/hydrate (about 12), PTY fallback-read clocks over SDK streams |
| `extensions/sandbox/test_vercel.py` | 40 | 2 | 3 | 35 | ported: mount failure never repeats the transformed secret, absolute symlink destination rejected before upload. Covered: mount command timeout includes output (`the_exec_timeout_covers_output_collection`), mountpoint upgrade (`sandbox_e2b_daytona_vercel_tests.rs` dnf case), native snapshot resume (`...vercel_tests.rs:1457`). Remaining: `from_state`/constructor with Python's two-source trusted mounts and pydantic mutation of supplied manifests (Rust takes one owned manifest), traceback-frame-local redaction, hostile exception descriptors, asyncio lock/cancellation/shutdown races (about 20), env-var token lookup done inside the vendor SDK |
| `extensions/sandbox/test_vercel_provider_contract.py` | 5 | 1 | 0 | 4 | ported: resume whose status poll fails creates a replacement. Remaining: network policy round-trip through a state payload (Rust state keeps provider options in the client; the wire shapes are covered by the unit tests in `vercel/mod.rs`), cancelled creation (asyncio), unadopted-client close (httpx handle lifecycle) |
| `models/test_any_llm_model.py` | 21 | 2 | 5 | 14 | ported: `extra_body` fields reach the request once, native `reasoning_content`. Covered under renamed tests: stream via chat handler, default model at call time, both span-before-completed cases (`py_core6_modelsrt_stream_terminal_span.rs`), usage-present stream not double counted. Remaining: provider `aclose()` failure/cancellation/in-flight close (11), Google SDK `http_options`/role normalization (3), dynamically typed payload shapes, module fixture teardown |
| `sandbox/test_mount_lifecycle.py` | 4 | 0 | 0 | 4 | asyncio cancellation of mount teardown/restore |
| `sandbox/test_mount_security.py` | 130 | 4 | 1 | 125 | ported: the protected-error redactor keeps code/operation/retry flag, blanks the rest, and leaves credentialless manifests alone (source fix). Covered: credentialless hosted strategies (`py_sandbox_a_mount_security_3.rs::a_credentialless_external_mount_round_trips_without_a_redaction_marker`). Remaining: pydantic subclass/`__deepcopy__`/descriptor/`SystemExit`/exception-group/traceback-local hostility, custom `Mount` subclasses spoofing trust (Rust strategies are trait impls that declare their own authority), and about 45 cases that sanitize a manifest inside a state payload (Rust `SandboxSessionState` has only a manifest digest, so the guarantee is structural and covered by `state_json_never_contains_credentials_or_host_paths`) |
| `sandbox/test_session_sinks.py` | 4 | 0 | 0 | 4 | asyncio/test-helper (not in the requested pattern; r11b judged them) |
| `sandbox/test_session_state_roundtrip.py` | 7 | 0 | 3 | 4 | covered: host-path redaction and rebind, path-only grants, unmarked host path discarded (`py_sandbox_a_snapshot_1.rs`). Remaining: pydantic subclass registry, env-value subclasses, deepcopy of extension state |
| `sandbox/test_session_utils.py` | 2 | 0 | 0 | 2 | `io.IOBase` seekability probes |
| `sandbox/test_snapshot.py` | 10 | 0 | 4 | 6 | covered: remote snapshot persist/restore/restorable, remote snapshot in state, session dependencies reach the remote client (`py_sandbox_a_snapshot_2.rs`, the client registry replaces `set_dependencies`). Remaining: pydantic frozen/duplicate-registration/discriminator rules, stream `close()` accounting, atomic copy from a failing stream source |

Totals: before 329, ported 25, covered 42, remaining 262.

## Tests added

All under `tests/py_core12_r12a_*.rs` (27 tests): `e2b` (8), `cloud` (4: Cloudflare 3, Daytona 1), `vercel` (3), `modal` (6), `mount_security` (4), `any_llm` (2).

## Rust source fixes

- `src/extensions/sandbox/e2b/mod.rs`: `start` probes `test -d <root>` and calls the filesystem API only when the root is missing, and the manifest applier's root `mkdir` right after `start` no longer repeats the call (an unavailable files API no longer fails a start whose root exists). `mkdir` of the root goes straight to the files API without remote path validation (it must work when the root vanished), and `mkdir` without `parents` now probes the parent with a POSIX `test -d` and fails with the non-zero-exit error when the parent is missing (envd's MakeDir always created parents).
- `src/extensions/sandbox/cloudflare/mounts.rs`: bucket mount and unmount validate the mount path with the in-sandbox resolver (for write) and send the resolved path, as Python's `mount_bucket`/`unmount_bucket` do; a path the resolver refuses is never mounted.
- `src/sandbox/sandboxes/docker.rs::redact_for_protected_mounts`: the retry flag survives redaction. Before, a transient failure on a session with credentialed mounts always became non-retryable.

Existing tests adjusted for the E2B start probe (`py_sandbox_e2b.rs`, `py_core6_cfdaytona_e2b.rs`, `sandbox_e2b_daytona_vercel_tests.rs`): their mock sandbox now answers the new `test -d` probe with exit 1 (a missing root, like Python's fake), and the "tolerates an existing root" case calls `mkdir` explicitly.

## Findings that were not divergences

- E2B failed native snapshot followed by a failed re-attach: no detach gap. The mounts are detached in order, the platform snapshot fails, the failing re-attach is recorded as the cause, the remaining mount is still attached, and the archive failure is kept in `snapshot_error_before_remount_corruption` (tests `a_failed_remount_keeps_attaching_and_wins_over_the_archive_error` and `a_failed_native_snapshot_then_a_failed_remount_still_attaches_the_rest`).
- any-llm `extra_body`: Python asserts the SDK call arguments; over HTTP a key that collides with an explicit setting (temperature) takes the `extra_body` value, which is the OpenAI client's merge rule.
- Vercel resume with a failing status poll already replaced the sandbox correctly; Daytona PTY teardown was already bounded.

## Breaking changes

- E2B `mkdir(path, parents=false)` now errors when the parent is missing (previously it created parents silently). `mkdir(.., true)` and the manifest applier are unchanged.
- Protected-mount errors now carry the original `retryable` flag.

## Concerns / unresolved

- Provider options (Vercel runtime, network policy, interactive, timeouts; Modal builder version and resources; E2B mcp) live in the Rust client options, not in `SandboxSessionState`, so the Python state-payload round-trip/default tests have no direct counterpart; resume uses the client's options. Persisting them into `provider_metadata` would be a design change, not done here.
- The E2B/Daytona PTY per-call `timeout` argument has no Rust field; a stalled PTY start is bounded only by the HTTP client defaults.
- Ledger counts and unmatched-name lists disagree for four files (see Method); the table follows the ledger, the real unmatched sets were all read.
- Not covered: `tests/models/test_litellm_*` (11 Python-only cases), outside the requested list.

Status: DONE_WITH_CONCERNS
Summary: 329 cases before; 25 ported (27 Rust tests, 3 source fixes), 42 covered, 262 remain valid Python-only.
Concerns: see above; final gate results are appended below.

## Gate results

- `cargo test --workspace --all-features --no-fail-fast`: 7403 passed, 0 failed.
- `cargo clippy --workspace --all-features --all-targets -D warnings` and the default-feature run: clean after allowing the unused harness items in `py_core12_r12a_vercel.rs`.
- rustfmt applied to the new test files only.
