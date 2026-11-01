# py-test-parity round 8, r8g2: tests/sandbox second half

CSV: `plans/reports/py-test-parity-core8-r8g2.csv` (before - ported - covered = remaining on every row).

## Per-file table

"Before" is the ledger's python_only count. Counts for parametrized tests follow my reading of the Python parameter lists, which does not always match the ledger's expansion exactly (noted where it matters).

| file | before | ported | covered | remaining | remaining reason (one per group) |
|---|---|---|---|---|---|
| test_mount_security.py | 135 | 4 | 1 | 130 | hostile pydantic subclass/`__deepcopy__`/traceback/exception-group redaction; raw-dict `model_validate` sanitizers (Rust state has no manifest, so a payload that embeds one is parsed without it and cannot be "rejected"); `redact_mount_error_data` decorator internals; contextvars/asyncio retained failures; `examples/` import |
| test_mounts.py | 2 | 2 | 0 | 0 | |
| test_pty_output.py | 1 | 0 | 0 | 1 | monkeypatched `time.monotonic` plus injected poll/settle callbacks |
| test_pty_types.py | 2 | 0 | 0 | 2 | asyncio cancel-reason handling in `_settle_pty_cleanup` |
| test_retry.py | 2 | 0 | 0 | 2 | `__cause__`/`__context__` exception-chain walking (Rust errors have no chain protocol) |
| test_runtime.py | 94 | 18 | 0 | 76 | see "test_runtime remaining" below |
| test_runtime_agent_preparation.py | 4 | 3 | 0 | 1 | dict/pydantic coercion of `model_settings` and `run_as` at the `SandboxAgent` constructor |
| test_sandboxes_import.py | 3 | 0 | 0 | 3 | Windows `monkeypatch` of the import system; `__all__` export check |
| test_session_manager.py | 3 | 3 | 0 | 0 | |
| test_session_sinks.py | 5 | 1 | 0 | 4 | Python test helper; callback second parameter (bound session) has no Rust API; asyncio cancellation of a finish sink; best-effort task flushing (Rust sinks have no delivery modes) |
| test_session_state_roundtrip.py | 17 | 7 | 0 | 10 | pydantic subclass registry / `parse()` / `exclude_unset` / `model_validate_json` (7); manifest-embedded `EnvValue` payloads (2, state has no manifest); no-deepcopy (1) |
| test_session_utils.py | 2 | 0 | 0 | 2 | `best_effort_stream_len` on Python `io` streams |
| test_snapshot.py | 10 | 0 | 0 | 10 | pydantic `exclude_unset`/frozen/duplicate-registration/missing-`type` class machinery; stream `close()` counts; duck-typed client without `exists`; `SandboxSession` wrapper delegation |
| test_tar_utils.py | 5 | 2 | 1 | 2 | stream-object failure injection (source read / output write fail); Rust strips in-memory bytes |
| test_unix_local.py | 12 | 2 | 0 | 10 | asyncio cancellation waits (6), monkeypatched tty fd close, `sudo` worker design (mkdir/rm as user), `tarfile.addfile` mid-capture patching |
| test_unix_local_create.py | 12 | 4 | 0 | 8 | `shutil.copyfileobj`/`os.fdopen` injection (close failure), replace-the-target-inside-the-copy variant, `bound-user` fixture runs Python's `sudo python3` worker |
| test_unix_local_file_io.py | 10 | 4 | 0 | 6 | streaming read handle (no Rust handle API); open-parent-stays-pinned via monkeypatched `os.open` (the Rust equivalent is the after-pin race test, which does not map one to one); thread-pool cancellation; manager-private seam; `example-user` legs (sudo worker) |
| test_unix_local_special_files.py | 5 | 2 | 0 | 3 | Linux `F_SETLEASE` (2, not runnable on macOS); `os.open` monkeypatch for lease retry |
| test_unix_local_user_file_io.py | 22 | 3 | 0 | 19 | the `sudo -u <user> python3 -I -S -c <worker>` body executed in-process with mocked `os`/`subprocess`; Rust runs user operations as `exec` commands |
| test_workspace_jsonl_sink.py | 10 | 9 | 0 | 1 | `loop.time` monkeypatch (backend-owned read deadline) |
| test_workspace_paths.py | 6 | 1 | 0 | 5 | `PureWindowsPath` argument types (3), `Path.exists` monkeypatch, `posix_path_as_path` pathlib identity |
| test_workspace_payloads.py | 9 | 0 | 0 | 9 | Python `io.IOBase` write coercion (`write` takes `&[u8]`) |
| **total** | **371** | **65** | **2** | **304** | |

### test_runtime: moved
* 16 cases: the macOS `sandbox-exec` confinement tests (`exec_confines_commands_to_workspace_root`, `exec_denies_launch_services`, `exec_rejects_when_confinement_is_unavailable`, `exec_runs_without_wrapper_on_linux`, `confined_exec_command_*` incl. parameter sets, `darwin_exec_profile_*`). The earlier report filed them Python-only ("feature absent"); the unix-local helper later ported them in `tests/py_sandbox_a_unix_local_3.rs`. Reclassified to ported by citation, not by new code.
* 4 cases (`test_unix_local_file_io.py`): the leaf-swap cases for rm, rmtree, mv and same_file. First counted as covered by a static-symlink test; now real race-injection ports (see "Race closing" below).
* 2 cases: `test_runner_owned_cleanup_redacts_client_delete_failure` (both parameter sets), ported in `tests/py_core8_r8g2_teardown_logging.rs`.

### test_runtime remaining (76)
Python `SandboxSession.aclose`/pre-stop-hook/dependency-close wrapper API and its asyncio cancellation and exception redaction (Rust teardown is `SandboxSessionGuard::close` -> `client.delete`; no pre-stop hooks exist); `__deepcopy__` and capability deepcopy; hooks that mutate `agent.tools`/`agent.name` in place (Rust hooks take `&Agent`); `run_loop_task`; monkeypatched `importlib.resources`; private `_nested_history_owned_session_item_refs`; `RunState._sandbox` dict shape; Python exception `__cause__` chaining in `protected_start`; duplicate-agent-name resume keys (see unresolved questions).

## New tests (all pass)
| file | tests | python cases |
|---|---|---|
| `tests/py_core8_r8g2_sinks_state.rs` | 6 | raising sink, header snapshot, legacy read failure, malformed-state redaction (7), embedded-manifest parse (4 mount-security rows, 1 covered by existing), tar payload (2) |
| `tests/py_core8_r8g2_sink_logging.rs` | 2 | logged sink failure x2, jsonl errors-no-payload x8 |
| `tests/py_core8_r8g2_create_failures.rs` | 3 | create race x2, failed create x2 (via `RLIMIT_FSIZE` and the after-validate hook) |
| `tests/py_core8_r8g2_unix_local_misc.rs` | 2 | FIFO replacement x2, PTY exit flush x2 |
| `tests/py_core8_r8g2_user_file_io.rs` | 2 | user write failure/no-fallback x3 |
| `tests/py_core8_r8g2_teardown_logging.rs` | 1 | client delete failure x2 |
| `tests/py_core8_r8g2_pinned_ops.rs` | 2 | leaf swap x4 (rm, rmtree, mv, same_file) counted; parent swap x4 for the same operations is extra coverage the ledger count cannot absorb |
| `tests/py_core8_r8g2_manifest_payloads.rs` | 5 | mounts x2, grants x1, agent-prep x3 (needs `Manifest::to_value`/`from_untrusted_value` from the manifest-serde work) |

## Source bugs fixed
* `src/sandbox/state.rs` `SandboxSessionState::from_json_str`: the error quoted serde's text, which echoes offending values (`invalid type: string "<secret>"`) and the schema version. Now `sandbox session state payload is invalid` / `... JSON is invalid`, as Python.
* `src/sandbox/events.rs`: a failing sink was logged as `... delivery failed: <code>` at warn. Now `Sandbox event sink failed (ignored)` at error; the failure's text appears only when tool data may be logged (`dont_log_tool_data` false). `Instrumentation::send` carries that text in a `cause` context only under the same condition.
* `src/sandbox/runtime.rs`: suspend/teardown failures were logged with the raw provider error. A session holding a credentialed mount now logs the sentence alone; otherwise the tool-data policy applies. New helpers `session_protects_mounts`, `log_sandbox_failure`.
* `src/sandbox/sandboxes/unix_local.rs` `write_new_file`: a failure after the name was claimed returned `WorkspaceWriteTypeError`. Now `WorkspaceArchiveWriteError`, `retryable=false`, Python's "Inspect the destination ..." message; the partial file is never deleted.
* `src/sandbox/mount_patterns.rs` `S3FilesOptions.extra_options`: serialized as an array of pairs; Python writes a dict. Now an ordered object (array still accepted on read).

* `src/sandbox/sandboxes/pinned_ops.rs` (new, unix) and `unix_local.rs` `mkdir`, `ls`, `rm`, `mv`, `same_file`: see "Race closing".

## Race closing (check-then-use)
The earlier mitigation only compared the parent's resolution before and after a window, so a swap after the second resolution still redirected `rm`, `mv`, `mkdir`, `ls` and `same_file`. Now each operation resolves the entry's parent after validation, re-authorizes that resolution, and opens it once with `rustix` descriptor-relative `openat(O_NOFOLLOW | O_DIRECTORY)` walked from `/` (`pin_parent`, `pinned_ops::open_dir_pinned`). The entry is then inspected and changed relative to that descriptor (`statat`, `unlinkat`, `renameat`, `mkdirat`, a descriptor-based recursive remove, directory listing). A component swapped before the open is refused (the walk meets a link in a canonical path); one swapped afterwards cannot matter. `mkdir` pins the deepest existing ancestor and creates the missing components below it, descending through each with `O_NOFOLLOW`. A leaf link is never followed, except that `rm` still follows an alias that stays inside the workspace and `mkdir -p` still accepts a link that resolves to an authorized directory (existing behaviour). `read_file`, `write_file` and exclusive create keep their open-then-verify check. Non-unix behaviour is unchanged (the module is `cfg(unix)`).

Race tests: the earlier-window ports are `py_core6_modelsrt_unix_local_swaps.rs` (swap between validation and use, for all six operations). New hidden hook `UnixLocalSandboxSession::with_after_pin_hook` runs after the parent is pinned and before the entry is acted on; `py_core8_r8g2_pinned_ops.rs` swaps the parent or the leaf there for rm, rmtree, mv and same_file and checks the outside entry is untouched and the pinned directory's entry is the one affected. No hook exists for `mkdir` and `ls` after the pin because their descriptor is opened inside one blocking call with no await point to inject at; the swap before the open is covered by the after-validate ports.

## Pre-stop hooks and `aclose`
No Rust equivalent exists: sessions have no pre-stop hook registry and no `aclose`, and the `Dependencies` container is not wired into session teardown (teardown is `SandboxSessionGuard::close` then `client.delete`). The portable lifecycle half (stop and shutdown once) is already ported as `py_sandbox_a_runtime_2::a_completed_run_stops_and_shuts_down_the_session_once`. The pre-stop and dependency-close redaction cases stay python-only as a documented feature gap, as do the duplicate-agent-name handoff graph and the sink-bound-session decisions.

## Breaking changes
* `SandboxSessionState::from_json_str` error text changed (code unchanged).
* The persisted S3 Files pattern payload writes `extra_options` as an object.
* A failed exclusive create reports a different error code and message.
* Sink/teardown log text changed.

## Gates
* New binaries: 19 tests pass. Regression: `sandbox::` lib (295), `sandbox_tests` (65), `sandbox_core_parity_tests`, `py_sandbox_a_runtime_4`, `py_sandbox_a_runtime_unix_local`, `py_sandbox_a_unix_local_{1,2,5}`, `py_sandbox_b_{sinks,session_sinks,sink_traces,jsonl_sink}`, `py_sandbox_a_entries_{1..4}`, `py_core8_r8g1_manifest` all pass.
* Clippy: default-feature `--lib` plus my targets clean. `--all-features` fails in `src/memory/advanced_sqlite_session.rs` (needless_borrow, not mine); with that one lint allowed, lib plus my targets are clean. Default `--all-targets` fails only in another agent's `tests/py_core8_r8g6_docstrings.rs`.
* rustfmt clean on my files. `src/sandbox/runtime.rs` has a pre-existing format diff that I did not reformat.

## Unresolved questions / concerns
* Duplicate agent names in a sandbox handoff graph are rejected in Rust; Python gives them resume keys (about 8 runtime cases). Design divergence, kept python-only.
* Custom `impl MountStrategy` reporting no credential authority is trusted (mount-security rows 15-40); no provenance allowlist.
* No pre-stop hooks or `aclose` equivalent exist in Rust (about 10 runtime cases); a feature gap rather than a valid Python-only reason.
* A callback sink cannot receive the bound session (`test_callback_sink_receives_bound_inner_session`); feature gap.
* Residual: `read_file`, `write_file` and exclusive create still use open-then-verify rather than a descriptor walk (an A-to-B-to-A swap before the verify is not seen).
* Linux lease tests are not runnable here; unported.
* `test_snapshot` #24 (`system_state_preserved`) may be coverable by `py_sandbox_a_runtime_6` but I did not verify an exact match, so it stays python-only.
* Ledger-expansion mismatch: my per-parameter reading may differ from the ledger for `test_workspace_jsonl_sink` and `test_unix_local_file_io`; totals follow the ledger.

Status: DONE_WITH_CONCERNS
Summary: 22 files audited, 371 python-only before; 65 ported, 2 covered, 304 remain; 5 source bugs fixed plus the descriptor-pinned unix-local operations.
Concerns: the unresolved items above (races, design divergences, feature gaps).
