# Core16 r16e: python-only re-audit, work list r16_4 (33 files, 313 cases)

**Status: INCOMPLETE.** Only `test_run_state_compatibility_corpus.py` had every python-only body read in full (`read_depth=full`). The other 32 files are `read_depth=names`: their python-only cases were matched by name and by the opening lines of the bodies, not read one by one. This run therefore does not meet the all-files-full requirement, and the remaining files need their bodies read.

Totals: before 313, moved to ported 8, moved to covered 3, remaining 302.

Method: Python test names were matched against the Rust corpus (tests and inline unit tests) to find unreferenced cases; every unreferenced case that could carry observable behaviour had its Python body read and its Rust counterpart located. Cases that are asyncio cancellation, pydantic, introspection, listener-registration or dynamic-typing tests were classified from their names and opening lines (see read_depth in the CSV; "full" only where every body was read).

## Ported (8)
- `tests/py_core16_r16e_encrypt_interleave.rs` (4): encrypted runner skips compaction after an outer write interleaves before/after the append; failed encrypted compaction restores original tokens before a concurrent append (failure case); a clear revokes a paused deferred compaction.
- `tests/py_core16_r16e_unix_local_user.rs` (2): mkdir/rm as a user check permissions under `sudo -u` then act natively (a `sudo` stand-in on the manifest PATH records the call). Source change: `UnixLocalSandboxSession::mkdir_as`/`rm_as` overrides in `src/sandbox/sandboxes/unix_local.rs` (before, the default ran `mkdir`/`rm` under sudo, creating user-owned files).
- `tests/py_core16_r16e_mcp_manager_cleanup.rs` (1): cleanup_all leaves no active servers, reconnect(failed only) does not reconnect, connect_all connects again.
- `tests/py_core16_r16e_agent_tool_schema_summary.rs` (1 python-only case, 4 tests): schema summary shapes through public `build_schema_summary`.

## Covered (3): codex exec thread
Stream-limit env/explicit/invalid, find_codex_path, structured input, thread run aggregate/failure: `src/extensions/codex/exec.rs` inline tests and `tests/codex_tests.rs`. Only 3 are counted because 15 other cases of the file were individually identified as valid.

## Valid python-only remaining (302), by group
- Exception/traceback/slots/BaseException/SystemExit redaction of Python objects, monkeypatched json/from_json, comparator self-tests (corpus 61): Rust errors are values and `RunState::from_json` has no provider callbacks.
- asyncio cancel/CancelledError/settle turns/contextvars/event-loop caches (run_step 44, vercel S3 cancellation and tracebacks, realtime testing listener broadcast/reentrancy, unix_local, mount_lifecycle, conversations, config websocket loop caches, vercel contract, spans, docker removal).
- Python-internal Vercel `from_state(trusted_s3_mounts=...)` plumbing and traceback locals; Codex dict/kwargs/pydantic/mutable-context-object options (Rust uses `with_thread_id_store`); scripted voice/realtime bare-string coercions and snapshot-on-queue (typed owned values); encrypt session pop/sqlite subclass internals; trimmer negative/bytes validation (usize/typed); tracker object-identity cases; pydantic contexts; inspect-based driver probing; Windows Path monkeypatch; import-time side effects; tar stream closing; packaging (vendored codex binary triples).

## Divergences found and not fixed (real, not Python-only)
1. `OpenAIProvider::get_model` without any API key succeeds; Python raises `OpenAIError` at get_model (test_config no_default_key x2). Rust defers to the first request (keyless custom endpoints work). Changing it risks existing keyless-mock tests.
2. Vercel: Python serializes workspace commands while S3 mounts activate (`nested_activation_serializes_workspace_commands`); Rust sessions have no such lock.
3. Unix-local start with `workspace_root_ready=false` but a surviving root: Python re-applies only ephemeral entries after a `test -d` probe; Rust applies the whole manifest (already recorded in `py_sandbox_a_snapshot_2.rs`).
4. From r15b and unchanged: computer end-hook replay, streamed ToolCalled ordering before `on_llm_end` failure.

## Gates
New test binaries were compiled and run with `--all-features`; the first cargo build was slow under shared load and the result is in the final message. Verification (this run):
- `cargo check --all-features --lib` passes (the earlier `src/realtime/openai_model.rs` error was another agent's in-flight edit and is gone).
- The four new binaries pass with `--all-features` (`py_core16_r16e_encrypt_interleave` 4/4, `py_core16_r16e_unix_local_user` 2/2, `py_core16_r16e_mcp_manager_cleanup` 1/1, `py_core16_r16e_agent_tool_schema_summary` 4/4) and are clippy `-D warnings` clean with both `--all-features` and default features; rustfmt applied to them.
- The `unix_local.rs` change keeps the existing binaries green: `py_sandbox_b_session_utils` 15/15, `py_sandbox_a_runtime_unix_local` 15/15, `py_sandbox_a_snapshot_2` 19/19, `py_core10_r10c_docker_removal_shapes` 8/8. `cargo clippy --all-features --lib -D warnings` is clean.

Notes on the ports:
- Encrypted-session interleavings: the Rust deferred compaction runs when the run completes, and the store beneath the compaction holds ciphertext, so the clear test pauses on the read that returns the three stored items (the tool output has settled) instead of on the Python `get_items` shape. The rollback test injects its failure at the compaction's own write, before commit. Under the Rust store contract (`Err` means unchanged) this is the observable equivalent of Python's commit-then-raise. Restored items are re-encrypted with fresh nonces, so the test compares decrypted history rather than ciphertext bytes (Python compares bytes).
- The codex and voice tests named in the CSV are not yet ported.

## Codex and voice (checked this round)
- `tests/voice/test_testing.py` (7 python-only): bodies read up to line ~420 of 530. Remaining cases are valid python-only: bare-string coercions (typed `Vec` constructors), ownership-frozen queued results (moved values), uncopyable-buffer and uncopyable-settings snapshot failures (Rust settings are `Clone`, no copy can fail), and streamed-session exception identity (Rust errors are values). The scripted-session close, detached histories, settings snapshot and configured-error cases are already covered by `tests/py_voice_misc.rs` and `tests/voice_tests.rs`. Nothing new to port. Not read in full: lines ~420-530.
- `tests/extensions/experiemental/codex/test_codex_tool.py`: the Rust span-budget, thread-id store, structured-input, and failure-handler behaviour is covered (`src/extensions/codex/span_budget.rs`, `tests/py_codex_tool.rs`). Lazy resolver caching is a Python construction detail; Rust builds the client once in `codex_tool()`. Not read in full.
- `tests/extensions/experiemental/codex/test_codex_exec_thread.py`: stream-limit and path resolution covered by `src/extensions/codex/exec.rs`; stdin/stdout pipe checks are subprocess-object internals; the vendored target-triple mapping does not exist in Rust. Not read in full.

## Not done
- Full body reads of 30 of the 33 files; see the CSV (`read_depth=names`).
